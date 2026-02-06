"""
Background task processor for document ingestion.

Provides BackgroundProcessor class that manages an asyncio queue for processing
documents with retry logic and graceful shutdown.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .document_processor import DocumentProcessor, DocumentProcessingError
from .vector_store import VectorStore
from .embeddings import EmbeddingService
from .maintenance import MaintenanceService


logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """
    Represents a task item in the background queue.

    Attributes:
        file_path: Path to the file to process
        attempt: Current attempt count (starts at 1)
    """
    file_path: str
    attempt: int = 1


class BackgroundProcessor:
    """
    Background task processor using asyncio.Queue for document ingestion.

    Manages a worker loop that processes files using DocumentProcessor with
    retry logic (max 3 attempts). Failed tasks are requeued with exponential
    backoff delay. Provides graceful shutdown via asyncio.Event.

    Attributes:
        max_retries: Maximum number of retry attempts per task
        retry_delay: Base delay in seconds between retries (doubles each attempt)
        queue: asyncio.Queue holding TaskItem objects
        shutdown_event: asyncio.Event for graceful shutdown
        processor: DocumentProcessor instance for file processing
        _worker_task: Reference to the worker coroutine
        _running: Boolean indicating if processor is active
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        maintenance_service: Optional[MaintenanceService] = None,
    ):
        """
        Initialize the background processor.

        Args:
            max_retries: Maximum retry attempts for failed tasks (default: 3)
            retry_delay: Base delay in seconds between retries (default: 1.0)
            chunk_size: Target chunk size for DocumentProcessor
            chunk_overlap: Overlap between chunks for DocumentProcessor
            vector_store: VectorStore instance for document storage
            embedding_service: EmbeddingService instance for generating embeddings
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.queue: asyncio.Queue[TaskItem] = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self.maintenance_service = maintenance_service

    async def start(self) -> None:
        """
        Start the background processor worker loop.

        Creates and starts the worker task that processes items from the queue.
        Safe to call multiple times - will not create duplicate workers.
        """
        if self._running:
            logger.warning("Background processor is already running")
            return

        self._running = True
        self.shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Background processor started")

    async def stop(self) -> None:
        """
        Stop the background processor gracefully.

        Signals the worker to shut down and waits for it to complete.
        Pending queue items are not processed after shutdown is initiated.
        """
        if not self._running:
            logger.warning("Background processor is not running")
            return

        logger.info("Stopping background processor...")
        self.shutdown_event.set()

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Worker task did not stop gracefully, cancelling...")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

        self._running = False
        logger.info("Background processor stopped")

    async def enqueue(self, file_path: str) -> None:
        """
        Add a file to the processing queue.

        Args:
            file_path: Path to the file to process

        Note:
            If the processor is not running, the item will still be queued
            and processed when start() is called.
        """
        if self.maintenance_service and self.maintenance_service.get_flag().enabled:
            raise DocumentProcessingError("Maintenance mode prevents enqueueing")
        task = TaskItem(file_path=file_path, attempt=1)
        await self.queue.put(task)
        logger.debug(f"Enqueued file: {file_path}")

    async def _worker_loop(self) -> None:
        """
        Main worker loop that processes items from the queue.

        Continuously processes tasks until shutdown_event is set.
        Handles retries with exponential backoff on failure.
        """
        while not self.shutdown_event.is_set():
            try:
                # Wait for task with timeout to check shutdown periodically
                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            if task is None:
                continue

            await self._process_task(task)
            self.queue.task_done()

    async def _process_task(self, task: TaskItem) -> None:
        """
        Process a single task with retry logic.

        Args:
            task: TaskItem containing file path and attempt count

        On failure, requeues the task with incremented attempt count
        and exponential backoff delay if retries remain.
        """
        logger.info(f"Processing file: {task.file_path} (attempt {task.attempt})")

        try:
            await self.processor.process_file(task.file_path)
            logger.info(f"Successfully processed: {task.file_path}")

        except DocumentProcessingError as e:
            logger.error(f"Processing error for {task.file_path}: {e}")
            await self._handle_failure(task, str(e))

        except Exception as e:
            logger.error(f"Unexpected error processing {task.file_path}: {e}")
            await self._handle_failure(task, str(e))

    async def _handle_failure(self, task: TaskItem, error_message: str) -> None:
        """
        Handle task failure with retry logic.

        Args:
            task: The failed task
            error_message: Error message from the failure

        Requeues the task with incremented attempt count if retries remain,
        otherwise logs the permanent failure.
        """
        if task.attempt < self.max_retries:
            # Calculate exponential backoff delay
            delay = self.retry_delay * (2 ** (task.attempt - 1))
            logger.warning(
                f"Task failed for {task.file_path}, "
                f"retrying in {delay}s (attempt {task.attempt + 1}/{self.max_retries})"
            )

            # Wait before requeuing
            await asyncio.sleep(delay)

            # Requeue with incremented attempt count
            new_task = TaskItem(
                file_path=task.file_path,
                attempt=task.attempt + 1
            )
            await self.queue.put(new_task)
        else:
            logger.error(
                f"Task permanently failed for {task.file_path} "
                f"after {self.max_retries} attempts: {error_message}"
            )

    @property
    def is_running(self) -> bool:
        """Check if the background processor is currently running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get the current number of items in the queue."""
        return self.queue.qsize()
