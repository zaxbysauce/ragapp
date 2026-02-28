"""Query transformation service for step-back prompting."""
import logging
from typing import List

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Transforms queries using step-back prompting for broader retrieval."""
    
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client
    
    async def transform(self, query: str) -> List[str]:
        """
        Transform a query into [original, step_back] versions.
        
        Args:
            query: Original user query
            
        Returns:
            List containing original query and step-back variant.
            On LLM error, returns [original] only.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a query transformation assistant. Your task is to generate "
                        "a broader, more general version of the user's question that captures "
                        "the high-level intent and underlying concepts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate a broader, more general version of this question that "
                        f"captures the underlying concept:\n"
                        f"Original: {query}\n"
                        f"Step-back:"
                    )
                }
            ]
            
            step_back = await self._llm_client.chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            
            if step_back and step_back.strip():
                logger.debug("Query transformation: original='%s', step_back='%s'", query, step_back)
                return [query, step_back.strip()]
            else:
                logger.warning("Query transformation returned empty response, using original only")
                return [query]
                
        except Exception as e:
            logger.warning("Query transformation failed: %s, using original query only", e)
            return [query]
