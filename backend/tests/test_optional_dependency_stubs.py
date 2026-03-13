"""Shared helpers for stubbing optional backend test dependencies."""

import sys
import types


def install_optional_dependency_stubs() -> None:
    """Stub heavy optional imports used by backend services in unit tests."""
    if "lancedb" not in sys.modules:
        sys.modules["lancedb"] = types.ModuleType("lancedb")

    if "pyarrow" not in sys.modules:
        sys.modules["pyarrow"] = types.ModuleType("pyarrow")

    if "unstructured" in sys.modules:
        return

    unstructured = types.ModuleType("unstructured")
    unstructured.partition = types.ModuleType("unstructured.partition")
    unstructured.partition.auto = types.ModuleType("unstructured.partition.auto")
    unstructured.partition.auto.partition = lambda *args, **kwargs: []
    unstructured.chunking = types.ModuleType("unstructured.chunking")
    unstructured.chunking.title = types.ModuleType("unstructured.chunking.title")
    unstructured.chunking.title.chunk_by_title = lambda *args, **kwargs: []
    unstructured.documents = types.ModuleType("unstructured.documents")
    unstructured.documents.elements = types.ModuleType("unstructured.documents.elements")
    unstructured.documents.elements.Element = type("Element", (), {})
    sys.modules["unstructured"] = unstructured
    sys.modules["unstructured.partition"] = unstructured.partition
    sys.modules["unstructured.partition.auto"] = unstructured.partition.auto
    sys.modules["unstructured.chunking"] = unstructured.chunking
    sys.modules["unstructured.chunking.title"] = unstructured.chunking.title
    sys.modules["unstructured.documents"] = unstructured.documents
    sys.modules["unstructured.documents.elements"] = unstructured.documents.elements
