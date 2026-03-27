"""
Test configuration — sets required environment variables BEFORE any app
modules are imported (conftest.py is loaded prior to test collection).

This prevents the Settings model_validator from rejecting default credential
values during testing; tests must not use production credentials.

Python version note: this project targets Python 3.11 (see Dockerfile).
When run under Python 3.12+ the `unstructured` package has incompatible native
extensions that crash at the C level during import.  We stub it out here so
that test modules that transitively import `app.main` (which pulls in
`document_processor` → `unstructured`) can be collected and executed.
The stub is applied BEFORE any test module is collected, so it is safe to
apply unconditionally on Python 3.12+.
"""
import sys
import types
import os

# ---------------------------------------------------------------------------
# 1. Credentials — must be set before any app module is imported
# ---------------------------------------------------------------------------
os.environ.setdefault("ADMIN_SECRET_TOKEN", "test-admin-secret-token-for-ci-only-not-production")
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-jwt-secret-key-for-ci-only-not-production-minimum-32-characters",
)
os.environ.setdefault("HEALTH_CHECK_API_KEY", "test-health-api-key-for-ci-only")

# ---------------------------------------------------------------------------
# 2. Stub `unstructured` on Python 3.12+ (native extensions crash at C level)
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 12) and "unstructured" not in sys.modules:
    def _make_pkg_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]  # marks as a package
        mod.__spec__ = None  # type: ignore[assignment]
        return mod

    # All sub-modules referenced directly or transitively by the app
    _stub_names = [
        "unstructured",
        "unstructured.chunking",
        "unstructured.chunking.title",
        "unstructured.partition",
        "unstructured.partition.auto",
        "unstructured.partition.text",
        "unstructured.documents",
        "unstructured.documents.elements",
    ]
    for _name in _stub_names:
        _mod = _make_pkg_stub(_name)
        sys.modules[_name] = _mod
        # Attach as an attribute on the parent so `pkg.sub` attribute access works
        if "." in _name:
            _parent, _child = _name.rsplit(".", 1)
            setattr(sys.modules[_parent], _child, _mod)

    # Stub callable symbols used at module-level in the app
    sys.modules["unstructured.partition.auto"].partition = lambda *a, **kw: []  # type: ignore[attr-defined]
    sys.modules["unstructured.partition.text"].partition_text = lambda *a, **kw: []  # type: ignore[attr-defined]
    sys.modules["unstructured.chunking.title"].chunk_by_title = lambda *a, **kw: []  # type: ignore[attr-defined]

    # Element base class stub (used by chunking.py for isinstance checks)
    class _ElementStub:  # noqa: N801
        """Minimal stub for unstructured.documents.elements.Element."""
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["unstructured.documents.elements"].Element = _ElementStub  # type: ignore[attr-defined]
