"""Simple secret manager helper."""

import os
from typing import Optional


class SecretManagerError(RuntimeError):
    """Raised when a secret cannot be retrieved."""


class SecretManager:
    """Helper for retrieving secrets such as audit HMAC keys."""

    def __init__(self) -> None:
        self.default_hmac_version = os.getenv("AUDIT_HMAC_KEY_VERSION", "v1")
        self.default_aes_version = os.getenv("AES_KEY_VERSION", "v1")

    def get_hmac_key(self, version: Optional[str] = None) -> tuple[bytes, str]:
        """Return the HMAC key and version."""
        version = (version or self.default_hmac_version).lower()
        env_name = f"AUDIT_HMAC_KEY_{version.upper()}"
        key = os.getenv(env_name) or os.getenv("AUDIT_HMAC_KEY")
        if not key:
            raise SecretManagerError(
                f"HMAC key for version '{version}' is not configured"
            )
        return key.encode("utf-8"), version

    def get_aes_key(self, version: Optional[str] = None) -> tuple[bytes, str]:
        """Return the AES key and version."""
        version = (version or self.default_aes_version).lower()
        env_name = f"AES_KEY_{version.upper()}"
        key = os.getenv(env_name) or os.getenv("AES_KEY")
        if not key:
            raise SecretManagerError(
                f"AES key for version '{version}' is not configured"
            )
        return key.encode("utf-8"), version
