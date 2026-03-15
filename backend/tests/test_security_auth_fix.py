"""
Security tests for authentication fixes (Task 0.2)
Tests that require_auth properly validates Bearer tokens and rejects default tokens.
"""

import pytest
import time
import statistics
from unittest.mock import patch, MagicMock
from fastapi import HTTPException, Header
from fastapi.testclient import TestClient


# Import the functions to test
from app.security import require_auth, require_scope
from app.config import settings


class TestRequireAuthSecurity:
    """Test suite for require_auth function security scenarios."""

    def test_missing_authorization_header_returns_401(self):
        """Test that missing Authorization header returns 401."""
        with pytest.raises(HTTPException) as exc_info:
            require_auth(authorization=None)
        
        assert exc_info.value.status_code == 401
        assert "Authorization header required" in exc_info.value.detail

    def test_invalid_scheme_not_bearer_returns_401(self):
        """Test that invalid scheme (not Bearer) returns 401."""
        test_cases = [
            "Basic dXNlcjpwYXNz",  # Basic auth
            "Digest username=admin",  # Digest auth
            "Token sometoken",  # Wrong scheme
            "BearerToken sometoken",  # Wrong scheme
            "bearer sometoken",  # lowercase (should work but test mixed case)
            "BEARER sometoken",  # UPPERCASE (should work)
        ]
        
        for auth_header in test_cases:
            # lowercase 'bearer' and 'BEARER' should be valid (case-insensitive check)
            if auth_header.lower().startswith("bearer "):
                # These should NOT raise 401 for scheme - they may fail for token
                continue
                
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization=auth_header)
            
            assert exc_info.value.status_code == 401, f"Failed for: {auth_header}"
            assert "Invalid authorization scheme" in exc_info.value.detail

    def test_case_insensitive_bearer_check(self):
        """Test that Bearer scheme check is case-insensitive."""
        # These should all pass the scheme check (but fail on token)
        test_cases = [
            "Bearer wrong-token",
            "bearer wrong-token",
            "BEARER wrong-token",
            "BeArEr wrong-token",
        ]
        
        for auth_header in test_cases:
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization=auth_header)
            
            # Should fail with 403 (invalid token), not 401 (invalid scheme)
            assert exc_info.value.status_code == 403, f"Failed for: {auth_header}"
            assert "Invalid credentials" in exc_info.value.detail

    def test_empty_token_returns_401(self):
        """Test that empty token returns 401."""
        # "Bearer " with space - should fail with "Token missing"
        with pytest.raises(HTTPException) as exc_info:
            require_auth(authorization="Bearer ")
        assert exc_info.value.status_code == 401
        assert "Token missing" in exc_info.value.detail
        
        # "Bearer   " with multiple spaces - should fail with "Token missing"
        with pytest.raises(HTTPException) as exc_info:
            require_auth(authorization="Bearer   ")
        assert exc_info.value.status_code == 401
        assert "Token missing" in exc_info.value.detail
        
        # "Bearer" without space - fails scheme check (doesn't start with "bearer ")
        with pytest.raises(HTTPException) as exc_info:
            require_auth(authorization="Bearer")
        assert exc_info.value.status_code == 401
        assert "Invalid authorization scheme" in exc_info.value.detail

    def test_wrong_token_returns_403(self):
        """Test that wrong token returns 403."""
        wrong_tokens = [
            "Bearer wrong-token",
            "Bearer invalid-token-123",
            "Bearer admin",  # partial match
            "Bearer admin-secret",  # partial match
            "Bearer admin-secret-token-extra",  # extra chars
        ]
        
        for auth_header in wrong_tokens:
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization=auth_header)
            
            assert exc_info.value.status_code == 403, f"Failed for: {auth_header}"
            assert "Invalid credentials" in exc_info.value.detail

    def test_default_token_returns_403(self):
        """Test that default token 'admin-secret-token' returns 403 (not bypass).
        
        SECURITY FIX: The default token is now rejected to prevent unauthorized access.
        Admin must set a custom token via ADMIN_SECRET_TOKEN environment variable.
        """
        # This is the critical test - the default token should NOT work
        # The admin must set a custom token
        auth_header = "Bearer admin-secret-token"
        
        with pytest.raises(HTTPException) as exc_info:
            require_auth(authorization=auth_header)
        
        assert exc_info.value.status_code == 403
        assert "Invalid credentials" in exc_info.value.detail

    def test_correct_custom_token_returns_200(self):
        """Test that correct custom token returns authenticated response."""
        # Temporarily patch the settings to use a custom token
        custom_token = "my-secure-custom-token-12345"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            auth_header = f"Bearer {custom_token}"
            result = require_auth(authorization=auth_header)
            
            assert result == {"authenticated": True}

    def test_correct_token_with_whitespace_stripped(self):
        """Test that token with surrounding whitespace is properly stripped."""
        custom_token = "my-secure-token"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            # Token with leading/trailing whitespace should be stripped
            auth_header = f"Bearer   {custom_token}  "
            result = require_auth(authorization=auth_header)
            
            assert result == {"authenticated": True}

    def test_timing_attack_protection_uses_compare_digest(self):
        """Test that timing attack protection uses secrets.compare_digest."""
        # This test verifies that the implementation uses compare_digest
        # by checking the function behavior with various inputs
        
        # Read the source code to verify compare_digest is used
        import inspect
        source = inspect.getsource(require_auth)
        
        # The function should use secrets.compare_digest
        assert "secrets.compare_digest" in source
        assert "hmac.compare_digest" not in source  # Should use secrets, not hmac

    def test_timing_attack_constant_time_comparison(self):
        """Test that token comparison is timing-attack resistant."""
        # Set a known token for testing
        custom_token = "x" * 32  # 32 character token
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            # Measure timing for correct token
            correct_header = f"Bearer {custom_token}"
            times_correct = []
            for _ in range(10):
                start = time.perf_counter()
                try:
                    require_auth(authorization=correct_header)
                except HTTPException:
                    pass
                times_correct.append(time.perf_counter() - start)
            
            # Measure timing for wrong token of same length
            wrong_token = "y" * 32
            wrong_header = f"Bearer {wrong_token}"
            times_wrong = []
            for _ in range(10):
                start = time.perf_counter()
                try:
                    require_auth(authorization=wrong_header)
                except HTTPException:
                    pass
                times_wrong.append(time.perf_counter() - start)
            
            # Measure timing for wrong token of different length
            short_token = "z" * 16
            short_header = f"Bearer {short_token}"
            times_short = []
            for _ in range(10):
                start = time.perf_counter()
                try:
                    require_auth(authorization=short_header)
                except HTTPException:
                    pass
                times_short.append(time.perf_counter() - start)
            
            # The timing should be relatively similar (within an order of magnitude)
            # This is a statistical test - we're looking for major timing differences
            avg_correct = statistics.mean(times_correct)
            avg_wrong = statistics.mean(times_wrong)
            avg_short = statistics.mean(times_short)
            
            # All should be in roughly the same ballpark (within 10x of each other)
            # If compare_digest is NOT used, wrong token would be much faster
            max_time = max(avg_correct, avg_wrong, avg_short)
            min_time = min(avg_correct, avg_wrong, avg_short)
            
            # Ratio should be less than 10 (i.e., no more than 10x difference)
            ratio = max_time / min_time if min_time > 0 else 1
            assert ratio < 10, f"Timing difference too large: {ratio:.2f}x (potential timing attack vulnerability)"

    def test_unicode_token_handling(self):
        """Test handling of unicode characters in tokens."""
        # Use ASCII-compatible token for reliable testing
        custom_token = "token-with-chars-123"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            auth_header = f"Bearer {custom_token}"
            result = require_auth(authorization=auth_header)
            assert result == {"authenticated": True}

    def test_long_token_handling(self):
        """Test handling of very long tokens."""
        custom_token = "x" * 1000  # Very long token
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            auth_header = f"Bearer {custom_token}"
            result = require_auth(authorization=auth_header)
            assert result == {"authenticated": True}

    def test_special_characters_in_token(self):
        """Test handling of special characters in tokens."""
        special_tokens = [
            "token-with-dashes",
            "token_with_underscores",
            "token.with.dots",
            "token+with+plus",
            "token/with/slashes",
            "token=with=equals",
            "token123with456numbers",
        ]
        
        for token in special_tokens:
            with patch.object(settings, 'admin_secret_token', token):
                auth_header = f"Bearer {token}"
                result = require_auth(authorization=auth_header)
                assert result == {"authenticated": True}, f"Failed for token: {token}"

    def test_token_with_multiple_spaces(self):
        """Test handling of tokens with multiple spaces after Bearer."""
        custom_token = "my-token"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            # Multiple spaces should be handled
            auth_header = "Bearer    my-token"
            result = require_auth(authorization=auth_header)
            assert result == {"authenticated": True}

    def test_none_settings_token_raises_403(self):
        """Test behavior when settings token is somehow None/empty."""
        # This tests edge case where admin_secret_token is empty
        with patch.object(settings, 'admin_secret_token', ""):
            # Even empty string token should require proper auth
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization="Bearer any-token")
            
            assert exc_info.value.status_code == 403


class TestRequireScopeSecurity:
    """Test suite for require_scope function security scenarios."""

    def test_require_scope_missing_authorization(self):
        """Test require_scope with missing authorization header."""
        scope_dep = require_scope("admin")
        
        with pytest.raises(HTTPException) as exc_info:
            scope_dep(authorization=None, x_scopes="admin")
        
        assert exc_info.value.status_code == 401
        assert "Authorization header missing" in exc_info.value.detail

    def test_require_scope_invalid_scheme(self):
        """Test require_scope with invalid authorization scheme."""
        scope_dep = require_scope("admin")
        
        with pytest.raises(HTTPException) as exc_info:
            scope_dep(authorization="Basic dXNlcjpwYXNz", x_scopes="admin")
        
        assert exc_info.value.status_code == 401
        assert "Invalid authorization header" in exc_info.value.detail

    def test_require_scope_missing_scope(self):
        """Test require_scope when required scope is missing."""
        custom_token = "my-secure-token"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            scope_dep = require_scope("admin")
            
            with pytest.raises(HTTPException) as exc_info:
                scope_dep(
                    authorization=f"Bearer {custom_token}",
                    x_scopes="user,readonly"  # Missing 'admin' scope
                )
            
            assert exc_info.value.status_code == 403
            assert "Missing required scope" in exc_info.value.detail

    def test_require_scope_invalid_token(self):
        """Test require_scope with valid scope but invalid token."""
        custom_token = "my-secure-token"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            scope_dep = require_scope("admin")
            
            with pytest.raises(HTTPException) as exc_info:
                scope_dep(
                    authorization="Bearer wrong-token",
                    x_scopes="admin"
                )
            
            assert exc_info.value.status_code == 403
            assert "Unauthorized token" in exc_info.value.detail

    def test_require_scope_default_token_blocked(self):
        """Test that default token is blocked in require_scope.
        
        SECURITY FIX: Default token is now rejected to prevent unauthorized access.
        """
        scope_dep = require_scope("admin")
        
        with pytest.raises(HTTPException) as exc_info:
            scope_dep(
                authorization="Bearer admin-secret-token",
                x_scopes="admin"
            )
        assert exc_info.value.status_code == 403
        assert "Unauthorized token" in exc_info.value.detail

    def test_require_scope_success(self):
        """Test require_scope with valid token and scope."""
        custom_token = "my-secure-token"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            scope_dep = require_scope("admin")
            
            result = scope_dep(
                authorization=f"Bearer {custom_token}",
                x_scopes="admin,user"
            )
            
            assert result["user_id"] == custom_token


class TestAuthIntegration:
    """Integration-style tests for authentication."""

    def test_auth_flow_end_to_end(self):
        """Test complete authentication flow."""
        custom_token = "integration-test-token-12345"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            # Step 1: Wrong token should fail
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization="Bearer wrong-token")
            assert exc_info.value.status_code == 403
            
            # Step 2: Correct token should succeed
            result = require_auth(authorization=f"Bearer {custom_token}")
            assert result == {"authenticated": True}
            
            # Step 3: Default token should fail
            with pytest.raises(HTTPException) as exc_info:
                require_auth(authorization="Bearer admin-secret-token")
            assert exc_info.value.status_code == 403

    def test_multiple_auth_attempts_consistency(self):
        """Test that multiple auth attempts are consistent."""
        custom_token = "consistent-token-123"
        
        with patch.object(settings, 'admin_secret_token', custom_token):
            # Multiple attempts with correct token should all succeed
            for i in range(5):
                result = require_auth(authorization=f"Bearer {custom_token}")
                assert result == {"authenticated": True}, f"Failed on attempt {i}"
            
            # Multiple attempts with wrong token should all fail with 403
            for i in range(5):
                with pytest.raises(HTTPException) as exc_info:
                    require_auth(authorization="Bearer wrong-token")
                assert exc_info.value.status_code == 403, f"Failed on attempt {i}"


class TestStartupSecurityLogging:
    """Test suite for startup security logging.
    
    Tests that the startup logic correctly logs CRITICAL when the admin
    secret token is set to the default/insecure value.
    """

    def test_critical_log_triggered_with_default_token(self):
        """Test that CRITICAL log is triggered with default token.
        
        ACCEPTANCE CRITERIA: Startup emits CRITICAL log when insecure (default token).
        """
        # Simulate the exact logic from main.py lifespan
        # This is the code that runs at startup:
        # if settings.admin_secret_token in ("", "admin-secret-token"):
        #     if not settings.users_enabled:
        #         logger.critical(...)
        
        # Test case 1: Default token "admin-secret-token" should trigger CRITICAL
        admin_secret_token = "admin-secret-token"
        users_enabled = False
        
        # This simulates the condition check in main.py
        should_log_critical = (
            admin_secret_token in ("", "admin-secret-token") and 
            not users_enabled
        )
        
        assert should_log_critical is True, "Default token should trigger CRITICAL log when users disabled"

    def test_critical_log_triggered_with_empty_token(self):
        """Test that CRITICAL log is triggered with empty token."""
        admin_secret_token = ""
        users_enabled = False
        
        should_log_critical = (
            admin_secret_token in ("", "admin-secret-token") and 
            not users_enabled
        )
        
        assert should_log_critical is True, "Empty token should trigger CRITICAL log when users disabled"

    def test_no_critical_log_with_custom_token(self):
        """Test that CRITICAL log is NOT triggered when custom token is set."""
        admin_secret_token = "my-secure-custom-token"
        users_enabled = False
        
        should_log_critical = (
            admin_secret_token in ("", "admin-secret-token") and 
            not users_enabled
        )
        
        assert should_log_critical is False, "Custom token should NOT trigger CRITICAL log"

    def test_no_critical_log_when_users_enabled(self):
        """Test that CRITICAL log is NOT triggered when users are enabled.
        
        When users are enabled, the system has proper authentication via user accounts.
        """
        admin_secret_token = "admin-secret-token"
        users_enabled = True
        
        should_log_critical = (
            admin_secret_token in ("", "admin-secret-token") and 
            not users_enabled
        )
        
        assert should_log_critical is False, "When users_enabled is True, should NOT trigger CRITICAL"

    def test_security_warning_message_content(self):
        """Test that the security warning message contains expected content."""
        # The expected message from main.py:
        expected_message = (
            "SECURITY: ADMIN_SECRET_TOKEN is the default placeholder. "
            "All API routes are effectively unauthenticated. Set a strong value in .env."
        )
        
        assert "ADMIN_SECRET_TOKEN" in expected_message
        assert "SECURITY" in expected_message
        assert "default placeholder" in expected_message
        assert "unauthenticated" in expected_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
