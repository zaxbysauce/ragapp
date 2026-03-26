"""Tests for password_strength_check function in auth_service.py"""

import pytest
import sys
import os

# Add the backend directory to the path so we can import the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.auth_service import password_strength_check


class TestPasswordStrengthCheck:
    """Test cases for password_strength_check function."""

    def test_password_too_short_7_chars(self):
        """Test that 7-character password raises ValueError about length."""
        # "passwor" is exactly 7 characters
        with pytest.raises(ValueError) as exc_info:
            password_strength_check("passwor")
        assert "8 characters" in str(exc_info.value)

    def test_password_no_uppercase(self):
        """Test that password without uppercase raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            password_strength_check("password1")
        assert "uppercase" in str(exc_info.value).lower()

    def test_password_no_digit(self):
        """Test that password without digit raises ValueError."""
        # "PASSWORD" has no digit (removed the "1" from "PASSWORD1")
        with pytest.raises(ValueError) as exc_info:
            password_strength_check("PASSWORD")
        assert "digit" in str(exc_info.value).lower()

    def test_valid_password(self):
        """Test that valid password Pass1234 does not raise."""
        # Should not raise any exception
        password_strength_check("Pass1234")

    def test_empty_password(self):
        """Test that empty password raises ValueError about being empty."""
        with pytest.raises(ValueError) as exc_info:
            password_strength_check("")
        assert "empty" in str(exc_info.value).lower()

    def test_password_too_short_3_chars(self):
        """Test that 3-character password raises ValueError about length."""
        with pytest.raises(ValueError) as exc_info:
            password_strength_check("Ab1")
        assert "8 characters" in str(exc_info.value)
