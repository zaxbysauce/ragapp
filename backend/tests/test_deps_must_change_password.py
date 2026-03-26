"""Tests for Task 1.7: must_change_password field in deps.py"""

import unittest
import re


class TestMustChangePasswordFix(unittest.TestCase):
    """Test cases for must_change_password field verification in deps.py."""

    @classmethod
    def setUpClass(cls):
        """Read the deps.py file source code."""
        import os

        deps_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "api", "deps.py"
        )
        with open(deps_path, "r", encoding="utf-8") as f:
            cls.source_code = f.read()

    # ===== Test 1: Import verification via syntax check =====
    def test_deps_file_has_valid_python_syntax(self):
        """Verify deps.py has valid Python syntax."""
        try:
            compile(self.source_code, "deps.py", "exec")
        except SyntaxError as e:
            self.fail(f"deps.py has invalid syntax: {e}")

    # ===== Test 2: SELECT query includes must_change_password =====
    def test_select_query_includes_must_change_password(self):
        """Verify SELECT query includes must_change_password column."""
        # Look for SELECT statement that includes must_change_password
        # The query should have: SELECT ... must_change_password FROM users
        pattern = r"SELECT\s+[^;]*must_change_password\s+FROM\s+users"
        match = re.search(pattern, self.source_code, re.IGNORECASE | re.DOTALL)

        self.assertIsNotNone(
            match,
            "SELECT query must include must_change_password column from users table",
        )

    # ===== Test 3: User dict includes must_change_password field =====
    def test_user_dict_has_must_change_password_field(self):
        """Verify user dict includes must_change_password field."""
        # Look for the user dict construction with must_change_password
        # Should have: "must_change_password": bool(row[5]) if row[5] is not None else False
        pattern = r'"must_change_password":\s*bool\(row\[5\]\)\s*if\s*row\[5\]\s*is\s*not\s*None\s*else\s*False'
        match = re.search(pattern, self.source_code)

        self.assertIsNotNone(
            match,
            "User dict must include must_change_password: bool(row[5]) if row[5] is not None else False",
        )

    # ===== Test 4: Pseudo-user dict has must_change_password: False =====
    def test_pseudo_user_has_must_change_password_false(self):
        """Verify pseudo-user dict includes must_change_password: False."""
        # Look for pseudo-user dict (when users_enabled=False)
        # Should have: "must_change_password": False
        pattern = r'"must_change_password":\s*False'
        match = re.search(pattern, self.source_code)

        self.assertIsNotNone(
            match, "Pseudo-user dict must include must_change_password: False"
        )

    # ===== Test 5: Full user dict structure verification =====
    def test_user_dict_has_all_required_fields(self):
        """Verify user dict has all required fields including must_change_password."""
        # Look for user dict with id, username, full_name, role, is_active, must_change_password
        required_fields = [
            '"id":',  # user id
            '"username":',  # username
            '"full_name":',  # full_name
            '"role":',  # role
            '"is_active":',  # is_active
            '"must_change_password":',  # must_change_password (new field)
        ]

        for field in required_fields:
            self.assertIn(
                field, self.source_code, f"User dict must include {field} field"
            )


if __name__ == "__main__":
    unittest.main()
