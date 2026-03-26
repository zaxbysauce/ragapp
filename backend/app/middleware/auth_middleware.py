"""Auth middleware dependencies."""

from fastapi import Depends, HTTPException

from app.api.deps import get_current_active_user


async def check_password_change_required(
    user: dict = Depends(get_current_active_user),
) -> dict:
    """
    Dependency that checks if the user must change their password.

    If must_change_password is True, raises HTTPException 403 with
    X-Redirect header pointing to the password change page.

    Returns the user dict if no password change is required.
    """
    if user.get("must_change_password"):
        raise HTTPException(
            status_code=403,
            detail="Password change required",
            headers={"X-Redirect": "/settings?action=change-password"},
        )
    return user
