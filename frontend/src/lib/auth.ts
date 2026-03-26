import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";

/**
 * Hook that redirects to /settings?action=change-password when
 * must_change_password is true in the auth store.
 * Use this in pages that require a valid auth session.
 */
export function useRequirePasswordChange() {
  const navigate = useNavigate();
  const mustChangePassword = useAuthStore((s) => s.mustChangePassword);

  useEffect(() => {
    if (mustChangePassword) {
      navigate("/settings?action=change-password", { replace: true });
    }
  }, [mustChangePassword, navigate]);
}
