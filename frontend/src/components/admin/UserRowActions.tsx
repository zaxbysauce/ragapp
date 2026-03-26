"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Key, Users, Copy, Check } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetFooter,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { User } from "./UserTable";
import apiClient from "@/lib/api";

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface Group {
  id: number;
  name: string;
  description: string | null;
}

interface EditUserFormData {
  username: string;
  full_name: string;
  role: "superadmin" | "admin" | "member" | "viewer";
}

interface FormErrors {
  username?: string;
  full_name?: string;
  role?: string;
  general?: string;
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

async function fetchGroups(): Promise<Group[]> {
  const response = await apiClient.get<Group[]>("/groups");
  return response.data;
}

async function fetchUserGroups(userId: number): Promise<number[]> {
  const response = await apiClient.get<number[]>(`/users/${userId}/groups`);
  return response.data;
}

async function updateUser(
  userId: number,
  data: Partial<EditUserFormData>
): Promise<void> {
  await apiClient.patch(`/users/${userId}`, data);
}

async function resetUserPassword(userId: number, newPassword: string): Promise<void> {
  await apiClient.post(`/users/${userId}/reset-password`, { new_password: newPassword });
}

async function toggleUserActive(userId: number, isActive: boolean): Promise<void> {
  await apiClient.patch(`/users/${userId}/active`, { is_active: isActive });
}

async function deleteUser(userId: number): Promise<void> {
  await apiClient.delete(`/users/${userId}`);
}

async function updateUserGroups(userId: number, groupIds: number[]): Promise<void> {
  await apiClient.put(`/users/${userId}/groups`, { group_ids: groupIds });
}

const VALID_ROLES: Array<"superadmin" | "admin" | "member" | "viewer"> = [
  "superadmin",
  "admin",
  "member",
  "viewer",
];

// ============================================================================
// EDIT USER MODAL
// ============================================================================

interface EditUserModalProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function EditUserModal({
  user,
  open,
  onOpenChange,
  onSuccess,
}: EditUserModalProps): JSX.Element {
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState<EditUserFormData>({
    username: "",
    full_name: "",
    role: "member",
  });
  const [errors, setErrors] = useState<FormErrors>({});

  // Reset form when user changes
  useState(() => {
    if (user) {
      setFormData({
        username: user.username,
        full_name: user.full_name || "",
        role: user.role,
      });
      setErrors({});
    }
  });

  const updateMutation = useMutation<void, Error, EditUserFormData>({
    mutationFn: (data) => updateUser(user!.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
    onError: (error) => {
      setErrors({ general: error.message || "Failed to update user" });
    },
  });

  const validateForm = (data: EditUserFormData): FormErrors => {
    const errs: FormErrors = {};
    if (!data.username.trim()) {
      errs.username = "Username is required";
    } else if (data.username.length < 3) {
      errs.username = "Username must be at least 3 characters";
    }
    if (data.full_name.trim() && data.full_name.length < 2) {
      errs.full_name = "Display name must be at least 2 characters";
    }
    return errs;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationErrors = validateForm(formData);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    updateMutation.mutate(formData);
  };

  const updateField = <K extends keyof EditUserFormData>(
    field: K,
    value: EditUserFormData[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit User</DialogTitle>
          <DialogDescription>
            Update user details for @{user?.username}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {errors.general && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
                {errors.general}
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="edit-username">
                Username <span className="text-destructive">*</span>
              </Label>
              <Input
                id="edit-username"
                value={formData.username}
                onChange={(e) => updateField("username", e.target.value)}
                aria-invalid={!!errors.username}
                aria-describedby={errors.username ? "edit-username-error" : undefined}
              />
              {errors.username && (
                <p id="edit-username-error" className="text-sm text-destructive">
                  {errors.username}
                </p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-fullname">Display Name</Label>
              <Input
                id="edit-fullname"
                value={formData.full_name}
                onChange={(e) => updateField("full_name", e.target.value)}
                placeholder="Enter display name"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-role">Role</Label>
              <Select
                value={formData.role}
                onValueChange={(value) =>
                  updateField("role", value as EditUserFormData["role"])
                }
              >
                <SelectTrigger id="edit-role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VALID_ROLES.map((role) => (
                    <SelectItem key={role} value={role}>
                      {role.charAt(0).toUpperCase() + role.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={updateMutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={updateMutation.isPending}>
              {updateMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Changes"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================================
// RESET PASSWORD DIALOG
// ============================================================================

interface ResetPasswordDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function ResetPasswordDialog({
  user,
  open,
  onOpenChange,
  onSuccess,
}: ResetPasswordDialogProps): JSX.Element {
  const [newPassword, setNewPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resetMutation = useMutation<void, Error, string>({
    mutationFn: (password) => resetUserPassword(user!.id, password),
    onSuccess: () => {
      onSuccess();
      // Don't close immediately - let user copy the password
    },
    onError: (err) => {
      setError(err.message || "Failed to reset password");
    },
  });

  const handleGeneratePassword = () => {
    // Generate a secure random password
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";
    let password = "";
    for (let i = 0; i < 16; i++) {
      password += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    setNewPassword(password);
    setError(null);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(newPassword);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSubmit = () => {
    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    resetMutation.mutate(newPassword);
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setNewPassword("");
      setShowPassword(false);
      setCopied(false);
      setError(null);
      resetMutation.reset();
    }
    onOpenChange(open);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            Reset Password
          </DialogTitle>
          <DialogDescription>
            Set a new password for @{user?.username}. They will be required to change it on next login.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          )}
          {resetMutation.isSuccess ? (
            <div className="rounded-md border border-green-500/50 bg-green-500/10 p-4 space-y-3">
              <p className="text-sm text-green-700">
                Password reset successfully! Copy this password and share it securely:
              </p>
              <div className="flex gap-2">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={newPassword}
                  readOnly
                  className="font-mono"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? "Hide" : "Show"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={handleCopy}
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-600" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex gap-2">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={newPassword}
                  onChange={(e) => {
                    setNewPassword(e.target.value);
                    setError(null);
                  }}
                  placeholder="Enter new password"
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? "Hide" : "Show"}
                </Button>
              </div>
              <Button
                type="button"
                variant="secondary"
                onClick={handleGeneratePassword}
                className="w-full"
              >
                Generate Secure Password
              </Button>
            </>
          )}
        </div>
        <DialogFooter>
          {resetMutation.isSuccess ? (
            <Button onClick={() => handleOpenChange(false)}>Close</Button>
          ) : (
            <>
              <Button
                variant="outline"
                onClick={() => handleOpenChange(false)}
                disabled={resetMutation.isPending}
              >
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={!newPassword || resetMutation.isPending}
              >
                {resetMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Resetting...
                  </>
                ) : (
                  "Reset Password"
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================================
// MANAGE GROUPS SHEET
// ============================================================================

interface ManageGroupsSheetProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function ManageGroupsSheet({
  user,
  open,
  onOpenChange,
  onSuccess,
}: ManageGroupsSheetProps): JSX.Element {
  const queryClient = useQueryClient();
  const [selectedGroups, setSelectedGroups] = useState<number[]>([]);

  // Fetch all available groups
  const { data: groups = [] } = useQuery<Group[]>({
    queryKey: ["groups"],
    queryFn: fetchGroups,
    enabled: open,
  });

  // Fetch user's current groups
  const { data: userGroups = [], isLoading: isLoadingUserGroups } = useQuery<number[]>({
    queryKey: ["user-groups", user?.id],
    queryFn: () => fetchUserGroups(user!.id),
    enabled: open && !!user,
  });

  // Sync selected groups when userGroups loads
  useState(() => {
    if (userGroups.length > 0) {
      setSelectedGroups(userGroups);
    }
  });

  const updateMutation = useMutation<void, Error, number[]>({
    mutationFn: (groupIds) => updateUserGroups(user!.id, groupIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      queryClient.invalidateQueries({ queryKey: ["user-groups", user?.id] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const toggleGroup = (groupId: number) => {
    setSelectedGroups((prev) =>
      prev.includes(groupId)
        ? prev.filter((id) => id !== groupId)
        : [...prev, groupId]
    );
  };

  const handleSave = () => {
    updateMutation.mutate(selectedGroups);
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setSelectedGroups([]);
      updateMutation.reset();
    }
    onOpenChange(open);
  };

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent className="w-full sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Manage Groups
          </SheetTitle>
          <SheetDescription>
            Assign groups for @{user?.username}
          </SheetDescription>
        </SheetHeader>
        <div className="py-6">
          {isLoadingUserGroups ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-10 animate-pulse rounded bg-muted" />
              ))}
            </div>
          ) : groups.length === 0 ? (
            <p className="text-center text-sm text-muted-foreground">
              No groups available. Create groups first.
            </p>
          ) : (
            <div className="space-y-2">
              {groups.map((group) => (
                <label
                  key={group.id}
                  className="flex items-start gap-3 rounded-lg border p-3 hover:bg-muted/50 cursor-pointer transition-colors"
                >
                  <Checkbox
                    checked={selectedGroups.includes(group.id)}
                    onCheckedChange={() => toggleGroup(group.id)}
                    aria-label={`Select ${group.name}`}
                  />
                  <div className="flex-1">
                    <p className="font-medium">{group.name}</p>
                    {group.description && (
                      <p className="text-sm text-muted-foreground">
                        {group.description}
                      </p>
                    )}
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>
        <SheetFooter className="flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2">
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={updateMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={updateMutation.isPending || isLoadingUserGroups}
          >
            {updateMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              "Save Changes"
            )}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

// ============================================================================
// DELETE USER CONFIRMATION
// ============================================================================

interface DeleteUserDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function DeleteUserDialog({
  user,
  open,
  onOpenChange,
  onSuccess,
}: DeleteUserDialogProps): JSX.Element {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation<void, Error>({
    mutationFn: () => deleteUser(user!.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const handleDelete = () => {
    deleteMutation.mutate();
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      deleteMutation.reset();
    }
    onOpenChange(open);
  };

  return (
    <AlertDialog open={open} onOpenChange={handleOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete User</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to delete <strong>@{user?.username}</strong>?
            This action cannot be undone. All data associated with this user will be permanently removed.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {deleteMutation.isError && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {deleteMutation.error?.message || "Failed to delete user"}
          </div>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleteMutation.isPending}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {deleteMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Deleting...
              </>
            ) : (
              "Delete User"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// ============================================================================
// TOGGLE ACTIVE CONFIRMATION
// ============================================================================

interface ToggleActiveDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function ToggleActiveDialog({
  user,
  open,
  onOpenChange,
  onSuccess,
}: ToggleActiveDialogProps): JSX.Element {
  const queryClient = useQueryClient();
  const isActivating = !user?.is_active;

  const toggleMutation = useMutation<void, Error>({
    mutationFn: () => toggleUserActive(user!.id, isActivating),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const handleToggle = () => {
    toggleMutation.mutate();
  };

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      toggleMutation.reset();
    }
    onOpenChange(open);
  };

  return (
    <AlertDialog open={open} onOpenChange={handleOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {isActivating ? "Activate User" : "Deactivate User"}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {isActivating ? (
              <>
                Are you sure you want to activate <strong>@{user?.username}</strong>?
                They will be able to log in and access the system.
              </>
            ) : (
              <>
                Are you sure you want to deactivate <strong>@{user?.username}</strong>?
                They will no longer be able to log in or access the system.
              </>
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {toggleMutation.isError && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {toggleMutation.error?.message || "Failed to update user status"}
          </div>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={toggleMutation.isPending}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleToggle}
            disabled={toggleMutation.isPending}
            className={isActivating ? "" : "bg-destructive text-destructive-foreground hover:bg-destructive/90"}
          >
            {toggleMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {isActivating ? "Activating..." : "Deactivating..."}
              </>
            ) : isActivating ? (
              "Activate User"
            ) : (
              "Deactivate User"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// ============================================================================
// USER ACTIONS HOOK
// ============================================================================

interface UseUserActionsReturn {
  // Modal states
  editModalOpen: boolean;
  resetPasswordOpen: boolean;
  manageGroupsOpen: boolean;
  deleteDialogOpen: boolean;
  toggleActiveOpen: boolean;
  // Current user being acted upon
  selectedUser: User | null;
  // Action handlers
  openEditModal: (user: User) => void;
  openResetPassword: (user: User) => void;
  openManageGroups: (user: User) => void;
  openDeleteDialog: (user: User) => void;
  openToggleActive: (user: User) => void;
  // Modal components (render these in your page)
  EditUserModalComponent: JSX.Element;
  ResetPasswordDialogComponent: JSX.Element;
  ManageGroupsSheetComponent: JSX.Element;
  DeleteUserDialogComponent: JSX.Element;
  ToggleActiveDialogComponent: JSX.Element;
}

export function useUserActions(onSuccess: () => void): UseUserActionsReturn {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [resetPasswordOpen, setResetPasswordOpen] = useState(false);
  const [manageGroupsOpen, setManageGroupsOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [toggleActiveOpen, setToggleActiveOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  const openEditModal = (user: User) => {
    setSelectedUser(user);
    setEditModalOpen(true);
  };

  const openResetPassword = (user: User) => {
    setSelectedUser(user);
    setResetPasswordOpen(true);
  };

  const openManageGroups = (user: User) => {
    setSelectedUser(user);
    setManageGroupsOpen(true);
  };

  const openDeleteDialog = (user: User) => {
    setSelectedUser(user);
    setDeleteDialogOpen(true);
  };

  const openToggleActive = (user: User) => {
    setSelectedUser(user);
    setToggleActiveOpen(true);
  };

  return {
    editModalOpen,
    resetPasswordOpen,
    manageGroupsOpen,
    deleteDialogOpen,
    toggleActiveOpen,
    selectedUser,
    openEditModal,
    openResetPassword,
    openManageGroups,
    openDeleteDialog,
    openToggleActive,
    EditUserModalComponent: (
      <EditUserModal
        user={selectedUser}
        open={editModalOpen}
        onOpenChange={setEditModalOpen}
        onSuccess={onSuccess}
      />
    ),
    ResetPasswordDialogComponent: (
      <ResetPasswordDialog
        user={selectedUser}
        open={resetPasswordOpen}
        onOpenChange={setResetPasswordOpen}
        onSuccess={onSuccess}
      />
    ),
    ManageGroupsSheetComponent: (
      <ManageGroupsSheet
        user={selectedUser}
        open={manageGroupsOpen}
        onOpenChange={setManageGroupsOpen}
        onSuccess={onSuccess}
      />
    ),
    DeleteUserDialogComponent: (
      <DeleteUserDialog
        user={selectedUser}
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        onSuccess={onSuccess}
      />
    ),
    ToggleActiveDialogComponent: (
      <ToggleActiveDialog
        user={selectedUser}
        open={toggleActiveOpen}
        onOpenChange={setToggleActiveOpen}
        onSuccess={onSuccess}
      />
    ),
  };
}
