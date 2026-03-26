"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Loader2, UserCheck, UserX } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";
import { UserTable, User } from "@/components/admin/UserTable";
import apiClient from "@/lib/api";

interface RegisterRequest {
  username: string;
  password: string;
  full_name?: string;
}

interface RegisterResponse {
  id: number;
  username: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
}

interface NewUserFormData {
  username: string;
  display_name: string;
  initial_password: string;
  role: "superadmin" | "admin" | "member" | "viewer";
}

interface FormErrors {
  username?: string;
  display_name?: string;
  initial_password?: string;
  role?: string;
  general?: string;
}

const VALID_ROLES: Array<"superadmin" | "admin" | "member" | "viewer"> = [
  "superadmin",
  "admin",
  "member",
  "viewer",
];

async function registerUser(request: RegisterRequest): Promise<RegisterResponse> {
  const response = await apiClient.post<RegisterResponse>("/auth/register", request);
  return response.data;
}

async function updateUserRole(userId: number, role: string): Promise<void> {
  await apiClient.patch(`/users/${userId}/role`, undefined, { params: { role } });
}

// User row action API functions
async function updateUser(userId: number, data: { username: string; full_name: string; role: string }): Promise<void> {
  await apiClient.patch(`/users/${userId}`, data);
}

async function resetUserPassword(userId: number, password: string): Promise<void> {
  await apiClient.patch(`/users/${userId}/password`, { password });
}

async function toggleUserActive(userId: number, isActive: boolean): Promise<void> {
  await apiClient.patch(`/users/${userId}/active`, undefined, { params: { is_active: isActive } });
}

async function deleteUser(userId: number): Promise<void> {
  await apiClient.delete(`/users/${userId}`);
}

async function fetchUserGroups(userId: number): Promise<string[]> {
  const response = await apiClient.get<{ groups: string[] }>(`/users/${userId}/groups`);
  return response.data.groups;
}

async function updateUserGroups(userId: number, groups: string[]): Promise<void> {
  await apiClient.put(`/users/${userId}/groups`, { groups });
}

async function fetchAllGroups(): Promise<string[]> {
  const response = await apiClient.get<{ groups: string[] }>("/groups");
  return response.data.groups;
}

// ============================================================================
// Row Action Dialog Components
// ============================================================================

interface EditUserModalProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function EditUserModal({ user, open, onOpenChange, onSuccess }: EditUserModalProps): JSX.Element {
  const [formData, setFormData] = useState({
    username: "",
    full_name: "",
    role: "member" as User["role"],
  });
  const [errors, setErrors] = useState<{ general?: string; username?: string }>({});
  const queryClient = useQueryClient();

  useEffect(() => {
    if (user && open) {
      setFormData({
        username: user.username,
        full_name: user.full_name || "",
        role: user.role,
      });
      setErrors({});
    }
  }, [user, open]);

  const mutation = useMutation<void, Error, typeof formData>({
    mutationFn: (data) => {
      if (!user) throw new Error("No user selected");
      return updateUser(user.id, {
        username: data.username.trim(),
        full_name: data.full_name.trim(),
        role: data.role,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
    onError: (error) => {
      setErrors({ general: error.message || "Failed to update user" });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.username.trim()) {
      setErrors({ username: "Username is required" });
      return;
    }
    mutation.mutate(formData);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit User</DialogTitle>
          <DialogDescription>Update user details for @{user?.username}.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {errors.general && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
                {errors.general}
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="edit-username">Username</Label>
              <Input
                id="edit-username"
                value={formData.username}
                onChange={(e) => {
                  setFormData((prev) => ({ ...prev, username: e.target.value }));
                  setErrors((prev) => ({ ...prev, username: undefined }));
                }}
                placeholder="Enter username"
              />
              {errors.username && <p className="text-sm text-destructive">{errors.username}</p>}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-display-name">Display Name</Label>
              <Input
                id="edit-display-name"
                value={formData.full_name}
                onChange={(e) => setFormData((prev) => ({ ...prev, full_name: e.target.value }))}
                placeholder="Enter display name"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-role">Role</Label>
              <Select
                value={formData.role}
                onValueChange={(value) => setFormData((prev) => ({ ...prev, role: value as User["role"] }))}
              >
                <SelectTrigger id="edit-role">
                  <SelectValue placeholder="Select a role" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="superadmin">Super Admin</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="member">Member</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? (
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

interface ResetPasswordDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ResetPasswordDialog({ user, open, onOpenChange, onSuccess }: ResetPasswordDialogProps): JSX.Element {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (open) {
      setPassword("");
      setError(null);
    }
  }, [open]);

  const mutation = useMutation<void, Error, string>({
    mutationFn: (newPassword) => {
      if (!user) throw new Error("No user selected");
      return resetUserPassword(user.id, newPassword);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
    onError: (err) => {
      setError(err.message || "Failed to reset password");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!password || password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    mutation.mutate(password);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>Reset Password</DialogTitle>
          <DialogDescription>Set a new password for @{user?.username}.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {error && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="new-password">New Password</Label>
              <Input
                id="new-password"
                type="password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
                placeholder="Enter new password"
                autoComplete="new-password"
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Resetting...
                </>
              ) : (
                "Reset Password"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

interface ManageGroupsSheetProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ManageGroupsSheet({ user, open, onOpenChange, onSuccess }: ManageGroupsSheetProps): JSX.Element {
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const { data: allGroups = [] } = useQuery<string[]>({
    queryKey: ["groups"],
    queryFn: fetchAllGroups,
    enabled: open,
  });

  const { data: userGroups = [] } = useQuery<string[]>({
    queryKey: ["users", user?.id, "groups"],
    queryFn: () => fetchUserGroups(user!.id),
    enabled: open && !!user,
  });

  useEffect(() => {
    if (open && userGroups) {
      setSelectedGroups(userGroups);
    }
  }, [open, userGroups]);

  const mutation = useMutation<void, Error, string[]>({
    mutationFn: (groups) => {
      if (!user) throw new Error("No user selected");
      return updateUserGroups(user.id, groups);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      queryClient.invalidateQueries({ queryKey: ["users", user?.id, "groups"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const toggleGroup = (group: string) => {
    setSelectedGroups((prev) =>
      prev.includes(group) ? prev.filter((g) => g !== group) : [...prev, group]
    );
  };

  const handleSave = () => {
    mutation.mutate(selectedGroups);
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Manage Groups</SheetTitle>
          <SheetDescription>Manage group memberships for @{user?.username}.</SheetDescription>
        </SheetHeader>
        <div className="py-6">
          {allGroups.length === 0 ? (
            <p className="text-sm text-muted-foreground">No groups available.</p>
          ) : (
            <div className="space-y-4">
              {allGroups.map((group) => (
                <div key={group} className="flex items-center space-x-3">
                  <Checkbox
                    id={`group-${group}`}
                    checked={selectedGroups.includes(group)}
                    onCheckedChange={() => toggleGroup(group)}
                  />
                  <Label htmlFor={`group-${group}`} className="cursor-pointer">
                    {group}
                  </Label>
                </div>
              ))}
            </div>
          )}
        </div>
        <SheetFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={mutation.isPending}>
            {mutation.isPending ? (
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

interface ToggleActiveDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ToggleActiveDialog({ user, open, onOpenChange, onSuccess }: ToggleActiveDialogProps): JSX.Element {
  const queryClient = useQueryClient();

  const mutation = useMutation<void, Error, boolean>({
    mutationFn: (isActive) => {
      if (!user) throw new Error("No user selected");
      return toggleUserActive(user.id, isActive);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const isActivating = !user?.is_active;
  const actionText = isActivating ? "Activate" : "Deactivate";
  const Icon = isActivating ? UserCheck : UserX;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>
            {actionText} User
          </DialogTitle>
          <DialogDescription>
            Are you sure you want to {actionText.toLowerCase()} @{user?.username}?
            {isActivating
              ? " They will be able to log in again."
              : " They will no longer be able to log in."}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
            Cancel
          </Button>
          <Button
            onClick={() => mutation.mutate(isActivating)}
            disabled={mutation.isPending}
            variant={isActivating ? "default" : "destructive"}
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {actionText}ing...
              </>
            ) : (
              <>
                <Icon className="mr-2 h-4 w-4" />
                {actionText}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface DeleteUserDialogProps {
  user: User | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function DeleteUserDialog({ user, open, onOpenChange, onSuccess }: DeleteUserDialogProps): JSX.Element {
  const queryClient = useQueryClient();

  const mutation = useMutation<void, Error, void>({
    mutationFn: () => {
      if (!user) throw new Error("No user selected");
      return deleteUser(user.id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>Delete User</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete @{user?.username}? This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={mutation.isPending}>
            Cancel
          </Button>
          <Button onClick={() => mutation.mutate()} disabled={mutation.isPending} variant="destructive">
            {mutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Deleting...
              </>
            ) : (
              "Delete User"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function validateForm(data: NewUserFormData): FormErrors {
  const errors: FormErrors = {};

  if (!data.username.trim()) {
    errors.username = "Username is required";
  } else if (data.username.length < 3) {
    errors.username = "Username must be at least 3 characters";
  } else if (!/^[a-zA-Z0-9_-]+$/.test(data.username)) {
    errors.username = "Username can only contain letters, numbers, underscores, and hyphens";
  }

  if (data.display_name.trim() && data.display_name.length < 2) {
    errors.display_name = "Display name must be at least 2 characters";
  }

  if (!data.initial_password) {
    errors.initial_password = "Initial password is required";
  } else if (data.initial_password.length < 8) {
    errors.initial_password = "Password must be at least 8 characters";
  }

  if (!data.role) {
    errors.role = "Role is required";
  } else if (!VALID_ROLES.includes(data.role)) {
    errors.role = "Invalid role selected";
  }

  return errors;
}

function NewUserModal({ onSuccess }: { onSuccess: () => void }): JSX.Element {
  const [open, setOpen] = useState(false);
  const [formData, setFormData] = useState<NewUserFormData>({
    username: "",
    display_name: "",
    initial_password: "",
    role: "member",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const queryClient = useQueryClient();

  const createUserMutation = useMutation<RegisterResponse, Error, NewUserFormData>({
    mutationFn: async (data) => {
      // First, register the user
      const registeredUser = await registerUser({
        username: data.username.trim(),
        password: data.initial_password,
        full_name: data.display_name.trim() || undefined,
      });

      // Then, update the role if it's not the default
      if (data.role !== registeredUser.role) {
        await updateUserRole(registeredUser.id, data.role);
      }

      return registeredUser;
    },
    onSuccess: () => {
      // Invalidate users query to refresh the list
      queryClient.invalidateQueries({ queryKey: ["users"] });
      setOpen(false);
      setFormData({
        username: "",
        display_name: "",
        initial_password: "",
        role: "member",
      });
      setErrors({});
      onSuccess();
    },
    onError: (error) => {
      setErrors({
        general: error.message || "Failed to create user. Please try again.",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});

    const validationErrors = validateForm(formData);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    createUserMutation.mutate(formData);
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    if (!newOpen) {
      // Reset form when closing
      setFormData({
        username: "",
        display_name: "",
        initial_password: "",
        role: "member",
      });
      setErrors({});
      createUserMutation.reset();
    }
  };

  const updateField = <K extends keyof NewUserFormData>(
    field: K,
    value: NewUserFormData[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          New User
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New User</DialogTitle>
          <DialogDescription>
            Add a new user to the system. They will be required to change their password on first login.
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
              <Label htmlFor="username">
                Username <span className="text-destructive">*</span>
              </Label>
              <Input
                id="username"
                value={formData.username}
                onChange={(e) => updateField("username", e.target.value)}
                placeholder="Enter username"
                autoComplete="off"
                aria-invalid={!!errors.username}
                aria-describedby={errors.username ? "username-error" : undefined}
              />
              {errors.username && (
                <p id="username-error" className="text-sm text-destructive">
                  {errors.username}
                </p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="display_name">Display Name</Label>
              <Input
                id="display_name"
                value={formData.display_name}
                onChange={(e) => updateField("display_name", e.target.value)}
                placeholder="Enter display name (optional)"
                autoComplete="off"
                aria-invalid={!!errors.display_name}
                aria-describedby={errors.display_name ? "display-name-error" : undefined}
              />
              {errors.display_name && (
                <p id="display-name-error" className="text-sm text-destructive">
                  {errors.display_name}
                </p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="initial_password">
                Initial Password <span className="text-destructive">*</span>
              </Label>
              <Input
                id="initial_password"
                type="password"
                value={formData.initial_password}
                onChange={(e) => updateField("initial_password", e.target.value)}
                placeholder="Enter initial password"
                autoComplete="new-password"
                aria-invalid={!!errors.initial_password}
                aria-describedby={errors.initial_password ? "password-error" : undefined}
              />
              {errors.initial_password && (
                <p id="password-error" className="text-sm text-destructive">
                  {errors.initial_password}
                </p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="role">
                Role <span className="text-destructive">*</span>
              </Label>
              <Select
                value={formData.role}
                onValueChange={(value) =>
                  updateField("role", value as NewUserFormData["role"])
                }
              >
                <SelectTrigger
                  id="role"
                  aria-invalid={!!errors.role}
                  aria-describedby={errors.role ? "role-error" : undefined}
                >
                  <SelectValue placeholder="Select a role" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="superadmin">Super Admin</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="member">Member</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
              {errors.role && (
                <p id="role-error" className="text-sm text-destructive">
                  {errors.role}
                </p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={createUserMutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={createUserMutation.isPending}>
              {createUserMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create User"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function UsersPage(): JSX.Element {
  const [refreshKey, setRefreshKey] = useState(0);

  // Modal states for row actions
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [resetPasswordOpen, setResetPasswordOpen] = useState(false);
  const [manageGroupsOpen, setManageGroupsOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [toggleActiveOpen, setToggleActiveOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  const handleUserCreated = () => {
    // Trigger a refresh of the user table
    setRefreshKey((prev) => prev + 1);
  };

  const handleActionSuccess = () => {
    setRefreshKey((prev) => prev + 1);
  };

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Users</h1>
          <p className="text-sm text-muted-foreground">
            Manage user accounts and their permissions
          </p>
        </div>
        <NewUserModal onSuccess={handleUserCreated} />
      </div>
      <UserTable
        key={refreshKey}
        onEdit={openEditModal}
        onResetPassword={openResetPassword}
        onManageGroups={openManageGroups}
        onToggleActive={openToggleActive}
        onDelete={openDeleteDialog}
      />

      {/* Row Action Modals */}
      <EditUserModal
        user={selectedUser}
        open={editModalOpen}
        onOpenChange={setEditModalOpen}
        onSuccess={handleActionSuccess}
      />
      <ResetPasswordDialog
        user={selectedUser}
        open={resetPasswordOpen}
        onOpenChange={setResetPasswordOpen}
        onSuccess={handleActionSuccess}
      />
      <ManageGroupsSheet
        user={selectedUser}
        open={manageGroupsOpen}
        onOpenChange={setManageGroupsOpen}
        onSuccess={handleActionSuccess}
      />
      <DeleteUserDialog
        user={selectedUser}
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        onSuccess={handleActionSuccess}
      />
      <ToggleActiveDialog
        user={selectedUser}
        open={toggleActiveOpen}
        onOpenChange={setToggleActiveOpen}
        onSuccess={handleActionSuccess}
      />
    </div>
  );
}
