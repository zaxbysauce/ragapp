"use client";

import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Loader2, Users, FolderOpen, Search } from "lucide-react";
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
import { Textarea } from "@/components/ui/textarea";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GroupTable, Group } from "@/components/admin/GroupTable";
import apiClient from "@/lib/api";

// ============================================================================
// API Types
// ============================================================================

interface User {
  id: number;
  username: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
}

interface Vault {
  id: number;
  name: string;
  description: string | null;
}

interface NewGroupFormData {
  name: string;
  description: string;
}

interface FormErrors {
  name?: string;
  description?: string;
  general?: string;
}

// ============================================================================
// API Functions
// ============================================================================

async function createGroup(data: NewGroupFormData): Promise<Group> {
  const response = await apiClient.post<Group>("/groups", data);
  return response.data;
}

async function updateGroup(
  groupId: number,
  data: { name: string; description: string }
): Promise<void> {
  await apiClient.patch(`/groups/${groupId}`, data);
}

async function deleteGroup(groupId: number): Promise<void> {
  await apiClient.delete(`/groups/${groupId}`);
}

async function fetchGroupMembers(groupId: number): Promise<User[]> {
  const response = await apiClient.get<{ users: User[] }>(`/groups/${groupId}/members`);
  return response.data.users;
}

async function updateGroupMembers(groupId: number, userIds: number[]): Promise<void> {
  await apiClient.put(`/groups/${groupId}/members`, { user_ids: userIds });
}

async function fetchAllUsers(): Promise<User[]> {
  const response = await apiClient.get<{ users: User[] }>("/users");
  return response.data.users;
}

async function fetchGroupVaults(groupId: number): Promise<number[]> {
  const response = await apiClient.get<{ vault_ids: number[] }>(`/groups/${groupId}/vaults`);
  return response.data.vault_ids;
}

async function updateGroupVaults(groupId: number, vaultIds: number[]): Promise<void> {
  await apiClient.put(`/groups/${groupId}/vaults`, { vault_ids: vaultIds });
}

async function fetchAllVaults(): Promise<Vault[]> {
  const response = await apiClient.get<{ vaults: Vault[] }>("/vaults");
  return response.data.vaults;
}

// ============================================================================
// New Group Modal
// ============================================================================

function NewGroupModal({ onSuccess }: { onSuccess: () => void }): JSX.Element {
  const [open, setOpen] = useState(false);
  const [formData, setFormData] = useState<NewGroupFormData>({
    name: "",
    description: "",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const queryClient = useQueryClient();

  const mutation = useMutation<Group, Error, NewGroupFormData>({
    mutationFn: createGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      setOpen(false);
      setFormData({ name: "", description: "" });
      setErrors({});
      onSuccess();
    },
    onError: (error) => {
      setErrors({ general: error.message || "Failed to create group" });
    },
  });

  const validateForm = (data: NewGroupFormData): FormErrors => {
    const errs: FormErrors = {};
    if (!data.name.trim()) {
      errs.name = "Group name is required";
    } else if (data.name.length < 2) {
      errs.name = "Group name must be at least 2 characters";
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
    mutation.mutate({
      name: formData.name.trim(),
      description: formData.description.trim() || undefined,
    });
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    if (!newOpen) {
      setFormData({ name: "", description: "" });
      setErrors({});
      mutation.reset();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
          New Group
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New Group</DialogTitle>
          <DialogDescription>
            Create a new group to organize users and manage vault access.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {errors.general && (
              <div
                role="alert"
                className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
              >
                {errors.general}
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="group-name">
                Group Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="group-name"
                value={formData.name}
                onChange={(e) => {
                  setFormData((prev) => ({ ...prev, name: e.target.value }));
                  setErrors((prev) => ({ ...prev, name: undefined }));
                }}
                placeholder="Enter group name"
                autoComplete="off"
                aria-invalid={!!errors.name}
                aria-describedby={errors.name ? "group-name-error" : undefined}
              />
              {errors.name && (
                <p id="group-name-error" className="text-sm text-destructive">
                  {errors.name}
                </p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="group-description">Description</Label>
              <Textarea
                id="group-description"
                value={formData.description}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, description: e.target.value }))
                }
                placeholder="Enter group description (optional)"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={mutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                  Creating...
                </>
              ) : (
                "Create Group"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================================
// Edit Group Modal
// ============================================================================

interface EditGroupModalProps {
  group: Group | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function EditGroupModal({ group, open, onOpenChange, onSuccess }: EditGroupModalProps): JSX.Element {
  const [formData, setFormData] = useState({ name: "", description: "" });
  const [errors, setErrors] = useState<{ general?: string; name?: string }>({});
  const queryClient = useQueryClient();

  useEffect(() => {
    if (group && open) {
      setFormData({
        name: group.name,
        description: group.description || "",
      });
      setErrors({});
    }
  }, [group, open]);

  const mutation = useMutation<void, Error, typeof formData>({
    mutationFn: (data) => {
      if (!group) throw new Error("No group selected");
      return updateGroup(group.id, {
        name: data.name.trim(),
        description: data.description.trim(),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      onOpenChange(false);
      onSuccess();
    },
    onError: (error) => {
      setErrors({ general: error.message || "Failed to update group" });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      setErrors({ name: "Group name is required" });
      return;
    }
    mutation.mutate(formData);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit Group</DialogTitle>
          <DialogDescription>Update group details for {group?.name}.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            {errors.general && (
              <div
                role="alert"
                className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive"
              >
                {errors.general}
              </div>
            )}
            <div className="grid gap-2">
              <Label htmlFor="edit-group-name">Group Name</Label>
              <Input
                id="edit-group-name"
                value={formData.name}
                onChange={(e) => {
                  setFormData((prev) => ({ ...prev, name: e.target.value }));
                  setErrors((prev) => ({ ...prev, name: undefined }));
                }}
                placeholder="Enter group name"
              />
              {errors.name && <p className="text-sm text-destructive">{errors.name}</p>}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-group-description">Description</Label>
              <Textarea
                id="edit-group-description"
                value={formData.description}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, description: e.target.value }))
                }
                placeholder="Enter group description"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={mutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
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
// Manage Members Sheet
// ============================================================================

interface ManageMembersSheetProps {
  group: Group | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ManageMembersSheet({ group, open, onOpenChange, onSuccess }: ManageMembersSheetProps): JSX.Element {
  const [selectedUserIds, setSelectedUserIds] = useState<number[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const queryClient = useQueryClient();

  const { data: allUsers = [] } = useQuery<User[]>({
    queryKey: ["users"],
    queryFn: fetchAllUsers,
    enabled: open,
  });

  const { data: groupMembers = [] } = useQuery<User[]>({
    queryKey: ["groups", group?.id, "members"],
    queryFn: () => fetchGroupMembers(group!.id),
    enabled: open && !!group,
  });

  useEffect(() => {
    if (open && groupMembers) {
      setSelectedUserIds(groupMembers.map((u) => u.id));
    }
  }, [open, groupMembers]);

  const mutation = useMutation<void, Error, number[]>({
    mutationFn: (userIds) => {
      if (!group) throw new Error("No group selected");
      return updateGroupMembers(group.id, userIds);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      queryClient.invalidateQueries({ queryKey: ["groups", group?.id, "members"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const toggleUser = (userId: number) => {
    setSelectedUserIds((prev) =>
      prev.includes(userId) ? prev.filter((id) => id !== userId) : [...prev, userId]
    );
  };

  const handleSave = () => {
    mutation.mutate(selectedUserIds);
  };

  const filteredUsers = allUsers.filter(
    (user) =>
      user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (user.full_name && user.full_name.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-[400px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" aria-hidden="true" />
            Manage Members
          </SheetTitle>
          <SheetDescription>
            Manage members for <strong>{group?.name}</strong>. Select users to add or remove from
            this group.
          </SheetDescription>
        </SheetHeader>
        <div className="py-4">
          <div className="relative mb-4">
            <Search
              className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              placeholder="Search users..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
              aria-label="Search users"
            />
          </div>
          <ScrollArea className="h-[calc(100vh-280px)]">
            {filteredUsers.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">
                {searchQuery ? "No users match your search" : "No users available"}
              </p>
            ) : (
              <div className="space-y-2 pr-4">
                {filteredUsers.map((user) => (
                  <div
                    key={user.id}
                    className="flex items-center space-x-3 rounded-md border p-3 hover:bg-muted/50 transition-colors"
                  >
                    <Checkbox
                      id={`user-${user.id}`}
                      checked={selectedUserIds.includes(user.id)}
                      onCheckedChange={() => toggleUser(user.id)}
                      aria-label={`Select ${user.username}`}
                    />
                    <Label
                      htmlFor={`user-${user.id}`}
                      className="flex-1 cursor-pointer space-y-0.5"
                    >
                      <div className="font-medium">{user.username}</div>
                      {user.full_name && (
                        <div className="text-sm text-muted-foreground">{user.full_name}</div>
                      )}
                      <div className="text-xs text-muted-foreground capitalize">{user.role}</div>
                    </Label>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
          <div className="mt-4 text-sm text-muted-foreground">
            {selectedUserIds.length} member{selectedUserIds.length !== 1 ? "s" : ""} selected
          </div>
        </div>
        <SheetFooter className="flex-col gap-2 sm:flex-row">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
            className="w-full sm:w-auto"
          >
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={mutation.isPending} className="w-full sm:w-auto">
            {mutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
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
// Manage Vault Access Sheet
// ============================================================================

interface ManageVaultAccessSheetProps {
  group: Group | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function ManageVaultAccessSheet({
  group,
  open,
  onOpenChange,
  onSuccess,
}: ManageVaultAccessSheetProps): JSX.Element {
  const [selectedVaultIds, setSelectedVaultIds] = useState<number[]>([]);
  const queryClient = useQueryClient();

  const { data: allVaults = [] } = useQuery<Vault[]>({
    queryKey: ["vaults"],
    queryFn: fetchAllVaults,
    enabled: open,
  });

  const { data: groupVaults = [] } = useQuery<number[]>({
    queryKey: ["groups", group?.id, "vaults"],
    queryFn: () => fetchGroupVaults(group!.id),
    enabled: open && !!group,
  });

  useEffect(() => {
    if (open && groupVaults) {
      setSelectedVaultIds(groupVaults);
    }
  }, [open, groupVaults]);

  const mutation = useMutation<void, Error, number[]>({
    mutationFn: (vaultIds) => {
      if (!group) throw new Error("No group selected");
      return updateGroupVaults(group.id, vaultIds);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      queryClient.invalidateQueries({ queryKey: ["groups", group?.id, "vaults"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const toggleVault = (vaultId: number) => {
    setSelectedVaultIds((prev) =>
      prev.includes(vaultId) ? prev.filter((id) => id !== vaultId) : [...prev, vaultId]
    );
  };

  const handleSave = () => {
    mutation.mutate(selectedVaultIds);
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-[400px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5" aria-hidden="true" />
            Manage Vault Access
          </SheetTitle>
          <SheetDescription>
            Select vaults that <strong>{group?.name}</strong> should have access to.
          </SheetDescription>
        </SheetHeader>
        <div className="py-6">
          {allVaults.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">No vaults available.</p>
          ) : (
            <ScrollArea className="h-[calc(100vh-260px)]">
              <div className="space-y-3 pr-4">
                {allVaults.map((vault) => (
                  <div
                    key={vault.id}
                    className="flex items-center space-x-3 rounded-md border p-3 hover:bg-muted/50 transition-colors"
                  >
                    <Checkbox
                      id={`vault-${vault.id}`}
                      checked={selectedVaultIds.includes(vault.id)}
                      onCheckedChange={() => toggleVault(vault.id)}
                      aria-label={`Select ${vault.name}`}
                    />
                    <Label
                      htmlFor={`vault-${vault.id}`}
                      className="flex-1 cursor-pointer space-y-0.5"
                    >
                      <div className="font-medium">{vault.name}</div>
                      {vault.description && (
                        <div className="text-sm text-muted-foreground line-clamp-2">
                          {vault.description}
                        </div>
                      )}
                    </Label>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
          <div className="mt-4 text-sm text-muted-foreground">
            {selectedVaultIds.length} vault{selectedVaultIds.length !== 1 ? "s" : ""} selected
          </div>
        </div>
        <SheetFooter className="flex-col gap-2 sm:flex-row">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
            className="w-full sm:w-auto"
          >
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={mutation.isPending} className="w-full sm:w-auto">
            {mutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
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
// Delete Group Dialog
// ============================================================================

interface DeleteGroupDialogProps {
  group: Group | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function DeleteGroupDialog({ group, open, onOpenChange, onSuccess }: DeleteGroupDialogProps): JSX.Element {
  const queryClient = useQueryClient();

  const mutation = useMutation<void, Error, void>({
    mutationFn: () => {
      if (!group) throw new Error("No group selected");
      return deleteGroup(group.id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>Delete Group</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete <strong>{group?.name}</strong>? This action cannot be
            undone. Members will lose access to vaults through this group.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="flex-col gap-2 sm:flex-row">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
            className="w-full sm:w-auto"
          >
            Cancel
          </Button>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            variant="destructive"
            className="w-full sm:w-auto"
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                Deleting...
              </>
            ) : (
              "Delete Group"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================================
// Main Groups Page
// ============================================================================

export default function GroupsPage(): JSX.Element {
  const [refreshKey, setRefreshKey] = useState(0);

  // Modal states for row actions
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [manageMembersOpen, setManageMembersOpen] = useState(false);
  const [manageVaultAccessOpen, setManageVaultAccessOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<Group | null>(null);

  const handleGroupCreated = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const handleActionSuccess = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const openEditModal = (group: Group) => {
    setSelectedGroup(group);
    setEditModalOpen(true);
  };

  const openManageMembers = (group: Group) => {
    setSelectedGroup(group);
    setManageMembersOpen(true);
  };

  const openManageVaultAccess = (group: Group) => {
    setSelectedGroup(group);
    setManageVaultAccessOpen(true);
  };

  const openDeleteDialog = (group: Group) => {
    setSelectedGroup(group);
    setDeleteDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Groups</h1>
          <p className="text-sm text-muted-foreground">
            Manage user groups and their vault access permissions
          </p>
        </div>
        <NewGroupModal onSuccess={handleGroupCreated} />
      </div>

      <GroupTable
        key={refreshKey}
        onEdit={openEditModal}
        onManageMembers={openManageMembers}
        onManageVaultAccess={openManageVaultAccess}
        onDelete={openDeleteDialog}
      />

      {/* Row Action Modals */}
      <EditGroupModal
        group={selectedGroup}
        open={editModalOpen}
        onOpenChange={setEditModalOpen}
        onSuccess={handleActionSuccess}
      />
      <ManageMembersSheet
        group={selectedGroup}
        open={manageMembersOpen}
        onOpenChange={setManageMembersOpen}
        onSuccess={handleActionSuccess}
      />
      <ManageVaultAccessSheet
        group={selectedGroup}
        open={manageVaultAccessOpen}
        onOpenChange={setManageVaultAccessOpen}
        onSuccess={handleActionSuccess}
      />
      <DeleteGroupDialog
        group={selectedGroup}
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        onSuccess={handleActionSuccess}
      />
    </div>
  );
}
