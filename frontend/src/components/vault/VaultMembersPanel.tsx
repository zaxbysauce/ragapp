import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Plus, Trash2, Loader2, Search, UserPlus, Shield } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

interface VaultMember {
  user_id: number;
  username: string;
  full_name: string;
  permission: 'read' | 'write' | 'admin';
  granted_at: string;
  granted_by: number | null;
}

interface VaultMembersPanelProps {
  vaultId: number;
  vaultName: string;
}

const permissionLabels: Record<string, string> = {
  read: 'Read',
  write: 'Write',
  admin: 'Admin',
};

const permissionColors: Record<string, string> = {
  read: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
  write: 'bg-green-500/10 text-green-500 border-green-500/20',
  admin: 'bg-purple-500/10 text-purple-500 border-purple-500/20',
};

export default function VaultMembersPanel({ vaultId, vaultName }: VaultMembersPanelProps) {
  const { user: currentUser } = useAuthStore();
  const [members, setMembers] = useState<VaultMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Dialog states
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [removeDialogOpen, setRemoveDialogOpen] = useState(false);
  const [selectedMember, setSelectedMember] = useState<VaultMember | null>(null);
  
  // Form states
  const [newUserId, setNewUserId] = useState('');
  const [newPermission, setNewPermission] = useState<string>('read');
  const [saving, setSaving] = useState(false);
  const [removing, setRemoving] = useState(false);

  const isAdmin = currentUser?.role === 'superadmin' || currentUser?.role === 'admin';

  // Helper to create auth headers - only includes Authorization when token exists
  const getAuthHeaders = (): Record<string, string> => {
    const token = useAuthStore.getState().accessToken;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  };

  useEffect(() => {
    fetchMembers();
  }, [vaultId]);

  async function fetchMembers() {
    try {
      const response = await fetch(`/api/vaults/${vaultId}/members`, {
        headers: getAuthHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch members');
      }
      
      const data = await response.json();
      setMembers(data.members || []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load members');
    } finally {
      setLoading(false);
    }
  }

  const filteredMembers = members.filter(m => 
    m.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.full_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  function openAddDialog() {
    setNewUserId('');
    setNewPermission('read');
    setAddDialogOpen(true);
  }

  function openRemoveDialog(member: VaultMember) {
    setSelectedMember(member);
    setRemoveDialogOpen(true);
  }

  async function handleAddMember() {
    const userId = parseInt(newUserId, 10);
    if (!userId || isNaN(userId)) {
      toast.error('Please enter a valid user ID');
      return;
    }

    setSaving(true);
    try {
      const response = await fetch(`/api/vaults/${vaultId}/members`, {
        method: 'POST',
headers: getAuthHeaders(),
        body: JSON.stringify({
          member_user_id: userId,
          permission: newPermission,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ message: 'Failed to add member' }));
        throw new Error(data.message);
      }

      toast.success('Member added successfully');
      setAddDialogOpen(false);
      fetchMembers();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to add member');
    } finally {
      setSaving(false);
    }
  }

  async function handleUpdatePermission(member: VaultMember, newPerm: string) {
    try {
      const response = await fetch(`/api/vaults/${vaultId}/members/${member.user_id}`, {
        method: 'PATCH',
headers: getAuthHeaders(),
        body: JSON.stringify({ permission: newPerm }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ message: 'Failed to update permission' }));
        throw new Error(data.message);
      }

      toast.success('Permission updated successfully');
      fetchMembers();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update permission');
    }
  }

  async function handleRemoveMember() {
    if (!selectedMember) return;

    setRemoving(true);
    try {
      const response = await fetch(`/api/vaults/${vaultId}/members/${selectedMember.user_id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ message: 'Failed to remove member' }));
        throw new Error(data.message);
      }

      toast.success('Member removed successfully');
      setRemoveDialogOpen(false);
      fetchMembers();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to remove member');
    } finally {
      setRemoving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Vault Members
          </h3>
          <p className="text-sm text-muted-foreground">
            Manage access to &quot;{vaultName}&quot;
          </p>
        </div>
        {isAdmin && (
          <Button onClick={openAddDialog} size="sm">
            <UserPlus className="mr-2 h-4 w-4" /> Add Member
          </Button>
        )}
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search members..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Members Table */}
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Member</TableHead>
              <TableHead>Permission</TableHead>
              <TableHead>Granted</TableHead>
              {isAdmin && <TableHead className="text-right">Actions</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredMembers.length === 0 ? (
              <TableRow>
                <TableCell 
                  colSpan={isAdmin ? 4 : 3} 
                  className="text-center py-8 text-muted-foreground"
                >
                  {searchQuery ? 'No members match your search' : 'No members found'}
                </TableCell>
              </TableRow>
            ) : (
              filteredMembers.map((member) => (
                <TableRow key={member.user_id}>
                  <TableCell>
                    <div className="flex flex-col">
                      <span className="font-medium">{member.full_name || member.username}</span>
                      <span className="text-sm text-muted-foreground">@{member.username}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    {isAdmin ? (
                      <Select
                        value={member.permission}
                        onValueChange={(value: string) => handleUpdatePermission(member, value)}
                      >
                        <SelectTrigger className="w-28">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="read">Read</SelectItem>
                          <SelectItem value="write">Write</SelectItem>
                          <SelectItem value="admin">Admin</SelectItem>
                        </SelectContent>
                      </Select>
                    ) : (
                      <Badge variant="outline" className={permissionColors[member.permission]}>
                        {permissionLabels[member.permission]}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {member.granted_at 
                      ? new Date(member.granted_at).toLocaleDateString() 
                      : 'Unknown'}
                  </TableCell>
                  {isAdmin && (
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => openRemoveDialog(member)}
                        className="text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  )}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Add Member Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Member</DialogTitle>
            <DialogDescription>
              Add a user to &quot;{vaultName}&quot; with the specified permission level.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">User ID</label>
              <Input
                type="number"
                placeholder="Enter user ID"
                value={newUserId}
                onChange={(e) => setNewUserId(e.target.value)}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Find user IDs in the User Management page
              </p>
            </div>
            <div>
              <label className="text-sm font-medium">Permission</label>
              <Select value={newPermission} onValueChange={setNewPermission}>
                <SelectTrigger>
                  <SelectValue placeholder="Select permission" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="read">Read - Can view documents</SelectItem>
                  <SelectItem value="write">Write - Can add/edit documents</SelectItem>
                  <SelectItem value="admin">Admin - Full vault access</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddMember} disabled={saving}>
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Adding...
                </>
              ) : (
                <>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Member
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Remove Confirmation Dialog */}
      <Dialog open={removeDialogOpen} onOpenChange={setRemoveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove Member</DialogTitle>
            <DialogDescription>
              Are you sure you want to remove &quot;{selectedMember?.full_name || selectedMember?.username}&quot; 
              from this vault? They will lose access to all documents in &quot;{vaultName}&quot;.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRemoveDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleRemoveMember} disabled={removing}>
              {removing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Removing...
                </>
              ) : (
                'Remove Member'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
