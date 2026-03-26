"use client";

import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  MoreHorizontal,
  Edit,
  Trash2,
  UserCheck,
  UserX,
  ChevronLeft,
  ChevronRight,
  Search,
  Key,
  Users,
} from "lucide-react";
import { useDebounce } from "@/hooks/useDebounce";
import apiClient from "@/lib/api";

export interface User {
  id: number;
  username: string;
  full_name: string | null;
  role: "superadmin" | "admin" | "member" | "viewer";
  is_active: boolean;
  groups: string[];
  last_login_at: string | null;
  created_at: string;
}

export interface UsersResponse {
  users: User[];
  total: number;
  page: number;
  per_page: number;
}

interface UserTableProps {
  onEdit?: (user: User) => void;
  onDelete?: (user: User) => void;
  onToggleActive?: (user: User) => void;
  onResetPassword?: (user: User) => void;
  onManageGroups?: (user: User) => void;
  currentUserId?: number;
}

const roleLabels: Record<string, string> = {
  superadmin: "Super Admin",
  admin: "Admin",
  member: "Member",
  viewer: "Viewer",
};

const roleColors: Record<string, string> = {
  superadmin: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  admin: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  member: "bg-green-500/10 text-green-500 border-green-500/20",
  viewer: "bg-gray-500/10 text-gray-500 border-gray-500/20",
};

const PAGE_SIZE = 20;

async function fetchUsers(
  page: number,
  search: string,
  perPage: number = PAGE_SIZE
): Promise<UsersResponse> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (search) {
    params.append("search", search);
  }
  const response = await apiClient.get<UsersResponse>(`/users?${params.toString()}`);
  return response.data;
}

function useUsers(page: number, search: string) {
  return useQuery<UsersResponse, Error>({
    queryKey: ["users", page, search],
    queryFn: () => fetchUsers(page, search),
    placeholderData: (previousData) => previousData,
  });
}

function formatDate(dateString: string | null): string {
  if (!dateString) return "Never";
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function SkeletonRow(): JSX.Element {
  return (
    <TableRow>
      <TableCell>
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-24" />
        </div>
      </TableCell>
      <TableCell>
        <Skeleton className="h-4 w-24" />
      </TableCell>
      <TableCell>
        <Skeleton className="h-5 w-16 rounded-full" />
      </TableCell>
      <TableCell>
        <Skeleton className="h-4 w-20" />
      </TableCell>
      <TableCell>
        <Skeleton className="h-4 w-32" />
      </TableCell>
      <TableCell className="text-right">
        <Skeleton className="h-8 w-8 ml-auto rounded-md" />
      </TableCell>
    </TableRow>
  );
}

export function UserTable({
  onEdit,
  onDelete,
  onToggleActive,
  onResetPassword,
  onManageGroups,
  currentUserId,
}: UserTableProps): JSX.Element {
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, isSearchPending] = useDebounce(searchInput, 300);

  const { data, isLoading, isFetching, error } = useUsers(page, debouncedSearch);

  const users = data?.users ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchInput(e.target.value);
    setPage(1);
  }, []);

  const handlePreviousPage = useCallback(() => {
    setPage((prev) => Math.max(1, prev - 1));
  }, []);

  const handleNextPage = useCallback(() => {
    setPage((prev) => Math.min(totalPages, prev + 1));
  }, [totalPages]);

  const handleEdit = useCallback(
    (user: User) => {
      onEdit?.(user);
    },
    [onEdit]
  );

  const handleDelete = useCallback(
    (user: User) => {
      onDelete?.(user);
    },
    [onDelete]
  );

  const handleToggleActive = useCallback(
    (user: User) => {
      onToggleActive?.(user);
    },
    [onToggleActive]
  );

  const handleResetPassword = useCallback(
    (user: User) => {
      onResetPassword?.(user);
    },
    [onResetPassword]
  );

  const handleManageGroups = useCallback(
    (user: User) => {
      onManageGroups?.(user);
    },
    [onManageGroups]
  );

  if (error) {
    return (
      <div className="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-center text-destructive">
        Failed to load users: {error.message}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search by username..."
          value={searchInput}
          onChange={handleSearchChange}
          className="pl-10"
        />
        {isSearchPending && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
            Searching...
          </span>
        )}
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Username</TableHead>
              <TableHead>Display Name</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Groups</TableHead>
              <TableHead>Last Login</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, index) => (
                <SkeletonRow key={index} />
              ))
            ) : users.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={7}
                  className="h-32 text-center text-muted-foreground"
                >
                  {debouncedSearch
                    ? "No users match your search"
                    : "No users found"}
                </TableCell>
              </TableRow>
            ) : (
              users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell className="font-medium">@{user.username}</TableCell>
                  <TableCell>{user.full_name || "—"}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={roleColors[user.role]}>
                      {roleLabels[user.role]}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {user.is_active ? (
                      <span className="inline-flex items-center gap-1.5 text-sm text-green-600">
                        <UserCheck className="h-4 w-4" />
                        Active
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 text-sm text-gray-500">
                        <UserX className="h-4 w-4" />
                        Inactive
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    {user.groups.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {user.groups.slice(0, 3).map((group) => (
                          <Badge key={group} variant="secondary" className="text-xs">
                            {group}
                          </Badge>
                        ))}
                        {user.groups.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{user.groups.length - 3}
                          </Badge>
                        )}
                      </div>
                    ) : (
                      <span className="text-sm text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(user.last_login_at)}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          disabled={user.id === currentUserId}
                        >
                          <MoreHorizontal className="h-4 w-4" />
                          <span className="sr-only">Open menu</span>
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleEdit(user)}>
                          <Edit className="mr-2 h-4 w-4" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleResetPassword(user)}>
                          <Key className="mr-2 h-4 w-4" />
                          Reset Password
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleManageGroups(user)}>
                          <Users className="mr-2 h-4 w-4" />
                          Manage Groups
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleToggleActive(user)}>
                          {user.is_active ? (
                            <>
                              <UserX className="mr-2 h-4 w-4" />
                              Deactivate
                            </>
                          ) : (
                            <>
                              <UserCheck className="mr-2 h-4 w-4" />
                              Activate
                            </>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => handleDelete(user)}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {(page - 1) * PAGE_SIZE + 1} to{" "}
            {Math.min(page * PAGE_SIZE, total)} of {total} users
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePreviousPage}
              disabled={page === 1 || isFetching}
            >
              <ChevronLeft className="mr-1 h-4 w-4" />
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              Page {page} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleNextPage}
              disabled={page === totalPages || isFetching}
            >
              Next
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserTable;
