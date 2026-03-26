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
  Users,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  Search,
} from "lucide-react";
import { useDebounce } from "@/hooks/useDebounce";
import apiClient from "@/lib/api";

export interface Group {
  id: number;
  name: string;
  description: string | null;
  member_count: number;
  vaults: string[];
  created_at: string;
}

export interface GroupsResponse {
  groups: Group[];
  total: number;
  page: number;
  per_page: number;
}

interface GroupTableProps {
  onEdit?: (group: Group) => void;
  onDelete?: (group: Group) => void;
  onManageMembers?: (group: Group) => void;
  onManageVaultAccess?: (group: Group) => void;
}

const PAGE_SIZE = 20;

async function fetchGroups(
  page: number,
  search: string,
  perPage: number = PAGE_SIZE
): Promise<GroupsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (search) {
    params.append("search", search);
  }
  const response = await apiClient.get<GroupsResponse>(`/groups?${params.toString()}`);
  return response.data;
}

function useGroups(page: number, search: string) {
  return useQuery<GroupsResponse, Error>({
    queryKey: ["groups", page, search],
    queryFn: () => fetchGroups(page, search),
    placeholderData: (previousData) => previousData,
  });
}

function SkeletonRow(): JSX.Element {
  return (
    <TableRow>
      <TableCell>
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-48" />
        </div>
      </TableCell>
      <TableCell>
        <Skeleton className="h-4 w-40" />
      </TableCell>
      <TableCell>
        <Skeleton className="h-5 w-16 rounded-full" />
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

export function GroupTable({
  onEdit,
  onDelete,
  onManageMembers,
  onManageVaultAccess,
}: GroupTableProps): JSX.Element {
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, isSearchPending] = useDebounce(searchInput, 300);

  const { data, isLoading, isFetching, error } = useGroups(page, debouncedSearch);

  const groups = data?.groups ?? [];
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
    (group: Group) => {
      onEdit?.(group);
    },
    [onEdit]
  );

  const handleDelete = useCallback(
    (group: Group) => {
      onDelete?.(group);
    },
    [onDelete]
  );

  const handleManageMembers = useCallback(
    (group: Group) => {
      onManageMembers?.(group);
    },
    [onManageMembers]
  );

  const handleManageVaultAccess = useCallback(
    (group: Group) => {
      onManageVaultAccess?.(group);
    },
    [onManageVaultAccess]
  );

  if (error) {
    return (
      <div
        role="alert"
        aria-live="polite"
        className="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-center text-destructive"
      >
        Failed to load groups: {error.message}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search
          className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <Input
          placeholder="Search by group name..."
          value={searchInput}
          onChange={handleSearchChange}
          className="pl-10"
          aria-label="Search groups"
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
              <TableHead>Group Name</TableHead>
              <TableHead>Description</TableHead>
              <TableHead>Members</TableHead>
              <TableHead>Vault Access</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, index) => (
                <SkeletonRow key={index} />
              ))
            ) : groups.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={5}
                  className="h-32 text-center text-muted-foreground"
                >
                  {debouncedSearch
                    ? "No groups match your search"
                    : "No groups found"}
                </TableCell>
              </TableRow>
            ) : (
              groups.map((group) => (
                <TableRow key={group.id}>
                  <TableCell className="font-medium">{group.name}</TableCell>
                  <TableCell className="max-w-xs truncate text-muted-foreground">
                    {group.description || "—"}
                  </TableCell>
                  <TableCell>
                    <span className="inline-flex items-center gap-1.5 text-sm">
                      <Users className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                      {group.member_count}
                    </span>
                  </TableCell>
                  <TableCell>
                    {group.vaults.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {group.vaults.slice(0, 3).map((vault) => (
                          <Badge key={vault} variant="secondary" className="text-xs">
                            <FolderOpen className="mr-1 h-3 w-3" aria-hidden="true" />
                            {vault}
                          </Badge>
                        ))}
                        {group.vaults.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{group.vaults.length - 3}
                          </Badge>
                        )}
                      </div>
                    ) : (
                      <span className="text-sm text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          aria-label={`Actions for ${group.name}`}
                        >
                          <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleEdit(group)}>
                          <Edit className="mr-2 h-4 w-4" aria-hidden="true" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleManageMembers(group)}>
                          <Users className="mr-2 h-4 w-4" aria-hidden="true" />
                          Manage Members
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleManageVaultAccess(group)}>
                          <FolderOpen className="mr-2 h-4 w-4" aria-hidden="true" />
                          Manage Vault Access
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => handleDelete(group)}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="mr-2 h-4 w-4" aria-hidden="true" />
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
            {Math.min(page * PAGE_SIZE, total)} of {total} groups
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePreviousPage}
              disabled={page === 1 || isFetching}
              aria-label="Previous page"
            >
              <ChevronLeft className="mr-1 h-4 w-4" aria-hidden="true" />
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
              aria-label="Next page"
            >
              Next
              <ChevronRight className="ml-1 h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export default GroupTable;
