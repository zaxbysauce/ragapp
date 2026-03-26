"use client";

import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, FolderOpen, Users, ChevronLeft, ChevronRight, Search, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import apiClient from "@/lib/api";
import { useDebounce } from "@/hooks/useDebounce";

// ============================================================================
// API Types
// ============================================================================

interface Vault {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  file_count: number;
  memory_count: number;
  session_count: number;
}

interface VaultsResponse {
  vaults: Vault[];
  total: number;
  page: number;
  per_page: number;
}

interface Group {
  id: number;
  name: string;
  description: string | null;
}

interface GroupsResponse {
  groups: Group[];
  total: number;
  page: number;
  per_page: number;
}

interface VaultGroupsResponse {
  group_ids: number[];
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchVaults(
  page: number,
  search: string,
  perPage: number = 20
): Promise<VaultsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (search) {
    params.append("search", search);
  }
  const response = await apiClient.get<VaultsResponse>(`/vaults?${params.toString()}`);
  return response.data;
}

async function fetchAllGroups(): Promise<Group[]> {
  const response = await apiClient.get<GroupsResponse>("/groups?per_page=1000");
  return response.data.groups;
}

async function fetchVaultGroups(vaultId: number): Promise<number[]> {
  const response = await apiClient.get<VaultGroupsResponse>(`/vaults/${vaultId}/groups`);
  return response.data.group_ids;
}

async function updateVaultGroups(vaultId: number, groupIds: number[]): Promise<void> {
  await apiClient.put(`/vaults/${vaultId}/groups`, { group_ids: groupIds });
}

// ============================================================================
// Skeleton Row Component
// ============================================================================

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
        <Skeleton className="h-4 w-16" />
      </TableCell>
      <TableCell>
        <Skeleton className="h-4 w-32" />
      </TableCell>
      <TableCell className="text-right">
        <Skeleton className="h-8 w-28 ml-auto rounded-md" />
      </TableCell>
    </TableRow>
  );
}

// ============================================================================
// Assign Groups Sheet
// ============================================================================

interface AssignGroupsSheetProps {
  vault: Vault | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

function AssignGroupsSheet({
  vault,
  open,
  onOpenChange,
  onSuccess,
}: AssignGroupsSheetProps): JSX.Element {
  const [selectedGroupIds, setSelectedGroupIds] = useState<number[]>([]);
  const queryClient = useQueryClient();

  const { data: allGroups = [] } = useQuery<Group[]>({
    queryKey: ["groups", "all"],
    queryFn: fetchAllGroups,
    enabled: open,
  });

  const { data: vaultGroups = [] } = useQuery<number[]>({
    queryKey: ["vaults", vault?.id, "groups"],
    queryFn: () => fetchVaultGroups(vault!.id),
    enabled: open && !!vault,
  });

  useEffect(() => {
    if (open && vaultGroups) {
      setSelectedGroupIds(vaultGroups);
    }
  }, [open, vaultGroups]);

  const mutation = useMutation<void, Error, number[]>({
    mutationFn: (groupIds) => {
      if (!vault) throw new Error("No vault selected");
      return updateVaultGroups(vault.id, groupIds);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vaults"] });
      queryClient.invalidateQueries({ queryKey: ["vaults", vault?.id, "groups"] });
      onOpenChange(false);
      onSuccess();
    },
  });

  const toggleGroup = (groupId: number) => {
    setSelectedGroupIds((prev) =>
      prev.includes(groupId) ? prev.filter((id) => id !== groupId) : [...prev, groupId]
    );
  };

  const handleSave = () => {
    mutation.mutate(selectedGroupIds);
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-[400px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" aria-hidden="true" />
            Assign Groups
          </SheetTitle>
          <SheetDescription>
            Select groups that should have access to <strong>{vault?.name}</strong>.
          </SheetDescription>
        </SheetHeader>
        <div className="py-6">
          {allGroups.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">No groups available.</p>
          ) : (
            <ScrollArea className="h-[calc(100vh-260px)]">
              <div className="space-y-3 pr-4">
                {allGroups.map((group) => (
                  <div
                    key={group.id}
                    className="flex items-center space-x-3 rounded-md border p-3 hover:bg-muted/50 transition-colors"
                  >
                    <Checkbox
                      id={`group-${group.id}`}
                      checked={selectedGroupIds.includes(group.id)}
                      onCheckedChange={() => toggleGroup(group.id)}
                      aria-label={`Select ${group.name}`}
                    />
                    <Label
                      htmlFor={`group-${group.id}`}
                      className="flex-1 cursor-pointer space-y-0.5"
                    >
                      <div className="font-medium">{group.name}</div>
                      {group.description && (
                        <div className="text-sm text-muted-foreground line-clamp-2">
                          {group.description}
                        </div>
                      )}
                    </Label>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
          <div className="mt-4 text-sm text-muted-foreground">
            {selectedGroupIds.length} group{selectedGroupIds.length !== 1 ? "s" : ""} selected
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
// Main Vaults Page
// ============================================================================

export default function VaultsPage(): JSX.Element {
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, isSearchPending] = useDebounce(searchInput, 300);
  const [refreshKey, setRefreshKey] = useState(0);

  // Sheet state
  const [assignGroupsOpen, setAssignGroupsOpen] = useState(false);
  const [selectedVault, setSelectedVault] = useState<Vault | null>(null);

  const { data, isLoading, isFetching, error } = useQuery<VaultsResponse, Error>({
    queryKey: ["vaults", page, debouncedSearch, refreshKey],
    queryFn: () => fetchVaults(page, debouncedSearch),
    placeholderData: (previousData) => previousData,
  });

  const vaults = data?.vaults ?? [];
  const total = data?.total ?? 0;
  const perPage = data?.per_page ?? 20;
  const totalPages = Math.ceil(total / perPage);

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

  const openAssignGroups = useCallback((vault: Vault) => {
    setSelectedVault(vault);
    setAssignGroupsOpen(true);
  }, []);

  const handleActionSuccess = useCallback(() => {
    setRefreshKey((prev) => prev + 1);
  }, []);

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Vaults</h1>
          <p className="text-sm text-muted-foreground">
            Manage vaults and their group access permissions
          </p>
        </div>
        <div
          role="alert"
          aria-live="polite"
          className="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-center text-destructive"
        >
          Failed to load vaults: {error.message}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Vaults</h1>
          <p className="text-sm text-muted-foreground">
            Manage vaults and their group access permissions
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            placeholder="Search by vault name..."
            value={searchInput}
            onChange={handleSearchChange}
            className="pl-10"
            aria-label="Search vaults"
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
                <TableHead>Vault Name</TableHead>
                <TableHead>Documents</TableHead>
                <TableHead>Groups with Access</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, index) => (
                  <SkeletonRow key={index} />
                ))
              ) : vaults.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={4}
                    className="h-32 text-center text-muted-foreground"
                  >
                    {debouncedSearch
                      ? "No vaults match your search"
                      : "No vaults found"}
                  </TableCell>
                </TableRow>
              ) : (
                vaults.map((vault) => (
                  <TableRow key={vault.id}>
                    <TableCell>
                      <div className="font-medium">{vault.name}</div>
                      {vault.description && (
                        <div className="text-sm text-muted-foreground line-clamp-1 max-w-xs">
                          {vault.description}
                        </div>
                      )}
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5 text-sm">
                        <FolderOpen className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                        {vault.file_count}
                      </span>
                    </TableCell>
                    <TableCell>
                      <VaultGroupsCell vaultId={vault.id} />
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openAssignGroups(vault)}
                      >
                        <Shield className="mr-2 h-4 w-4" aria-hidden="true" />
                        Assign Groups
                      </Button>
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
              Showing {(page - 1) * perPage + 1} to{" "}
              {Math.min(page * perPage, total)} of {total} vaults
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

      {/* Assign Groups Sheet */}
      <AssignGroupsSheet
        vault={selectedVault}
        open={assignGroupsOpen}
        onOpenChange={setAssignGroupsOpen}
        onSuccess={handleActionSuccess}
      />
    </div>
  );
}

// ============================================================================
// Vault Groups Cell Component (fetches and displays group names)
// ============================================================================

function VaultGroupsCell({ vaultId }: { vaultId: number }): JSX.Element {
  const { data: groupIds = [], isLoading } = useQuery<number[]>({
    queryKey: ["vaults", vaultId, "groups"],
    queryFn: () => fetchVaultGroups(vaultId),
  });

  const { data: allGroups = [] } = useQuery<Group[]>({
    queryKey: ["groups", "all"],
    queryFn: fetchAllGroups,
    enabled: groupIds.length > 0,
  });

  if (isLoading) {
    return <Skeleton className="h-4 w-24" />;
  }

  if (groupIds.length === 0) {
    return <span className="text-sm text-muted-foreground">—</span>;
  }

  const groupNames = groupIds
    .map((id) => allGroups.find((g) => g.id === id)?.name)
    .filter(Boolean) as string[];

  return (
    <div className="flex flex-wrap gap-1">
      {groupNames.slice(0, 3).map((name) => (
        <Badge key={name} variant="secondary" className="text-xs">
          <Users className="mr-1 h-3 w-3" aria-hidden="true" />
          {name}
        </Badge>
      ))}
      {groupNames.length > 3 && (
        <Badge variant="secondary" className="text-xs">
          +{groupNames.length - 3}
        </Badge>
      )}
    </div>
  );
}
