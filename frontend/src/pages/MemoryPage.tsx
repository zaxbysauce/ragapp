import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Brain, Plus, Search, Trash2, Loader2 } from "lucide-react";
import { searchMemories, addMemory, deleteMemory, listMemories, type MemoryResult } from "@/lib/api";
import { useDebounce } from "@/hooks/useDebounce";
import { useVaultStore } from "@/stores/useVaultStore";
import { VaultSelector } from "@/components/vault/VaultSelector";

const MAX_MEMORY_CONTENT_LENGTH = 10000;

export default function MemoryPage() {
  const [memories, setMemories] = useState<MemoryResult[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearchQuery] = useDebounce(searchQuery, 300);
  const [loading, setLoading] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newMemory, setNewMemory] = useState({
    content: "",
    category: "",
    tags: "",
    source: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { activeVaultId } = useVaultStore();

  const handleSearch = useCallback(async () => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setLoading(true);
    try {
      if (debouncedSearchQuery.trim()) {
        // Search mode — use POST /memories/search
        const response = await searchMemories(
          { query: debouncedSearchQuery, limit: 50 },
          abortController.signal,
          activeVaultId ?? undefined
        );
        if (!abortController.signal.aborted) {
          setMemories(response.results);
        }
      } else {
        // List mode — use GET /memories
        const response = await listMemories(activeVaultId ?? undefined);
        if (!abortController.signal.aborted) {
          setMemories(response.memories);
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      console.error("Failed to load memories:", err);
      toast.error(err instanceof Error ? err.message : "Failed to load memories");
    } finally {
      if (!abortController.signal.aborted) {
        setLoading(false);
      }
    }
  }, [debouncedSearchQuery, activeVaultId]);

  useEffect(() => {
    handleSearch();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [handleSearch]);

  const validateContent = (content: string): boolean => {
    if (content.length > MAX_MEMORY_CONTENT_LENGTH) {
      setContentError(`Content exceeds maximum length of ${MAX_MEMORY_CONTENT_LENGTH} characters`);
      return false;
    }
    setContentError(null);
    return true;
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setNewMemory({ ...newMemory, content: value });
    if (value.length > MAX_MEMORY_CONTENT_LENGTH) {
      setContentError(`Content exceeds maximum length of ${MAX_MEMORY_CONTENT_LENGTH} characters`);
    } else {
      setContentError(null);
    }
  };

  const handleAddMemory = async () => {
    if (!newMemory.content.trim()) return;
    if (!validateContent(newMemory.content)) return;

    setIsSubmitting(true);
    try {
      await addMemory({
        content: newMemory.content,
        category: newMemory.category || undefined,
        tags: newMemory.tags ? newMemory.tags.split(",").map((t) => t.trim()).filter(Boolean) : [],
        source: newMemory.source || undefined,
      }, activeVaultId ?? undefined);
      toast.success("Memory added successfully");
      // Reset form and close dialog only on success
      setNewMemory({ content: "", category: "", tags: "", source: "" });
      setContentError(null);
      setIsAddDialogOpen(false);
      handleSearch();
    } catch (err) {
      console.error("Failed to add memory:", err);
      toast.error(err instanceof Error ? err.message : "Failed to add memory");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      handleAddMemory();
    }
  };

  const handleDeleteMemory = async (id: string) => {
    if (!confirm("Are you sure you want to delete this memory?")) return;

    setIsDeleting(id);
    try {
      await deleteMemory(id);
      toast.success("Memory deleted successfully");
      handleSearch();
    } catch (err) {
      console.error("Failed to delete memory:", err);
      toast.error(err instanceof Error ? err.message : "Failed to delete memory");
    } finally {
      setIsDeleting(null);
    }
  };

  const getCategoryFromMetadata = (metadata?: Record<string, unknown>) => {
    return (metadata?.category as string) || "Uncategorized";
  };

  const getTagsFromMetadata = (metadata?: Record<string, unknown>) => {
    const tags = metadata?.tags;
    if (Array.isArray(tags)) return tags;
    return [];
  };

  const getSourceFromMetadata = (metadata?: Record<string, unknown>) => {
    return (metadata?.source as string) || "";
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Memory</h1>
          <p className="text-muted-foreground mt-1">View and manage AI memory and context</p>
        </div>
        <div className="flex items-center gap-2">
          <VaultSelector />
          <Button onClick={() => setIsAddDialogOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Add Memory
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search memories..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
        </div>
        <Button onClick={handleSearch} disabled={loading}>
          {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Search className="w-4 h-4 mr-2" />}
          Search
        </Button>
        <Badge variant="secondary">{memories.length} memories</Badge>
      </div>

      {isAddDialogOpen && (
        <Card>
          <CardHeader>
            <CardTitle>Add New Memory</CardTitle>
            <CardDescription>Create a new memory entry for the AI</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Content *</label>
              <Textarea
                placeholder="Enter memory content..."
                className={`min-h-[100px] ${contentError ? "border-destructive focus-visible:ring-destructive" : ""}`}
                value={newMemory.content}
                onChange={handleContentChange}
                onKeyDown={handleKeyDown}
              />
              {contentError && (
                <span className="text-xs text-destructive">{contentError}</span>
              )}
              <div className="flex justify-end">
                <span className={`text-xs ${newMemory.content.length > MAX_MEMORY_CONTENT_LENGTH ? "text-destructive" : "text-muted-foreground"}`}>
                  {newMemory.content.length}/{MAX_MEMORY_CONTENT_LENGTH}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Category</label>
                <Input
                  placeholder="e.g., facts, preferences"
                  value={newMemory.category}
                  onChange={(e) => setNewMemory({ ...newMemory, category: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Source</label>
                <Input
                  placeholder="e.g., user input, document"
                  value={newMemory.source}
                  onChange={(e) => setNewMemory({ ...newMemory, source: e.target.value })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Tags</label>
              <Input
                placeholder="Enter tags separated by commas..."
                value={newMemory.tags}
                onChange={(e) => setNewMemory({ ...newMemory, tags: e.target.value })}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddMemory} disabled={isSubmitting || !newMemory.content.trim()}>
                {isSubmitting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Plus className="w-4 h-4 mr-2" />}
                Add Memory
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {loading && memories.length === 0 ? (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-[90%]" />
                    <Skeleton className="h-4 w-[60%]" />
                    <div className="flex flex-wrap items-center gap-2 pt-2">
                      <Skeleton className="h-5 w-[70px]" />
                      <Skeleton className="h-5 w-[50px]" />
                      <Skeleton className="h-5 w-[60px]" />
                    </div>
                  </div>
                  <Skeleton className="h-8 w-8 shrink-0" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : memories.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              {searchQuery ? "No memories match your search" : "No memories yet. Add some to get started."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {memories.map((memory) => (
            <Card key={memory.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-2">
                    <p className="text-sm whitespace-pre-wrap">{memory.content}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline">{getCategoryFromMetadata(memory.metadata)}</Badge>
                      {getTagsFromMetadata(memory.metadata).map((tag, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {getSourceFromMetadata(memory.metadata) && (
                        <span className="text-xs text-muted-foreground">
                          Source: {getSourceFromMetadata(memory.metadata)}
                        </span>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={() => handleDeleteMemory(memory.id)}
                    disabled={isDeleting === memory.id}
                  >
                    {isDeleting === memory.id ? (
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    ) : (
                      <Trash2 className="w-4 h-4 text-destructive" />
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
