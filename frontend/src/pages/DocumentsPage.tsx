import { useState, useEffect, useCallback, useMemo } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { FileText, Upload, Search, Trash2, ScanLine, AlertCircle, Loader2, X } from "lucide-react";
import { listDocuments, uploadDocument, scanDocuments, deleteDocument, getDocumentStats, type Document, type DocumentStatsResponse } from "@/lib/api";
import { formatFileSize, formatDate } from "@/lib/formatters";
import { useDebounce } from "@/hooks/useDebounce";
import { useVaultStore } from "@/stores/useVaultStore";
import { VaultSelector } from "@/components/vault/VaultSelector";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { DocumentCard } from "@/components/shared/DocumentCard";

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<DocumentStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearchQuery, isSearching] = useDebounce(searchQuery, 300);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [isUploading, setIsUploading] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);

  // Upload queue state
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [pendingUploads, setPendingUploads] = useState<Set<string>>(new Set());

  const { activeVaultId } = useVaultStore();

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await listDocuments(activeVaultId ?? undefined);
      setDocuments(response.documents);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
      toast.error(err instanceof Error ? err.message : "Failed to load documents");
    }
  }, [activeVaultId]);

  const fetchStats = useCallback(async () => {
    try {
      const response = await getDocumentStats(activeVaultId ?? undefined);
      setStats(response);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
      toast.error(err instanceof Error ? err.message : "Failed to load document stats");
    }
  }, [activeVaultId]);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchDocuments(), fetchStats()]);
      setLoading(false);
    };
    loadData();
  }, [fetchDocuments, fetchStats]);

  // Status polling for documents in processing state
  useEffect(() => {
    const hasProcessingDocs = documents.some(
      (doc) => doc.metadata?.status === "processing" || doc.metadata?.status === "pending"
    );

    if (!hasProcessingDocs) return;

    const interval = setInterval(() => {
      fetchDocuments();
      fetchStats();
    }, 5000);

    return () => clearInterval(interval);
  }, [documents, fetchDocuments, fetchStats]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    // Add files to queue
    setUploadQueue((prev) => [...prev, ...acceptedFiles]);
    setPendingUploads((prev) => {
      const next = new Set(prev);
      acceptedFiles.forEach((f) => next.add(f.name));
      return next;
    });
    setRejectedFiles([]);
    
    const newProgress: Record<string, number> = {};
    for (const file of acceptedFiles) {
      newProgress[file.name] = 0;
    }
    setUploadProgress((prev) => ({ ...prev, ...newProgress }));
    setTotalFiles((prev) => prev + acceptedFiles.length);
  }, []);

  // Process upload queue
  useEffect(() => {
    if (isUploading || uploadQueue.length === 0) return;

    const processQueue = async () => {
      setIsUploading(true);
      
      while (uploadQueue.length > 0) {
        const file = uploadQueue[0];
        
        // Check if file was cancelled (removed from pending)
        if (!pendingUploads.has(file.name)) {
          setUploadQueue((prev) => prev.slice(1));
          setCurrentFileIndex((prev) => prev + 1);
          continue;
        }

        setCurrentFileIndex((prev) => prev + 1);
        
        try {
          await uploadDocument(file, (progress) => {
            setUploadProgress((prev) => ({
              ...prev,
              [file.name]: progress,
            }));
          }, activeVaultId ?? undefined);
          
          // Remove from queue on success
          setUploadQueue((prev) => prev.filter((f) => f.name !== file.name));
          setPendingUploads((prev) => {
            const next = new Set(prev);
            next.delete(file.name);
            return next;
          });
        } catch (err) {
          toast.error(`Failed to upload ${file.name}: ${err instanceof Error ? err.message : "Unknown error"}`);
          // Remove from queue on error
          setUploadQueue((prev) => prev.filter((f) => f.name !== file.name));
          setPendingUploads((prev) => {
            const next = new Set(prev);
            next.delete(file.name);
            return next;
          });
        }
      }

      setIsUploading(false);
      setUploadProgress({});
      setCurrentFileIndex(0);
      setTotalFiles(0);
      
      // Refresh documents
      try {
        await Promise.all([fetchDocuments(), fetchStats()]);
        toast.success("Upload queue completed");
      } catch (err) {
        console.error("Failed to refresh documents/stats after upload:", err);
      }
    };

    processQueue();
  }, [uploadQueue, isUploading, activeVaultId, fetchDocuments, fetchStats, pendingUploads]);

  const cancelUpload = useCallback((fileName: string) => {
    setPendingUploads((prev) => {
      const next = new Set(prev);
      next.delete(fileName);
      return next;
    });
    setUploadQueue((prev) => prev.filter((f) => f.name !== fileName));
    setUploadProgress((prev) => {
      const next = { ...prev };
      delete next[fileName];
      return next;
    });
    toast.info(`Cancelled upload for ${fileName}`);
  }, []);

  const onDropRejected = useCallback((rejected: FileRejection[]) => {
    const rejectedNames = rejected.map((r) => `${r.file.name} (${r.errors.map((e) => e.message).join(', ')})`);
    setRejectedFiles(rejectedNames);
    rejectedNames.forEach((name) => {
      toast.error(`File rejected: ${name}`);
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/markdown': ['.md'],
    },
    maxSize: MAX_FILE_SIZE,
    disabled: isUploading,
  });

  const handleScan = async () => {
    setIsScanning(true);
    try {
      const result = await scanDocuments(activeVaultId ?? undefined);
      toast.success(`Scan complete: ${result.added} new document(s) added, ${result.scanned} scanned`);
      await Promise.all([fetchDocuments(), fetchStats()]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setIsScanning(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    if (!confirm("Are you sure you want to delete this document? This will also remove all associated chunks.")) return;
    try {
      await deleteDocument(docId);
      toast.success("Document deleted successfully");
      await Promise.all([fetchDocuments(), fetchStats()]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete document");
    }
  };

  const filteredDocuments = useMemo(
    () => documents.filter((doc) =>
      doc.filename.toLowerCase().includes(debouncedSearchQuery.toLowerCase())
    ),
    [documents, debouncedSearchQuery]
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
          <p className="text-muted-foreground mt-1">Manage your knowledge base documents</p>
        </div>
        <div className="flex items-center gap-2">
          <VaultSelector />
          <Button onClick={handleScan} disabled={isScanning}>
            {isScanning ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <ScanLine className="w-4 h-4 mr-2" />
            )}
            Scan Directory
          </Button>
        </div>
      </div>

      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Documents</CardDescription>
              <CardTitle className="text-3xl">{stats.total_documents}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Chunks</CardDescription>
              <CardTitle className="text-3xl">{stats.total_chunks}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Size</CardDescription>
              <CardTitle className="text-3xl">{formatFileSize(stats.total_size_bytes)}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Processed</CardDescription>
              <CardTitle className="text-3xl">{stats.documents_by_status?.processed || 0}</CardTitle>
            </CardHeader>
          </Card>
        </div>
      )}

      <Card
        {...getRootProps()}
        className={`border-2 border-dashed cursor-pointer transition-colors ${
          isDragActive ? "border-primary bg-primary/5" : "border-border"
        } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input {...getInputProps()} />
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center text-center">
            <Upload className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">
              {isDragActive ? "Drop files here..." : "Drag & drop files here, or click to select"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Supports PDF, DOCX, TXT, MD files (max 50MB each)
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Upload Queue */}
      {(Object.keys(uploadProgress).length > 0 || uploadQueue.length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Upload Queue</CardTitle>
            {totalFiles > 1 && (
              <CardDescription>
                File {currentFileIndex} of {totalFiles}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Currently uploading / completed */}
            {Object.entries(uploadProgress).map(([filename, progress]) => {
              const isPending = pendingUploads.has(filename);
              return (
                <div key={filename} className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="truncate max-w-[250px]" title={filename}>{filename}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">
                        {!isPending ? 'Cancelled' : progress > 0 ? `${progress}%` : 'Uploading...'}
                      </span>
                      {isPending && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => cancelUpload(filename)}
                          aria-label={`Cancel upload for ${filename}`}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                  {isPending && <Progress value={progress} className="h-2" />}
                </div>
              );
            })}
            {/* Pending in queue */}
            {uploadQueue
              .filter((file) => !uploadProgress[file.name])
              .map((file) => (
                <div key={file.name} className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="truncate max-w-[250px]" title={file.name}>{file.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Pending</span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => cancelUpload(file.name)}
                        aria-label={`Cancel upload for ${file.name}`}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
          </CardContent>
        </Card>
      )}

      {rejectedFiles.length > 0 && (
        <div className="p-4 bg-amber-500/10 text-amber-700 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Some files were rejected:</span>
          </div>
          <ul className="list-disc pl-5 space-y-1">
            {rejectedFiles.map((file, index) => (
              <li key={index} className="text-sm">{file}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {isSearching && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground animate-spin" />
          )}
        </div>
        <Badge variant="secondary">{filteredDocuments.length} documents</Badge>
      </div>

      {loading ? (
        <>
          {/* Desktop Table Skeleton (hidden on mobile) */}
          <Card className="hidden sm:block">
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <caption className="sr-only">Documents List</caption>
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th scope="col" className="text-left p-4 font-medium">Filename</th>
                      <th scope="col" className="text-left p-4 font-medium">Status</th>
                      <th scope="col" className="text-left p-4 font-medium">Chunks</th>
                      <th scope="col" className="text-left p-4 font-medium">Size</th>
                      <th scope="col" className="text-left p-4 font-medium">Uploaded</th>
                      <th scope="col" className="text-right p-4 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...Array(5)].map((_, i) => (
                      <tr key={i} className="border-b">
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <Skeleton className="h-4 w-4" />
                            <Skeleton className="h-4 w-[180px]" />
                          </div>
                        </td>
                        <td className="p-4"><Skeleton className="h-5 w-[80px]" /></td>
                        <td className="p-4"><Skeleton className="h-4 w-[40px]" /></td>
                        <td className="p-4"><Skeleton className="h-4 w-[60px]" /></td>
                        <td className="p-4"><Skeleton className="h-4 w-[80px]" /></td>
                        <td className="p-4 text-right"><Skeleton className="h-11 w-11 ml-auto" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Mobile Cards Skeleton (hidden on desktop) */}
          <div className="grid grid-cols-1 gap-3 sm:hidden">
            {[...Array(3)].map((_, i) => (
              <Card key={i} className="w-full">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <Skeleton className="h-11 w-11 rounded-md" />
                      <div className="min-w-0">
                        <Skeleton className="h-5 w-32 mb-1" />
                        <Skeleton className="h-4 w-24" />
                      </div>
                    </div>
                    <Skeleton className="h-11 w-11 rounded-full" />
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="space-y-1">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-6 w-20" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-20" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                  </div>
                  <div className="mt-4 flex sm:hidden">
                    <Skeleton className="h-11 w-full rounded-md" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      ) : filteredDocuments.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              {searchQuery ? "No documents match your search" : "No documents yet. Upload some files to get started."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Desktop Table View (hidden on mobile) */}
          <Card className="hidden sm:block">
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <caption className="sr-only">Documents List</caption>
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th scope="col" className="text-left p-4 font-medium">Filename</th>
                      <th scope="col" className="text-left p-4 font-medium">Status</th>
                      <th scope="col" className="text-left p-4 font-medium">Chunks</th>
                      <th scope="col" className="text-left p-4 font-medium">Size</th>
                      <th scope="col" className="text-left p-4 font-medium">Uploaded</th>
                      <th scope="col" className="text-right p-4 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDocuments.map((doc) => (
                      <tr key={doc.id} className="border-b hover:bg-muted/50">
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-muted-foreground" />
                            <span className="font-medium truncate max-w-[200px]">{doc.filename}</span>
                          </div>
                        </td>
                        <td className="p-4"><StatusBadge status={doc.metadata?.status as string} /></td>
                        <td className="p-4">{String(doc.metadata?.chunk_count ?? 0)}</td>
                        <td className="p-4">{formatFileSize(doc.size)}</td>
                        <td className="p-4 text-muted-foreground">{formatDate(doc.created_at)}</td>
                        <td className="p-4 text-right">
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="min-w-[44px] min-h-[44px]" 
                            onClick={() => handleDeleteDocument(String(doc.id))}
                          >
                            <Trash2 className="w-4 h-4 text-destructive" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Mobile Cards View (hidden on desktop) */}
          <div className="grid grid-cols-1 gap-3 sm:hidden">
            {filteredDocuments.map((doc) => (
              <DocumentCard
                key={doc.id}
                document={doc}
                onDelete={(id) => handleDeleteDocument(String(id))}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
