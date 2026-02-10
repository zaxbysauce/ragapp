import { useState, useEffect, useCallback, useMemo } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { FileText, Upload, Search, Trash2, ScanLine, AlertCircle, CheckCircle, Clock, Loader2 } from "lucide-react";
import { listDocuments, uploadDocument, scanDocuments, getDocumentStats, type Document, type DocumentStatsResponse } from "@/lib/api";
import { useDebounce } from "@/hooks/useDebounce";
import { useVaultStore } from "@/stores/useVaultStore";
import { VaultSelector } from "@/components/vault/VaultSelector";

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

  const { activeVaultId } = useVaultStore();

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await listDocuments(activeVaultId ?? undefined);
      setDocuments(response.documents);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }, [activeVaultId]);

  const fetchStats = useCallback(async () => {
    try {
      const response = await getDocumentStats(activeVaultId ?? undefined);
      setStats(response);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
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

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setIsUploading(true);
    setRejectedFiles([]);
    const newProgress: Record<string, number> = {};

    for (const file of acceptedFiles) {
      newProgress[file.name] = 0;
    }
    setUploadProgress(newProgress);
    setTotalFiles(acceptedFiles.length);
    setCurrentFileIndex(0);

    try {
      for (let i = 0; i < acceptedFiles.length; i++) {
        const file = acceptedFiles[i];
        setCurrentFileIndex(i + 1);
        try {
          await uploadDocument(file, (progress) => {
            setUploadProgress((prev) => ({
              ...prev,
              [file.name]: progress,
            }));
          }, activeVaultId ?? undefined);
        } catch (err) {
          // Reset progress for failed file
          setUploadProgress((prev) => ({
            ...prev,
            [file.name]: 0,
          }));
          throw err;
        }
      }
      toast.success(`Uploaded ${acceptedFiles.length} file(s) successfully`);
      try {
        await Promise.all([fetchDocuments(), fetchStats()]);
      } catch (err) {
        console.error("Failed to refresh documents/stats after upload:", err);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      setUploadProgress({});
      setCurrentFileIndex(0);
      setTotalFiles(0);
    }
  }, [fetchDocuments, fetchStats, activeVaultId]);

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
      const result = await scanDocuments();
      toast.success(`Scan complete: ${result.added} new document(s) added, ${result.scanned} scanned`);
      await Promise.all([fetchDocuments(), fetchStats()]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setIsScanning(false);
    }
  };

  const filteredDocuments = useMemo(
    () => documents.filter((doc) =>
      doc.filename.toLowerCase().includes(debouncedSearchQuery.toLowerCase())
    ),
    [documents, debouncedSearchQuery]
  );

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "processed":
        return (
          <Badge variant="default" className="bg-green-500">
            <CheckCircle className="w-3 h-3 mr-1" />
            Processed
          </Badge>
        );
      case "processing":
        return (
          <Badge variant="secondary">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Processing
          </Badge>
        );
      case "pending":
        return (
          <Badge variant="outline">
            <Clock className="w-3 h-3 mr-1" />
            Pending
          </Badge>
        );
      case "error":
        return (
          <Badge variant="destructive">
            <AlertCircle className="w-3 h-3 mr-1" />
            Error
          </Badge>
        );
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "Unknown";
    return new Date(dateStr).toLocaleDateString();
  };

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

      {Object.keys(uploadProgress).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Upload Progress</CardTitle>
            {totalFiles > 1 && (
              <CardDescription>
                File {currentFileIndex} of {totalFiles}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(uploadProgress).map(([filename, progress]) => (
              <div key={filename} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="truncate max-w-[300px]">{filename}</span>
                  <span>{progress > 0 ? `${progress}%` : 'Uploading...'}</span>
                </div>
                <Progress value={progress} className="h-2" />
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
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-medium">Filename</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-left p-4 font-medium">Chunks</th>
                    <th className="text-left p-4 font-medium">Size</th>
                    <th className="text-left p-4 font-medium">Uploaded</th>
                    <th className="text-right p-4 font-medium">Actions</th>
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
                      <td className="p-4 text-right"><Skeleton className="h-8 w-8 ml-auto" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
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
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-medium">Filename</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-left p-4 font-medium">Chunks</th>
                    <th className="text-left p-4 font-medium">Size</th>
                    <th className="text-left p-4 font-medium">Uploaded</th>
                    <th className="text-right p-4 font-medium">Actions</th>
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
                      <td className="p-4">{getStatusBadge(doc.metadata?.status as string)}</td>
                      <td className="p-4">{String(doc.metadata?.chunk_count ?? 0)}</td>
                      <td className="p-4">{formatFileSize(doc.size)}</td>
                      <td className="p-4 text-muted-foreground">{formatDate(doc.created_at)}</td>
                      <td className="p-4 text-right">
                        <Button variant="ghost" size="icon" className="h-8 w-8">
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
      )}
    </div>
  );
}
