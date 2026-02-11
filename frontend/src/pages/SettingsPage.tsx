import { useState, useEffect } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Loader2 } from "lucide-react";
import { getSettings, updateSettings, testConnections, type ConnectionTestResult } from "@/lib/api";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { ConnectionStatusBadges } from "@/components/shared/ConnectionStatusBadges";
import type { HealthStatus } from "@/types/health";
import { useHealthCheck } from "@/hooks/useHealthCheck";

// Internal component that renders the settings form
function SettingsPageContent() {
  const {
    settings,
    formData,
    loading,
    saving,
    error,
    errors,
    setSettings,
    initializeForm,
    setLoading,
    setSaving,
    setError,
    updateFormField,
    validateForm,
    hasChanges,
  } = useSettingsStore();

  const [apiKey, setApiKey] = useState(() => {
    try {
      return localStorage.getItem("kv_api_key") || "";
    } catch {
      return "";
    }
  });
  const [apiKeySaved, setApiKeySaved] = useState(false);

  const handleApiKeySave = () => {
    try {
      localStorage.setItem("kv_api_key", apiKey);
      setApiKeySaved(true);
      toast.success("API key saved");
      setTimeout(() => setApiKeySaved(false), 2000);
    } catch (err) {
      toast.error("Failed to save API key");
    }
  };

  useEffect(() => {
    let mounted = true;
    getSettings()
      .then((data) => {
        if (mounted) {
          setSettings(data);
          initializeForm(data);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load settings");
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, [setSettings, initializeForm, setError, setLoading]);

  const handleInputChange = (field: keyof typeof formData, value: string | boolean) => {
    if (typeof value === "boolean") {
      updateFormField(field, value);
    } else {
      const numValue = parseFloat(value);
      if (!isNaN(numValue)) {
        updateFormField(field, numValue);
      }
    }
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const updated = await updateSettings({
        chunk_size: formData.chunk_size,
        chunk_overlap: formData.chunk_overlap,
        max_context_chunks: formData.max_context_chunks,
        auto_scan_enabled: formData.auto_scan_enabled,
        auto_scan_interval_minutes: formData.auto_scan_interval_minutes,
        rag_relevance_threshold: formData.rag_relevance_threshold,
      });
      setSettings(updated);
      toast.success("Settings saved successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
      toast.error(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {loading && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-[180px]" />
              <Skeleton className="h-4 w-[250px]" />
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[120px]" />
                <Skeleton className="h-10 w-full" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-[150px]" />
              <Skeleton className="h-4 w-[200px]" />
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Skeleton className="h-4 w-[80px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[140px]" />
                <div className="flex items-center gap-4">
                  <Skeleton className="h-10 w-24" />
                  <Skeleton className="h-2 flex-1" />
                </div>
              </div>
              <div className="flex items-center gap-2 pt-4">
                <Skeleton className="h-4 w-4" />
                <Skeleton className="h-4 w-[120px]" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-8">
            <p className="text-destructive text-center">Error: {error}</p>
          </CardContent>
        </Card>
      )}

      {!loading && !error && (
        <Tabs defaultValue="ai" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="ai">AI</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="api-key">API Key</TabsTrigger>
          </TabsList>

          <TabsContent value="ai">
            <Card>
              <CardHeader>
                <CardTitle>AI Configuration</CardTitle>
                <CardDescription>Configure AI model and behavior (read-only, set via environment variables)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chat Model</label>
                  <Input
                    value={settings?.chat_model || "Not configured"}
                    readOnly
                    className="bg-muted"
                  />
                  <p className="text-xs text-muted-foreground">
                    LLM model used for chat responses (set via CHAT_MODEL env var)
                  </p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Embedding Model</label>
                  <Input
                    value={settings?.embedding_model || "Not configured"}
                    readOnly
                    className="bg-muted"
                  />
                  <p className="text-xs text-muted-foreground">
                    Model used for document embeddings (set via EMBEDDING_MODEL env var)
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>



          <TabsContent value="advanced" className="space-y-4">
            <div className="rounded-lg border border-amber-200 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800 p-3 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
              <p className="text-sm text-amber-800 dark:text-amber-200">
                Note: Settings updates apply to the running session only.
              </p>
            </div>
            <Card>
              <CardHeader>
                <CardTitle>Advanced Settings</CardTitle>
                <CardDescription>Configure document processing and RAG parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Chunk Size */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chunk Size</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.chunk_size}
                    onChange={(e) => handleInputChange("chunk_size", e.target.value)}
                    className={errors.chunk_size ? "border-destructive" : ""}
                  />
                  {errors.chunk_size && (
                    <p className="text-xs text-destructive">{errors.chunk_size}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Number of characters per document chunk
                  </p>
                </div>

                {/* Chunk Overlap */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chunk Overlap</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.chunk_overlap}
                    onChange={(e) => handleInputChange("chunk_overlap", e.target.value)}
                    className={errors.chunk_overlap ? "border-destructive" : ""}
                  />
                  {errors.chunk_overlap && (
                    <p className="text-xs text-destructive">{errors.chunk_overlap}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Number of overlapping characters between chunks (must be less than chunk size)
                  </p>
                </div>

                {/* Max Context Chunks */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Context Chunks</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.max_context_chunks}
                    onChange={(e) => handleInputChange("max_context_chunks", e.target.value)}
                    className={errors.max_context_chunks ? "border-destructive" : ""}
                  />
                  {errors.max_context_chunks && (
                    <p className="text-xs text-destructive">{errors.max_context_chunks}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Maximum number of chunks to include in RAG context
                  </p>
                </div>

                {/* RAG Relevance Threshold */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">RAG Relevance Threshold</label>
                  <div className="flex items-center gap-4">
                    <Input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={formData.rag_relevance_threshold}
                      onChange={(e) => handleInputChange("rag_relevance_threshold", e.target.value)}
                      className={`w-24 ${errors.rag_relevance_threshold ? "border-destructive" : ""}`}
                    />
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={formData.rag_relevance_threshold}
                      onChange={(e) => handleInputChange("rag_relevance_threshold", e.target.value)}
                      className="flex-1"
                    />
                  </div>
                  {errors.rag_relevance_threshold && (
                    <p className="text-xs text-destructive">{errors.rag_relevance_threshold}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Minimum relevance score (0-1) for chunks to be included in context
                  </p>
                </div>

                {/* Auto Scan Enabled */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="auto_scan_enabled"
                      checked={formData.auto_scan_enabled}
                      onChange={(e) => handleInputChange("auto_scan_enabled", e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <label htmlFor="auto_scan_enabled" className="text-sm font-medium">
                      Enable Auto Scan
                    </label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Automatically scan for new documents at regular intervals
                  </p>
                </div>

                {/* Auto Scan Interval */}
                {formData.auto_scan_enabled && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Auto Scan Interval (minutes)</label>
                    <Input
                      type="number"
                      min={1}
                      value={formData.auto_scan_interval_minutes}
                      onChange={(e) => handleInputChange("auto_scan_interval_minutes", e.target.value)}
                      className={errors.auto_scan_interval_minutes ? "border-destructive" : ""}
                    />
                    {errors.auto_scan_interval_minutes && (
                      <p className="text-xs text-destructive">{errors.auto_scan_interval_minutes}</p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      How often to scan for new documents (in minutes)
                    </p>
                  </div>
                )}

                {/* Save Button and Status */}
                <div className="flex items-center gap-4 pt-4 border-t">
                  <Button
                    onClick={handleSave}
                    disabled={saving || !hasChanges()}
                  >
                    {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Save Changes
                  </Button>

                  {hasChanges() && (
                    <span className="text-sm text-muted-foreground">You have unsaved changes</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="api-key">
            <Card>
              <CardHeader>
                <CardTitle>API Key</CardTitle>
                <CardDescription>Configure API key for authenticated requests</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">API Key</label>
                  <Input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Enter your API key (if configured)"
                  />
                  <p className="text-xs text-muted-foreground">
                    Optional. Set this if your server requires authentication. The key is stored in your browser's localStorage.
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  <Button onClick={handleApiKeySave}>
                    Save API Key
                  </Button>
                  {apiKeySaved && (
                    <span className="text-sm text-green-600 dark:text-green-400">Saved</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </>
  );
}

// Wrapper that provides health status and connection test
function SettingsPageWithStatus({ health }: { health: HealthStatus }) {
  const formatLastChecked = (date: Date | null) => {
    if (!date) return "Not checked";
    return `Last checked: ${date.toLocaleTimeString()}`;
  };

  const [connectionResult, setConnectionResult] = useState<ConnectionTestResult | null>(null);
  const [isTestingConnections, setIsTestingConnections] = useState(false);

  const handleConnectionTest = async () => {
    setIsTestingConnections(true);
    try {
      const result = await testConnections();
      setConnectionResult(result);
      toast.success("Connection test completed");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Connection test failed";
      toast.error(message);
      setConnectionResult(null);
    } finally {
      setIsTestingConnections(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground mt-1">Configure your application preferences</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <ConnectionStatusBadges health={health} />
          <span className="text-xs text-muted-foreground">{formatLastChecked(health.lastChecked)}</span>
          <Button size="sm" variant="outline" onClick={handleConnectionTest} disabled={isTestingConnections}>
            {isTestingConnections ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              "Test Connections"
            )}
          </Button>
          {connectionResult && (
            <div className="flex gap-2">
              {Object.entries(connectionResult).map(([service, info]) => (
                <Badge key={service} variant={info.ok ? "outline" : "destructive"} className="text-xs">
                  {service}: {info.ok ? "OK" : "Fail"}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
      <SettingsPageContent />
    </div>
  );
}

// Main SettingsPage that checks health
export default function SettingsPage() {
  const health = useHealthCheck();
  return <SettingsPageWithStatus health={health} />;
}
