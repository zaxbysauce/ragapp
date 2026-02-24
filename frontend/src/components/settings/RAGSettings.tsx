import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { SettingsFormData, SettingsErrors } from "@/stores/useSettingsStore";

interface RAGSettingsProps {
  formData: SettingsFormData;
  errors: SettingsErrors;
  onChange: (field: keyof SettingsFormData, value: string | boolean) => void;
}

export function RAGSettings({
  formData,
  errors,
  onChange,
}: RAGSettingsProps) {
  return (
    <>
      {/* Max Distance Threshold */}
      <Card>
        <CardHeader>
          <CardTitle>RAG Retrieval Settings</CardTitle>
          <CardDescription>Configure RAG retrieval behavior and filtering</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Max Distance Threshold</label>
            <div className="flex items-center gap-4">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={formData.max_distance_threshold}
                onChange={(e) => onChange("max_distance_threshold", e.target.value)}
                className={`w-24 ${errors.max_distance_threshold ? "border-destructive" : ""}`}
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={formData.max_distance_threshold}
                onChange={(e) => onChange("max_distance_threshold", e.target.value)}
                className="flex-1"
              />
            </div>
            {errors.max_distance_threshold && (
              <p className="text-xs text-destructive">{errors.max_distance_threshold}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Maximum distance (1-0) for chunks to be included in context (lower = more strict)
            </p>
          </div>

          {/* Retrieval Window */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Retrieval Window</label>
            <Input
              type="number"
              min={0}
              max={3}
              value={formData.retrieval_window}
              onChange={(e) => onChange("retrieval_window", e.target.value)}
              className={errors.retrieval_window ? "border-destructive" : ""}
            />
            {errors.retrieval_window && (
              <p className="text-xs text-destructive">{errors.retrieval_window}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Number of adjacent chunks to include (0-3)
            </p>
          </div>

          {/* Vector Metric */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Vector Metric</label>
            <select
              value={formData.vector_metric}
              onChange={(e) => onChange("vector_metric", e.target.value)}
              className={`w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${errors.vector_metric ? "border-destructive" : ""}`}
            >
              <option value="cosine">Cosine Similarity</option>
              <option value="euclidean">Euclidean Distance</option>
              <option value="dot_product">Dot Product</option>
            </select>
            {errors.vector_metric && (
              <p className="text-xs text-destructive">{errors.vector_metric}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Distance metric used for vector similarity search
            </p>
          </div>

          {/* Embedding Batch Size */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Embedding Batch Size</label>
            <Input
              type="number"
              min={64}
              max={2048}
              step={64}
              value={formData.embedding_batch_size}
              onChange={(e) => onChange("embedding_batch_size", e.target.value)}
              className={errors.embedding_batch_size ? "border-destructive" : ""}
            />
            {errors.embedding_batch_size && (
              <p className="text-xs text-destructive">{errors.embedding_batch_size}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Number of chunks to embed per API request. Higher values = better GPU utilization.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Embedding Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Advanced Embedding Settings</CardTitle>
          <CardDescription>Configure embedding prefixes for different content types</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Embedding Document Prefix</label>
            <textarea
              value={formData.embedding_doc_prefix}
              onChange={(e) => onChange("embedding_doc_prefix", e.target.value)}
              className={`w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 min-h-[80px] ${errors.embedding_doc_prefix ? "border-destructive" : ""}`}
              placeholder="Passage: "
            />
            {errors.embedding_doc_prefix && (
              <p className="text-xs text-destructive">{errors.embedding_doc_prefix}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Prefix added to document chunks before embedding (default: &quot;Passage: &quot;)
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Embedding Query Prefix</label>
            <textarea
              value={formData.embedding_query_prefix}
              onChange={(e) => onChange("embedding_query_prefix", e.target.value)}
              className={`w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 min-h-[80px] ${errors.embedding_query_prefix ? "border-destructive" : ""}`}
              placeholder="Query: "
            />
            {errors.embedding_query_prefix && (
              <p className="text-xs text-destructive">{errors.embedding_query_prefix}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Prefix added to queries before embedding (default: &quot;Query: &quot;)
            </p>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
