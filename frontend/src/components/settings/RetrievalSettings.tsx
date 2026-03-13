import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { SettingsFormData, SettingsErrors } from "@/stores/useSettingsStore";

export interface RetrievalSettingsProps {
  formData: SettingsFormData;
  errors: SettingsErrors;
  onChange: (field: keyof SettingsFormData, value: string | boolean) => void;
}

export function RetrievalSettings({
  formData,
  errors,
  onChange,
}: RetrievalSettingsProps) {
  return (
    <>
      {/* Reranking Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Reranking</CardTitle>
          <CardDescription>Configure document reranking for improved retrieval quality</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Enable Reranking Toggle */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="reranking_enabled"
                checked={formData.reranking_enabled || false}
                onChange={(e) => onChange("reranking_enabled", e.target.checked)}
                className="h-4 w-4 rounded border-gray-300"
              />
              <label htmlFor="reranking_enabled" className="text-sm font-medium">
                Enable Reranking
              </label>
            </div>
            <p className="text-xs text-muted-foreground">
              Apply reranking to improve the relevance of retrieved documents
            </p>
          </div>

          {/* Reranker URL */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Reranker URL</label>
            <Input
              type="text"
              value={formData.reranker_url || ""}
              onChange={(e) => onChange("reranker_url", e.target.value)}
              placeholder="http://host.docker.internal:8082 or leave empty for local"
              className={errors.reranker_url ? "border-destructive" : ""}
            />
            {errors.reranker_url && (
              <p className="text-xs text-destructive">{errors.reranker_url}</p>
            )}
            <p className="text-xs text-muted-foreground">
              URL to the reranking service (leave empty to use local reranker)
            </p>
          </div>

          {/* Reranker Model */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Reranker Model</label>
            <Input
              type="text"
              value={formData.reranker_model || ""}
              onChange={(e) => onChange("reranker_model", e.target.value)}
              placeholder="BAAI/bge-reranker-v2-m3"
              className={errors.reranker_model ? "border-destructive" : ""}
            />
            {errors.reranker_model && (
              <p className="text-xs text-destructive">{errors.reranker_model}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Model name for the reranking service
            </p>
          </div>

          {/* Initial Retrieval Top-K */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Initial Retrieval Top-K</label>
            <Input
              type="number"
              min={5}
              max={100}
              value={formData.initial_retrieval_top_k || 20}
              onChange={(e) => onChange("initial_retrieval_top_k", e.target.value)}
              className={errors.initial_retrieval_top_k ? "border-destructive" : ""}
            />
            {errors.initial_retrieval_top_k && (
              <p className="text-xs text-destructive">{errors.initial_retrieval_top_k}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Number of initial documents to retrieve before reranking (5-100)
            </p>
          </div>

          {/* Reranker Top-N */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Reranker Top-N</label>
            <Input
              type="number"
              min={1}
              max={20}
              value={formData.reranker_top_n || 5}
              onChange={(e) => onChange("reranker_top_n", e.target.value)}
              className={errors.reranker_top_n ? "border-destructive" : ""}
            />
            {errors.reranker_top_n && (
              <p className="text-xs text-destructive">{errors.reranker_top_n}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Number of top documents to keep after reranking (1-20)
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Hybrid Search Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Hybrid Search</CardTitle>
          <CardDescription>Configure hybrid search combining vector and keyword search</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Enable Hybrid Search Toggle */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="hybrid_search_enabled"
                checked={formData.hybrid_search_enabled || false}
                onChange={(e) => onChange("hybrid_search_enabled", e.target.checked)}
                className="h-4 w-4 rounded border-gray-300"
              />
              <label htmlFor="hybrid_search_enabled" className="text-sm font-medium">
                Enable Hybrid Search
              </label>
            </div>
            <p className="text-xs text-muted-foreground">
              Combine vector similarity search with keyword-based search
            </p>
          </div>

          {/* Hybrid Alpha */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Hybrid Alpha</label>
            <div className="flex items-center gap-4">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.1}
                value={formData.hybrid_alpha || 0.5}
                onChange={(e) => onChange("hybrid_alpha", e.target.value)}
                className={`w-24 ${errors.hybrid_alpha ? "border-destructive" : ""}`}
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={formData.hybrid_alpha || 0.5}
                onChange={(e) => onChange("hybrid_alpha", e.target.value)}
                className="flex-1"
              />
            </div>
            {errors.hybrid_alpha && (
              <p className="text-xs text-destructive">{errors.hybrid_alpha}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Weight for vector search vs keyword search (0 = keyword only, 1 = vector only)
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Advanced Retrieval Features</CardTitle>
          <CardDescription>Configure query expansion, distillation, sparse retrieval, and recency bias</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {[
            {
              field: "query_transformation_enabled" as const,
              label: "Enable Query Transformation",
              help: "Generate broader step-back query variants before retrieval",
            },
            {
              field: "retrieval_evaluation_enabled" as const,
              label: "Enable Retrieval Evaluation",
              help: "Classify retrieved context quality before prompt assembly",
            },
            {
              field: "context_distillation_enabled" as const,
              label: "Enable Context Distillation",
              help: "Deduplicate overlapping retrieved sentences before generation",
            },
            {
              field: "context_distillation_synthesis_enabled" as const,
              label: "Enable Distillation Synthesis",
              help: "Allow an LLM synthesis pass when retrieval quality is weak",
            },
            {
              field: "hyde_enabled" as const,
              label: "Enable HyDE",
              help: "Generate a hypothetical answer passage as an extra query variant",
            },
            {
              field: "tri_vector_search_enabled" as const,
              label: "Enable Tri-Vector Sparse Retrieval",
              help: "Use FlagEmbedding sparse vectors alongside dense retrieval",
            },
          ].map(({ field, label, help }) => (
            <div key={field} className="space-y-2">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id={field}
                  checked={formData[field] || false}
                  onChange={(e) => onChange(field, e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <label htmlFor={field} className="text-sm font-medium">
                  {label}
                </label>
              </div>
              <p className="text-xs text-muted-foreground">{help}</p>
            </div>
          ))}

          <div className="space-y-2">
            <label className="text-sm font-medium">FlagEmbedding URL</label>
            <Input
              type="text"
              value={formData.flag_embedding_url || ""}
              onChange={(e) => onChange("flag_embedding_url", e.target.value)}
              placeholder="http://embedding-server:18080"
              className={errors.flag_embedding_url ? "border-destructive" : ""}
            />
            {errors.flag_embedding_url && (
              <p className="text-xs text-destructive">{errors.flag_embedding_url}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Endpoint used for BGE-M3 sparse query generation when tri-vector search is enabled
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Context Distillation Threshold</label>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={formData.context_distillation_dedup_threshold || 0.92}
              onChange={(e) => onChange("context_distillation_dedup_threshold", e.target.value)}
              className={errors.context_distillation_dedup_threshold ? "border-destructive" : ""}
            />
            {errors.context_distillation_dedup_threshold && (
              <p className="text-xs text-destructive">{errors.context_distillation_dedup_threshold}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Sentence cosine-similarity threshold for deduplication (0-1)
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Sparse Search Max Candidates</label>
            <Input
              type="number"
              min={10}
              value={formData.sparse_search_max_candidates || 1000}
              onChange={(e) => onChange("sparse_search_max_candidates", e.target.value)}
              className={errors.sparse_search_max_candidates ? "border-destructive" : ""}
            />
            {errors.sparse_search_max_candidates && (
              <p className="text-xs text-destructive">{errors.sparse_search_max_candidates}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Limit candidate rows scanned during sparse dot-product retrieval
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Retrieval Recency Weight</label>
            <div className="flex items-center gap-4">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={formData.retrieval_recency_weight || 0.1}
                onChange={(e) => onChange("retrieval_recency_weight", e.target.value)}
                className={`w-24 ${errors.retrieval_recency_weight ? "border-destructive" : ""}`}
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={formData.retrieval_recency_weight || 0.1}
                onChange={(e) => onChange("retrieval_recency_weight", e.target.value)}
                className="flex-1"
              />
            </div>
            {errors.retrieval_recency_weight && (
              <p className="text-xs text-destructive">{errors.retrieval_recency_weight}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Blend normalized recency into fused ranking to favor newer material
            </p>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
