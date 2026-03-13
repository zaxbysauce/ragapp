import { create } from "zustand";
import type { SettingsResponse } from "@/lib/api";

export interface SettingsFormData {
  chunk_size_chars: number;
  chunk_overlap_chars: number;
  retrieval_top_k: number;
  auto_scan_enabled: boolean;
  auto_scan_interval_minutes: number;
  max_distance_threshold: number;
  retrieval_window: number;
  vector_metric: string;
  embedding_doc_prefix: string;
  embedding_query_prefix: string;
  embedding_batch_size: number;
  // Retrieval settings
  reranking_enabled?: boolean;
  reranker_url?: string;
  reranker_model?: string;
  initial_retrieval_top_k?: number;
  reranker_top_n?: number;
  hybrid_search_enabled?: boolean;
  hybrid_alpha?: number;
  query_transformation_enabled?: boolean;
  retrieval_evaluation_enabled?: boolean;
  context_distillation_enabled?: boolean;
  context_distillation_dedup_threshold?: number;
  context_distillation_synthesis_enabled?: boolean;
  hyde_enabled?: boolean;
  tri_vector_search_enabled?: boolean;
  flag_embedding_url?: string;
  sparse_search_max_candidates?: number;
  retrieval_recency_weight?: number;
}

export interface SettingsErrors {
  chunk_size_chars?: string;
  chunk_overlap_chars?: string;
  retrieval_top_k?: string;
  auto_scan_interval_minutes?: string;
  max_distance_threshold?: string;
  retrieval_window?: string;
  vector_metric?: string;
  embedding_doc_prefix?: string;
  embedding_query_prefix?: string;
  embedding_batch_size?: string;
  reranker_url?: string;
  reranker_model?: string;
  initial_retrieval_top_k?: string;
  reranker_top_n?: string;
  hybrid_alpha?: string;
  context_distillation_dedup_threshold?: string;
  sparse_search_max_candidates?: string;
  retrieval_recency_weight?: string;
  flag_embedding_url?: string;
}

export interface SettingsState {
  // Data state
  settings: SettingsResponse | null;
  formData: SettingsFormData;
  
  // Loading state
  loading: boolean;
  saving: boolean;
  
  // Error/Status state
  error: string | null;
  errors: SettingsErrors;
  saveStatus: "idle" | "success" | "error";

  // Actions
  setSettings: (settings: SettingsResponse | null) => void;
  setFormData: (formData: SettingsFormData | ((prev: SettingsFormData) => SettingsFormData)) => void;
  updateFormField: <K extends keyof SettingsFormData>(
    field: K,
    value: SettingsFormData[K]
  ) => void;
  setLoading: (loading: boolean) => void;
  setSaving: (saving: boolean) => void;
  setError: (error: string | null) => void;
  setErrors: (errors: SettingsErrors) => void;
  setSaveStatus: (status: "idle" | "success" | "error") => void;
  
  // Initialize form from settings
  initializeForm: (settings: SettingsResponse) => void;
  
  // Validation
  validateForm: () => boolean;
  
  // Check if form has changes
  hasChanges: () => boolean;
  
  // Reset state
  resetState: () => void;
}

const defaultFormData: SettingsFormData = {
  chunk_size_chars: 2000,
  chunk_overlap_chars: 200,
  retrieval_top_k: 5,
  auto_scan_enabled: false,
  auto_scan_interval_minutes: 60,
  max_distance_threshold: 0.7,
  retrieval_window: 1,
  vector_metric: "cosine",
  embedding_doc_prefix: "Passage: ",
  embedding_query_prefix: "Query: ",
  embedding_batch_size: 512,
  reranking_enabled: false,
  reranker_url: "",
  reranker_model: "",
  initial_retrieval_top_k: 20,
  reranker_top_n: 5,
  hybrid_search_enabled: false,
  hybrid_alpha: 0.5,
  query_transformation_enabled: false,
  retrieval_evaluation_enabled: false,
  context_distillation_enabled: false,
  context_distillation_dedup_threshold: 0.92,
  context_distillation_synthesis_enabled: false,
  hyde_enabled: false,
  tri_vector_search_enabled: false,
  flag_embedding_url: "http://embedding-server:18080",
  sparse_search_max_candidates: 1000,
  retrieval_recency_weight: 0.1,
};

export const useSettingsStore = create<SettingsState>((set, get) => ({
  // Initial state
  settings: null,
  formData: { ...defaultFormData },
  loading: true,
  saving: false,
  error: null,
  errors: {},
  saveStatus: "idle",

  // Actions
  setSettings: (settings) => {
    set({ settings });
  },

  setFormData: (formData) => {
    if (typeof formData === "function") {
      set((state) => ({ formData: formData(state.formData) }));
    } else {
      set({ formData });
    }
  },

  updateFormField: (field, value) => {
    set((state) => ({
      formData: { ...state.formData, [field]: value },
      saveStatus: "idle",
    }));
  },

  setLoading: (loading) => {
    set({ loading });
  },

  setSaving: (saving) => {
    set({ saving });
  },

  setError: (error) => {
    set({ error });
  },

  setErrors: (errors) => {
    set({ errors });
  },

  setSaveStatus: (saveStatus) => {
    set({ saveStatus });
  },

  initializeForm: (settings) => {
    set({
      formData: {
        chunk_size_chars: settings.chunk_size_chars ?? 2000,
        chunk_overlap_chars: settings.chunk_overlap_chars ?? 200,
        retrieval_top_k: settings.retrieval_top_k ?? 5,
        auto_scan_enabled: settings.auto_scan_enabled ?? false,
        auto_scan_interval_minutes: settings.auto_scan_interval_minutes ?? 60,
        max_distance_threshold: settings.max_distance_threshold ?? 0.7,
        retrieval_window: settings.retrieval_window ?? 1,
        vector_metric: settings.vector_metric ?? "cosine",
        embedding_doc_prefix: settings.embedding_doc_prefix ?? "Passage: ",
        embedding_query_prefix: settings.embedding_query_prefix ?? "Query: ",
        embedding_batch_size: settings.embedding_batch_size ?? 512,
        reranking_enabled: settings.reranking_enabled ?? false,
        reranker_url: settings.reranker_url ?? "",
        reranker_model: settings.reranker_model ?? "",
        initial_retrieval_top_k: settings.initial_retrieval_top_k ?? 20,
        reranker_top_n: settings.reranker_top_n ?? 5,
        hybrid_search_enabled: settings.hybrid_search_enabled ?? false,
        hybrid_alpha: settings.hybrid_alpha ?? 0.5,
        query_transformation_enabled: settings.query_transformation_enabled ?? false,
        retrieval_evaluation_enabled: settings.retrieval_evaluation_enabled ?? false,
        context_distillation_enabled: settings.context_distillation_enabled ?? false,
        context_distillation_dedup_threshold: settings.context_distillation_dedup_threshold ?? 0.92,
        context_distillation_synthesis_enabled: settings.context_distillation_synthesis_enabled ?? false,
        hyde_enabled: settings.hyde_enabled ?? false,
        tri_vector_search_enabled: settings.tri_vector_search_enabled ?? false,
        flag_embedding_url: settings.flag_embedding_url ?? "http://embedding-server:18080",
        sparse_search_max_candidates: settings.sparse_search_max_candidates ?? 1000,
        retrieval_recency_weight: settings.retrieval_recency_weight ?? 0.1,
      },
      loading: false,
      error: null,
    });
  },

  validateForm: () => {
    const { formData } = get();
    const newErrors: SettingsErrors = {};

    // Validate positive integers
    if (formData.chunk_size_chars <= 0) {
      newErrors.chunk_size_chars = "Chunk size must be a positive integer";
    }
    if (formData.chunk_overlap_chars <= 0) {
      newErrors.chunk_overlap_chars = "Chunk overlap must be a positive integer";
    }
    if (formData.retrieval_top_k <= 0) {
      newErrors.retrieval_top_k = "Retrieval top-k must be a positive integer";
    }
    if (formData.auto_scan_interval_minutes <= 0) {
      newErrors.auto_scan_interval_minutes = "Scan interval must be a positive integer";
    }
    if (formData.embedding_batch_size < 64 || formData.embedding_batch_size > 2048) {
      newErrors.embedding_batch_size = "Embedding batch size must be between 64 and 2048";
    }

    // Validate overlap < size
    if (formData.chunk_overlap_chars >= formData.chunk_size_chars) {
      newErrors.chunk_overlap_chars = "Chunk overlap must be less than chunk size";
    }

    // Validate threshold is between 0 and 1
    if (formData.max_distance_threshold < 0 || formData.max_distance_threshold > 1) {
      newErrors.max_distance_threshold = "Distance threshold must be between 0 and 1";
    }

    // Validate retrieval window (0-3)
    if (formData.retrieval_window < 0 || formData.retrieval_window > 3) {
      newErrors.retrieval_window = "Retrieval window must be between 0 and 3";
    }

    // Validate vector metric
    const validMetrics = ["cosine", "euclidean", "dot_product"];
    if (!validMetrics.includes(formData.vector_metric)) {
      newErrors.vector_metric = "Vector metric must be cosine, euclidean, or dot_product";
    }

    // Validate retrieval settings
    if (formData.initial_retrieval_top_k !== undefined && (formData.initial_retrieval_top_k < 5 || formData.initial_retrieval_top_k > 100)) {
      newErrors.initial_retrieval_top_k = "Initial retrieval top-k must be between 5 and 100";
    }
    if (formData.reranker_top_n !== undefined && (formData.reranker_top_n < 1 || formData.reranker_top_n > 20)) {
      newErrors.reranker_top_n = "Reranker top-n must be between 1 and 20";
    }
    if (formData.hybrid_alpha !== undefined && (formData.hybrid_alpha < 0 || formData.hybrid_alpha > 1)) {
      newErrors.hybrid_alpha = "Hybrid alpha must be between 0 and 1";
    }
    if (
      formData.context_distillation_dedup_threshold !== undefined &&
      (formData.context_distillation_dedup_threshold < 0 || formData.context_distillation_dedup_threshold > 1)
    ) {
      newErrors.context_distillation_dedup_threshold = "Distillation threshold must be between 0 and 1";
    }
    if (
      formData.sparse_search_max_candidates !== undefined &&
      formData.sparse_search_max_candidates < 10
    ) {
      newErrors.sparse_search_max_candidates = "Sparse search candidates must be at least 10";
    }
    if (
      formData.retrieval_recency_weight !== undefined &&
      (formData.retrieval_recency_weight < 0 || formData.retrieval_recency_weight > 1)
    ) {
      newErrors.retrieval_recency_weight = "Recency weight must be between 0 and 1";
    }

    set({ errors: newErrors });
    return Object.keys(newErrors).length === 0;
  },

  hasChanges: () => {
    const { settings, formData } = get();
    if (!settings) return false;
    return (
      formData.chunk_size_chars !== (settings.chunk_size_chars ?? 2000) ||
      formData.chunk_overlap_chars !== (settings.chunk_overlap_chars ?? 200) ||
      formData.retrieval_top_k !== (settings.retrieval_top_k ?? 5) ||
      formData.auto_scan_enabled !== (settings.auto_scan_enabled ?? false) ||
      formData.auto_scan_interval_minutes !== (settings.auto_scan_interval_minutes ?? 60) ||
      formData.max_distance_threshold !== (settings.max_distance_threshold ?? 0.7) ||
      formData.retrieval_window !== (settings.retrieval_window ?? 1) ||
      formData.vector_metric !== (settings.vector_metric ?? "cosine") ||
      formData.embedding_doc_prefix !== (settings.embedding_doc_prefix ?? "Passage: ") ||
      formData.embedding_query_prefix !== (settings.embedding_query_prefix ?? "Query: ") ||
      formData.embedding_batch_size !== (settings.embedding_batch_size ?? 512) ||
      formData.reranking_enabled !== (settings.reranking_enabled ?? false) ||
      formData.reranker_url !== (settings.reranker_url ?? "") ||
      formData.reranker_model !== (settings.reranker_model ?? "") ||
      formData.initial_retrieval_top_k !== (settings.initial_retrieval_top_k ?? 20) ||
      formData.reranker_top_n !== (settings.reranker_top_n ?? 5) ||
      formData.hybrid_search_enabled !== (settings.hybrid_search_enabled ?? false) ||
      formData.hybrid_alpha !== (settings.hybrid_alpha ?? 0.5) ||
      formData.query_transformation_enabled !== (settings.query_transformation_enabled ?? false) ||
      formData.retrieval_evaluation_enabled !== (settings.retrieval_evaluation_enabled ?? false) ||
      formData.context_distillation_enabled !== (settings.context_distillation_enabled ?? false) ||
      formData.context_distillation_dedup_threshold !== (settings.context_distillation_dedup_threshold ?? 0.92) ||
      formData.context_distillation_synthesis_enabled !== (settings.context_distillation_synthesis_enabled ?? false) ||
      formData.hyde_enabled !== (settings.hyde_enabled ?? false) ||
      formData.tri_vector_search_enabled !== (settings.tri_vector_search_enabled ?? false) ||
      formData.flag_embedding_url !== (settings.flag_embedding_url ?? "http://embedding-server:18080") ||
      formData.sparse_search_max_candidates !== (settings.sparse_search_max_candidates ?? 1000) ||
      formData.retrieval_recency_weight !== (settings.retrieval_recency_weight ?? 0.1)
    );
  },

  resetState: () => {
    set({
      settings: null,
      formData: { ...defaultFormData },
      loading: true,
      saving: false,
      error: null,
      errors: {},
      saveStatus: "idle",
    });
  },

  // Alias for resetState to match task requirements
  reset: () => {
    set({
      settings: null,
      formData: { ...defaultFormData },
      loading: true,
      saving: false,
      error: null,
      errors: {},
      saveStatus: "idle",
    });
  },
}));
