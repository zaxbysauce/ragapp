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
      formData.embedding_batch_size !== (settings.embedding_batch_size ?? 512)
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
}));
