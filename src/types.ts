export interface ClassPrediction {
  class_index: number;
  class_label: string;
  display_name: string;
  fruit_type: string;
  freshness: 'fresh' | 'rotten' | string;
  confidence: number;
}

export interface SampleGroundTruth {
  ground_truth_class: string;
  display_name: string;
  split: string;
  matched_ground_truth: boolean;
}

export interface PredictionResult {
  predicted_class: string;
  predicted_index: number;
  display_name: string;
  fruit_type: string;
  freshness_status: 'fresh' | 'rotten' | string;
  confidence: number;
  confidence_tier: 'high' | 'moderate' | 'low';
  is_low_confidence: boolean;
  confidence_warning: string | null;
  aggregated_freshness: {
    fresh_probability: number;
    rotten_probability: number;
  };
  normalized_entropy: number;
  top_predictions: ClassPrediction[];
  image_metadata: {
    original_width: number;
    original_height: number;
    original_mode: string;
    original_format: string;
    byte_size: number;
  };
  preprocessing: {
    target_size: [number, number];
    color_mode: string;
    normalization: string;
  };
  inference_time_ms: number;
  food_safety_disclaimer: string;
  sample_ground_truth?: SampleGroundTruth;
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  sourceLabel: string;
  previewUrl: string;
  result: PredictionResult;
}

export interface SampleTestItem {
  id: string;
  filename: string;
  ground_truth_class: string;
  display_name: string;
  fruit_type: string;
  freshness: 'fresh' | 'rotten' | string;
  split: string;
  original_dimensions: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  tensorflow_version: string;
  model_ready: boolean;
  missing_artifacts: string[];
  supported_extensions: string[];
  num_classes: number;
  classes: string[];
  food_safety_disclaimer: string;
}

export interface DatasetSummary {
  dataset_metadata: {
    name: string;
    hf_repo_id: string;
    hf_config: string;
    hf_url: string;
    original_source: string;
    authors: string;
    license: string;
    supported_fruits: string[];
    freshness_states: string[];
  };
  raw_directory: string;
  discovered_classes: string[];
  num_classes: number;
  empty_class_directories: string[];
  total_files_scanned: number;
  valid_readable_images: number;
  corrupted_or_unreadable_count: number;
  unsupported_file_count: number;
  total_size_mb: number;
  image_dimensions: {
    width_min: number;
    width_max: number;
    width_mean: number;
    width_median: number;
    height_min: number;
    height_max: number;
    height_mean: number;
    height_median: number;
  };
  color_modes: Record<string, number>;
  file_formats: Record<string, number>;
  duplicates: {
    exact_sha256_duplicate_groups: number;
    exact_sha256_duplicate_images: number;
    perceptual_dhash_duplicate_groups: number;
    perceptual_dhash_duplicate_images: number;
  };
  class_distribution: Record<string, number>;
  fruit_distribution: Record<string, number>;
  freshness_distribution: Record<string, number>;
  imbalance_ratio_max_to_min: number;
  splits?: {
    seed: number;
    ratios: { train: number; val: number; test: number };
    counts: { train: number; val: number; test: number };
    per_class: {
      train: Record<string, number>;
      val: Record<string, number>;
      test: Record<string, number>;
    };
    deduplication_and_leakage: {
      input_valid_images: number;
      exact_duplicates_removed: number;
      conflicting_label_duplicates_quarantined: number;
      near_duplicate_clusters: number;
      near_duplicate_extra_images_grouped: number;
      cross_class_perceptual_clusters: number;
      final_clean_images: number;
      leakage_verified_zero_overlap: boolean;
    };
    class_weights: Record<string, number>;
  };
}

export interface TrainingEpoch {
  epoch: number;
  stage: 'head_warmup' | 'fine_tuning';
  loss: number;
  val_loss: number;
  accuracy: number;
  val_accuracy: number;
}

export interface TrainingHistory {
  model_name: string;
  backbone: string;
  input_shape: [number, number, number];
  num_classes: number;
  random_seed: number;
  batch_size: number;
  stage1_lr: number;
  stage2_lr: number;
  fine_tune_top_layers: number;
  stage1_epochs_completed: number;
  total_epochs_completed: number;
  training_duration_seconds: number;
  model_file: string;
  model_size_mb: number;
  best_val_accuracy: number;
  best_val_loss: number;
  epochs: TrainingEpoch[];
}

export interface PerClassMetric {
  class_index: number;
  class_label: string;
  display_name: string;
  fruit_type: string;
  freshness: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
  correct: number;
}

export interface EvaluationReport {
  evaluated_split: string;
  num_test_samples: number;
  num_classes: number;
  evaluation_duration_seconds: number;
  overall_metrics: {
    accuracy: number;
    test_log_loss: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_precision: number;
    weighted_recall: number;
    weighted_f1: number;
  };
  roc_auc: {
    appropriate: boolean;
    methodology?: string;
    ovr_macro_roc_auc?: number;
    ovr_weighted_roc_auc?: number;
    reason?: string;
  };
  binary_fresh_vs_rotten: {
    accuracy: number;
    f1_rotten_detection: number;
    roc_auc: number | null;
    confusion_matrix_labels: string[];
    confusion_matrix: number[][];
  };
  per_class_metrics: PerClassMetric[];
  confusion_matrix: {
    labels: string[];
    display_labels: string[];
    matrix: number[][];
  };
  top_confused_pairs: Array<{
    true_class: string;
    true_display: string;
    predicted_class: string;
    predicted_display: string;
    count: number;
  }>;
  limitations_and_imbalance_summary: string[];
  text_classification_report: string;
}

export interface AnalyticsBundle {
  available: boolean;
  unavailable_reasons: string[];
  dataset_summary: DatasetSummary | null;
  training_history: TrainingHistory | null;
  evaluation_report: EvaluationReport | null;
}

export interface GitHubStatusResponse {
  oauth_configured: boolean;
  authenticated: boolean;
  username: string | null;
  avatar_url: string | null;
  target_repo_name: string;
  dev_callback_url: string;
  shared_callback_url: string;
}
