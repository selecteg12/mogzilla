export type Box = { x: number; y: number; width: number; height: number }
export type Point = { id: number; x: number; y: number }
export type Score = { samples: number; nme: number; median_nme: number; p90_nme: number; mean_localization_error_px: number; failure_rate: number; failure_threshold: number }
export type Metric = { id: string; title: string; description: string; category: string; value: number | null; unit: string; point_ids: number[]; points: Point[]; formula: string; substitution: Record<string, unknown>; limitation: string }
export type ModelMeta = { architecture: string; dataset: string; landmarks: number; epochs?: number; parameters?: number; initialization: string; validation?: Score; test?: Score; manifest_sha256?: string }
export type ModelStatus = { ready: boolean; reason: string | null; metadata: ModelMeta | null; active_profile?: 'personal' | 'scut'; region_finder?: { ready: boolean } }
export type Localization = { status: string; method?: string; message?: string; candidates?: { box: number[]; score: number }[]; score?: number; framing_check?: { complete: boolean; score: number; box_agreement_iou: number; predicted_face_coverage_in_input: number } }
export type Analysis = {
  source: { width: number; height: number; bytes: number }; box: number[];
  preprocessed_image: string; transform: { origin: number[]; side: number; size: number; margin: number; normalization: string };
  quality: { laplacian_variance: number; mean_luminance: number; luminance_std: number; warnings: string[]; interpretation: string };
  stages: { name: string; detail: string; milliseconds: number; status: string }[];
  landmarks?: Point[];
  geometry?: { metrics: Metric[]; unavailable: { term: string; reason: string }[]; roll_degrees: number; axis: { origin: number[]; direction: number[] }; warnings: string[]; paths: number[][]; coordinate_convention: string };
  reliability?: { warnings: string[]; flip_disagreement_nme?: number; validation?: Score; test?: Score; interpretation: string; geometry_error: Record<string, {mean_absolute_error: number; p90_absolute_error: number; samples: number}> };
  model?: ModelMeta; total_milliseconds?: number; heatmap_image?: string; heatmap_description?: string;
  localization?: Localization;
  rating?: RatingResult; rating_settings?: RatingSettings;
  holistic?: HolisticResult; visual_model_image?: string;
}
export type RegionObservation = { id: string; title: string; subtitle: string; summary: string; kind: string; metrics: { id: string; title: string; value: number | null; unit: string; point_ids: number[] }[] }
export type PersonalMetrics = { n: number; mae: number; rmse: number; within_one: number; pearson: number | null }
export type PersonalProfile = { feature_mode: string; training_method?: string; fit_images: number; test_images: number; total_rated_images: number; validation: { original: PersonalMetrics; personal: PersonalMetrics; scalar_calibration?: PersonalMetrics; mean_baseline?: PersonalMetrics; ranking?: { original: { spearman: number }; personal: { spearman: number } }; preview_only?: boolean; evaluation_reused?: boolean; fresh_comparison?: { previous: PersonalMetrics & { spearman: number }; updated: PersonalMetrics & { spearman: number } } } }
export type HolisticResult = { status: string; value: number; raw_value: number; kind: string; personal?: PersonalProfile; baseline?: { value: number; raw_value: number; model_id: string }; scale: number[]; source_scale: number[]; conversion: string; interpretation: string; model_id: string; revision: string; explanation: string; limitations: string[]; regions: RegionObservation[]; stability: { low: number; high: number; flip_raw_delta: number; status: string; interpretation: string } }
export type RatingRule = { id: string; low: number; high: number; falloff: number; weight: number }
export type RatingSettings = { criteria: (RatingRule & { title: string; description: string; unit: string; metrics: string[] })[]; preset_name: string; source: string; interpretation: string; formula: string; coverage: string }
export type Annotation = { a: number[]; b: number[]; label: string; kind: 'measure' | 'reference' }
export type RatingPart = { id: string; title: string; description: string; observed: number; low: number; high: number; falloff: number; weight: number; unit: string; score: number; contribution: number; lost_contribution: number; metric_ids: string[]; point_ids: number[]; annotations: Annotation[]; deviation: { direction: 'below' | 'within' | 'above'; distance: number; deduction: number }; summary: string }
export type RatingResult = { status: string; value: number | null; message?: string; provisional?: boolean; review_message?: string; parts?: RatingPart[]; used_criteria?: number; sensitivity?: { low: number; high: number; interpretation: string } | null; unavailable?: { id: string; reason: string }[] }
export type Research = { status: string; message?: string; parameters?: number; selected_architecture?: string; test: Score | null; baselines?: { mean_training_shape: Score; shuffled_image_prediction_nme: number }; dataset?: { dataset: string; counts: Record<string, number>; protocol: string; unique_images: number; split_policy: string }; comparison: { architecture: string; validation: Score; parameters: number }[] }
export type ReferenceValues = {
  reference: { kind: string; width: number; height: number; points: Point[]; paths: number[][]; rating: RatingResult; criteria: RatingSettings['criteria']; interpretation: string };
  gallery: { examples: { id: number; title: string; image_url: string; source_image: string; dataset: string; value: number; all_ten: boolean; parts: RatingPart[]; landmark_nme: number }[]; generated_at?: string; scan_mode?: string; scanned_images?: number; filtered_candidates?: number; all_ten_count?: number; selection_note?: string; crop_note?: string; source_url?: string; message?: string };
}
