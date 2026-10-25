import React, { useRef, useState } from 'react';
import {
  Upload,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  RotateCcw,
  Download,
  Trash2,
  FileWarning,
  Image as ImageIcon,
} from 'lucide-react';
import type {
  HealthResponse,
  HistoryEntry,
  PredictionResult,
  SampleTestItem,
} from '../types';

interface ScannerWorkspaceProps {
  health: HealthResponse | null;
  healthError: string | null;
  samples: SampleTestItem[];
  history: HistoryEntry[];
  onAddHistory: (entry: HistoryEntry) => void;
  onClearHistory: () => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}

const SUPPORTED_FRUITS = [
  'Apple',
  'Banana',
  'Grape',
  'Guava',
  'Jujube',
  'Orange',
  'Pomegranate',
  'Strawberry',
];

export const ScannerWorkspace: React.FC<ScannerWorkspaceProps> = ({
  health,
  healthError,
  samples,
  history,
  onAddHistory,
  onClearHistory,
  fileInputRef,
}) => {
  const [selectedPreview, setSelectedPreview] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [sampleFilter, setSampleFilter] = useState<'all' | 'fresh' | 'rotten'>('all');
  const [brokenSampleIds, setBrokenSampleIds] = useState<Record<string, boolean>>({});
  const previewFallbackRef = useRef(false);
  const [previewFailed, setPreviewFailed] = useState(false);

  const runUploadPrediction = async (file: File) => {
    setPredictionError(null);
    setCurrentResult(null);
    setPreviewFailed(false);
    previewFallbackRef.current = false;

    const objectUrl = URL.createObjectURL(file);
    setSelectedPreview(objectUrl);
    setSelectedName(file.name);
    setIsPredicting(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/ml/predict', {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail || payload.error || `Prediction failed with HTTP ${response.status}`,
        );
      }

      const result = payload as PredictionResult;
      setCurrentResult(result);
      onAddHistory({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        sourceLabel: file.name,
        previewUrl: objectUrl,
        result,
      });
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsPredicting(false);
    }
  };

  const runSamplePrediction = async (sample: SampleTestItem) => {
    setPredictionError(null);
    setCurrentResult(null);
    setPreviewFailed(false);
    const imgUrl = `/api/ml/samples/image/${encodeURIComponent(sample.filename)}`;
    setSelectedPreview(imgUrl);
    setSelectedName(`${sample.display_name} (${sample.filename})`);
    setIsPredicting(true);

    try {
      const response = await fetch(`/api/ml/predict-sample/${encodeURIComponent(sample.id)}`, {
        method: 'POST',
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail || payload.error || `Sample inference failed with HTTP ${response.status}`,
        );
      }
      const result = payload as PredictionResult;
      setCurrentResult(result);
      onAddHistory({
        id: `${Date.now()}-${sample.id}`,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        sourceLabel: `Held-out Test: ${sample.display_name}`,
        previewUrl: imgUrl,
        result,
      });
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsPredicting(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void runUploadPrediction(file);
    }
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void runUploadPrediction(file);
    }
  };

  const handleExportHistoryCsv = () => {
    if (history.length === 0) return;
    const header = [
      'Timestamp',
      'Source',
      'Predicted_Class',
      'Freshness_Status',
      'Fruit_Type',
      'Confidence_Pct',
      'Fresh_Prob_Pct',
      'Rotten_Prob_Pct',
      'Inference_ms',
    ];
    const rows = history.map((h) => [
      h.timestamp,
      `"${h.sourceLabel.replace(/"/g, '""')}"`,
      h.result.predicted_class,
      h.result.freshness_status,
      h.result.fruit_type,
      (h.result.confidence * 100).toFixed(2),
      (h.result.aggregated_freshness.fresh_probability * 100).toFixed(2),
      (h.result.aggregated_freshness.rotten_probability * 100).toFixed(2),
      h.result.inference_time_ms.toFixed(2),
    ]);
    const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `freshscan_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const filteredSamples = samples.filter((s) =>
    sampleFilter === 'all' ? true : s.freshness === sampleFilter,
  );

  return (
    <div className="space-y-12">
      {/* Editorial Hero & System Calibration Summary */}
      <section className="pt-2 pb-6 border-b border-[#DCE2D7]">
        <div className="max-w-3xl space-y-3">
          <p className="text-xs font-medium text-[#526358]">
            MobileNetV2 Transfer Classifier · 224×224 RGB · 8 Supported Fruit Species · 16 Classes
          </p>
          <h1
            className="text-3xl sm:text-4xl font-bold tracking-tight text-[#14241B]"
            style={{ textWrap: 'balance' }}
          >
            Optical Fruit Freshness &amp; Surface Spoilage Inspection
          </h1>
          <p className="text-base text-[#3D4F44] leading-relaxed max-w-2xl">
            Upload a photograph or select a held-out test specimen below to execute live
            TensorFlow/Keras inference. Every image undergoes format validation, white-matte RGB
            compositing, 224×224 bilinear resizing, and MobileNetV2 normalization prior to softmax
            scoring.
          </p>
          <div className="pt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-[#526358]">
            <span>Supported produce: {SUPPORTED_FRUITS.join(', ')}</span>
            <span aria-hidden="true">·</span>
            {healthError ? (
              <span className="text-[#B45309] font-medium">Backend status: {healthError}</span>
            ) : health?.model_ready ? (
              <span className="text-[#166534] font-medium">
                Model loaded (TensorFlow {health.tensorflow_version} · {health.num_classes} classes)
              </span>
            ) : (
              <span className="text-[#B45309] font-medium">
                Model training in progress or awaiting artifacts
              </span>
            )}
          </div>
        </div>
      </section>

      {/* Main 2-Column Asymmetric Inspection Console */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload Dropzone & Image Preview */}
        <div className="lg:col-span-5 bg-white border border-[#DCE2D7] rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-[#14241B]">Input Specimen</h2>
            {selectedPreview && (
              <button
                type="button"
                onClick={() => {
                  setSelectedPreview(null);
                  setSelectedName('');
                  setCurrentResult(null);
                  setPredictionError(null);
                }}
                className="inline-flex items-center gap-1.5 text-xs font-medium text-[#526358] hover:text-[#14241B] transition-colors whitespace-nowrap shrink-0"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
            onChange={handleFileChange}
            className="hidden"
          />

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`cursor-pointer rounded-lg border border-dashed p-6 text-center transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1B4332] ${
              isDragging
                ? 'border-[#1B4332] bg-[#EFF4F0]'
                : 'border-[#B8C4B9] bg-[#F7F8F5] hover:border-[#1B4332]'
            }`}
          >
            <Upload className="w-6 h-6 mx-auto text-[#1B4332] mb-2.5" />
            <p className="text-sm font-semibold text-[#14241B]">
              Drop a fruit image here or click to browse
            </p>
            <p className="text-xs text-[#526358] mt-1">
              JPEG, PNG, WebP, or BMP · Up to 15 MB · Minimum 16×16 px
            </p>
          </div>

          {selectedPreview && (
            <div className="pt-4 border-t border-[#E6EBE1] space-y-3">
              <div className="aspect-[4/3] w-full rounded-lg overflow-hidden bg-[#F7F8F5] border border-[#E6EBE1] flex items-center justify-center">
                {!previewFailed ? (
                  <img
                    src={selectedPreview}
                    alt={selectedName || 'Selected fruit specimen'}
                    referrerPolicy="no-referrer"
                    onError={() => setPreviewFailed(true)}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-6 text-[#526358]">
                    <FileWarning className="w-8 h-8 mx-auto mb-2 text-[#B45309]" />
                    <p className="text-xs font-medium">Preview unavailable for unreadable file</p>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between text-xs text-[#526358]">
                <span className="truncate max-w-[260px]" title={selectedName}>
                  {selectedName}
                </span>
                {currentResult && (
                  <span className="font-mono tabular-nums shrink-0">
                    {currentResult.image_metadata.original_width}×
                    {currentResult.image_metadata.original_height} px ·{' '}
                    {currentResult.image_metadata.original_mode}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Live Inference Output & Probability Breakdown */}
        <div className="lg:col-span-7 bg-white border border-[#DCE2D7] rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-[#14241B]">Inference Report</h2>
            {currentResult && (
              <span className="text-xs font-mono tabular-nums text-[#526358]">
                Latency: {currentResult.inference_time_ms.toFixed(1)} ms · Entropy:{' '}
                {currentResult.normalized_entropy.toFixed(3)}
              </span>
            )}
          </div>

          {/* Pending / Loading State */}
          {isPredicting && (
            <div className="py-14 text-center space-y-3">
              <div className="w-7 h-7 border-2 border-[#1B4332] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm font-medium text-[#14241B]">
                Decoding RGB tensor and running MobileNetV2 forward pass...
              </p>
            </div>
          )}

          {/* Error State (Invalid image, missing model, or backend offline) */}
          {!isPredicting && predictionError && (
            <div className="rounded-lg border border-[#F3D2CE] bg-[#FDF5F4] p-4 space-y-2">
              <div className="flex items-center gap-2 text-sm font-semibold text-[#991B1B]">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>Unable to Complete Prediction</span>
              </div>
              <p className="text-xs text-[#7F1D1D] leading-relaxed">{predictionError}</p>
            </div>
          )}

          {/* Empty State */}
          {!isPredicting && !predictionError && !currentResult && (
            <div className="py-12 text-center space-y-3 border border-dashed border-[#E6EBE1] rounded-lg bg-[#FAFBF9] px-6">
              <ImageIcon className="w-7 h-7 text-[#6A7B70] mx-auto" />
              <p className="text-sm font-medium text-[#14241B]">No Specimen Evaluated Yet</p>
              <p className="text-xs text-[#526358] max-w-md mx-auto leading-relaxed">
                Upload a fruit photograph from your device or click one of the held-out test set
                specimens below to inspect real softmax class probabilities.
              </p>
            </div>
          )}

          {/* Populated Prediction Result */}
          {!isPredicting && currentResult && (
            <div className="space-y-6">
              {/* Primary Classification Readout */}
              <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#E6EBE1]">
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    {currentResult.freshness_status === 'fresh' ? (
                      <CheckCircle2 className="w-5 h-5 text-[#166534] shrink-0" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-[#B91C1C] shrink-0" />
                    )}
                    <span
                      className={`text-xs font-semibold ${
                        currentResult.freshness_status === 'fresh'
                          ? 'text-[#166534]'
                          : 'text-[#B91C1C]'
                      }`}
                    >
                      {currentResult.freshness_status === 'fresh'
                        ? 'Classified Fresh'
                        : 'Classified Rotten / Spoiled'}
                    </span>
                    <span className="text-xs text-[#526358]" aria-hidden="true">
                      ·
                    </span>
                    <span className="text-xs text-[#526358] capitalize">
                      Species: {currentResult.fruit_type}
                    </span>
                  </div>
                  <h3 className="text-2xl sm:text-3xl font-bold text-[#14241B]">
                    {currentResult.display_name}
                  </h3>
                  {currentResult.sample_ground_truth && (
                    <p className="text-xs text-[#526358]">
                      Held-out ground truth:{' '}
                      <strong className="font-semibold text-[#14241B]">
                        {currentResult.sample_ground_truth.display_name}
                      </strong>{' '}
                      ·{' '}
                      {currentResult.sample_ground_truth.matched_ground_truth
                        ? 'Correct prediction on unseen test image'
                        : 'Misclassified test sample'}
                    </p>
                  )}
                </div>

                <div className="sm:text-right">
                  <p className="text-xs text-[#526358]">Top-1 Softmax Confidence</p>
                  <p className="text-3xl font-mono font-bold tabular-nums text-[#14241B]">
                    {(currentResult.confidence * 100).toFixed(2)}
                    <span className="text-sm font-mono text-[#526358] ml-1">%</span>
                  </p>
                </div>
              </div>

              {/* Low / Moderate Confidence Warning */}
              {currentResult.confidence_warning && (
                <div className="rounded-lg border border-[#EBD6B3] bg-[#FBF7F0] p-3.5 text-xs text-[#78350F] leading-relaxed flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-[#B45309] shrink-0 mt-0.5" />
                  <span>{currentResult.confidence_warning}</span>
                </div>
              )}

              {/* Aggregated Binary Fresh vs Rotten Probability Balance */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-[#166534] font-mono tabular-nums">
                    Aggregate Fresh:{' '}
                    {(currentResult.aggregated_freshness.fresh_probability * 100).toFixed(1)}%
                  </span>
                  <span className="font-medium text-[#B91C1C] font-mono tabular-nums">
                    Aggregate Rotten:{' '}
                    {(currentResult.aggregated_freshness.rotten_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 w-full rounded-full bg-[#F3E2E0] overflow-hidden flex">
                  <div
                    className="h-full bg-[#1B4332] transition-opacity duration-150"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, currentResult.aggregated_freshness.fresh_probability * 100),
                      )}%`,
                    }}
                  />
                </div>
              </div>

              {/* Top-5 Class Distribution Table */}
              <div className="pt-2 space-y-2.5">
                <h4 className="text-xs font-semibold text-[#526358]">
                  Top-5 Class Softmax Probabilities
                </h4>
                <div className="divide-y divide-[#E6EBE1] border-t border-b border-[#E6EBE1]">
                  {currentResult.top_predictions.map((pred) => {
                    const pct = pred.confidence * 100;
                    return (
                      <div
                        key={pred.class_label}
                        className="py-2 flex items-center justify-between gap-4 text-xs"
                      >
                        <div className="min-w-0 flex items-center gap-2">
                          <span className="font-medium text-[#14241B] truncate">
                            {pred.display_name}
                          </span>
                          <span className="text-[#6A7B70] font-mono">({pred.class_label})</span>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <div className="w-28 h-1.5 bg-[#EFF2EC] rounded-full overflow-hidden hidden sm:block">
                            <div
                              className={`h-full ${
                                pred.freshness === 'fresh' ? 'bg-[#1B4332]' : 'bg-[#B91C1C]'
                              }`}
                              style={{ width: `${Math.min(100, pct)}%` }}
                            />
                          </div>
                          <span className="font-mono tabular-nums font-medium text-[#14241B] w-14 text-right">
                            {pct.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Explicit Food Safety Disclaimer */}
              <div className="rounded-lg border border-[#DCE2D7] bg-[#F7F8F5] p-3.5 flex items-start gap-2.5 text-xs text-[#3D4F44] leading-relaxed">
                <ShieldAlert className="w-4 h-4 text-[#1B4332] shrink-0 mt-0.5" />
                <p>
                  <strong className="font-semibold text-[#14241B]">Food Safety Notice: </strong>
                  {currentResult.food_safety_disclaimer}
                </p>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Held-Out Test Set Sample-Image Testing */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-[#14241B]">
              Held-Out Test Set Specimens (Live Inference)
            </h2>
            <p className="text-xs text-[#526358] mt-0.5">
              Real images extracted strictly from the unseen 15% test split · Clicking any specimen
              executes live TensorFlow model inference
            </p>
          </div>

          <div className="flex items-center gap-1 p-1 bg-[#EFF2EC] rounded-lg self-start">
            {(['all', 'fresh', 'rotten'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setSampleFilter(mode)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize whitespace-nowrap shrink-0 ${
                  sampleFilter === mode
                    ? 'bg-white text-[#14241B] shadow-xs'
                    : 'text-[#526358] hover:text-[#14241B]'
                }`}
              >
                {mode === 'all' ? 'All Specimens' : mode}
              </button>
            ))}
          </div>
        </div>

        {samples.length === 0 ? (
          <div className="bg-white border border-[#DCE2D7] rounded-xl p-8 text-center text-xs text-[#526358]">
            Held-out test samples will appear automatically once dataset preparation (`train_model.py`) completes.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            {filteredSamples.map((sample) => {
              const imgUrl = `/api/ml/samples/image/${encodeURIComponent(sample.filename)}`;
              const isBroken = Boolean(brokenSampleIds[sample.id]);
              return (
                <button
                  key={sample.id}
                  type="button"
                  disabled={isPredicting}
                  onClick={() => void runSamplePrediction(sample)}
                  className="group text-left bg-white border border-[#DCE2D7] rounded-lg p-2 hover:border-[#1B4332] transition-colors disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-[#1B4332]"
                >
                  <div className="aspect-square w-full rounded bg-[#F7F8F5] overflow-hidden mb-2 flex items-center justify-center">
                    {!isBroken ? (
                      <img
                        src={imgUrl}
                        alt={sample.display_name}
                        referrerPolicy="no-referrer"
                        onError={() =>
                          setBrokenSampleIds((prev) => ({ ...prev, [sample.id]: true }))
                        }
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-150"
                      />
                    ) : (
                      <ImageIcon className="w-5 h-5 text-[#6A7B70]" />
                    )}
                  </div>
                  <p className="text-xs font-semibold text-[#14241B] truncate">
                    {sample.display_name}
                  </p>
                  <p className="text-[11px] text-[#526358] font-mono tabular-nums truncate">
                    {sample.original_dimensions}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* Session Prediction History */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-[#14241B]">Session Prediction Log</h2>
            <p className="text-xs text-[#526358]">
              {history.length} inspection{history.length === 1 ? '' : 's'} recorded in current
              browser session
            </p>
          </div>
          {history.length > 0 && (
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={handleExportHistoryCsv}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#14241B] bg-white border border-[#DCE2D7] rounded-lg hover:bg-[#EFF2EC] transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
              <button
                type="button"
                onClick={onClearHistory}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#991B1B] bg-white border border-[#DCE2D7] rounded-lg hover:bg-[#FDF5F4] transition-colors whitespace-nowrap shrink-0"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear Log
              </button>
            </div>
          )}
        </div>

        {history.length === 0 ? (
          <div className="bg-white border border-[#DCE2D7] rounded-xl p-6 text-xs text-[#526358]">
            Completed predictions from uploaded files and held-out test specimens will be logged here during your session.
          </div>
        ) : (
          <div className="bg-white border border-[#DCE2D7] rounded-xl overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#DCE2D7] text-xs font-semibold text-[#526358] bg-[#FAFBF9]">
                  <th className="py-2.5 px-4">Time</th>
                  <th className="py-2.5 px-4">Specimen</th>
                  <th className="py-2.5 px-4">Predicted Class</th>
                  <th className="py-2.5 px-4">Status</th>
                  <th className="py-2.5 px-4 text-right">Confidence</th>
                  <th className="py-2.5 px-4 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E6EBE1] text-xs">
                {history.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => {
                      setSelectedPreview(item.previewUrl);
                      setSelectedName(item.sourceLabel);
                      setCurrentResult(item.result);
                      setPredictionError(null);
                    }}
                    className="cursor-pointer hover:bg-[#F7F8F5] transition-colors"
                  >
                    <td className="py-2.5 px-4 font-mono tabular-nums text-[#526358] whitespace-nowrap">
                      {item.timestamp}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-[#14241B] max-w-[240px] truncate">
                      {item.sourceLabel}
                    </td>
                    <td className="py-2.5 px-4 text-[#14241B] font-medium">
                      {item.result.display_name}
                    </td>
                    <td className="py-2.5 px-4">
                      <span
                        className={`font-medium ${
                          item.result.freshness_status === 'fresh'
                            ? 'text-[#166534]'
                            : 'text-[#B91C1C]'
                        }`}
                      >
                        {item.result.freshness_status === 'fresh' ? 'Fresh' : 'Rotten'}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums font-semibold text-[#14241B]">
                      {(item.result.confidence * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#526358]">
                      {item.result.inference_time_ms.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
