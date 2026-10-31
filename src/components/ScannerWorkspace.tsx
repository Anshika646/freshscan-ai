import React, { useEffect, useRef, useState } from 'react';
import {
  Upload,
  Camera,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  RotateCcw,
  Download,
  Trash2,
  FileWarning,
  Image as ImageIcon,
  Search,
  Play,
  X,
} from 'lucide-react';
import type {
  HealthResponse,
  HistoryEntry,
  PredictionResult,
  SampleTestItem,
  SupportedFruitId,
} from '../types';

interface ScannerWorkspaceProps {
  health: HealthResponse | null;
  healthError: string | null;
  samples: SampleTestItem[];
  history: HistoryEntry[];
  onAddHistory: (entry: HistoryEntry) => void;
  onClearHistory: () => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  embedded?: boolean;
  externalSampleRequest?: SampleTestItem | null;
  onHandledExternalSample?: () => void;
  onScanningChange?: (scanning: boolean) => void;
  onPredictedFruit?: (fruit: SupportedFruitId | null) => void;
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

const CATEGORY_FILTERS: Array<{ id: string; label: string }> = [
  { id: 'all', label: 'All 8 Produce' },
  { id: 'apple', label: 'Apple' },
  { id: 'banana', label: 'Banana' },
  { id: 'grape', label: 'Grape' },
  { id: 'guava', label: 'Guava' },
  { id: 'jujube', label: 'Jujube' },
  { id: 'orange', label: 'Orange' },
  { id: 'pomegranate', label: 'Pomegranate' },
  { id: 'strawberry', label: 'Strawberry' },
];

export const ScannerWorkspace: React.FC<ScannerWorkspaceProps> = ({
  health,
  healthError,
  samples,
  history,
  onAddHistory,
  onClearHistory,
  fileInputRef,
  embedded = false,
  externalSampleRequest = null,
  onHandledExternalSample,
  onScanningChange,
  onPredictedFruit,
}) => {
  const [inputMode, setInputMode] = useState<'upload' | 'camera'>('upload');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [selectedPreview, setSelectedPreview] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string>('');
  const [lastUploadedFile, setLastUploadedFile] = useState<File | null>(null);
  const [lastSampleItem, setLastSampleItem] = useState<SampleTestItem | null>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);

  const [sampleFilter, setSampleFilter] = useState<'all' | 'fresh' | 'rotten'>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [sampleQuery, setSampleQuery] = useState<string>('');
  const [brokenSampleIds, setBrokenSampleIds] = useState<Record<string, boolean>>({});
  const [previewFailed, setPreviewFailed] = useState(false);

  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    onScanningChange?.(isPredicting);
  }, [isPredicting, onScanningChange]);

  const startCameraStream = async () => {
    setCameraError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Camera access is not supported in this browser context. Please upload an image file.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      });
    } catch (err) {
      setCameraError(
        err instanceof Error
          ? `Unable to access camera (${err.message}). Check browser camera permissions or use File Upload.`
          : 'Unable to access camera. Check permissions or use File Upload.',
      );
    }
  };

  const handleCaptureCameraFrame = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const capturedFile = new File([blob], `camera_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        stopCameraStream();
        setInputMode('upload');
        void runUploadPrediction(capturedFile);
      },
      'image/jpeg',
      0.92,
    );
  };

  const runUploadPrediction = async (file: File) => {
    setPredictionError(null);
    setCurrentResult(null);
    setPreviewFailed(false);
    setLastUploadedFile(file);
    setLastSampleItem(null);

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
      onPredictedFruit?.(result.fruit_type as SupportedFruitId);
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
    stopCameraStream();
    setInputMode('upload');
    setPredictionError(null);
    setCurrentResult(null);
    setPreviewFailed(false);
    setLastSampleItem(sample);
    setLastUploadedFile(null);

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
      onPredictedFruit?.(result.fruit_type as SupportedFruitId);
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

  useEffect(() => {
    if (externalSampleRequest) {
      void runSamplePrediction(externalSampleRequest);
      onHandledExternalSample?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalSampleRequest]);

  const handleRetryLast = () => {
    if (lastUploadedFile) {
      void runUploadPrediction(lastUploadedFile);
    } else if (lastSampleItem) {
      void runSamplePrediction(lastSampleItem);
    }
  };

  const handleResetScanner = () => {
    stopCameraStream();
    setSelectedPreview(null);
    setSelectedName('');
    setCurrentResult(null);
    setPredictionError(null);
    setLastUploadedFile(null);
    setLastSampleItem(null);
    onPredictedFruit?.(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      stopCameraStream();
      setInputMode('upload');
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

  const filteredSamples = samples.filter((s) => {
    const matchesFreshness = sampleFilter === 'all' ? true : s.freshness === sampleFilter;
    const matchesCategory = categoryFilter === 'all' ? true : s.fruit_type === categoryFilter;
    const q = sampleQuery.trim().toLowerCase();
    const matchesQuery =
      !q ||
      s.display_name.toLowerCase().includes(q) ||
      s.fruit_type.toLowerCase().includes(q) ||
      s.ground_truth_class.toLowerCase().includes(q);
    return matchesFreshness && matchesCategory && matchesQuery;
  });

  const handleTestRandomSpecimen = () => {
    const pool = filteredSamples.length > 0 ? filteredSamples : samples;
    if (pool.length === 0) return;
    const randomSample = pool[Math.floor(Math.random() * pool.length)];
    void runSamplePrediction(randomSample);
  };

  return (
    <div className="space-y-12">
      {/* Header (shown in standalone Scanner view or compact in embedded Overview) */}
      {!embedded ? (
        <section className="pt-2 pb-6 border-b border-[#18221D]">
          <div className="max-w-3xl space-y-3">
            <p className="text-xs font-medium text-[#8FA396]">
              MobileNetV2 Transfer Classifier · 224×224 RGB · 8 Supported Fruit Species · 16 Classes
            </p>
            <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-[#F3F7F4] text-balance">
              Optical Produce Freshness &amp; Spoilage Scanner
            </h1>
            <p className="text-base text-[#A3B5AA] leading-relaxed max-w-2xl">
              Upload a fruit photograph, capture directly from your camera, or select an unseen
              held-out test specimen below to execute real-time TensorFlow/Keras inference.
            </p>
            <div className="pt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-[#8FA396]">
              <span>Supported produce: {SUPPORTED_FRUITS.join(', ')}</span>
              <span aria-hidden="true">·</span>
              {healthError ? (
                <span className="text-[#F59E0B] font-medium">Engine status: {healthError}</span>
              ) : health?.model_ready ? (
                <span className="text-[#34D399] font-medium">
                  TensorFlow {health.tensorflow_version} active · {health.num_classes} calibrated classes
                </span>
              ) : (
                <span className="text-[#F59E0B] font-medium">
                  Warming up TensorFlow model artifacts...
                </span>
              )}
            </div>
          </div>
        </section>
      ) : (
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-2">
          <div className="space-y-2">
            <p className="text-xs font-medium text-[#34D399]">
              Live TensorFlow / Keras Inference Workbench
            </p>
            <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#F3F7F4] text-balance">
              01. Real-Time Produce Optical Scanner
            </h2>
            <p className="text-sm text-[#9EB0A5] max-w-2xl">
              Drop any supported fruit photo (JPG, PNG, WEBP) or test an unseen specimen from our
              15% held-out benchmark split. Every score comes live from MobileNetV2.
            </p>
          </div>
          <div className="text-xs text-[#8FA396] shrink-0">
            {health?.model_ready ? (
              <span className="text-[#34D399] font-medium">
                Engine Ready · TensorFlow {health.tensorflow_version}
              </span>
            ) : (
              <span className="text-[#F59E0B] font-medium">Warming up ML service...</span>
            )}
          </div>
        </div>
      )}

      {/* Main 2-Column Balanced Inspection Workbench */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        {/* Left Column: Input Specimen (Upload or Camera) */}
        <div className="lg:col-span-5 bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 flex flex-col justify-between space-y-6">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-base font-semibold text-[#F3F7F4]">Input Specimen</h3>

              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 p-1 bg-[#121B16] border border-[#1B2821] rounded-lg">
                  <button
                    type="button"
                    onClick={() => {
                      stopCameraStream();
                      setInputMode('upload');
                    }}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      inputMode === 'upload'
                        ? 'bg-[#059669] text-white'
                        : 'text-[#8FA396] hover:text-[#F3F7F4]'
                    }`}
                  >
                    <Upload className="w-3.5 h-3.5" />
                    Upload
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setInputMode('camera');
                      void startCameraStream();
                    }}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      inputMode === 'camera'
                        ? 'bg-[#059669] text-white'
                        : 'text-[#8FA396] hover:text-[#F3F7F4]'
                    }`}
                  >
                    <Camera className="w-3.5 h-3.5" />
                    Camera
                  </button>
                </div>

                {(selectedPreview || predictionError) && (
                  <button
                    type="button"
                    onClick={handleResetScanner}
                    className="inline-flex items-center gap-1 text-xs font-medium text-[#8FA396] hover:text-[#F3F7F4] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Reset
                  </button>
                )}
              </div>
            </div>

            {!embedded && (
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
                onChange={handleFileChange}
                className="hidden"
              />
            )}
            {embedded && (
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
                onChange={handleFileChange}
                className="hidden"
              />
            )}

            {inputMode === 'camera' ? (
              <div className="space-y-3">
                {cameraError ? (
                  <div className="rounded-lg border border-[#5B2121] bg-[#1F1012] p-4 space-y-2.5 text-xs text-[#FCA5A5]">
                    <p className="font-semibold text-[#FECACA]">Camera Unavailable</p>
                    <p className="leading-relaxed">{cameraError}</p>
                    <button
                      type="button"
                      onClick={() => setInputMode('upload')}
                      className="px-3 py-1.5 text-xs font-medium text-[#F3F7F4] bg-[#141E18] border border-[#24352C] rounded-md hover:border-[#10B981] cursor-pointer whitespace-nowrap shrink-0"
                    >
                      Switch to File Upload
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="aspect-[4/3] w-full rounded-lg overflow-hidden bg-[#060907] border border-[#1E2C24] relative flex items-center justify-center">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <button
                        type="button"
                        disabled={!cameraActive}
                        onClick={handleCaptureCameraFrame}
                        className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold text-white bg-[#059669] rounded-lg hover:bg-[#10B981] transition-colors disabled:opacity-50 whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        <Camera className="w-4 h-4" />
                        Capture &amp; Analyze
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          stopCameraStream();
                          setInputMode('upload');
                        }}
                        className="inline-flex items-center gap-1 px-3 py-2.5 text-xs font-medium text-[#9EB0A5] bg-[#121B16] border border-[#1E2C24] rounded-lg hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
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
                className={`cursor-pointer rounded-lg border border-dashed p-6 text-center transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#10B981] ${
                  isDragging
                    ? 'border-[#10B981] bg-[#0F241B]'
                    : 'border-[#25362D] bg-[#090E0B] hover:border-[#10B981]/70 hover:bg-[#0D1611]'
                }`}
              >
                <Upload className="w-6 h-6 mx-auto text-[#10B981] mb-2" />
                <p className="text-sm font-semibold text-[#F3F7F4]">
                  Drop a produce photo here or click to browse
                </p>
                <p className="text-xs text-[#8FA396] mt-1">
                  JPG, JPEG, PNG, WEBP, or BMP · Max 15 MB · Min 16×16 px
                </p>
              </div>
            )}

            {selectedPreview && inputMode === 'upload' && (
              <div className="pt-4 border-t border-[#18221D] space-y-3">
                <div className="aspect-[4/3] w-full rounded-lg overflow-hidden bg-[#060907] border border-[#1E2C24] relative flex items-center justify-center">
                  {!previewFailed ? (
                    <img
                      src={selectedPreview}
                      alt={selectedName || 'Selected fruit specimen'}
                      referrerPolicy="no-referrer"
                      onError={() => setPreviewFailed(true)}
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-6 text-[#8FA396]">
                      <FileWarning className="w-8 h-8 mx-auto mb-2 text-[#F59E0B]" />
                      <p className="text-xs font-medium">Preview unavailable for unreadable file</p>
                    </div>
                  )}

                  {/* Animated Optical Laser Sweep Overlay during Real Model Inference */}
                  {isPredicting && (
                    <div
                      aria-hidden="true"
                      className="absolute inset-0 bg-[#070A08]/45 pointer-events-none overflow-hidden"
                    >
                      <div className="freshscan-laser-line w-full h-12 bg-gradient-to-b from-transparent via-[#10B981]/45 to-transparent border-b border-[#34D399]" />
                      <div className="absolute bottom-2.5 left-3 right-3 flex items-center justify-between text-[11px] font-mono text-[#A7F3D0] bg-[#070A08]/85 px-2.5 py-1 rounded border border-[#10B981]/40">
                        <span>SCANNING 224×224 TENSOR</span>
                        <span>MOBILENETV2</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between gap-2 text-xs text-[#8FA396]">
                  <span className="truncate max-w-[220px]" title={selectedName}>
                    {selectedName}
                  </span>
                  {currentResult && (
                    <span className="font-mono tabular-nums shrink-0 text-[#A3B5AA]">
                      {currentResult.image_metadata.original_width}×
                      {currentResult.image_metadata.original_height} px ·{' '}
                      {currentResult.image_metadata.original_mode}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-[#18221D] text-xs text-[#8FA396] space-y-1">
            <div className="font-medium text-[#D5E1D9]">Deterministic Tensor Preprocessing</div>
            <p className="font-mono text-[11px] text-[#8FA396] leading-relaxed">
              RGB (#FFFFFF alpha composite) → 224×224 Bilinear → (x / 127.5) − 1.0
            </p>
          </div>
        </div>

        {/* Right Column: Live Inference Output & Probability Breakdown */}
        <div className="lg:col-span-7 bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 flex flex-col justify-between space-y-6">
          <div className="space-y-6">
            <div className="flex items-center justify-between gap-4">
              <h3 className="text-base font-semibold text-[#F3F7F4]">Diagnostic Output</h3>
              {currentResult && (
                <span className="text-xs font-mono tabular-nums text-[#8FA396]">
                  Latency: {currentResult.inference_time_ms.toFixed(1)} ms · Entropy:{' '}
                  {currentResult.normalized_entropy.toFixed(3)}
                </span>
              )}
            </div>

            {/* Pending / Scanning State */}
            {isPredicting && (
              <div className="py-16 text-center space-y-3 border border-[#1B2821] rounded-lg bg-[#080C0A]">
                <div className="w-7 h-7 border-2 border-[#10B981] border-t-transparent rounded-full animate-spin mx-auto" />
                <p className="text-sm font-medium text-[#F3F7F4]">
                  Decoding RGB tensor and running MobileNetV2 forward pass...
                </p>
                <p className="text-xs text-[#8FA396]">
                  Evaluating calibrated softmax probabilities across 16 produce states
                </p>
              </div>
            )}

            {/* Error State with Retry & Reset */}
            {!isPredicting && predictionError && (
              <div className="rounded-lg border border-[#5B2121] bg-[#1B0E10] p-4 space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-[#FCA5A5]">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Unable to Complete Prediction</span>
                </div>
                <p className="text-xs text-[#FECACA]/90 leading-relaxed">{predictionError}</p>
                <div className="flex items-center gap-3 pt-1">
                  {(lastUploadedFile || lastSampleItem) && (
                    <button
                      type="button"
                      onClick={handleRetryLast}
                      className="px-3 py-1.5 text-xs font-semibold text-white bg-[#059669] rounded-md hover:bg-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Retry Analysis
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleResetScanner}
                    className="px-3 py-1.5 text-xs font-medium text-[#D5E1D9] bg-[#141E18] border border-[#24352C] rounded-md hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Clear File
                  </button>
                </div>
              </div>
            )}

            {/* Empty State with Direct One-Click Actions */}
            {!isPredicting && !predictionError && !currentResult && (
              <div className="py-14 text-center space-y-4 border border-dashed border-[#213028] rounded-lg bg-[#080C0A] px-6">
                <ImageIcon className="w-7 h-7 text-[#6E8577] mx-auto" />
                <div className="space-y-1.5">
                  <p className="text-sm font-semibold text-[#F3F7F4]">No Specimen Evaluated Yet</p>
                  <p className="text-xs text-[#8FA396] max-w-md mx-auto leading-relaxed">
                    Upload a fruit photograph from your device or run live inference on a random
                    unseen specimen from the held-out test set.
                  </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-3 pt-1">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-[#059669] rounded-lg hover:bg-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    Upload Photo
                  </button>
                  {samples.length > 0 && (
                    <button
                      type="button"
                      onClick={handleTestRandomSpecimen}
                      className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-[#E2ECE5] bg-[#121B16] border border-[#25362D] rounded-lg hover:border-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5 text-[#10B981]" />
                      Test Unseen Specimen
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Populated Prediction Result */}
            {!isPredicting && currentResult && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#18221D]">
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      {currentResult.freshness_status === 'fresh' ? (
                        <CheckCircle2 className="w-5 h-5 text-[#34D399] shrink-0" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-[#F87171] shrink-0" />
                      )}
                      <span
                        className={`text-xs font-semibold ${
                          currentResult.freshness_status === 'fresh'
                            ? 'text-[#34D399]'
                            : 'text-[#F87171]'
                        }`}
                      >
                        {currentResult.freshness_status === 'fresh'
                          ? 'Classified Fresh'
                          : 'Classified Spoiled / Rotten'}
                      </span>
                      <span className="text-xs text-[#6E8577]" aria-hidden="true">
                        ·
                      </span>
                      <span className="text-xs text-[#9EB0A5] capitalize">
                        Detected category: {currentResult.fruit_type}
                      </span>
                    </div>
                    <h4 className="font-display text-2xl sm:text-3xl font-bold text-[#F3F7F4]">
                      {currentResult.display_name}
                    </h4>
                    {currentResult.sample_ground_truth && (
                      <p className="text-xs text-[#8FA396]">
                        Held-out ground truth:{' '}
                        <strong className="font-semibold text-[#E2ECE5]">
                          {currentResult.sample_ground_truth.display_name}
                        </strong>{' '}
                        ·{' '}
                        <span
                          className={
                            currentResult.sample_ground_truth.matched_ground_truth
                              ? 'text-[#34D399] font-medium'
                              : 'text-[#FBBF24] font-medium'
                          }
                        >
                          {currentResult.sample_ground_truth.matched_ground_truth
                            ? 'Matches unseen test label'
                            : 'Differs from test label'}
                        </span>
                      </p>
                    )}
                  </div>

                  <div className="sm:text-right">
                    <p className="text-xs text-[#8FA396]">Top-1 Softmax Confidence</p>
                    <p className="text-3xl font-mono font-bold tabular-nums text-[#F3F7F4]">
                      {(currentResult.confidence * 100).toFixed(2)}
                      <span className="text-sm font-mono text-[#8FA396] ml-1">%</span>
                    </p>
                  </div>
                </div>

                {/* Honest Explanation & Uncertainty Warning */}
                <p className="text-xs text-[#A3B5AA] leading-relaxed">
                  {currentResult.freshness_status === 'fresh'
                    ? `Optical surface features align with intact ${currentResult.fruit_type} epidermis (${(
                        currentResult.aggregated_freshness.fresh_probability * 100
                      ).toFixed(1)}% cumulative fresh probability across species).`
                    : `Optical surface features indicate peel discoloration, bruising, or lesion patterns associated with spoiled ${currentResult.fruit_type} (${(
                        currentResult.aggregated_freshness.rotten_probability * 100
                      ).toFixed(1)}% cumulative spoilage probability).`}
                </p>

                {currentResult.confidence_warning && (
                  <div className="rounded-lg border border-[#593E14] bg-[#1C1509] p-3.5 text-xs text-[#FDE68A] leading-relaxed flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-[#F59E0B] shrink-0 mt-0.5" />
                    <span>{currentResult.confidence_warning}</span>
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-[#34D399] font-mono tabular-nums">
                      Aggregate Fresh:{' '}
                      {(currentResult.aggregated_freshness.fresh_probability * 100).toFixed(1)}%
                    </span>
                    <span className="font-medium text-[#F87171] font-mono tabular-nums">
                      Aggregate Spoiled:{' '}
                      {(currentResult.aggregated_freshness.rotten_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <svg
                    className="h-2.5 w-full rounded-full overflow-hidden block"
                    viewBox="0 0 100 6"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <rect x="0" y="0" width="100" height="6" fill="#3B181B" />
                    <rect
                      x="0"
                      y="0"
                      width={Math.max(
                        0,
                        Math.min(100, currentResult.aggregated_freshness.fresh_probability * 100),
                      )}
                      height="6"
                      fill="#10B981"
                    />
                  </svg>
                </div>

                <div className="pt-1 space-y-2.5">
                  <h5 className="text-xs font-semibold text-[#8FA396]">
                    Top-5 Class Softmax Breakdown
                  </h5>
                  <div className="divide-y divide-[#18221D] border-t border-b border-[#18221D]">
                    {currentResult.top_predictions.map((pred) => {
                      const pct = pred.confidence * 100;
                      return (
                        <div
                          key={pred.class_label}
                          className="py-2 flex items-center justify-between gap-4 text-xs"
                        >
                          <div className="min-w-0 flex items-center gap-2">
                            <span className="font-medium text-[#E2ECE5] truncate">
                              {pred.display_name}
                            </span>
                            <span className="text-[#6E8577] font-mono hidden sm:inline">
                              ({pred.class_label})
                            </span>
                          </div>
                          <div className="flex items-center gap-3 shrink-0">
                            <svg
                              className="w-28 h-1.5 rounded-full overflow-hidden hidden sm:block"
                              viewBox="0 0 100 4"
                              preserveAspectRatio="none"
                              aria-hidden="true"
                            >
                              <rect x="0" y="0" width="100" height="4" fill="#141F19" />
                              <rect
                                x="0"
                                y="0"
                                width={Math.min(100, pct)}
                                height="4"
                                fill={pred.freshness === 'fresh' ? '#10B981' : '#EF4444'}
                              />
                            </svg>
                            <span className="font-mono tabular-nums font-medium text-[#F3F7F4] w-14 text-right">
                              {pct.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Persistent Food Safety Notice */}
          <div className="rounded-lg border border-[#1B2821] bg-[#090E0B] p-3.5 flex items-start gap-2.5 text-xs text-[#9EB0A5] leading-relaxed">
            <ShieldAlert className="w-4 h-4 text-[#10B981] shrink-0 mt-0.5" />
            <p>
              <strong className="font-semibold text-[#E2ECE5]">Food Safety Limitation: </strong>
              {currentResult?.food_safety_disclaimer ||
                'Visual classification evaluates external peel appearance only. It cannot detect internal rot, microbial pathogens, or chemical residue, and never guarantees food safety.'}
            </p>
          </div>
        </div>
      </section>

      {/* Held-Out Test Set Sample-Image Testing with Produce Category Filter */}
      <section className="space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <h3 className="font-display text-lg font-bold text-[#F3F7F4]">
              Held-Out Test Specimens (16 Unseen Benchmark Photos)
            </h3>
            <p className="text-xs text-[#8FA396] mt-0.5">
              Zero overlap with training or validation splits · Click any specimen to execute live
              TensorFlow inference
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="sr-only" htmlFor="produce-category-select">
              Filter by produce category
            </label>
            <select
              id="produce-category-select"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="px-3 py-1.5 text-xs font-medium text-[#E2ECE5] bg-[#0C120F] border border-[#1E2C24] rounded-lg focus:outline-none focus:border-[#10B981] cursor-pointer"
            >
              {CATEGORY_FILTERS.map((cat) => (
                <option key={cat.id} value={cat.id} className="bg-[#0C120F] text-[#E2ECE5]">
                  {cat.label}
                </option>
              ))}
            </select>

            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#8FA396] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="search"
                value={sampleQuery}
                onChange={(e) => setSampleQuery(e.target.value)}
                placeholder="Search specimen..."
                aria-label="Search test specimens"
                className="pl-8 pr-3 py-1.5 text-xs text-[#E2ECE5] placeholder-[#6E8577] bg-[#0C120F] border border-[#1E2C24] rounded-lg focus:outline-none focus:border-[#10B981] w-40 sm:w-44"
              />
            </div>

            <div className="flex items-center gap-1 p-1 bg-[#121B16] border border-[#1B2821] rounded-lg">
              {(['all', 'fresh', 'rotten'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setSampleFilter(mode)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors capitalize whitespace-nowrap shrink-0 cursor-pointer ${
                    sampleFilter === mode
                      ? 'bg-[#059669] text-white'
                      : 'text-[#8FA396] hover:text-[#F3F7F4]'
                  }`}
                >
                  {mode === 'all' ? 'All' : mode === 'rotten' ? 'Spoiled' : 'Fresh'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {samples.length === 0 ? (
          <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-8 text-center text-xs text-[#8FA396]">
            Held-out test specimens are being prepared by the dataset pipeline and will appear automatically.
          </div>
        ) : filteredSamples.length === 0 ? (
          <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-8 text-center space-y-2.5 text-xs text-[#8FA396]">
            <p>No held-out test specimens match the active filter criteria.</p>
            <button
              type="button"
              onClick={() => {
                setSampleQuery('');
                setSampleFilter('all');
                setCategoryFilter('all');
              }}
              className="px-3 py-1.5 text-xs font-medium text-[#E2ECE5] bg-[#121B16] border border-[#24352C] rounded-md hover:border-[#10B981] cursor-pointer whitespace-nowrap shrink-0"
            >
              Reset Filters
            </button>
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
                  className="group text-left bg-[#0C120F]/92 border border-[#1B2821] rounded-lg p-2 hover:border-[#10B981] transition-colors disabled:opacity-60 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#10B981]"
                >
                  <div className="aspect-square w-full rounded bg-[#070A08] overflow-hidden mb-2 flex items-center justify-center border border-[#15201A]">
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
                      <div className="flex flex-col items-center justify-center p-2 text-center text-[#8FA396]">
                        <ImageIcon className="w-5 h-5 mb-1" />
                        <span className="text-[10px] capitalize">{sample.fruit_type}</span>
                      </div>
                    )}
                  </div>
                  <p className="text-xs font-semibold text-[#E8F0EB] truncate">
                    {sample.display_name}
                  </p>
                  <p className="text-[11px] text-[#7E9486] font-mono tabular-nums truncate">
                    {sample.original_dimensions}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* Session Prediction History (shown in standalone view or once user has scanned an item) */}
      {(!embedded || history.length > 0) && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="font-display text-lg font-bold text-[#F3F7F4]">
                Session Inspection Log
              </h3>
              <p className="text-xs text-[#8FA396]">
                {history.length} inspection{history.length === 1 ? '' : 's'} recorded in current
                browser session
              </p>
            </div>
            {history.length > 0 && (
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={handleExportHistoryCsv}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#E2ECE5] bg-[#121B16] border border-[#24352C] rounded-lg hover:border-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export CSV
                </button>
                <button
                  type="button"
                  onClick={onClearHistory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#FCA5A5] bg-[#1B0E10] border border-[#4C1D21] rounded-lg hover:border-[#F87171] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear Log
                </button>
              </div>
            )}
          </div>

          {history.length === 0 ? (
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 text-xs text-[#8FA396]">
              Completed predictions from uploaded images, camera captures, and held-out test
              specimens will appear here during your session.
            </div>
          ) : (
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#1B2821] text-xs font-semibold text-[#8FA396] bg-[#090E0B]">
                    <th className="py-2.5 px-4">Time</th>
                    <th className="py-2.5 px-4">Specimen</th>
                    <th className="py-2.5 px-4">Predicted Class</th>
                    <th className="py-2.5 px-4">Condition</th>
                    <th className="py-2.5 px-4 text-right">Confidence</th>
                    <th className="py-2.5 px-4 text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#16201B] text-xs">
                  {history.map((item) => (
                    <tr
                      key={item.id}
                      onClick={() => {
                        stopCameraStream();
                        setInputMode('upload');
                        setSelectedPreview(item.previewUrl);
                        setSelectedName(item.sourceLabel);
                        setCurrentResult(item.result);
                        setPredictionError(null);
                      }}
                      className="cursor-pointer hover:bg-[#111A15] transition-colors"
                    >
                      <td className="py-2.5 px-4 font-mono tabular-nums text-[#8FA396] whitespace-nowrap">
                        {item.timestamp}
                      </td>
                      <td className="py-2.5 px-4 font-medium text-[#E2ECE5] max-w-[240px] truncate">
                        {item.sourceLabel}
                      </td>
                      <td className="py-2.5 px-4 text-[#F3F7F4] font-medium">
                        {item.result.display_name}
                      </td>
                      <td className="py-2.5 px-4">
                        <span
                          className={`font-medium ${
                            item.result.freshness_status === 'fresh'
                              ? 'text-[#34D399]'
                              : 'text-[#F87171]'
                          }`}
                        >
                          {item.result.freshness_status === 'fresh' ? 'Fresh' : 'Spoiled'}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums font-semibold text-[#F3F7F4]">
                        {(item.result.confidence * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#8FA396]">
                        {item.result.inference_time_ms.toFixed(1)} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
};
