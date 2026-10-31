import React, { useState } from 'react';
import { Search } from 'lucide-react';
import type { DatasetSummary, TrainingHistory } from '../types';

interface DatasetExplorerProps {
  summary: DatasetSummary | null;
  trainingHistory?: TrainingHistory | null;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({
  summary,
  trainingHistory,
}) => {
  const [query, setQuery] = useState('');

  const classRows = summary
    ? Object.entries(summary.class_distribution)
        .map(([cls, count]) => ({
          cls,
          display: cls.replace('_', ' '),
          isFresh: cls.startsWith('fresh_'),
          rawCount: count,
          trainCount: summary.splits?.per_class.train[cls] ?? 0,
          valCount: summary.splits?.per_class.val[cls] ?? 0,
          testCount: summary.splits?.per_class.test[cls] ?? 0,
          weight: summary.splits?.class_weights[cls] ?? 1.0,
        }))
        .filter((row) => !query.trim() || row.display.toLowerCase().includes(query.trim().toLowerCase()))
    : [];

  return (
    <div className="space-y-10">
      <section className="space-y-2 pb-6 border-b border-[#18221D] max-w-3xl">
        <p className="text-xs font-medium text-[#34D399]">
          CC BY 4.0 Open License · Project AgML Raw Benchmark · DOI 10.17632/bdd69gyhv8.1
        </p>
        <h1 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#F3F7F4] text-balance">
          Model Architecture, Dataset Provenance &amp; API Reference
        </h1>
        <p className="text-sm text-[#A3B5AA] leading-relaxed">
          Complete technical specification covering dataset exploration, SHA-256 &amp; perceptual
          dHash leak prevention, MobileNetV2 two-stage transfer architecture, REST API endpoints,
          and reproducible CLI training commands.
        </p>
      </section>

      {!summary ? (
        <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-8 text-sm text-[#8FA396]">
          Dataset inspection summary (<code className="font-mono text-xs">models/dataset_summary.json</code>) is not yet available. Run <code className="font-mono text-xs">python -m src.data_loader</code> to generate it.
        </div>
      ) : (
        <>
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-5 space-y-1">
              <div className="text-xs text-[#8FA396]">Valid Raw Images Scanned</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-[#F3F7F4]">
                {summary.valid_readable_images.toLocaleString()}
              </div>
              <div className="text-xs text-[#7E9486] font-mono tabular-nums">
                {summary.total_size_mb.toFixed(1)} MB raw inspected
              </div>
            </div>

            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-5 space-y-1">
              <div className="text-xs text-[#8FA396]">Corrupted / Unreadable</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-[#F3F7F4]">
                {summary.corrupted_or_unreadable_count}
              </div>
              <div className="text-xs text-[#7E9486] font-mono tabular-nums">
                Unsupported files: {summary.unsupported_file_count}
              </div>
            </div>

            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-5 space-y-1">
              <div className="text-xs text-[#8FA396]">Raw Resolution Range</div>
              <div className="text-xl font-bold font-mono tabular-nums text-[#F3F7F4]">
                {summary.image_dimensions.width_min}px – {summary.image_dimensions.width_max}px
              </div>
              <div className="text-xs text-[#7E9486] font-mono tabular-nums">
                Mean {summary.image_dimensions.width_mean}×{summary.image_dimensions.height_mean}px
              </div>
            </div>

            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-5 space-y-1">
              <div className="text-xs text-[#8FA396]">Duplicates Audited</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-[#34D399]">
                {summary.duplicates.exact_sha256_duplicate_images}
              </div>
              <div className="text-xs text-[#7E9486] font-mono tabular-nums">
                Exact SHA-256 · {summary.duplicates.perceptual_dhash_duplicate_groups} dHash clusters
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
              <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
                Dataset Source &amp; CC BY 4.0 License
              </h2>
              <dl className="divide-y divide-[#18221D] text-xs">
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Dataset Title</dt>
                  <dd className="font-medium text-[#E2ECE5] text-right">{summary.dataset_metadata.name}</dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Authors</dt>
                  <dd className="font-medium text-[#E2ECE5] text-right">{summary.dataset_metadata.authors}</dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Peer-Reviewed Source</dt>
                  <dd className="font-mono text-[#E2ECE5] text-right">{summary.dataset_metadata.original_source}</dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Hugging Face Mirror</dt>
                  <dd className="font-mono text-[#34D399] text-right">
                    <a
                      href={summary.dataset_metadata.hf_url}
                      target="_blank"
                      rel="noreferrer"
                      className="hover:underline"
                    >
                      {summary.dataset_metadata.hf_repo_id} ({summary.dataset_metadata.hf_config})
                    </a>
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">License</dt>
                  <dd className="font-semibold text-[#F3F7F4] text-right">{summary.dataset_metadata.license}</dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Supported Fruits (8)</dt>
                  <dd className="font-medium text-[#E2ECE5] text-right capitalize">
                    {summary.dataset_metadata.supported_fruits.join(', ')}
                  </dd>
                </div>
              </dl>
            </div>

            {summary.splits && (
              <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
                <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
                  Leakage-Safe Stratified Split Audit
                </h2>
                <dl className="divide-y divide-[#18221D] text-xs">
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Training Split (70%)</dt>
                    <dd className="font-mono tabular-nums font-semibold text-[#E2ECE5]">
                      {summary.splits.counts.train.toLocaleString()} images (Augmented dynamically)
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Validation Split (15%)</dt>
                    <dd className="font-mono tabular-nums font-semibold text-[#E2ECE5]">
                      {summary.splits.counts.val.toLocaleString()} images (Un-augmented)
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Held-Out Test Split (15%)</dt>
                    <dd className="font-mono tabular-nums font-semibold text-[#34D399]">
                      {summary.splits.counts.test.toLocaleString()} images (Un-augmented)
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Exact SHA-256 Duplicates Removed</dt>
                    <dd className="font-mono tabular-nums text-[#E2ECE5]">
                      {summary.splits.deduplication_and_leakage.exact_duplicates_removed}
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Perceptual dHash Clusters Grouped</dt>
                    <dd className="font-mono tabular-nums text-[#E2ECE5]">
                      {summary.splits.deduplication_and_leakage.near_duplicate_clusters} clusters (
                      {summary.splits.deduplication_and_leakage.near_duplicate_extra_images_grouped}{' '}
                      near-dups grouped)
                    </dd>
                  </div>
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-[#8FA396]">Zero Cross-Split Hash Overlap</dt>
                    <dd className="font-semibold text-[#34D399]">
                      {summary.splits.deduplication_and_leakage.leakage_verified_zero_overlap
                        ? 'Verified (0 overlap)'
                        : 'Unverified'}
                    </dd>
                  </div>
                </dl>
              </div>
            )}
          </section>

          {/* Model Architecture & Training Specification */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
              <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
                MobileNetV2 Transfer Architecture Specification
              </h2>
              <dl className="divide-y divide-[#18221D] text-xs">
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Backbone</dt>
                  <dd className="font-mono text-[#E2ECE5]">
                    {trainingHistory?.backbone || 'MobileNetV2 (ImageNet weights, alpha=1.0)'}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Classification Head</dt>
                  <dd className="font-mono text-[#E2ECE5] text-right">
                    GlobalAvgPool2D → Dropout(0.35) → Dense(16, softmax)
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Stage 1 Head Warm-up</dt>
                  <dd className="font-mono tabular-nums text-[#E2ECE5]">
                    {trainingHistory ? `${trainingHistory.stage1_epochs_completed} epochs @ lr=${trainingHistory.stage1_lr}` : '8 epochs @ lr=1e-3'}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Stage 2 Fine-Tuning</dt>
                  <dd className="font-mono tabular-nums text-[#E2ECE5] text-right">
                    Top {trainingHistory?.fine_tune_top_layers ?? 30} layers unfrozen (BatchNorm frozen) @ lr={trainingHistory?.stage2_lr ?? 1e-4}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-[#8FA396]">Saved Artifact</dt>
                  <dd className="font-mono tabular-nums text-[#34D399]">
                    models/freshscan_mobilenetv2.keras ({trainingHistory?.model_size_mb.toFixed(2) ?? '8.92'} MB)
                  </dd>
                </div>
              </dl>
            </div>

            {/* FastAPI REST Endpoints Reference */}
            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
              <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
                Production REST API Specification
              </h2>
              <div className="divide-y divide-[#18221D] text-xs">
                <div className="py-2.5 flex items-start justify-between gap-4">
                  <code className="font-mono text-[#34D399] shrink-0">GET /health</code>
                  <span className="text-[#A3B5AA] text-right">
                    Service liveness, TensorFlow version, model readiness, and upload limits
                  </span>
                </div>
                <div className="py-2.5 flex items-start justify-between gap-4">
                  <code className="font-mono text-[#34D399] shrink-0">GET /model-info</code>
                  <span className="text-[#A3B5AA] text-right">
                    Model version, input tensor shape, preprocessing config, and 16 class labels
                  </span>
                </div>
                <div className="py-2.5 flex items-start justify-between gap-4">
                  <code className="font-mono text-[#34D399] shrink-0">GET /metrics</code>
                  <span className="text-[#A3B5AA] text-right">
                    Held-out test accuracy, P/R/F1, 16×16 confusion matrix, and baseline comparison
                  </span>
                </div>
                <div className="py-2.5 flex items-start justify-between gap-4">
                  <code className="font-mono text-[#34D399] shrink-0">POST /predict</code>
                  <span className="text-[#A3B5AA] text-right">
                    Multipart image upload (`file`) returning calibrated softmax &amp; entropy report
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Per-Class Split & Class Weight Table */}
          <section className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="font-display text-base font-bold text-[#F3F7F4]">
                  16-Class Split Counts &amp; Balanced Training Weights
                </h2>
                <p className="text-xs text-[#8FA396]">
                  Exact sample allocation and inverse-frequency class weights computed on the
                  training split
                </p>
              </div>
              <div className="relative self-start">
                <Search className="w-3.5 h-3.5 text-[#8FA396] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search class..."
                  aria-label="Search dataset class"
                  className="pl-8 pr-3 py-1.5 text-xs text-[#E2ECE5] placeholder-[#6E8577] bg-[#0C120F] border border-[#1E2C24] rounded-lg focus:outline-none focus:border-[#10B981] w-48"
                />
              </div>
            </div>

            <div className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#1B2821] bg-[#090E0B] text-xs font-semibold text-[#8FA396]">
                    <th className="py-2.5 px-4">Class Label</th>
                    <th className="py-2.5 px-4">Condition</th>
                    <th className="py-2.5 px-4 text-right">Raw Valid</th>
                    <th className="py-2.5 px-4 text-right">Train (70%)</th>
                    <th className="py-2.5 px-4 text-right">Val (15%)</th>
                    <th className="py-2.5 px-4 text-right">Test (15%)</th>
                    <th className="py-2.5 px-4 text-right">Class Weight</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#16201B] text-xs">
                  {classRows.map((row) => (
                    <tr key={row.cls} className="hover:bg-[#111A15] transition-colors">
                      <td className="py-2.5 px-4 font-medium text-[#F3F7F4] capitalize">
                        {row.display}{' '}
                        <span className="font-mono text-[#6E8577] lowercase">({row.cls})</span>
                      </td>
                      <td className="py-2.5 px-4">
                        <span
                          className={
                            row.isFresh
                              ? 'text-[#34D399] font-medium'
                              : 'text-[#F87171] font-medium'
                          }
                        >
                          {row.isFresh ? 'Fresh' : 'Spoiled'}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#E2ECE5]">
                        {row.rawCount}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#8FA396]">
                        {row.trainCount}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#8FA396]">
                        {row.valCount}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums font-semibold text-[#34D399]">
                        {row.testCount}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#E2ECE5]">
                        {row.weight.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* Documented Preprocessing & Reproduction Steps */}
      <section className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-5">
        <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
          Documented Preprocessing Decisions &amp; Reproducible Retraining CLI
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-[#A3B5AA] leading-relaxed">
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">01. White-Canvas Alpha Compositing</h3>
            <p>
              Transparent PNG/WebP fruit cutouts converted naively via{' '}
              <code className="font-mono text-[#E2ECE5]">.convert(&apos;RGB&apos;)</code> map
              alpha=0 pixels to pure black <code className="font-mono text-[#E2ECE5]">(0,0,0)</code>,
              creating dark borders that mimic necrotic tissue. FreshScan AI composites alpha masks
              onto <code className="font-mono text-[#E2ECE5]">#FFFFFF</code> first.
            </p>
          </div>
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">02. Color-Preserving Augmentation</h3>
            <p>
              Training batches apply horizontal flips, ±8% rotation, ±10% zoom, and ±10% contrast
              jitter. Random hue shifts are intentionally omitted because peel chromaticity is
              biologically diagnostic of oxidation and mold.
            </p>
          </div>
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">03. Dataset Bias &amp; Domain Scope</h3>
            <p>
              Studio benchmark photographs exhibit cleaner backgrounds than cluttered kitchen
              countertops. Cropping closely around a single fruit under neutral lighting yields the
              highest calibration accuracy.
            </p>
          </div>
        </div>

        <pre className="p-4 rounded-lg bg-[#070A08] border border-[#1B2821] text-xs font-mono text-[#D5E1D9] overflow-x-auto leading-relaxed">
{`# Reproduce dataset audit, leak-free split, 2-stage training, evaluation & tests:
python -m src.data_loader --max-per-class 45 --cleanup-raw --seed 42
python train_model.py --no-download --stage1-epochs 8 --stage2-epochs 6 --batch-size 32 --seed 42
python evaluate_model.py
pytest -v`}
        </pre>
      </section>
    </div>
  );
};
