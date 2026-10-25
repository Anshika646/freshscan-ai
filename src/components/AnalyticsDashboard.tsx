import React, { useState } from 'react';
import { RefreshCw, AlertCircle } from 'lucide-react';
import type { AnalyticsBundle, TrainingEpoch } from '../types';

interface AnalyticsDashboardProps {
  analytics: AnalyticsBundle | null;
  loading: boolean;
  onRefresh: () => void;
}

const SvgLineChart: React.FC<{
  title: string;
  subtitle: string;
  epochs: TrainingEpoch[];
  trainKey: 'loss' | 'accuracy';
  valKey: 'val_loss' | 'val_accuracy';
  stage1End: number;
  isPercentage?: boolean;
}> = ({ title, subtitle, epochs, trainKey, valKey, stage1End, isPercentage = false }) => {
  const width = 520;
  const height = 220;
  const padLeft = 48;
  const padRight = 20;
  const padTop = 18;
  const padBottom = 34;

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const trainVals = epochs.map((e) => (isPercentage ? e[trainKey] * 100 : e[trainKey]));
  const valVals = epochs.map((e) => (isPercentage ? e[valKey] * 100 : e[valKey]));
  const allVals = [...trainVals, ...valVals];
  const minVal = isPercentage ? Math.max(0, Math.floor(Math.min(...allVals) - 4)) : 0;
  const maxVal = isPercentage
    ? Math.min(100, Math.ceil(Math.max(...allVals) + 2))
    : Math.max(...allVals) * 1.08 || 1;

  const xCoord = (idx: number) =>
    epochs.length <= 1
      ? padLeft + plotW / 2
      : padLeft + (idx / (epochs.length - 1)) * plotW;

  const yCoord = (val: number) => {
    const span = Math.max(maxVal - minVal, 1e-6);
    return padTop + plotH - ((val - minVal) / span) * plotH;
  };

  const trainPath = trainVals
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xCoord(i).toFixed(1)} ${yCoord(v).toFixed(1)}`)
    .join(' ');
  const valPath = valVals
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xCoord(i).toFixed(1)} ${yCoord(v).toFixed(1)}`)
    .join(' ');

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => minVal + t * (maxVal - minVal));

  return (
    <div className="bg-white border border-[#DCE2D7] rounded-xl p-5 space-y-3">
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-[#14241B]">{title}</h3>
          <p className="text-xs text-[#526358]">{subtitle}</p>
        </div>
        <div className="flex items-center gap-4 text-xs text-[#526358] shrink-0">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#1B4332] inline-block" />
            Train
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#D97706] inline-block" />
            Validation
          </span>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto overflow-visible"
        role="img"
        aria-label={title}
      >
        {yTicks.map((tick, idx) => {
          const y = yCoord(tick);
          return (
            <g key={idx}>
              <line
                x1={padLeft}
                x2={width - padRight}
                y1={y}
                y2={y}
                stroke="#E6EBE1"
                strokeWidth="1"
              />
              <text
                x={padLeft - 8}
                y={y + 3}
                textAnchor="end"
                className="fill-[#526358] text-[10px] font-mono"
              >
                {isPercentage ? `${tick.toFixed(0)}%` : tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        {stage1End > 0 && stage1End < epochs.length && (
          <g>
            <line
              x1={xCoord(stage1End - 0.5)}
              x2={xCoord(stage1End - 0.5)}
              y1={padTop}
              y2={padTop + plotH}
              stroke="#64748B"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <text
              x={xCoord(stage1End - 0.5) + 4}
              y={padTop + 10}
              className="fill-[#526358] text-[9px] font-mono"
            >
              Fine-tune
            </text>
          </g>
        )}

        <path d={trainPath} fill="none" stroke="#1B4332" strokeWidth="2.2" />
        <path
          d={valPath}
          fill="none"
          stroke="#D97706"
          strokeWidth="2.2"
          strokeDasharray="4 3"
        />

        {epochs.map((ep, idx) => (
          <g key={ep.epoch}>
            <circle cx={xCoord(idx)} cy={yCoord(trainVals[idx])} r="3" fill="#1B4332" />
            <circle cx={xCoord(idx)} cy={yCoord(valVals[idx])} r="3" fill="#D97706" />
            <text
              x={xCoord(idx)}
              y={height - 10}
              textAnchor="middle"
              className="fill-[#526358] text-[10px] font-mono"
            >
              E{ep.epoch}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  analytics,
  loading,
  onRefresh,
}) => {
  const [metricFilter, setMetricFilter] = useState<'all' | 'fresh' | 'rotten'>('all');
  const [hoveredCell, setHoveredCell] = useState<{
    trueLabel: string;
    predLabel: string;
    count: number;
  } | null>(null);

  if (loading && !analytics) {
    return (
      <div className="bg-white border border-[#DCE2D7] rounded-xl p-12 text-center space-y-3">
        <div className="w-7 h-7 border-2 border-[#1B4332] border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-medium text-[#14241B]">Loading model evaluation artifacts...</p>
      </div>
    );
  }

  if (!analytics || !analytics.available || !analytics.evaluation_report || !analytics.training_history) {
    return (
      <div className="bg-white border border-[#DCE2D7] rounded-xl p-8 space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5 text-[#B45309]">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <h2 className="text-lg font-bold text-[#14241B]">
              Model Evaluation Results Currently Unavailable
            </h2>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-[#1B4332] rounded-lg hover:bg-[#143326] transition-colors whitespace-nowrap shrink-0"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Check Again
          </button>
        </div>
        <p className="text-sm text-[#3D4F44] leading-relaxed max-w-2xl">
          FreshScan AI never displays fabricated metrics. Training or held-out test evaluation is
          currently running or has not yet written its artifacts to disk:
        </p>
        {analytics?.unavailable_reasons && analytics.unavailable_reasons.length > 0 && (
          <ul className="list-disc pl-5 text-xs text-[#526358] space-y-1">
            {analytics.unavailable_reasons.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        )}
        <pre className="p-3.5 rounded-lg bg-[#F7F8F5] border border-[#DCE2D7] text-xs font-mono text-[#14241B] overflow-x-auto">
          python train_model.py --stage1-epochs 6 --stage2-epochs 5 --batch-size 32 --seed 42 &amp;&amp; python evaluate_model.py
        </pre>
      </div>
    );
  }

  const evalReport = analytics.evaluation_report;
  const trainHist = analytics.training_history;
  const cm = evalReport.confusion_matrix;
  const maxCmVal = Math.max(1, ...cm.matrix.flat());

  const filteredClasses = evalReport.per_class_metrics.filter((m) =>
    metricFilter === 'all' ? true : m.freshness === metricFilter,
  );

  return (
    <div className="space-y-10">
      {/* Header */}
      <section className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#DCE2D7]">
        <div className="space-y-1.5">
          <p className="text-xs text-[#526358]">
            Evaluated on {evalReport.evaluated_split} · {evalReport.num_test_samples} unseen test
            images · Seed {trainHist.random_seed}
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold text-[#14241B]">
            Held-Out Test Evaluation &amp; Training Diagnostics
          </h1>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-[#14241B] bg-white border border-[#DCE2D7] rounded-lg hover:bg-[#EFF2EC] transition-colors self-start whitespace-nowrap shrink-0"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Reload Artifacts
        </button>
      </section>

      {/* Primary Quantitative Metrics Strip */}
      <section className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white border border-[#DCE2D7] rounded-xl p-4 space-y-1">
          <p className="text-xs text-[#526358]">16-Class Test Accuracy</p>
          <p className="text-2xl font-mono font-bold tabular-nums text-[#14241B]">
            {(evalReport.overall_metrics.accuracy * 100).toFixed(2)}%
          </p>
          <p className="text-[11px] font-mono tabular-nums text-[#526358]">
            Log-loss: {evalReport.overall_metrics.test_log_loss.toFixed(4)}
          </p>
        </div>

        <div className="bg-white border border-[#DCE2D7] rounded-xl p-4 space-y-1">
          <p className="text-xs text-[#526358]">Macro F1-Score</p>
          <p className="text-2xl font-mono font-bold tabular-nums text-[#14241B]">
            {(evalReport.overall_metrics.macro_f1 * 100).toFixed(2)}%
          </p>
          <p className="text-[11px] font-mono tabular-nums text-[#526358]">
            P: {(evalReport.overall_metrics.macro_precision * 100).toFixed(1)}% · R:{' '}
            {(evalReport.overall_metrics.macro_recall * 100).toFixed(1)}%
          </p>
        </div>

        <div className="bg-white border border-[#DCE2D7] rounded-xl p-4 space-y-1">
          <p className="text-xs text-[#526358]">OvR Multi-Class ROC-AUC</p>
          <p className="text-2xl font-mono font-bold tabular-nums text-[#14241B]">
            {evalReport.roc_auc.appropriate && evalReport.roc_auc.ovr_macro_roc_auc !== undefined
              ? evalReport.roc_auc.ovr_macro_roc_auc.toFixed(4)
              : 'N/A'}
          </p>
          <p className="text-[11px] text-[#526358]">One-vs-Rest Macro AUC</p>
        </div>

        <div className="bg-white border border-[#DCE2D7] rounded-xl p-4 space-y-1">
          <p className="text-xs text-[#526358]">Binary Fresh/Rotten Acc</p>
          <p className="text-2xl font-mono font-bold tabular-nums text-[#166534]">
            {(evalReport.binary_fresh_vs_rotten.accuracy * 100).toFixed(2)}%
          </p>
          <p className="text-[11px] font-mono tabular-nums text-[#526358]">
            Binary AUC:{' '}
            {evalReport.binary_fresh_vs_rotten.roc_auc !== null
              ? evalReport.binary_fresh_vs_rotten.roc_auc.toFixed(4)
              : 'N/A'}
          </p>
        </div>

        <div className="bg-white border border-[#DCE2D7] rounded-xl p-4 space-y-1">
          <p className="text-xs text-[#526358]">Model Footprint</p>
          <p className="text-2xl font-mono font-bold tabular-nums text-[#14241B]">
            {trainHist.model_size_mb.toFixed(2)}
            <span className="text-xs text-[#526358] ml-1">MB</span>
          </p>
          <p className="text-[11px] font-mono tabular-nums text-[#526358]">
            {trainHist.total_epochs_completed} epochs · {trainHist.training_duration_seconds.toFixed(0)}s
          </p>
        </div>
      </section>

      {/* Loss & Accuracy Curves */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SvgLineChart
          title="Training vs. Validation Cross-Entropy Loss"
          subtitle={`Stage 1 Head Warm-up (E1–E${trainHist.stage1_epochs_completed}) + Stage 2 Fine-Tuning`}
          epochs={trainHist.epochs}
          trainKey="loss"
          valKey="val_loss"
          stage1End={trainHist.stage1_epochs_completed}
          isPercentage={false}
        />
        <SvgLineChart
          title="Training vs. Validation Accuracy"
          subtitle={`Best Validation Accuracy: ${(trainHist.best_val_accuracy * 100).toFixed(2)}%`}
          epochs={trainHist.epochs}
          trainKey="accuracy"
          valKey="val_accuracy"
          stage1End={trainHist.stage1_epochs_completed}
          isPercentage={true}
        />
      </section>

      {/* Confusion Matrix Heatmap */}
      <section className="bg-white border border-[#DCE2D7] rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-[#14241B]">
              16×16 Held-Out Test Confusion Matrix
            </h2>
            <p className="text-xs text-[#526358]">
              Rows represent true ground-truth classes; columns represent MobileNetV2 predictions
            </p>
          </div>
          <div className="text-xs font-mono tabular-nums text-[#14241B] h-5">
            {hoveredCell
              ? `True: ${hoveredCell.trueLabel} → Pred: ${hoveredCell.predLabel} (${hoveredCell.count})`
              : 'Hover any matrix cell to inspect pair count'}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="border-collapse text-[11px] font-mono tabular-nums mx-auto">
            <thead>
              <tr>
                <th className="p-1.5 text-left text-[#526358] font-sans font-medium">
                  True \ Pred
                </th>
                {cm.display_labels.map((lbl, idx) => (
                  <th
                    key={idx}
                    className="p-1 text-center text-[#526358] font-medium max-w-[52px] truncate"
                    title={lbl}
                  >
                    {lbl
                      .split(' ')
                      .map((w) => w.slice(0, 2))
                      .join('')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.matrix.map((row, rIdx) => (
                <tr key={rIdx}>
                  <td className="pr-3 py-1 font-sans text-xs text-[#14241B] whitespace-nowrap font-medium">
                    {cm.display_labels[rIdx]}
                  </td>
                  {row.map((val, cIdx) => {
                    const intensity = val / maxCmVal;
                    const isDiag = rIdx === cIdx;
                    let bgClass = 'bg-[#FAFBF9] text-[#8A998E]';
                    if (val > 0) {
                      if (isDiag) {
                        bgClass =
                          intensity > 0.55
                            ? 'bg-[#1B4332] text-white font-semibold'
                            : 'bg-[#74C69D]/50 text-[#14241B] font-semibold';
                      } else {
                        bgClass = 'bg-[#FDE8E8] text-[#991B1B] font-semibold';
                      }
                    }
                    return (
                      <td
                        key={cIdx}
                        onMouseEnter={() =>
                          setHoveredCell({
                            trueLabel: cm.display_labels[rIdx],
                            predLabel: cm.display_labels[cIdx],
                            count: val,
                          })
                        }
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`w-9 h-8 text-center border border-[#E6EBE1] transition-colors cursor-crosshair ${bgClass}`}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-Class Precision, Recall, F1 Table */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-[#14241B]">
              Per-Class Precision, Recall &amp; F1-Score
            </h2>
            <p className="text-xs text-[#526358]">
              Exact metrics computed on the held-out test split via scikit-learn
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-[#EFF2EC] rounded-lg self-start">
            {(['all', 'fresh', 'rotten'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setMetricFilter(mode)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize whitespace-nowrap shrink-0 ${
                  metricFilter === mode
                    ? 'bg-white text-[#14241B] shadow-xs'
                    : 'text-[#526358] hover:text-[#14241B]'
                }`}
              >
                {mode === 'all' ? 'All 16 Classes' : mode}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white border border-[#DCE2D7] rounded-xl overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#DCE2D7] bg-[#FAFBF9] text-xs font-semibold text-[#526358]">
                <th className="py-2.5 px-4">Class</th>
                <th className="py-2.5 px-4">State</th>
                <th className="py-2.5 px-4 text-right">Precision</th>
                <th className="py-2.5 px-4 text-right">Recall</th>
                <th className="py-2.5 px-4 text-right">F1-Score</th>
                <th className="py-2.5 px-4 text-right">Correct / Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E6EBE1] text-xs">
              {filteredClasses.map((cls) => (
                <tr key={cls.class_label} className="hover:bg-[#F7F8F5] transition-colors">
                  <td className="py-2.5 px-4 font-medium text-[#14241B]">
                    {cls.display_name}{' '}
                    <span className="text-[#6A7B70] font-mono">({cls.class_label})</span>
                  </td>
                  <td className="py-2.5 px-4">
                    <span
                      className={
                        cls.freshness === 'fresh'
                          ? 'text-[#166534] font-medium'
                          : 'text-[#B91C1C] font-medium'
                      }
                    >
                      {cls.freshness === 'fresh' ? 'Fresh' : 'Rotten'}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#14241B]">
                    {(cls.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#14241B]">
                    {(cls.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono tabular-nums font-semibold text-[#14241B]">
                    {(cls.f1_score * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-4 text-right font-mono tabular-nums text-[#526358]">
                    {cls.correct} / {cls.support}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Model Limitations & Class Balance Analysis */}
      <section className="bg-white border border-[#DCE2D7] rounded-xl p-6 space-y-3">
        <h2 className="text-base font-bold text-[#14241B]">
          Model Limitations &amp; Class Imbalance Assessment
        </h2>
        <ul className="space-y-2.5 text-xs text-[#3D4F44] leading-relaxed list-disc pl-5">
          {evalReport.limitations_and_imbalance_summary.map((point, idx) => (
            <li key={idx}>{point}</li>
          ))}
        </ul>
      </section>
    </div>
  );
};
