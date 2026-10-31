/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Navbar, type ActiveTab } from './components/Navbar';
import { OverviewExperience } from './components/OverviewExperience';
import { ScannerWorkspace } from './components/ScannerWorkspace';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { DatasetExplorer } from './components/DatasetExplorer';
import { GitHubDeployPanel } from './components/GitHubDeployPanel';
import { FruitBackground3D } from './three/FruitBackground3D';
import type {
  AnalyticsBundle,
  HealthResponse,
  HistoryEntry,
  SampleTestItem,
  SupportedFruitId,
} from './types';

const SESSION_HISTORY_KEY = 'freshscan_session_history_v1';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleTestItem[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsBundle | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false);

  const [focusedFruit, setFocusedFruit] = useState<SupportedFruitId | null>(null);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const scrollProgressRef = useRef<number>(0);

  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_HISTORY_KEY);
      return saved ? (JSON.parse(saved) as HistoryEntry[]) : [];
    } catch {
      return [];
    }
  });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchSamples = useCallback(async () => {
    try {
      const sResp = await fetch('/api/ml/samples');
      if (sResp.ok) {
        const sData = await sResp.json();
        if (Array.isArray(sData.samples)) {
          setSamples(sData.samples as SampleTestItem[]);
        }
      }
    } catch {
      // Non-fatal
    }
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const hResp = await fetch('/api/ml/health');
      const hData = await hResp.json();
      if (!hResp.ok) {
        setHealthError(hData.detail || `Backend returned HTTP ${hResp.status}`);
      } else {
        setHealth(hData as HealthResponse);
        setHealthError(null);
      }
    } catch (err) {
      setHealthError(err instanceof Error ? err.message : 'Unable to reach inference backend');
    }
  }, []);

  const fetchAnalytics = useCallback(async () => {
    setAnalyticsLoading(true);
    try {
      const resp = await fetch('/api/ml/analytics');
      if (resp.ok) {
        const data = (await resp.json()) as AnalyticsBundle;
        setAnalytics(data);
      }
    } catch {
      // Keep previous state or null
    } finally {
      setAnalyticsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchSamples();
    void fetchAnalytics();
    void fetchHealth();
  }, [fetchSamples, fetchAnalytics, fetchHealth]);

  // Poll until model_ready and analytics.available are both true
  useEffect(() => {
    if (health?.model_ready && analytics?.available && samples.length > 0) return;
    const timer = window.setInterval(() => {
      void fetchSamples();
      void fetchAnalytics();
      void fetchHealth();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [health?.model_ready, analytics?.available, samples.length, fetchSamples, fetchAnalytics, fetchHealth]);

  const handleSelectTab = useCallback((tab: ActiveTab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const handleAddHistory = useCallback((entry: HistoryEntry) => {
    setHistory((prev) => {
      const updated = [entry, ...prev].slice(0, 30);
      try {
        sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(updated));
      } catch {
        // Ignore storage quota errors
      }
      return updated;
    });
  }, []);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
    try {
      sessionStorage.removeItem(SESSION_HISTORY_KEY);
    } catch {
      // Ignore
    }
  }, []);

  const handleTriggerScanCta = () => {
    if (activeTab === 'overview') {
      const el = document.getElementById('live-scanner');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
    }
    if (activeTab !== 'scanner') {
      setActiveTab('scanner');
      window.scrollTo({ top: 0, behavior: 'smooth' });
      setTimeout(() => fileInputRef.current?.click(), 120);
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#09100C] text-[#EEF3EF] relative overflow-x-clip">
      {/* Interactive Three.js 3D Botanical Fruits Background Scene */}
      <FruitBackground3D
        scrollProgressRef={scrollProgressRef}
        focusedFruit={focusedFruit}
        onSelectFocusFruit={setFocusedFruit}
        isScanning={isScanning}
        isOverviewActive={activeTab === 'overview'}
      />

      {/* Foreground Content Layer above WebGL Canvas */}
      <div className="relative z-10 min-h-screen flex flex-col">
        <Navbar
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          onTriggerScanCta={handleTriggerScanCta}
        />

        <main className="flex-1 w-full max-w-[1320px] mx-auto px-4 sm:px-6 py-8">
          {activeTab === 'overview' && (
            <OverviewExperience
              analytics={analytics}
              health={health}
              healthError={healthError}
              samples={samples}
              history={history}
              onAddHistory={handleAddHistory}
              onClearHistory={handleClearHistory}
              fileInputRef={fileInputRef}
              scrollProgressRef={scrollProgressRef}
              focusedFruit={focusedFruit}
              onSelectFocusFruit={setFocusedFruit}
              onScanningChange={setIsScanning}
              onNavigateTab={handleSelectTab}
            />
          )}

          {activeTab === 'scanner' && (
            <ScannerWorkspace
              health={health}
              healthError={healthError}
              samples={samples}
              history={history}
              onAddHistory={handleAddHistory}
              onClearHistory={handleClearHistory}
              fileInputRef={fileInputRef}
              onScanningChange={setIsScanning}
              onPredictedFruit={setFocusedFruit}
            />
          )}

          {activeTab === 'analytics' && (
            <AnalyticsDashboard
              analytics={analytics}
              loading={analyticsLoading}
              onRefresh={() => void fetchAnalytics()}
            />
          )}

          {activeTab === 'dataset' && (
            <DatasetExplorer
              summary={analytics?.dataset_summary ?? null}
              trainingHistory={analytics?.training_history ?? null}
            />
          )}

          {activeTab === 'deploy' && <GitHubDeployPanel />}
        </main>

        <footer className="border-t border-[#18221D] bg-[#070A08]/95 py-6 px-4 sm:px-6 mt-12">
          <div className="max-w-[1320px] mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-[#8FA396]">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <span className="font-semibold text-[#E2ECE5]">FreshScan AI</span>
              <span aria-hidden="true">·</span>
              <span>MobileNetV2 Optical Produce Freshness Detection</span>
              <span aria-hidden="true">·</span>
              <span>Dataset: Sultana et al. (2022) CC BY 4.0</span>
            </div>

            <div className="flex flex-wrap items-center gap-5">
              <button
                type="button"
                onClick={() => handleSelectTab('dataset')}
                className="hover:text-[#F3F7F4] transition-colors cursor-pointer"
              >
                Documentation &amp; API
              </button>
              <button
                type="button"
                onClick={() => handleSelectTab('analytics')}
                className="hover:text-[#F3F7F4] transition-colors cursor-pointer"
              >
                Evaluation Metrics
              </button>
              <button
                type="button"
                onClick={() => handleSelectTab('deploy')}
                className="hover:text-[#F3F7F4] transition-colors cursor-pointer"
              >
                GitHub (FreshScan-AI)
              </button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
