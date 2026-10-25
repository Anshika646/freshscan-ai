/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Navbar, type ActiveTab } from './components/Navbar';
import { ScannerWorkspace } from './components/ScannerWorkspace';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { DatasetExplorer } from './components/DatasetExplorer';
import { GitHubDeployPanel } from './components/GitHubDeployPanel';
import type {
  AnalyticsBundle,
  GitHubStatusResponse,
  HealthResponse,
  HistoryEntry,
  SampleTestItem,
} from './types';

const SESSION_HISTORY_KEY = 'freshscan_session_history_v1';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('scanner');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleTestItem[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsBundle | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false);
  const [githubStatus, setGithubStatus] = useState<GitHubStatusResponse | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_HISTORY_KEY);
      return saved ? (JSON.parse(saved) as HistoryEntry[]) : [];
    } catch {
      return [];
    }
  });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchHealthAndSamples = useCallback(async () => {
    try {
      const hResp = await fetch('/api/ml/health');
      const hData = await hResp.json();
      if (!hResp.ok) {
        setHealthError(hData.detail || `Backend returned HTTP ${hResp.status}`);
      } else {
        setHealth(hData as HealthResponse);
        setHealthError(null);
      }

      const sResp = await fetch('/api/ml/samples');
      if (sResp.ok) {
        const sData = await sResp.json();
        if (Array.isArray(sData.samples)) {
          setSamples(sData.samples as SampleTestItem[]);
        }
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

  const fetchGitHubStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/github/status');
      if (resp.ok) {
        const data = (await resp.json()) as GitHubStatusResponse;
        setGithubStatus(data);
      }
    } catch {
      // Non-fatal
    }
  }, []);

  useEffect(() => {
    void fetchHealthAndSamples();
    void fetchAnalytics();
    void fetchGitHubStatus();
  }, [fetchHealthAndSamples, fetchAnalytics, fetchGitHubStatus]);

  // Poll health until the trained model is ready on disk
  useEffect(() => {
    if (health?.model_ready) return;
    const timer = window.setInterval(() => {
      void fetchHealthAndSamples();
      void fetchAnalytics();
    }, 6000);
    return () => window.clearInterval(timer);
  }, [health?.model_ready, fetchHealthAndSamples, fetchAnalytics]);

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

  const handleTriggerUpload = () => {
    if (activeTab !== 'scanner') {
      setActiveTab('scanner');
      setTimeout(() => fileInputRef.current?.click(), 50);
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F8F5] text-[#14241B]">
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onTriggerUpload={handleTriggerUpload}
      />

      <main className="flex-1 w-full max-w-[1320px] mx-auto px-6 py-8">
        {activeTab === 'scanner' && (
          <ScannerWorkspace
            health={health}
            healthError={healthError}
            samples={samples}
            history={history}
            onAddHistory={handleAddHistory}
            onClearHistory={handleClearHistory}
            fileInputRef={fileInputRef}
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
          <DatasetExplorer summary={analytics?.dataset_summary ?? null} />
        )}

        {activeTab === 'deploy' && <GitHubDeployPanel />}
      </main>

      <footer className="border-t border-[#DCE2D7] bg-[#F7F8F5] py-5 px-6 mt-12">
        <div className="max-w-[1320px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#526358]">
          <span>
            FreshScan AI — Optical Fruit Freshness Classification (MobileNetV2 · TensorFlow/Keras)
          </span>
          <span>
            Dataset: Project AgML / Sultana et al. (2022) · CC BY 4.0 · Visual predictions do not
            guarantee food safety
          </span>
        </div>
      </footer>
    </div>
  );
}
