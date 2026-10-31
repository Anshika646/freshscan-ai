import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Copy,
  ExternalLink,
  GitBranch,
  RefreshCw,
  Server,
} from 'lucide-react';
import type { DeployReadinessResponse, GitHubStatusResponse } from '../types';

export const GitHubDeployPanel: React.FC = () => {
  const [status, setStatus] = useState<GitHubStatusResponse | null>(null);
  const [readiness, setReadiness] = useState<DeployReadinessResponse | null>(null);
  const [repoName, setRepoName] = useState<string>('FreshScan-AI');
  const [isPrivate, setIsPrivate] = useState<boolean>(false);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [syncResult, setSyncResult] = useState<{
    repo_url: string;
    commit_sha: string;
    files_synced: number;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const loadPanelData = async () => {
    try {
      const [ghResp, rdResp] = await Promise.all([
        fetch('/api/github/status'),
        fetch('/api/deploy/readiness'),
      ]);
      if (ghResp.ok) {
        const ghData = (await ghResp.json()) as GitHubStatusResponse;
        setStatus(ghData);
        if (ghData.target_repo_name) setRepoName(ghData.target_repo_name);
      }
      if (rdResp.ok) {
        const rdData = (await rdResp.json()) as DeployReadinessResponse;
        setReadiness(rdData);
      }
    } catch {
      // Non-fatal during server reload
    }
  };

  useEffect(() => {
    void loadPanelData();

    const handlePopupMessage = (event: MessageEvent) => {
      if (event.data?.type === 'OAUTH_AUTH_SUCCESS') {
        void loadPanelData();
      }
    };
    window.addEventListener('message', handlePopupMessage);
    return () => window.removeEventListener('message', handlePopupMessage);
  }, []);

  const handleConnectGitHub = async () => {
    setErrorMsg(null);
    try {
      const origin = encodeURIComponent(window.location.origin);
      const resp = await fetch(`/api/github/auth/url?origin=${origin}`);
      const payload = await resp.json();
      if (!resp.ok || !payload.url) {
        throw new Error(payload.error || 'Unable to construct GitHub OAuth authorization URL.');
      }
      const popup = window.open(payload.url, 'github_oauth_popup', 'width=620,height=720');
      if (!popup) {
        setErrorMsg(
          'Popup window was blocked by the browser. Please allow popups for this domain and try again.',
        );
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const handleSyncRepository = async () => {
    setErrorMsg(null);
    setSyncResult(null);
    if (repoName.trim().toLowerCase().includes('webpearl')) {
      setErrorMsg(
        'Safety guard: Pushing FreshScan AI to webpearl-website is blocked. Keep repository target set to FreshScan-AI.',
      );
      return;
    }
    setSyncing(true);
    try {
      const resp = await fetch('/api/github/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoName: repoName.trim() || 'FreshScan-AI', private: isPrivate }),
      });
      const payload = await resp.json();
      if (!resp.ok) {
        throw new Error(payload.error || `GitHub sync failed (${resp.status})`);
      }
      setSyncResult(payload);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setSyncing(false);
    }
  };

  const handleCopy = (key: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1800);
  };

  const cloudRunCmd = `gcloud run deploy freshscan-ai \\
  --source . \\
  --region asia-east1 \\
  --memory 2Gi \\
  --cpu 2 \\
  --port 8080 \\
  --allow-unauthenticated`;

  const dockerCmd = `docker build -t freshscan-ai .
docker run --rm -p 8080:8080 freshscan-ai`;

  const gitCliCmd = `git remote add origin https://github.com/<YOUR_USERNAME>/FreshScan-AI.git
git branch -M main
git push -u origin main`;

  const readinessItems = readiness
    ? [
        {
          label: `Trained Keras Model (${readiness.model_size_mb.toFixed(2)} MB)`,
          ok: readiness.checks.trained_model_saved,
        },
        {
          label: '16-Class Label Encoder Mapping (class_indices.json)',
          ok: readiness.checks.class_mapping_verified,
        },
        {
          label: 'Preprocessing Configuration (224×224 RGB MobileNetV2)',
          ok: readiness.checks.preprocessing_config_saved,
        },
        {
          label: 'Held-Out Test Evaluation Report & Diagnostic PNGs',
          ok: readiness.checks.heldout_evaluation_saved,
        },
        {
          label: '16 Unseen Test Specimens Bundled (sample_test_images/)',
          ok: readiness.checks.heldout_samples_bundled,
        },
        {
          label: 'Frontend Production Static Bundle (dist/)',
          ok: readiness.checks.frontend_production_bundle,
        },
        {
          label: 'Multi-Runtime Production Dockerfile + .dockerignore',
          ok: readiness.checks.dockerfile_present,
        },
        {
          label: 'Python FastAPI TensorFlow Inference Service Online',
          ok: readiness.checks.python_inference_engine_online,
        },
      ]
    : [];

  return (
    <div className="space-y-10">
      <section className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b border-[#18221D]">
        <div className="space-y-2 max-w-3xl">
          <p className="text-xs font-medium text-[#34D399]">
            Repository Target: FreshScan-AI · Container Target: Google Cloud Run / Docker (Port 8080)
          </p>
          <h1 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#F3F7F4] text-balance">
            Production Deployment Readiness &amp; GitHub Sync
          </h1>
          <p className="text-sm text-[#A3B5AA] leading-relaxed">
            Verify live production artifacts, sync the clean source tree to{' '}
            <code className="font-mono text-xs text-[#E2ECE5]">FreshScan-AI</code> on GitHub
            (excluding raw datasets, virtualenvs, and secrets), and deploy via Google Cloud Run,
            Docker, or split static-frontend + FastAPI hosting.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void loadPanelData()}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-[#E2ECE5] bg-[#0C120F] border border-[#1E2C24] rounded-lg hover:border-[#10B981] transition-colors self-start whitespace-nowrap shrink-0 cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh Audit
        </button>
      </section>

      {/* Live Deployment Readiness Audit */}
      <section className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-[#10B981]" />
            <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
              01. Live Production &amp; Container Readiness Verification
            </h2>
          </div>
          {readiness && (
            <span
              className={`inline-flex items-center gap-1.5 text-xs font-semibold ${
                readiness.ready_for_production ? 'text-[#34D399]' : 'text-[#F59E0B]'
              }`}
            >
              {readiness.ready_for_production ? (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  <span>All 8 Production Checks Verified</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-4 h-4" />
                  <span>Artifact Verification In Progress</span>
                </>
              )}
            </span>
          )}
        </div>

        {!readiness ? (
          <p className="text-xs text-[#8FA396]">Checking production artifacts on disk...</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2.5 pt-1">
            {readinessItems.map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between gap-3 py-2 border-b border-[#18221D] text-xs"
              >
                <span className="text-[#E2ECE5] font-medium">{item.label}</span>
                <span
                  className={`inline-flex items-center gap-1 font-mono font-semibold shrink-0 ${
                    item.ok ? 'text-[#34D399]' : 'text-[#F59E0B]'
                  }`}
                >
                  {item.ok ? (
                    <>
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Ready
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-3.5 h-3.5" />
                      Pending
                    </>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* GitHub Sync + Cloud Run Commands */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        <div className="lg:col-span-6 bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-5 flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-2">
              <h2 className="font-display text-base font-semibold text-[#F3F7F4] flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-[#10B981]" />
                <span>02. GitHub OAuth Repository Sync</span>
              </h2>
              {status?.authenticated && (
                <span className="text-xs font-medium text-[#34D399]">
                  Connected as @{status.username}
                </span>
              )}
            </div>

            {!status?.oauth_configured ? (
              <div className="space-y-4 text-xs text-[#A3B5AA] leading-relaxed">
                {status?.placeholder_credentials_detected && (
                  <div className="p-3 rounded-lg bg-[#1C1509] border border-[#593E14] text-[#FDE68A]">
                    Placeholder GitHub OAuth credentials were detected in the environment. Replace{' '}
                    <code className="font-mono">GITHUB_CLIENT_ID</code> and{' '}
                    <code className="font-mono">GITHUB_CLIENT_SECRET</code> with valid credentials
                    from your GitHub OAuth App to enable browser popup push, or push directly via
                    the Git CLI commands on the right.
                  </div>
                )}
                <p>
                  To sync via browser OAuth, register a GitHub OAuth App at{' '}
                  <a
                    href="https://github.com/settings/developers"
                    target="_blank"
                    rel="noreferrer"
                    className="text-[#34D399] font-semibold hover:underline"
                  >
                    github.com/settings/developers
                  </a>{' '}
                  and configure <code className="font-mono text-[#E2ECE5]">GITHUB_CLIENT_ID</code>{' '}
                  and <code className="font-mono text-[#E2ECE5]">GITHUB_CLIENT_SECRET</code> with
                  the callback URLs below:
                </p>

                {status && (
                  <div className="space-y-2.5 p-3.5 rounded-lg bg-[#070A08] border border-[#1B2821]">
                    <div className="font-semibold text-[#F3F7F4]">OAuth Authorization Callback URLs:</div>
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <code className="font-mono text-[11px] text-[#D5E1D9] break-all">
                          {status.dev_callback_url}
                        </code>
                        <button
                          type="button"
                          onClick={() => handleCopy('dev', status.dev_callback_url)}
                          className="p-1.5 text-[#8FA396] hover:text-white cursor-pointer shrink-0"
                          title="Copy Development Callback URL"
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <code className="font-mono text-[11px] text-[#D5E1D9] break-all">
                          {status.shared_callback_url}
                        </code>
                        <button
                          type="button"
                          onClick={() => handleCopy('shared', status.shared_callback_url)}
                          className="p-1.5 text-[#8FA396] hover:text-white cursor-pointer shrink-0"
                          title="Copy Shared Callback URL"
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : !status.authenticated ? (
              <div className="space-y-4">
                <p className="text-xs text-[#A3B5AA] leading-relaxed">
                  GitHub OAuth credentials are configured. Authorize your GitHub account in a popup
                  window to create or update the <code className="font-mono text-[#E2ECE5]">FreshScan-AI</code>{' '}
                  repository.
                </p>
                <button
                  type="button"
                  onClick={() => void handleConnectGitHub()}
                  className="px-4 py-2.5 text-xs font-semibold text-white bg-[#059669] rounded-lg hover:bg-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  Authorize with GitHub
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <label className="space-y-1 text-xs">
                    <span className="font-medium text-[#E2ECE5]">Repository Name</span>
                    <input
                      type="text"
                      value={repoName}
                      onChange={(e) => setRepoName(e.target.value)}
                      className="w-full px-3 py-2 text-xs font-mono text-[#F3F7F4] bg-[#070A08] border border-[#1E2C24] rounded-lg focus:outline-none focus:border-[#10B981]"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs pt-5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isPrivate}
                      onChange={(e) => setIsPrivate(e.target.checked)}
                      className="rounded border-[#24352C] accent-[#10B981]"
                    />
                    <span className="text-[#E2ECE5] font-medium">Private Repository</span>
                  </label>
                </div>

                <button
                  type="button"
                  disabled={syncing}
                  onClick={() => void handleSyncRepository()}
                  className="inline-flex items-center gap-2 px-4 py-2.5 text-xs font-semibold text-white bg-[#059669] rounded-lg hover:bg-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-50"
                >
                  {syncing ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Pushing Clean Tree to GitHub...</span>
                    </>
                  ) : (
                    <span>Create &amp; Push to GitHub ({repoName})</span>
                  )}
                </button>
              </div>
            )}

            {errorMsg && (
              <div className="p-3 rounded-lg bg-[#1B0E10] border border-[#5B2121] text-xs text-[#FCA5A5]">
                {errorMsg}
              </div>
            )}

            {syncResult && (
              <div className="p-4 rounded-lg bg-[#0D2118] border border-[#10B981]/50 space-y-2 text-xs">
                <div className="flex items-center gap-1.5 font-semibold text-[#34D399]">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Repository Synced ({syncResult.files_synced} files committed)</span>
                </div>
                <div className="font-mono text-[11px] text-[#9EB0A5]">
                  Commit SHA: {syncResult.commit_sha.slice(0, 12)}
                </div>
                <a
                  href={syncResult.repo_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 font-semibold text-[#34D399] hover:underline"
                >
                  <span>{syncResult.repo_url}</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            )}
          </div>

          <p className="text-[11px] text-[#7E9486] pt-2 border-t border-[#18221D]">
            Safety Policy: Pushing to <code className="font-mono text-[#A3B5AA]">webpearl-website</code> is
            explicitly blocked at both client and server layers.
          </p>
        </div>

        <div className="lg:col-span-6 bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
              03. Cloud Run, Docker &amp; Git CLI Commands
            </h2>
            {copiedKey && (
              <span className="text-xs font-medium text-[#34D399]">Copied to clipboard</span>
            )}
          </div>

          <div className="space-y-3 text-xs">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[#E2ECE5]">
                  Deploy Unified Container to Google Cloud Run (Recommended):
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('cloudrun', cloudRunCmd)}
                  className="inline-flex items-center gap-1 text-[#34D399] hover:underline cursor-pointer"
                >
                  <Copy className="w-3 h-3" />
                  Copy
                </button>
              </div>
              <pre className="p-3 rounded-lg bg-[#070A08] border border-[#1B2821] text-[#D5E1D9] font-mono text-[11px] overflow-x-auto leading-relaxed">
                {cloudRunCmd}
              </pre>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[#E2ECE5]">
                  Build &amp; Run Production Docker Image Anywhere:
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('docker', dockerCmd)}
                  className="inline-flex items-center gap-1 text-[#34D399] hover:underline cursor-pointer"
                >
                  <Copy className="w-3 h-3" />
                  Copy
                </button>
              </div>
              <pre className="p-3 rounded-lg bg-[#070A08] border border-[#1B2821] text-[#D5E1D9] font-mono text-[11px] overflow-x-auto leading-relaxed">
                {dockerCmd}
              </pre>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[#E2ECE5]">
                  Push Clean Repository to GitHub (FreshScan-AI):
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('git', gitCliCmd)}
                  className="inline-flex items-center gap-1 text-[#34D399] hover:underline cursor-pointer"
                >
                  <Copy className="w-3 h-3" />
                  Copy
                </button>
              </div>
              <pre className="p-3 rounded-lg bg-[#070A08] border border-[#1B2821] text-[#D5E1D9] font-mono text-[11px] overflow-x-auto leading-relaxed">
                {gitCliCmd}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Honest Hosting Architecture Comparison */}
      <section className="bg-[#0C120F]/92 border border-[#1B2821] rounded-xl p-6 space-y-4">
        <h2 className="font-display text-base font-semibold text-[#F3F7F4]">
          04. Deployment Architecture &amp; Free-Tier Memory Considerations
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-[#A3B5AA] leading-relaxed">
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">Google Cloud Run (Recommended)</h3>
            <p>
              Deploys the single <code className="font-mono text-[#E2ECE5]">Dockerfile</code>{' '}
              (Express SPA + FastAPI TensorFlow CPU + 8.92 MB <code className="font-mono text-[#E2ECE5]">.keras</code>{' '}
              model) on 2 GiB RAM with generous monthly free-tier vCPU-seconds and automatic HTTPS.
            </p>
          </div>
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">Hugging Face Spaces (Docker CPU)</h3>
            <p>
              Offers 16 GiB RAM on the free CPU Basic tier—ideal for zero-cost TensorFlow 2.21
              hosting without 512 MiB OOM limits common on micro PaaS free plans.
            </p>
          </div>
          <div className="space-y-1.5">
            <h3 className="font-semibold text-[#F3F7F4]">Split Vercel Frontend + FastAPI API</h3>
            <p>
              Deploy <code className="font-mono text-[#E2ECE5]">dist/</code> to Vercel or Netlify
              and run <code className="font-mono text-[#E2ECE5]">uvicorn app:app</code> on a ≥1 GiB
              container with <code className="font-mono text-[#E2ECE5]">CORS_ORIGINS</code> set to
              your frontend origin.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
