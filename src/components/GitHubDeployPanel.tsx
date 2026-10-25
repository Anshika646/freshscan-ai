import React, { useEffect, useState } from 'react';
import { CheckCircle2, Copy, ExternalLink, GitBranch, RefreshCw } from 'lucide-react';
import type { GitHubStatusResponse } from '../types';

export const GitHubDeployPanel: React.FC = () => {
  const [status, setStatus] = useState<GitHubStatusResponse | null>(null);
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

  const loadGitHubStatus = async () => {
    try {
      const resp = await fetch('/api/github/status');
      if (resp.ok) {
        const data = (await resp.json()) as GitHubStatusResponse;
        setStatus(data);
        if (data.target_repo_name) setRepoName(data.target_repo_name);
      }
    } catch {
      // Handled silently if backend is reloading
    }
  };

  useEffect(() => {
    void loadGitHubStatus();

    const handlePopupMessage = (event: MessageEvent) => {
      if (event.data?.type === 'OAUTH_AUTH_SUCCESS') {
        void loadGitHubStatus();
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
        setErrorMsg('Popup window was blocked by the browser. Please allow popups for this domain and try again.');
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const handleSyncRepository = async () => {
    setErrorMsg(null);
    setSyncResult(null);
    setSyncing(true);
    try {
      const resp = await fetch('/api/github/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repoName, private: isPrivate }),
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

  return (
    <div className="space-y-10">
      <section className="space-y-2 pb-6 border-b border-[#DFE5DD] max-w-3xl">
        <p className="text-xs font-medium text-[#4A5D52]">
          Repository Target: FreshScan-AI · Container Target: Google Cloud Run (Dockerfile included)
        </p>
        <h1 className="text-3xl font-bold tracking-tight text-[#14241B] [text-wrap:balance]">
          GitHub Repository Sync & Cloud Run Deployment
        </h1>
        <p className="text-sm text-[#4A5D52] leading-relaxed">
          Authorize with GitHub OAuth to create and push the clean <code className="font-mono text-xs">FreshScan-AI</code> repository directly (automatically excluding raw training datasets, virtualenvs, and secrets), then deploy to Google Cloud Run.
        </p>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-6 bg-white border border-[#DFE5DD] rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-[#14241B] flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-[#1B4332]" />
              <span>01. GitHub OAuth Repository Sync</span>
            </h2>
            {status?.authenticated && (
              <span className="text-xs font-medium text-[#16A34A]">
                Connected as @{status.username}
              </span>
            )}
          </div>

          {!status?.oauth_configured ? (
            <div className="space-y-4 text-xs text-[#4A5D52] leading-relaxed">
              <p>
                To enable one-click popup authorization from inside this preview container, register a GitHub OAuth App at{' '}
                <a
                  href="https://github.com/settings/developers"
                  target="_blank"
                  rel="noreferrer"
                  className="text-[#1B4332] font-medium hover:underline"
                >
                  github.com/settings/developers
                </a>{' '}
                and add <code className="font-mono">GITHUB_CLIENT_ID</code> and <code className="font-mono">GITHUB_CLIENT_SECRET</code> in the AI Studio Secrets panel.
              </p>

              {status && (
                <div className="space-y-2.5 p-3.5 rounded-lg bg-[#F7F8F5] border border-[#DFE5DD]">
                  <div className="font-semibold text-[#14241B]">Required OAuth Callback URLs:</div>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <code className="font-mono text-[11px] text-[#14241B] break-all">
                        {status.dev_callback_url}
                      </code>
                      <button
                        type="button"
                        onClick={() => handleCopy('dev', status.dev_callback_url)}
                        className="p-1.5 text-[#4A5D52] hover:text-[#14241B] cursor-pointer shrink-0"
                        title="Copy Development Callback URL"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <code className="font-mono text-[11px] text-[#14241B] break-all">
                        {status.shared_callback_url}
                      </code>
                      <button
                        type="button"
                        onClick={() => handleCopy('shared', status.shared_callback_url)}
                        className="p-1.5 text-[#4A5D52] hover:text-[#14241B] cursor-pointer shrink-0"
                        title="Copy Shared Callback URL"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  {copiedKey && (
                    <div className="text-[11px] text-[#16A34A] font-medium">
                      Copied {copiedKey} callback URL to clipboard.
                    </div>
                  )}
                </div>
              )}

              <button
                type="button"
                onClick={() => void loadGitHubStatus()}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-[#14241B] bg-[#F7F8F5] border border-[#DFE5DD] rounded-lg hover:border-[#1B4332] cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Re-check OAuth Configuration
              </button>
            </div>
          ) : !status.authenticated ? (
            <div className="space-y-4">
              <p className="text-xs text-[#4A5D52] leading-relaxed">
                GitHub OAuth credentials are configured. Authorize your GitHub account in a popup window to create or update the <code className="font-mono">FreshScan-AI</code> repository.
              </p>
              <button
                type="button"
                onClick={() => void handleConnectGitHub()}
                className="px-4 py-2.5 text-xs font-semibold text-white bg-[#1B4332] rounded-lg hover:bg-[#143326] transition-colors cursor-pointer"
              >
                Authorize with GitHub
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <label className="space-y-1 text-xs">
                  <span className="font-medium text-[#14241B]">Repository Name</span>
                  <input
                    type="text"
                    value={repoName}
                    onChange={(e) => setRepoName(e.target.value)}
                    className="w-full px-3 py-2 text-xs font-mono bg-[#F7F8F5] border border-[#DFE5DD] rounded-lg focus:outline-none focus:border-[#1B4332]"
                  />
                </label>
                <label className="flex items-center gap-2 text-xs pt-5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isPrivate}
                    onChange={(e) => setIsPrivate(e.target.checked)}
                    className="rounded border-[#DFE5DD] accent-[#1B4332]"
                  />
                  <span className="text-[#14241B] font-medium">Create as Private Repository</span>
                </label>
              </div>

              <button
                type="button"
                disabled={syncing}
                onClick={() => void handleSyncRepository()}
                className="inline-flex items-center gap-2 px-4 py-2.5 text-xs font-semibold text-white bg-[#1B4332] rounded-lg hover:bg-[#143326] transition-colors cursor-pointer disabled:opacity-50"
              >
                {syncing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Pushing Clean Tree to GitHub...</span>
                  </>
                ) : (
                  <span>Create & Push to GitHub ({repoName})</span>
                )}
              </button>
            </div>
          )}

          {errorMsg && (
            <div className="p-3 rounded-lg bg-[#FEE2E2] text-xs text-[#991B1B]">{errorMsg}</div>
          )}

          {syncResult && (
            <div className="p-4 rounded-lg bg-[#EDF3EE] border border-[#B7C4BC] space-y-2 text-xs">
              <div className="flex items-center gap-1.5 font-semibold text-[#1B4332]">
                <CheckCircle2 className="w-4 h-4" />
                <span>Repository Synced ({syncResult.files_synced} files committed)</span>
              </div>
              <div className="font-mono text-[11px] text-[#4A5D52]">
                Commit SHA: {syncResult.commit_sha.slice(0, 12)}
              </div>
              <a
                href={syncResult.repo_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-semibold text-[#1B4332] hover:underline"
              >
                <span>{syncResult.repo_url}</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          )}
        </div>

        <div className="lg:col-span-6 bg-white border border-[#DFE5DD] rounded-xl p-6 space-y-5">
          <h2 className="text-base font-semibold text-[#14241B]">
            02. Standard Git CLI & Google Cloud Run Commands
          </h2>
          <p className="text-xs text-[#4A5D52] leading-relaxed">
            This workspace is pre-initialized as a clean Git repository with <code className="font-mono">.gitignore</code> and a production multi-runtime <code className="font-mono">Dockerfile</code> bundling Node 22 + Python TensorFlow CPU.
          </p>

          <div className="space-y-3 text-xs">
            <div className="space-y-1">
              <div className="font-medium text-[#14241B]">Push to GitHub via CLI / AI Studio Git Export:</div>
              <pre className="p-3 rounded-lg bg-[#14241B] text-[#E6EFE9] font-mono text-[11px] overflow-x-auto leading-relaxed">{`git remote add origin https://github.com/<YOUR_USERNAME>/FreshScan-AI.git
git branch -M main
git push -u origin main`}</pre>
            </div>

            <div className="space-y-1">
              <div className="font-medium text-[#14241B]">Deploy directly from source to Google Cloud Run:</div>
              <pre className="p-3 rounded-lg bg-[#14241B] text-[#E6EFE9] font-mono text-[11px] overflow-x-auto leading-relaxed">{`gcloud run deploy freshscan-ai \\
  --source . \\
  --region asia-east1 \\
  --memory 2Gi \\
  --cpu 2 \\
  --allow-unauthenticated`}</pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
