import express, { type Request, type Response } from 'express';
import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const ML_PORT = Number(process.env.ML_PORT || 8050);
const ML_HOST = '127.0.0.1';

const DEV_APP_URL =
  process.env.APP_URL ||
  'https://ais-dev-gipbnbopwii3plib5tjxkw-870394523421.asia-east1.run.app';
const SHARED_APP_URL =
  process.env.SHARED_APP_URL ||
  'https://ais-pre-gipbnbopwii3plib5tjxkw-870394523421.asia-east1.run.app';

const MODELS_DIR = process.env.FRESHSCAN_MODELS_DIR
  ? path.resolve(process.env.FRESHSCAN_MODELS_DIR)
  : path.join(__dirname, 'models');
const DATA_DIR = process.env.FRESHSCAN_DATA_DIR
  ? path.resolve(process.env.FRESHSCAN_DATA_DIR)
  : path.join(__dirname, 'data');

const ALLOWED_PLOTS = new Set([
  'confusion_matrix.png',
  'training_curves.png',
  'class_distribution.png',
]);

let mlProcess: ChildProcess | null = null;
let mlBootstrapping = false;
let mlOnline = false;
let shuttingDown = false;

function pythonHasMlDeps(pythonBin: string): boolean {
  try {
    const check = spawnSync(
      pythonBin,
      ['-c', 'import fastapi, uvicorn, PIL, numpy, tensorflow'],
      { timeout: 15000, stdio: 'ignore' },
    );
    return check.status === 0;
  } catch {
    return false;
  }
}

function resolveReadyPythonExecutable(): string | null {
  const candidates = [
    path.join(__dirname, '.venv', 'bin', 'python'),
    '/app/.venv/bin/python',
    '/app/applet/.venv/bin/python',
    '/tmp/freshscan_venv/bin/python',
    'python3',
  ];
  for (const candidate of candidates) {
    if (candidate === 'python3' || fs.existsSync(candidate)) {
      if (pythonHasMlDeps(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function bootstrapRuntimePythonEnv(onComplete: () => void): void {
  if (mlBootstrapping || shuttingDown) return;
  mlBootstrapping = true;
  console.log('[server] Bootstrapping Python ML runtime environment in /tmp/freshscan_venv...');

  const reqFile = path.join(__dirname, 'requirements.txt');
  const bootstrapCmd = [
    'rm -rf /tmp/freshscan_venv',
    'python3 -m venv --without-pip --copies /tmp/freshscan_venv',
    'export UV_INSTALL_DIR=/tmp/uv_bin',
    'curl -LsSf https://astral.sh/uv/install.sh | sh',
    `/tmp/uv_bin/uv pip install --no-cache --python /tmp/freshscan_venv/bin/python -r "${reqFile}"`,
  ].join(' && ');

  const proc = spawn('bash', ['-lc', bootstrapCmd], {
    cwd: __dirname,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  proc.stdout?.on('data', (chunk: Buffer) => {
    process.stdout.write(`[py-bootstrap] ${chunk.toString()}`);
  });
  proc.stderr?.on('data', (chunk: Buffer) => {
    process.stderr.write(`[py-bootstrap] ${chunk.toString()}`);
  });
  proc.on('exit', (code) => {
    mlBootstrapping = false;
    if (code === 0) {
      console.log('[server] Python ML runtime bootstrap succeeded.');
      onComplete();
    } else {
      console.error(`[server] Python ML runtime bootstrap exited with code ${code}.`);
    }
  });
}

function probeMlPort(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: ML_HOST,
        port: ML_PORT,
        path: '/api/ml/health',
        timeout: 1500,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      },
    );
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

function startPythonBackend(): void {
  if (shuttingDown || mlProcess || mlBootstrapping) return;
  const appPyPath = path.join(__dirname, 'app.py');
  if (!fs.existsSync(appPyPath)) {
    console.warn('[server] app.py not found; skipping Python ML backend spawn.');
    return;
  }

  const pythonBin = resolveReadyPythonExecutable();
  if (!pythonBin) {
    bootstrapRuntimePythonEnv(() => startPythonBackend());
    return;
  }

  console.log(`[server] Starting Python FastAPI ML service via ${pythonBin} on ${ML_HOST}:${ML_PORT}...`);
  mlProcess = spawn(pythonBin, [appPyPath], {
    cwd: __dirname,
    env: {
      ...process.env,
      ML_PORT: String(ML_PORT),
      PYTHONUNBUFFERED: '1',
      MALLOC_ARENA_MAX: '2',
      OMP_NUM_THREADS: '2',
      TF_NUM_INTRAOP_THREADS: '2',
      TF_NUM_INTEROP_THREADS: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  mlProcess.stdout?.on('data', (chunk: Buffer) => {
    const text = chunk.toString();
    if (text.includes('Application startup complete') || text.includes('Uvicorn running')) {
      mlOnline = true;
    }
    process.stdout.write(`[ml-backend] ${text}`);
  });
  mlProcess.stderr?.on('data', (chunk: Buffer) => {
    const text = chunk.toString();
    if (text.includes('Application startup complete') || text.includes('Uvicorn running')) {
      mlOnline = true;
    }
    process.stderr.write(`[ml-backend] ${text}`);
  });

  mlProcess.on('exit', (code, signal) => {
    console.warn(`[server] Python ML service exited (code=${code}, signal=${signal}).`);
    mlProcess = null;
    mlOnline = false;
    if (!shuttingDown) {
      setTimeout(() => startPythonBackend(), 3000);
    }
  });
}

function stopPythonBackend(): void {
  shuttingDown = true;
  if (mlProcess) {
    mlProcess.kill('SIGTERM');
    mlProcess = null;
  }
}

process.on('SIGINT', () => {
  stopPythonBackend();
  process.exit(0);
});
process.on('SIGTERM', () => {
  stopPythonBackend();
  process.exit(0);
});

function readJsonSafe<T = Record<string, unknown>>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
  } catch {
    return null;
  }
}

interface GitHubSession {
  accessToken: string;
  username: string;
  avatarUrl: string | null;
  createdAt: number;
}

const githubSessions = new Map<string, GitHubSession>();
const SESSION_COOKIE_NAME = 'freshscan_gh_sid';

function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!cookieHeader) return out;
  for (const part of cookieHeader.split(';')) {
    const idx = part.indexOf('=');
    if (idx > 0) {
      const k = part.slice(0, idx).trim();
      const v = decodeURIComponent(part.slice(idx + 1).trim());
      out[k] = v;
    }
  }
  return out;
}

function getSessionFromReq(req: Request): GitHubSession | null {
  const cookies = parseCookies(req.headers.cookie);
  const sid = cookies[SESSION_COOKIE_NAME];
  if (!sid) return null;
  return githubSessions.get(sid) || null;
}

function setIframeSafeCookie(res: Response, name: string, value: string, maxAgeSec: number): void {
  const isLocal = DEV_APP_URL.startsWith('http://localhost');
  const attrs = [
    `${name}=${encodeURIComponent(value)}`,
    'Path=/',
    'HttpOnly',
    `Max-Age=${maxAgeSec}`,
  ];
  if (!isLocal) {
    attrs.push('SameSite=None', 'Secure');
  } else {
    attrs.push('SameSite=Lax');
  }
  res.setHeader('Set-Cookie', attrs.join('; '));
}

const EXCLUDED_DIRS = new Set([
  'node_modules',
  '.venv',
  'venv',
  'dist',
  '.git',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.vite',
  '_tmp_shards',
]);

const EXCLUDED_REL_PREFIXES = ['data/raw', 'data/processed', 'models/checkpoints'];

function shouldIncludeFileForGitHub(relPath: string): boolean {
  const normalized = relPath.split(path.sep).join('/');
  for (const prefix of EXCLUDED_REL_PREFIXES) {
    if (normalized === prefix || normalized.startsWith(`${prefix}/`)) {
      return false;
    }
  }
  const base = path.basename(normalized);
  if (
    base === '.env' ||
    (base.startsWith('.env.') && base !== '.env.example') ||
    base === 'server.js' ||
    base.endsWith('.parquet') ||
    base.endsWith('.npz') ||
    base.endsWith('.h5') ||
    base.endsWith('.pyc') ||
    base.endsWith('.log') ||
    base.includes('credentials')
  ) {
    return false;
  }
  return true;
}

function collectProjectFiles(rootDir: string, currentDir = rootDir): string[] {
  const entries = fs.readdirSync(currentDir, { withFileTypes: true });
  const collected: string[] = [];
  for (const entry of entries) {
    if (entry.name === '.DS_Store') continue;
    const fullPath = path.join(currentDir, entry.name);
    const relPath = path.relative(rootDir, fullPath).split(path.sep).join('/');
    if (entry.isDirectory()) {
      if (EXCLUDED_DIRS.has(entry.name)) continue;
      if (!shouldIncludeFileForGitHub(relPath)) continue;
      collected.push(...collectProjectFiles(rootDir, fullPath));
    } else if (entry.isFile()) {
      if (shouldIncludeFileForGitHub(relPath)) {
        collected.push(relPath);
      }
    }
  }
  return collected;
}

function proxyToPythonWithRetry(req: Request, res: Response, retriesLeft = 6): void {
  const targetPath = `/api/ml${req.url === '/' ? '' : req.url}`;
  const chunks: Buffer[] = [];

  req.on('data', (chunk: Buffer) => chunks.push(chunk));
  req.on('end', () => {
    const bodyBuffer = Buffer.concat(chunks);

    const attempt = (remaining: number) => {
      const proxyReq = http.request(
        {
          hostname: ML_HOST,
          port: ML_PORT,
          path: targetPath,
          method: req.method,
          headers: {
            ...req.headers,
            host: `${ML_HOST}:${ML_PORT}`,
            'content-length': String(bodyBuffer.length),
          },
        },
        (proxyRes) => {
          mlOnline = true;
          res.status(proxyRes.statusCode || 502);
          for (const [key, value] of Object.entries(proxyRes.headers)) {
            if (value !== undefined) {
              res.setHeader(key, value);
            }
          }
          proxyRes.pipe(res);
        },
      );

      proxyReq.on('error', () => {
        if (remaining > 0 && !shuttingDown) {
          startPythonBackend();
          setTimeout(() => attempt(remaining - 1), 1500);
          return;
        }
        if (!res.headersSent) {
          res.status(503).json({
            detail: mlBootstrapping
              ? 'Python TensorFlow environment is bootstrapping for first-run deployment. Please retry in ~15 seconds.'
              : `Python TensorFlow inference service is warming up on port ${ML_PORT}. Please retry in a few seconds.`,
          });
        }
      });

      if (bodyBuffer.length > 0) {
        proxyReq.write(bodyBuffer);
      }
      proxyReq.end();
    };

    attempt(retriesLeft);
  });
}

async function bootstrapServer(): Promise<void> {
  startPythonBackend();

  const app = express();

  const MODEL_VERSION = '1.2.0-mobilenetv2';

  function getIsoMtime(filePath: string): string | null {
    if (!fs.existsSync(filePath)) return null;
    try {
      return new Date(fs.statSync(filePath).mtimeMs).toISOString().replace(/\.\d{3}Z$/, 'Z');
    } catch {
      return null;
    }
  }

  // Fast static serving for read-only ML artifacts (eliminates cold-start UI delay)
  app.get('/api/ml/analytics', (_req: Request, res: Response) => {
    const evalFile = path.join(MODELS_DIR, 'evaluation_report.json');
    const datasetSummary = readJsonSafe(path.join(MODELS_DIR, 'dataset_summary.json'));
    const trainingHistory = readJsonSafe(path.join(MODELS_DIR, 'training_history.json'));
    const evaluationReport = readJsonSafe(evalFile);
    const available = Boolean(datasetSummary && trainingHistory && evaluationReport);
    const unavailableReasons: string[] = [];
    if (!datasetSummary) {
      unavailableReasons.push('Dataset summary (models/dataset_summary.json) has not been generated yet.');
    }
    if (!trainingHistory) {
      unavailableReasons.push('Model training history (models/training_history.json) is unavailable.');
    }
    if (!evaluationReport) {
      unavailableReasons.push('Held-out evaluation report (models/evaluation_report.json) is unavailable.');
    }
    res.json({
      available,
      model_version: MODEL_VERSION,
      evaluated_at_utc: getIsoMtime(evalFile),
      unavailable_reasons: unavailableReasons,
      dataset_summary: datasetSummary,
      training_history: trainingHistory,
      evaluation_report: evaluationReport,
    });
  });

  app.get('/api/ml/samples', (_req: Request, res: Response) => {
    const manifest = readJsonSafe<unknown[]>(path.join(DATA_DIR, 'sample_test_images', 'manifest.json'));
    const samples = Array.isArray(manifest) ? manifest : [];
    res.json({ available: samples.length > 0, samples });
  });

  app.get('/api/ml/samples/image/:filename', (req: Request, res: Response) => {
    const safeName = path.basename(req.params.filename || '');
    if (safeName !== req.params.filename || !safeName.endsWith('.jpg')) {
      res.status(400).json({ detail: 'Invalid sample image filename.' });
      return;
    }
    const target = path.join(DATA_DIR, 'sample_test_images', safeName);
    if (!fs.existsSync(target)) {
      res.status(404).json({ detail: 'Sample image not found on disk.' });
      return;
    }
    res.type('image/jpeg').sendFile(target);
  });

  app.get('/api/ml/plots/:plotName', (req: Request, res: Response) => {
    const plotName = req.params.plotName || '';
    if (!ALLOWED_PLOTS.has(plotName)) {
      res.status(404).json({ detail: 'Requested diagnostic chart is not supported.' });
      return;
    }
    const target = path.join(MODELS_DIR, 'plots', plotName);
    if (!fs.existsSync(target)) {
      res.status(404).json({ detail: `Plot '${plotName}' has not been generated yet.` });
      return;
    }
    res.type('image/png').sendFile(target);
  });

  // Top-level REST aliases (/health, /model-info, /metrics, /predict) forwarded to FastAPI
  for (const routeAlias of ['/health', '/model-info', '/metrics', '/predict']) {
    app.all(routeAlias, (req: Request, res: Response) => {
      req.url = routeAlias;
      const retries = req.method === 'POST' ? 8 : 3;
      proxyToPythonWithRetry(req, res, retries);
    });
  }

  // Live inference & TensorFlow health proxied to FastAPI with cold-start retry
  app.use('/api/ml', (req: Request, res: Response) => {
    const retries = req.method === 'POST' ? 8 : 2;
    proxyToPythonWithRetry(req, res, retries);
  });

  app.use(express.json({ limit: '2mb' }));

  app.get('/api/deploy/readiness', async (_req: Request, res: Response) => {
    const modelFile = path.join(MODELS_DIR, 'freshscan_mobilenetv2.keras');
    const classMapFile = path.join(MODELS_DIR, 'class_indices.json');
    const prepConfigFile = path.join(MODELS_DIR, 'preprocessing_config.json');
    const evalReportFile = path.join(MODELS_DIR, 'evaluation_report.json');
    const manifestFile = path.join(DATA_DIR, 'sample_test_images', 'manifest.json');
    const distIndexFile = path.join(__dirname, 'dist', 'index.html');
    const dockerfilePath = path.join(__dirname, 'Dockerfile');

    const modelExists = fs.existsSync(modelFile);
    const modelSizeMb = modelExists
      ? Number((fs.statSync(modelFile).size / (1024 * 1024)).toFixed(2))
      : 0;
    const classMap = readJsonSafe<{ num_classes?: number }>(classMapFile);
    const samples = readJsonSafe<unknown[]>(manifestFile);
    const pythonHealthy = await probeMlPort();
    mlOnline = pythonHealthy;

    const checks = {
      trained_model_saved: modelExists && modelSizeMb > 1,
      class_mapping_verified: Boolean(classMap && classMap.num_classes === 16),
      preprocessing_config_saved: fs.existsSync(prepConfigFile),
      heldout_evaluation_saved: fs.existsSync(evalReportFile),
      heldout_samples_bundled: Array.isArray(samples) && samples.length === 16,
      frontend_production_bundle: fs.existsSync(distIndexFile),
      dockerfile_present: fs.existsSync(dockerfilePath),
      python_inference_engine_online: pythonHealthy,
    };

    const allReady = Object.values(checks).every(Boolean);

    res.json({
      ready_for_production: allReady,
      python_bootstrapping: mlBootstrapping,
      model_size_mb: modelSizeMb,
      checks,
    });
  });

  function hasValidGitHubOAuthConfig(): { configured: boolean; placeholderDetected: boolean } {
    const cid = (process.env.GITHUB_CLIENT_ID || '').trim();
    const csec = (process.env.GITHUB_CLIENT_SECRET || '').trim();
    if (!cid || !csec) {
      return { configured: false, placeholderDetected: false };
    }
    if (cid.length < 16 || csec.length < 20 || cid === 'MY_GITHUB_CLIENT_ID') {
      return { configured: false, placeholderDetected: true };
    }
    return { configured: true, placeholderDetected: false };
  }

  app.get('/api/github/status', (req: Request, res: Response) => {
    const session = getSessionFromReq(req);
    const baseAppUrl = DEV_APP_URL.replace(/\/+$/, '');
    const sharedUrl = SHARED_APP_URL.replace(/\/+$/, '');
    const oauthState = hasValidGitHubOAuthConfig();
    res.json({
      oauth_configured: oauthState.configured,
      placeholder_credentials_detected: oauthState.placeholderDetected,
      authenticated: Boolean(session),
      username: session?.username ?? null,
      avatar_url: session?.avatarUrl ?? null,
      target_repo_name: 'FreshScan-AI',
      dev_callback_url: `${baseAppUrl}/auth/callback`,
      shared_callback_url: `${sharedUrl}/auth/callback`,
    });
  });

  app.get('/api/github/auth/url', (req: Request, res: Response) => {
    const oauthState = hasValidGitHubOAuthConfig();
    const clientId = (process.env.GITHUB_CLIENT_ID || '').trim();
    if (!oauthState.configured) {
      res.status(400).json({
        error: oauthState.placeholderDetected
          ? 'Placeholder GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET detected. Provide valid 20-char GitHub OAuth App credentials.'
          : 'GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET are not configured in environment variables.',
      });
      return;
    }
    const originParam = typeof req.query.origin === 'string' ? req.query.origin : '';
    const base =
      originParam && (originParam.startsWith('https://') || originParam.startsWith('http://localhost'))
        ? originParam.replace(/\/+$/, '')
        : DEV_APP_URL.replace(/\/+$/, '');
    const redirectUri = `${base}/auth/callback`;

    const params = new URLSearchParams({
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: 'repo read:user',
    });

    res.json({
      url: `https://github.com/login/oauth/authorize?${params.toString()}`,
      redirect_uri: redirectUri,
    });
  });

  const oauthCallbackHandler = async (req: Request, res: Response): Promise<void> => {
    const code = typeof req.query.code === 'string' ? req.query.code : '';
    const clientId = process.env.GITHUB_CLIENT_ID;
    const clientSecret = process.env.GITHUB_CLIENT_SECRET;

    if (!code || !clientId || !clientSecret) {
      res.status(400).send('Missing OAuth code or server OAuth configuration.');
      return;
    }

    try {
      const tokenResp = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          client_id: clientId,
          client_secret: clientSecret,
          code,
        }),
      });
      const tokenJson = (await tokenResp.json()) as { access_token?: string; error_description?: string };
      if (!tokenJson.access_token) {
        res.status(400).send(`GitHub OAuth failed: ${tokenJson.error_description || 'No access token'}`);
        return;
      }

      const userResp = await fetch('https://api.github.com/user', {
        headers: {
          Authorization: `Bearer ${tokenJson.access_token}`,
          Accept: 'application/vnd.github+json',
          'User-Agent': 'FreshScan-AI-App',
        },
      });
      const userJson = (await userResp.json()) as { login?: string; avatar_url?: string };
      if (!userJson.login) {
        res.status(400).send('Unable to verify GitHub user identity.');
        return;
      }

      const sid = crypto.randomBytes(24).toString('hex');
      githubSessions.set(sid, {
        accessToken: tokenJson.access_token,
        username: userJson.login,
        avatarUrl: userJson.avatar_url || null,
        createdAt: Date.now(),
      });
      setIframeSafeCookie(res, SESSION_COOKIE_NAME, sid, 60 * 60 * 6);

      res.send(`<!doctype html>
<html>
  <body>
    <script>
      if (window.opener) {
        window.opener.postMessage({ type: 'OAUTH_AUTH_SUCCESS' }, '*');
        window.close();
      } else {
        window.location.href = '/';
      }
    </script>
    <p>GitHub authorization complete. Closing window...</p>
  </body>
</html>`);
    } catch (err) {
      res.status(500).send(`OAuth callback error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  app.get(['/auth/callback', '/auth/callback/'], oauthCallbackHandler);

  app.post('/api/github/disconnect', (_req: Request, res: Response) => {
    setIframeSafeCookie(res, SESSION_COOKIE_NAME, '', 0);
    res.json({ disconnected: true });
  });

  app.post('/api/github/sync', async (req: Request, res: Response) => {
    const session = getSessionFromReq(req);
    if (!session) {
      res.status(401).json({ error: 'Not authenticated with GitHub. Connect your GitHub account first.' });
      return;
    }

    const repoName =
      typeof req.body?.repoName === 'string' && /^[A-Za-z0-9_.-]+$/.test(req.body.repoName)
        ? req.body.repoName.trim()
        : 'FreshScan-AI';

    if (repoName.toLowerCase().includes('webpearl')) {
      res.status(400).json({
        error:
          'Repository guard: Syncing FreshScan AI to webpearl-website is strictly prohibited. Please target FreshScan-AI.',
      });
      return;
    }

    const isPrivate = Boolean(req.body?.private);

    const ghHeaders = {
      Authorization: `Bearer ${session.accessToken}`,
      Accept: 'application/vnd.github+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FreshScan-AI-Sync',
    };

    try {
      const owner = session.username;
      let repoCheck = await fetch(`https://api.github.com/repos/${owner}/${repoName}`, {
        headers: ghHeaders,
      });

      if (repoCheck.status === 404) {
        const createResp = await fetch('https://api.github.com/user/repos', {
          method: 'POST',
          headers: ghHeaders,
          body: JSON.stringify({
            name: repoName,
            description:
              'FreshScan AI — End-to-End Fruit Freshness Detection with MobileNetV2 Transfer Learning, FastAPI & React',
            private: isPrivate,
            auto_init: true,
          }),
        });
        if (!createResp.ok) {
          const errText = await createResp.text();
          res.status(createResp.status).json({ error: `Failed to create repository: ${errText}` });
          return;
        }
        repoCheck = createResp;
      }

      const repoData = (await repoCheck.json()) as { html_url: string; default_branch?: string };
      const branch = repoData.default_branch || 'main';

      const refResp = await fetch(
        `https://api.github.com/repos/${owner}/${repoName}/git/ref/heads/${branch}`,
        { headers: ghHeaders },
      );
      let parentCommitSha: string | null = null;
      if (refResp.ok) {
        const refData = (await refResp.json()) as { object?: { sha?: string } };
        parentCommitSha = refData.object?.sha || null;
      }

      const filesToSync = collectProjectFiles(__dirname);
      const treeItems: Array<{ path: string; mode: string; type: string; sha: string }> = [];

      for (const relFile of filesToSync) {
        const absFile = path.join(__dirname, relFile);
        const stat = fs.statSync(absFile);
        if (stat.size > 25 * 1024 * 1024) continue;
        const contentBase64 = fs.readFileSync(absFile).toString('base64');

        const blobResp = await fetch(`https://api.github.com/repos/${owner}/${repoName}/git/blobs`, {
          method: 'POST',
          headers: ghHeaders,
          body: JSON.stringify({ content: contentBase64, encoding: 'base64' }),
        });
        if (!blobResp.ok) {
          const bErr = await blobResp.text();
          throw new Error(`Failed uploading blob for ${relFile}: ${bErr}`);
        }
        const blobJson = (await blobResp.json()) as { sha: string };
        treeItems.push({
          path: relFile,
          mode: '100644',
          type: 'blob',
          sha: blobJson.sha,
        });
      }

      const treeResp = await fetch(`https://api.github.com/repos/${owner}/${repoName}/git/trees`, {
        method: 'POST',
        headers: ghHeaders,
        body: JSON.stringify({ tree: treeItems }),
      });
      if (!treeResp.ok) {
        throw new Error(`Failed creating git tree: ${await treeResp.text()}`);
      }
      const treeJson = (await treeResp.json()) as { sha: string };

      const commitPayload: Record<string, unknown> = {
        message: 'Sync FreshScan AI end-to-end ML pipeline, trained MobileNetV2 artifacts, and web app',
        tree: treeJson.sha,
      };
      if (parentCommitSha) {
        commitPayload.parents = [parentCommitSha];
      }

      const commitResp = await fetch(`https://api.github.com/repos/${owner}/${repoName}/git/commits`, {
        method: 'POST',
        headers: ghHeaders,
        body: JSON.stringify(commitPayload),
      });
      if (!commitResp.ok) {
        throw new Error(`Failed creating commit: ${await commitResp.text()}`);
      }
      const commitJson = (await commitResp.json()) as { sha: string };

      if (parentCommitSha) {
        await fetch(`https://api.github.com/repos/${owner}/${repoName}/git/refs/heads/${branch}`, {
          method: 'PATCH',
          headers: ghHeaders,
          body: JSON.stringify({ sha: commitJson.sha, force: true }),
        });
      } else {
        await fetch(`https://api.github.com/repos/${owner}/${repoName}/git/refs`, {
          method: 'POST',
          headers: ghHeaders,
          body: JSON.stringify({ ref: `refs/heads/${branch}`, sha: commitJson.sha }),
        });
      }

      res.json({
        synced: true,
        repo_url: repoData.html_url,
        branch,
        commit_sha: commitJson.sha,
        files_synced: treeItems.length,
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });

  const distPath = path.join(__dirname, 'dist');
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[server] FreshScan AI web server listening on http://0.0.0.0:${PORT}`);
  });
}

bootstrapServer().catch((err) => {
  console.error('[server] Fatal startup error:', err);
  process.exit(1);
});
