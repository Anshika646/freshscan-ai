import express, { type Request, type Response } from 'express';
import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
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

let mlProcess: ChildProcess | null = null;
let shuttingDown = false;

function resolvePythonExecutable(): string {
  const candidates = [
    path.join(__dirname, '.venv', 'bin', 'python'),
    '/app/.venv/bin/python',
    '/app/applet/.venv/bin/python',
    'python3',
  ];
  for (const candidate of candidates) {
    if (candidate === 'python3' || fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return 'python3';
}

function startPythonBackend(): void {
  if (shuttingDown || mlProcess) return;
  const pythonBin = resolvePythonExecutable();
  const appPyPath = path.join(__dirname, 'app.py');
  if (!fs.existsSync(appPyPath)) {
    console.warn('[server] app.py not found; skipping Python ML backend spawn.');
    return;
  }

  console.log(`[server] Starting Python FastAPI ML service via ${pythonBin} on ${ML_HOST}:${ML_PORT}...`);
  mlProcess = spawn(pythonBin, [appPyPath], {
    cwd: __dirname,
    env: {
      ...process.env,
      ML_PORT: String(ML_PORT),
      PYTHONUNBUFFERED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  mlProcess.stdout?.on('data', (chunk: Buffer) => {
    process.stdout.write(`[ml-backend] ${chunk.toString()}`);
  });
  mlProcess.stderr?.on('data', (chunk: Buffer) => {
    process.stderr.write(`[ml-backend] ${chunk.toString()}`);
  });

  mlProcess.on('exit', (code, signal) => {
    console.warn(`[server] Python ML service exited (code=${code}, signal=${signal}).`);
    mlProcess = null;
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

async function bootstrapServer(): Promise<void> {
  startPythonBackend();

  const app = express();

  app.use('/api/ml', (req: Request, res: Response) => {
    const targetPath = `/api/ml${req.url === '/' ? '' : req.url}`;
    const proxyReq = http.request(
      {
        hostname: ML_HOST,
        port: ML_PORT,
        path: targetPath,
        method: req.method,
        headers: {
          ...req.headers,
          host: `${ML_HOST}:${ML_PORT}`,
        },
      },
      (proxyRes) => {
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
      if (!res.headersSent) {
        res.status(503).json({
          detail:
            'Python TensorFlow inference service is initializing on port ' +
            ML_PORT +
            '. Please retry in a few seconds.',
        });
      }
    });

    req.pipe(proxyReq);
  });

  app.use(express.json({ limit: '2mb' }));

  app.get('/api/github/status', (req: Request, res: Response) => {
    const session = getSessionFromReq(req);
    const baseAppUrl = DEV_APP_URL.replace(/\/+$/, '');
    const sharedUrl = SHARED_APP_URL.replace(/\/+$/, '');
    res.json({
      oauth_configured: Boolean(process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET),
      authenticated: Boolean(session),
      username: session?.username ?? null,
      avatar_url: session?.avatarUrl ?? null,
      target_repo_name: 'FreshScan-AI',
      dev_callback_url: `${baseAppUrl}/auth/callback`,
      shared_callback_url: `${sharedUrl}/auth/callback`,
    });
  });

  app.get('/api/github/auth/url', (req: Request, res: Response) => {
    const clientId = process.env.GITHUB_CLIENT_ID;
    const clientSecret = process.env.GITHUB_CLIENT_SECRET;
    if (!clientId || !clientSecret) {
      res.status(400).json({
        error:
          'GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET are not configured in environment secrets.',
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
  <body style="font-family: sans-serif; padding: 24px; background: #F7F8F5; color: #14241B;">
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
        ? req.body.repoName
        : 'FreshScan-AI';
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

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
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
