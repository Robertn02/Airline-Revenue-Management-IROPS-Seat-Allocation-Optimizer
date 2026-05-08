# Deployment Guide

This guide walks through deploying Reroute as a public, always-on demo:
- **Static site** on GitHub Pages (the read-only demo)
- **Live API** on Render free tier (the interactive features)

End result: a public URL where anyone can click "New scenario" and watch the LP allocator run live.

Total time: about 30 minutes. Total cost: $0/month.

---

## Part 1 — Push to GitHub

```bash
cd reroute
git init
git add .
git commit -m "Initial commit"
gh repo create reroute --public --source=. --remote=origin --push
# or use the GitHub web UI to create the repo, then:
# git remote add origin https://github.com/YOUR_USERNAME/reroute.git
# git branch -M main
# git push -u origin main
```

Once pushed, GitHub Actions will run the test suite automatically on every push (`.github/workflows/tests.yml`).

---

## Part 2 — Deploy the API on Render (free tier)

### 2.1 — Create a Render account

Go to [render.com](https://render.com) and sign up. **No credit card required** for the free tier. You can sign in with GitHub, which makes step 2.3 easier.

### 2.2 — Create a new Web Service

From the Render dashboard:

1. Click **New +** → **Web Service**
2. Connect your GitHub account (if not already connected) and select the `reroute` repo
3. On the configuration page:
   - **Name**: `reroute-api` (or whatever you want — this becomes part of the URL)
   - **Region**: pick the region closest to most of your viewers
   - **Branch**: `main`
   - **Runtime**: should auto-detect as **Docker** (because of the `Dockerfile`)
   - **Plan**: select **Free**
4. Click **Create Web Service**

Render will start building the Docker image. This takes 8–12 minutes the first time because it's installing LightGBM and pre-training the model. You can watch the build logs.

When it's done, you'll get a public URL like:

```
https://reroute-api-XXXX.onrender.com
```

Test it:

```bash
curl https://reroute-api-XXXX.onrender.com/api/health
# → {"status":"ok","version":"0.1.0","model_loaded":true}
```

### 2.3 — What to expect from the free tier

- **Spins down after 15 minutes of inactivity.** First request after a quiet period takes 30–60 seconds while the container wakes up. The web frontend shows a "first request after idle may take 20–40s" message when this happens.
- **750 instance-hours per month.** More than enough for one always-running app.
- **512MB RAM.** Reroute uses about 170MB after model load — plenty of headroom.
- **Builds rebuild from scratch when you push to `main`.** Auto-deploy is on by default.

If you want zero cold starts, upgrade to Render's $7/month Starter tier later. Not needed for a portfolio demo.

---

## Part 3 — Deploy the static frontend to GitHub Pages

The frontend is in `web/`. It's already configured to detect the API automatically, but you need to tell it where the API lives.

### 3.1 — Tell GitHub Actions about your API URL

In your GitHub repo:

1. Go to **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. Go to **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
4. Click **New repository variable**:
   - Name: `REROUTE_API_URL`
   - Value: `https://reroute-api-XXXX.onrender.com` (your Render URL from Part 2.2)

### 3.2 — Trigger the Pages build

Either push any commit to `main`, or:
- Go to **Actions** → **deploy-pages** → **Run workflow**

The workflow (`.github/workflows/deploy-pages.yml`) will:
1. Train a fresh risk model
2. Generate 12 demo scenarios
3. Inject your API URL into `index.html`
4. Publish to GitHub Pages

When the workflow finishes, your demo is live at:

```
https://YOUR_USERNAME.github.io/reroute/
```

### 3.3 — Verify it works

Open the URL. You should see:
- The hero animation auto-running
- The "Static mode" badge in the toolbar should now read **"Live API connected"** (green) once the API responds to the health check
- Click **New scenario** — first click takes 30–60 seconds (cold start), subsequent clicks are instant

---

## Part 4 — Custom domain (optional)

If you want `reroute.example.com` instead of `username.github.io/reroute`:

1. In your domain registrar's DNS, add a CNAME record pointing your subdomain to `username.github.io`
2. In GitHub: **Settings** → **Pages** → **Custom domain** → enter your domain
3. Wait for DNS propagation (a few minutes to a few hours)
4. Once verified, enable **Enforce HTTPS**

---

## Troubleshooting

### "API not connecting" badge stays on Static mode

Open browser DevTools → Network tab. Try clicking "New scenario". Look for the `/api/health` request:
- **CORS error**: your API is running but blocking the request. Set `REROUTE_CORS_ORIGINS` env var on Render to your GitHub Pages URL (or `*` for any origin).
- **Connection refused / 404**: API URL is wrong. Check it's set correctly via the `REROUTE_API_URL` GitHub variable AND that your Render service is up.
- **Timeout (60s)**: The free tier is asleep. Wait and retry; the first request wakes it up.

### Render build keeps failing on memory

The 512MB free tier can OOM during model training in some edge cases. The Dockerfile is already designed to train during the build phase (where memory is more generous), not at runtime. If your build still fails:
- Reduce `n_scenarios` in the Dockerfile from 200 to 100
- Or upgrade to Render's $7/month Starter tier with 512MB → 1GB RAM

### Pages workflow fails with "permission denied"

Settings → Actions → General → Workflow permissions → set to **Read and write permissions**.

### Docker build is very slow

Normal — LightGBM compilation takes time. The good news is Render caches Docker layers, so subsequent builds (when you only change Python code) are much faster.

---

## What you should see at the end

A LinkedIn post linking to your GitHub Pages URL would show visitors:
- Auto-running hero animation that loads instantly
- A live, interactive simulator they can click on
- "New scenario" button that produces fresh solver outputs in real time (after ~30s cold start the first time)
- "Tune cost coefficients" sliders that change the optimization in real time

This puts you well above the bar for student portfolio projects.
