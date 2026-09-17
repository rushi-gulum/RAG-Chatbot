# Deployment Guide

## Architecture

```
Vercel        → Next.js frontend
Render        → FastAPI backend  (native Python runtime, no Docker needed)
Supabase/Neon → PostgreSQL via SQLAlchemy  (document metadata + chat history)
Firebase      → Authentication only  (Google sign-in, ID-token verification)
Cloudflare    → BGE-small-en-v1.5 embeddings  (384-dim, hosted Workers AI)
Qdrant Cloud  → Vector storage + similarity search
Groq          → LLM responses  (openai/gpt-oss-20b)
```

Uploaded PDFs are processed entirely in memory and deleted immediately after indexing.
No file storage service is required.

---

## Step 1 — Create the hosted services

### 1a. PostgreSQL — Supabase or Neon

**Supabase (recommended free tier)**
1. [app.supabase.com](https://app.supabase.com) → New project
2. Project Settings → Database → Connection string → **URI** tab
3. Copy the URI; it looks like:
   ```
   postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
   ```

**Neon**
1. [neon.tech](https://neon.tech) → New project
2. Dashboard → Connection string
3. Copy the URI; it looks like:
   ```
   postgresql://[user]:[password]@[host].neon.tech/[dbname]?sslmode=require
   ```

Tables are created automatically when the backend first starts (`SQLAlchemy Base.metadata.create_all`). No migration command is needed for a fresh database.

### 1b. Firebase — Authentication

1. [console.firebase.google.com](https://console.firebase.google.com) → your project
2. Authentication → Sign-in method → **Google** → Enable
3. Project Settings → General → Your apps → **Add app** → Web
   - Copy all `NEXT_PUBLIC_FIREBASE_*` values for Vercel (step 3)
4. Project Settings → **Service accounts** → Generate new private key
   - A JSON file is downloaded. Keep it secret.
   - Open the file, copy the entire contents as one line — this becomes `GOOGLE_APPLICATION_CREDENTIALS_JSON` on Render

### 1c. Qdrant Cloud

Already configured. Credentials are in `backend/.env`.
Payload indexes on `user_id` and `document_id` are created automatically by `_ensure_collection()`.

### 1d. Cloudflare Workers AI

Already configured. Credentials are in `backend/.env`.

### 1e. Groq

Already configured. Key is in `backend/.env`.

---

## Step 2 — Deploy the backend to Render

### 2a. Create the service

1. [dashboard.render.com](https://dashboard.render.com) → **New Web Service**
2. Connect your GitHub repository
3. Configure:
   | Field | Value |
   |---|---|
   | Root Directory | `backend` |
   | Runtime | **Python 3** |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
   | Health Check Path | `/health` |

   The `render.yaml` in the repo root contains these settings already; Render will pick them up automatically on import.

### 2b. Set environment variables

Go to **Environment** tab and add these secrets (do not commit them to the repo):

```env
NODE_ENV=production
AUTH_DISABLED=false

# PostgreSQL
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require

# Firebase (paste the full service-account JSON as one line)
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
FIREBASE_PROJECT_ID=your-firebase-project-id

# Groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b

# Cloudflare Workers AI
EMBEDDING_PROVIDER=cloudflare
EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...

# Qdrant Cloud
QDRANT_URL=https://your-cluster.region.aws.cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=rag_documents

# CORS — set to your Vercel URL after step 3
FRONTEND_URL=https://your-app.vercel.app
ALLOWED_ORIGINS=https://your-app.vercel.app

# Security
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
```

All `sync: false` variables in `render.yaml` are placeholders that Render will not sync — they must be set manually in the dashboard.

### 2c. Verify the backend

After the first deploy succeeds:
```
GET https://your-render-service.onrender.com/health
GET https://your-render-service.onrender.com/rag/status
```

Free-tier Render services spin down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds.

---

## Step 3 — Deploy the frontend to Vercel

### 3a. Import the project

1. [vercel.com](https://vercel.com) → **Add New Project** → Import your GitHub repo
2. Vercel auto-detects Next.js. Confirm:
   | Field | Value |
   |---|---|
   | Framework Preset | Next.js |
   | Root Directory | `frontend` |

   `vercel.json` at the repo root already sets `"root": "frontend"` — Vercel will apply this automatically.

### 3b. Set environment variables

In the Vercel project settings → **Environment Variables**:

```env
NEXT_PUBLIC_API_URL=https://your-render-service.onrender.com
NEXT_PUBLIC_AUTH_MODE=firebase

NEXT_PUBLIC_FIREBASE_API_KEY=AIzaSy...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
NEXT_PUBLIC_FIREBASE_APP_ID=1:...:web:...
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=G-...
```

Redeploy after saving the variables.

### 3c. Authorize the Vercel domain in Firebase

Firebase Console → Authentication → Settings → **Authorized domains** → Add domain:
```
your-app.vercel.app
```
Without this, Google sign-in will show an `auth/unauthorized-domain` error.

---

## Step 4 — Smoke test

```
# Backend health
GET https://your-render-service.onrender.com/health
GET https://your-render-service.onrender.com/rag/status

# Full flow (via the UI)
1. Open https://your-app.vercel.app
2. Sign in with Google
3. Upload a PDF
4. Select the document and ask a question
5. Confirm the answer has inline source citations
6. Sign in with a second Google account and confirm it cannot see the first account's documents
```

---

## Local development

```bash
# Backend
cd backend
cp .env.production.example .env   # fill in real values
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
python app.py

# Frontend
cd frontend
cp .env.local.example .env.local  # fill in local values
npm install
npm run dev
```

For local-only dev without Firebase auth, keep:
```env
# backend/.env
AUTH_DISABLED=true
AUTH_DEV_USER_ID=local-development-user

# frontend/.env.local
NEXT_PUBLIC_AUTH_MODE=local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## Security checklist

- [ ] `AUTH_DISABLED=false` on Render
- [ ] `GOOGLE_APPLICATION_CREDENTIALS_JSON` stored only as a Render secret, never committed
- [ ] All API keys set as secrets in Render / Vercel dashboards
- [ ] Firebase authorized domains updated with Vercel URL
- [ ] Vercel domain added before sharing the app publicly
- [ ] `SECRET_KEY` is a random 64-character hex string (not the default placeholder)
- [ ] Rotate any keys that were previously committed to the repo (Groq, Qdrant, Cloudflare, Firebase private key)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 401 on all API calls | Firebase token not accepted | Check `GOOGLE_APPLICATION_CREDENTIALS_JSON` on Render; ensure no newline inside the JSON |
| CORS error in browser | `ALLOWED_ORIGINS` mismatch | Set `FRONTEND_URL` and `ALLOWED_ORIGINS` to exact Vercel URL (no trailing slash) |
| "No relevant information" | Qdrant payload index missing | Indexes are auto-created by `_ensure_collection()`; re-upload a document to trigger |
| 503 on `/rag/*` | Qdrant env vars missing | Check `QDRANT_URL` and `QDRANT_API_KEY` on Render |
| Slow first request | Render free-tier cold start | Expected; upgrade to a paid instance or use a cron to keep it warm |
| `auth/unauthorized-domain` | Firebase missing Vercel domain | Add `your-app.vercel.app` to Firebase → Authentication → Authorized domains |
| DB tables not created | `DATABASE_URL` wrong format | Must start with `postgresql://`; the code rewrites it to `postgresql+psycopg://` automatically |
