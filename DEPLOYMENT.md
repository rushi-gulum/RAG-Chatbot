# Production Deployment

## Recommended free-tier architecture

```text
Next.js frontend  -> Vercel
FastAPI backend   -> Render web service
SQLAlchemy         -> Supabase or Neon PostgreSQL
Firebase           -> Authentication only
Embeddings         -> Cloudflare Workers AI BGE-small
Vector search      -> Chroma Cloud
LLM               -> Groq API
```

Firebase does not provide a SQLAlchemy relational database. Do not try to deploy the SQLite file to Firebase. Use Supabase or Neon for the SQLAlchemy database.

## 1. Create the hosted services

1. Create a Supabase or Neon PostgreSQL database and copy its connection string.
2. Create a Chroma Cloud database and copy its API key, tenant, and database values.
3. In Firebase Console, enable Authentication providers and create a Web App. Keep the frontend Firebase values for Vercel.
4. In Firebase Console, create a service-account JSON credential. Store its complete JSON content as a Render secret named `FIREBASE_SERVICE_ACCOUNT_JSON`. Never commit the JSON file.
5. Create a Groq API key.

## 2. Deploy the backend to Render

Create a Web Service from this repository. The checked-in `render.yaml` uses:

```text
Root directory: backend
Build command: pip install -r requirements.txt
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
Health path: /health
```

Set these Render environment variables:

```env
NODE_ENV=production
AUTH_DISABLED=false
DATABASE_URL=postgresql://...
FIREBASE_SERVICE_ACCOUNT_JSON={...complete JSON...}
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_PROVIDER=cloudflare
EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CHROMA_MODE=cloud
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
FRONTEND_URL=https://your-project.vercel.app
ALLOWED_ORIGINS=https://your-project.vercel.app
```

After deployment, verify:

```text
https://your-render-service.onrender.com/health
```

The first request on a free Render service may be slow while it wakes up. Large embedding model startup can also exceed free-tier memory/time limits; monitor the first deployment logs.

## 3. Deploy the frontend to Vercel

Import the repository into Vercel. Set the project root directory to `frontend`. The root `vercel.json` is frontend-only.

Set these Vercel environment variables:

```env
NEXT_PUBLIC_API_URL=https://your-render-service.onrender.com
NEXT_PUBLIC_AUTH_MODE=firebase
NEXT_PUBLIC_FIREBASE_API_KEY=...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=...
NEXT_PUBLIC_FIREBASE_PROJECT_ID=ragchatbot-f3ffd
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=...
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
NEXT_PUBLIC_FIREBASE_APP_ID=...
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=...
```

Redeploy after adding variables. In Firebase Console, add the Vercel domain to Authentication > Settings > Authorized domains.

## 4. Important data behavior

- PostgreSQL stores document metadata and chat history.
- Cloudflare Workers AI generates 384-dimensional BGE-small embeddings over HTTPS.
- Chroma Cloud stores document chunks and embeddings.
- Uploaded PDFs are processed temporarily by the backend and removed after indexing.
- SQLite and local `chroma_db/` are for development only.
- `AUTH_DISABLED=true` is local-only and must be `false` in production.

## 5. Smoke tests

```text
GET  https://your-render-service.onrender.com/health
GET  https://your-render-service.onrender.com/docs
GET  https://your-vercel-project.vercel.app
```

Then sign in through Firebase, upload a PDF, select it, and submit a question. Confirm the answer contains sources and that a second browser account cannot see the first account's documents.

## Security checklist

- Rotate any API keys or service-account keys that were previously exposed.
- Store secrets only in Render/Vercel environment settings.
- Do not set `AUTH_DISABLED=true` on Render.
- Do not deploy SQLite or local Chroma persistence as production storage.
- Do not mix existing `intfloat/e5-small` vectors with BGE vectors. Clear and re-index the Chroma collection after switching providers.
- Restrict Firebase Authentication authorized domains to the deployed frontend domains.
