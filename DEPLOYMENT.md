# 🚀 Cloud Deployment Guide

Complete deployment guide for the RAG Chatbot on production cloud infrastructure.

## 🏗️ Architecture Overview

- **Frontend**: Vercel (Next.js 15)
- **Backend**: Render (FastAPI + Python 3.11)
- **Authentication**: Firebase (Google OAuth + Email/Password)
- **Embeddings**: Cloudflare Workers AI (BGE-small-en-v1.5)
- **Vector Database**: Qdrant Cloud
- **Database**: Neon PostgreSQL
- **LLM**: Groq API (with OpenAI fallback)

---

## 📋 Pre-Deployment Checklist

### ✅ Service Accounts & API Keys Setup

#### 1. Firebase (Authentication)
- [ ] Create Firebase project: https://console.firebase.google.com/
- [ ] Enable Authentication → Google & Email/Password
- [ ] Generate service account key (Settings → Service accounts)
- [ ] Add authorized domains for production URLs

#### 2. Cloudflare Workers AI (Embeddings)
- [ ] Sign up: https://dash.cloudflare.com/
- [ ] Enable Workers AI
- [ ] Get Account ID & API Token
- [ ] Test BGE-small-en-v1.5 model access

#### 3. Qdrant Cloud (Vector Database)
- [ ] Create cluster: https://cloud.qdrant.io/
- [ ] Note cluster URL and API key
- [ ] Select region closest to backend deployment

#### 4. Neon PostgreSQL (Database)
- [ ] Create database: https://neon.tech/
- [ ] Get connection string (pooled connection recommended)
- [ ] Test connection from local environment

#### 5. Groq API (Primary LLM)
- [ ] Get API key: https://console.groq.com/
- [ ] Test model: `openai/gpt-oss-20b`
- [ ] Note rate limits (200K tokens/day free tier)

#### 6. OpenAI API (Fallback LLM)
- [ ] Get API key: https://platform.openai.com/
- [ ] Add billing method for reliable fallback
- [ ] Test GPT-3.5-turbo access

---

## 🖥️ Backend Deployment (Render)

### 1. Repository Setup
```bash
# Ensure all files are committed
git add .
git commit -m "Production deployment ready"
git push origin main
```

### 2. Render Service Creation
1. Go to https://dashboard.render.com/
2. **New → Web Service**
3. Connect GitHub repository
4. Configure:
   - **Name**: `rag-chatbot-backend`
   - **Region**: Oregon (or closest to users)
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables (Render Dashboard)
```bash
# Core Configuration
NODE_ENV=production
SECRET_KEY=<generate_with_python_secrets>
FRONTEND_URL=https://your-app.vercel.app
ALLOWED_ORIGINS=https://your-app.vercel.app

# Firebase Authentication
FIREBASE_PROJECT_ID=your-firebase-project
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}

# LLM APIs
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
OPENAI_API_KEY=sk-...

# Embeddings (Cloudflare)
EMBEDDING_PROVIDER=cloudflare
EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_token

# Vector Database (Qdrant)
QDRANT_URL=https://your-cluster.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=rag_documents

# PostgreSQL Database (Neon)
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### 4. Health Check
- **Path**: `/health`
- **Expected Response**: `{"status": "healthy", "timestamp": "..."}`

---

## 🌐 Frontend Deployment (Vercel)

### 1. Vercel Project Setup
1. Go to https://vercel.com/dashboard
2. **Import Project** from GitHub
3. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

### 2. Environment Variables (Vercel Dashboard)
```bash
# Backend API
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
NEXT_PUBLIC_AUTH_MODE=firebase

# Firebase Configuration (from Firebase Console → Project Settings)
NEXT_PUBLIC_FIREBASE_API_KEY=AIzaSy...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=123456789
NEXT_PUBLIC_FIREBASE_APP_ID=1:123:web:abc...
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=G-ABC123
```

### 3. Domain Setup
- **Custom Domain**: Configure your domain in Vercel dashboard
- **SSL**: Automatic via Vercel
- **Redirects**: HTTP → HTTPS automatic

---

## 🔧 Post-Deployment Configuration

### 1. Firebase Authorized Domains
Add your production URLs to Firebase:
- `your-app.vercel.app`
- `your-custom-domain.com` (if using custom domain)

### 2. CORS Configuration
Update backend `ALLOWED_ORIGINS` to include all frontend URLs.

### 3. Rate Limiting
Configure production rate limits in `backend/routes/*`:
```python
# Adjust for production traffic
@limiter.limit("100/minute")  # Increase from development limits
```

### 4. Database Migration
```bash
# Run migrations on first deployment (via Render logs or shell)
alembic upgrade head
```

---

## 🧪 Deployment Testing

### 1. Health Checks
```bash
# Backend health
curl https://your-backend.onrender.com/health

# Frontend health
curl https://your-app.vercel.app/api/health
```

### 2. Authentication Flow
1. Visit frontend URL
2. Sign in with Google OAuth
3. Verify JWT token validation
4. Check user session persistence

### 3. Document Upload Test
1. Upload small PDF file
2. Verify processing completes
3. Check Qdrant vector storage
4. Test query & response generation

### 4. API Integration Test
```bash
# Test vector search
curl -X POST https://your-backend.onrender.com/rag/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "test question", "top_k": 3}'
```

---

## 📊 Monitoring & Observability

### Render (Backend)
- **Logs**: Real-time via dashboard
- **Metrics**: CPU, Memory, Response time
- **Alerts**: Set up for 5xx errors, high latency

### Vercel (Frontend)  
- **Analytics**: Built-in Web Vitals
- **Functions**: Serverless function monitoring
- **Edge**: CDN performance metrics

### External Services
- **Groq**: Monitor token usage and rate limits
- **Qdrant**: Vector count and query performance
- **Neon**: Database connection pool and query performance
- **Firebase**: Authentication success rates

---

## 🚨 Troubleshooting

### Common Issues

**Backend Deployment Fails**
```bash
# Check requirements.txt for dependency conflicts
# Verify Python version compatibility
# Check Render logs for specific error
```

**Frontend Build Fails**
```bash
# Verify Node.js version (18.17+)
# Check for TypeScript errors: npm run type-check
# Validate environment variables
```

**Authentication Issues**
```bash
# Verify Firebase project ID matches
# Check authorized domains in Firebase
# Validate service account JSON formatting
```

**API Connection Failures**
```bash
# Test CORS configuration
# Verify backend URL in frontend env
# Check Render service status
```

**Database Connection Issues**
```bash
# Test connection string locally
# Verify Neon database is active
# Check connection pool limits
```

---

## 📈 Performance Optimization

### Backend (Render)
- **Plan**: Upgrade from Free to Starter for better performance
- **Region**: Deploy closest to primary user base
- **Database**: Use connection pooling (already configured)
- **Caching**: Implement Redis for session storage (future enhancement)

### Frontend (Vercel)
- **Edge Functions**: Leverage for dynamic content
- **Image Optimization**: Use Next.js built-in optimization
- **Bundle Analysis**: Monitor bundle size with `npm run build`

### Third-Party Services
- **Groq**: Upgrade to paid tier for higher rate limits
- **Qdrant**: Scale cluster for larger document collections
- **Neon**: Configure read replicas for high traffic

---

## 🔒 Security Considerations

### Environment Variables
- Never commit secrets to version control
- Use platform-specific secret management
- Rotate API keys regularly

### Network Security
- HTTPS enforced on all endpoints
- CORS properly configured
- Rate limiting on API endpoints

### Data Protection
- User data isolation via Firebase UIDs
- Document access control per user
- Regular security updates for dependencies

---

## 💰 Cost Optimization

### Free Tier Limits
- **Render**: 750 hours/month (sufficient for 1 service)
- **Vercel**: 100GB bandwidth, 100 serverless functions
- **Qdrant**: 1GB storage
- **Neon**: 500MB storage
- **Groq**: 200K tokens/day
- **Cloudflare Workers AI**: 10K requests/day

### Scaling Strategy
1. **Month 1-2**: Free tiers across all services
2. **Growth**: Upgrade Groq → Paid tier first
3. **Scale**: Render → Starter plan for better performance
4. **Enterprise**: Custom plans for high-volume usage

---

## 🎯 Success Metrics

- **Uptime**: >99.9% (monitor via status pages)
- **Response Time**: <2s end-to-end query processing
- **Error Rate**: <1% API errors
- **User Growth**: Track via Firebase Analytics
- **Document Processing**: Success rate >95%

---

## 🤝 Support & Maintenance

### Regular Tasks
- [ ] Weekly: Review logs for errors
- [ ] Monthly: Update dependencies
- [ ] Quarterly: Rotate API keys
- [ ] As needed: Scale services based on usage

### Emergency Contacts
- **Render Support**: https://render.com/support
- **Vercel Support**: https://vercel.com/support  
- **Firebase Support**: Firebase Console → Support tab

---

**🎉 Deployment Complete!**

Your RAG Chatbot is now running on production cloud infrastructure with enterprise-grade scalability, security, and monitoring.