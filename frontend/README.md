This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

I’ll audit the deployment boundaries first: Vercel frontend, FastAPI backend, SQLAlchemy database, file storage, and vector storage. One important correction up front: Firebase Authentication is suitable for identity, but it does not provide a SQLAlchemy-compatible relational database; for SQLAlchemy use a free PostgreSQL provider such as Supabase or Neon, and use Firebase only for auth.

The repository confirms the key production constraint: SQLite and local ChromaDB are filesystem-backed, while Vercel functions are not a durable home for either.

I’m preparing a split deployment: Vercel for frontend, Render for the FastAPI backend, Supabase Postgres for SQLAlchemy, and Qdrant Cloud for vectors; Firebase remains authentication only. I’ll add deployment config and production environment templates without committing secrets.



Steps:

Go to: https://vercel.com/dashboard
Select your project: rag-chatbot-4o35
Go to: Settings → Environment Variables
Add each variable above
Make sure they're enabled for Production environment
After adding all, click Redeploy from the Deployments tab
2. Update Render Environment Variables
Go to your Render service and update CORS settings:

Steps:

Go to: https://dashboard.render.com/
Select service: vw-brain-api (or your backend service name)
Go to: Environment tab
Update these variables:
FRONTEND_URL = https://rag-chatbot-4o35.vercel.app
ALLOWED_ORIGINS = https://rag-chatbot-4o35.vercel.app,http://localhost:3000
Click Save Changes (Render will auto-redeploy)
🧪 Test the Full Flow
Once both are redeployed, test:

Open: https://rag-chatbot-4o35.vercel.app/
Sign in with Google (should work now that domain is authorized)
Upload a PDF (test file < 5MB)
Ask a question about the PDF content
Check response includes citations
Troubleshooting:
If you see errors, check browser console (F12) and share:

Any CORS errors?
Any Firebase auth errors?
Any network request failures?
Let me know when the redeploys are done and we can test!

Vercel        → Next.js frontend
Render        → FastAPI backend
Supabase/Neon → PostgreSQL + SQLAlchemy
Firebase      → Authentication only
Chroma Cloud  → Vector database
Groq          → LLM responses