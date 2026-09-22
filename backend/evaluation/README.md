# Retrieval evaluation

Copy `dataset.example.json` to `dataset.json`. Each entry needs a question and
the UUID of at least one document that contains its answer. Document UUIDs are
returned by the upload endpoint and shown by the document-list endpoint.

Run the evaluator from the `backend` directory after setting the normal
Cloudflare and Qdrant environment variables:

```powershell
python scripts/evaluate_retrieval.py --dataset evaluation/dataset.json --user-id YOUR_FIREBASE_UID
```

The script reports Recall@K and MRR. Start with 30–50 real user questions and
run it before and after changing chunking, embedding, or retrieval settings.
