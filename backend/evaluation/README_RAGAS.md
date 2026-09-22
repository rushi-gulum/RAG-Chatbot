# RAGAS Evaluation System

RAGAS (Retrieval-Augmented Generation Assessment) is an advanced evaluation framework for RAG systems that uses LLMs to provide reference-free evaluation.

## 🎯 RAGAS Metrics

### 1. **Faithfulness**
- **What it measures**: Factual consistency between the generated answer and retrieved context
- **Scale**: 0.0 (completely unfaithful) to 1.0 (completely faithful)  
- **Good score**: ≥ 0.8
- **Improvement tips**: Better prompting, higher quality source documents

### 2. **Answer Relevancy**  
- **What it measures**: How well the answer addresses the original query
- **Scale**: 0.0 (irrelevant) to 1.0 (perfectly relevant)
- **Good score**: ≥ 0.8  
- **Improvement tips**: Better query understanding, improved answer generation

### 3. **Context Precision**
- **What it measures**: Relevance of retrieved context chunks to the query
- **Scale**: 0.0 (irrelevant context) to 1.0 (highly relevant context)
- **Good score**: ≥ 0.7
- **Improvement tips**: Better embedding models, improved retrieval parameters

### 4. **Context Recall** 
- **What it measures**: How well the context covers information needed to answer the query
- **Scale**: 0.0 (missing information) to 1.0 (complete information)
- **Good score**: ≥ 0.7
- **Improvement tips**: Increase top-k, better chunking strategy

## 🚀 Usage

### Command Line Evaluation

```bash
# Navigate to backend directory
cd backend

# Install RAGAS dependencies (optional)
pip install -r evaluation/requirements_ragas.txt

# Run RAGAS evaluation
python evaluation/ragas_evaluation.py \
  --dataset evaluation/dataset_ragas.json \
  --user-id YOUR_FIREBASE_UID \
  --top-k 5 \
  --output results_ragas.json
```

### API Endpoints

#### Quick Evaluation
```bash
POST /evaluation/ragas/quick
{
  "queries": [
    {
      "query": "What is artificial intelligence?",
      "expected_document_ids": ["doc-uuid-1"],
      "reference_answer": "AI is..." 
    }
  ],
  "top_k": 5
}
```

#### Health Check
```bash
GET /evaluation/ragas/health
```

#### Get Results
```bash  
GET /evaluation/ragas/{evaluation_id}
```

### Frontend Integration

The RAGAS evaluation UI is available at `/evaluation` (when integrated) and provides:
- Interactive query testing
- Real-time evaluation results
- Visual metrics dashboard
- Export functionality

## 📊 Interpreting Results

### Overall RAGAS Score Interpretation

| Score Range | Rating | Assessment |
|-------------|--------|------------|
| 0.8 - 1.0 | 🥇 Excellent | Production-ready RAG system |
| 0.7 - 0.8 | 🥈 Very Good | High-quality with minor improvements |
| 0.6 - 0.7 | 🥉 Good | Solid system, room for improvement |
| 0.5 - 0.6 | ⚠️ Fair | Functional but needs improvements |
| 0.0 - 0.5 | ❌ Poor | Major improvements required |

### Metric-Specific Recommendations

#### Low Faithfulness (< 0.6)
- Check for hallucinations in generated answers
- Improve prompting to stick to provided context
- Use more conservative generation parameters
- Verify context quality and relevance

#### Low Answer Relevancy (< 0.6)  
- Improve query understanding and processing
- Better instruction tuning for the LLM
- Review prompt templates for answer generation
- Consider query expansion techniques

#### Low Context Precision (< 0.6)
- Tune retrieval parameters (similarity thresholds)
- Experiment with different embedding models
- Improve query preprocessing and expansion
- Consider hybrid search (keyword + semantic)

#### Low Context Recall (< 0.6)
- Increase retrieval top-k parameter
- Improve document chunking strategy
- Consider overlapping chunks
- Review document preprocessing pipeline

## 🔧 Configuration

### Environment Variables
```bash
# Required for LLM-based evaluation
GROQ_API_KEY=your_groq_api_key

# RAG Pipeline Configuration  
EMBEDDING_PROVIDER=cloudflare
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

### Dataset Format
```json
[
  {
    "question": "What is machine learning?",
    "expected_document_ids": ["doc-uuid-1", "doc-uuid-2"],
    "reference_answer": "Machine learning is..." 
  }
]
```

### Evaluation Parameters
```python
# Adjust in ragas_evaluation.py
TOP_K = 5                    # Number of retrieved documents  
EVALUATION_MODEL = "llama3-70b-8192"  # LLM for evaluation
TEMPERATURE = 0.1            # Low temperature for consistent evaluation
```

## 📈 Performance Optimization

### Faster Evaluation
- Use smaller evaluation model (trade accuracy for speed)
- Reduce number of test queries for quick feedback
- Implement caching for repeated evaluations
- Run evaluations asynchronously

### Higher Quality Evaluation  
- Use larger, more capable evaluation models
- Increase number and diversity of test queries
- Include domain-specific evaluation criteria
- Combine with human evaluation for validation

## 🔍 Troubleshooting

### Common Issues

#### "RAGAS evaluation not available"
- Install dependencies: `pip install -r evaluation/requirements_ragas.txt`
- Check GROQ_API_KEY environment variable
- Verify LLM API connectivity

#### "No valid evaluation results"
- Ensure user has uploaded documents
- Check document processing status
- Verify user_id matches document ownership  

#### Low evaluation scores across all metrics
- Check document quality and relevance
- Verify embedding model performance
- Review LLM generation quality
- Validate test queries are appropriate

#### Evaluation timeout or errors
- Reduce batch size or number of queries
- Check API rate limits (especially LLM API)
- Verify network connectivity to all services

## 🎓 Best Practices

### Evaluation Design
1. **Diverse Test Set**: Include varied query types and difficulties
2. **Representative Queries**: Use real user questions when possible  
3. **Regular Evaluation**: Monitor metrics over time as system evolves
4. **Baseline Comparison**: Track improvements against previous versions

### Metric Interpretation
1. **Holistic View**: Consider all four metrics together
2. **Context Matters**: Interpret scores relative to your domain
3. **Trend Analysis**: Focus on improvement trends over absolute scores  
4. **Human Validation**: Correlate automated metrics with human evaluation

### System Improvement
1. **Iterative Approach**: Make incremental improvements based on metrics
2. **Component-wise**: Focus on retrieval OR generation based on weak metrics
3. **Data Quality**: High-quality source documents are fundamental
4. **User Feedback**: Combine RAGAS with user satisfaction metrics

## 📚 References

- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAG evaluation best practices](https://docs.ragas.io/)

## 🤝 Contributing

To improve the RAGAS evaluation system:

1. **Add new metrics**: Implement additional evaluation criteria
2. **Optimize performance**: Improve evaluation speed and accuracy  
3. **Enhance UI**: Better visualization and user experience
4. **Domain adaptation**: Customize for specific use cases