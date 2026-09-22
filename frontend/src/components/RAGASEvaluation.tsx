/**
 * RAGAS Evaluation Component
 * 
 * Provides UI for running and displaying RAGAS evaluation results
 */

"use client";

import React, { useState, useEffect } from 'react';
import { User } from 'firebase/auth';
import { 
  BarChart3, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Loader2, 
  Play,
  Download,
  TrendingUp,
  Target,
  Search,
  FileText
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface RAGASMetrics {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  overall_score: number;
}

interface RAGASResult {
  evaluation_id: string;
  timestamp: string;
  query_count: number;
  avg_faithfulness: number;
  avg_answer_relevancy: number;
  avg_context_precision: number;
  avg_context_recall: number;
  overall_ragas_score: number;
  individual_results: Array<{
    query: string;
    answer: string;
    faithfulness: number;
    answer_relevancy: number;
    context_precision: number;
    context_recall: number;
    overall_score: number;
    context_count: number;
  }>;
}

interface Props {
  user: User | null;
}

const RAGASEvaluation: React.FC<Props> = ({ user }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<RAGASResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [testQueries, setTestQueries] = useState<string[]>([
    "What is artificial intelligence?",
    "How does machine learning work?", 
    "What are neural networks?",
    "Explain deep learning",
    "What is natural language processing?"
  ]);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.8) return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (score >= 0.6) return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <XCircle className="w-5 h-5 text-red-600" />;
  };

  const getScoreDescription = (score: number) => {
    if (score >= 0.8) return "Excellent";
    if (score >= 0.6) return "Good";
    if (score >= 0.4) return "Fair";
    return "Poor";
  };

  const runEvaluation = async () => {
    if (!user || testQueries.length === 0) return;

    setIsRunning(true);
    setError(null);

    try {
      const token = await user.getIdToken();
      
      const response = await fetch(`${API_URL}/evaluation/ragas/quick`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          queries: testQueries.map(q => ({ query: q })),
          top_k: 5
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Evaluation failed');
      }

      const data = await response.json();
      setResult(data);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsRunning(false);
    }
  };

  const addQuery = () => {
    setTestQueries([...testQueries, ""]);
  };

  const updateQuery = (index: number, value: string) => {
    const newQueries = [...testQueries];
    newQueries[index] = value;
    setTestQueries(newQueries);
  };

  const removeQuery = (index: number) => {
    setTestQueries(testQueries.filter((_, i) => i !== index));
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center gap-3 mb-4">
          <BarChart3 className="w-8 h-8 text-blue-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">RAGAS Evaluation</h1>
            <p className="text-gray-600">Retrieval-Augmented Generation Assessment</p>
          </div>
        </div>
        
        <p className="text-gray-700 leading-relaxed">
          RAGAS provides reference-free evaluation of your RAG system using advanced LLM-based metrics:
          <strong> Faithfulness</strong> (factual consistency), 
          <strong> Answer Relevancy</strong> (query alignment),
          <strong> Context Precision</strong> (retrieval quality), and
          <strong> Context Recall</strong> (information coverage).
        </p>
      </div>

      {/* Test Queries Configuration */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <FileText className="w-5 h-5" />
          Test Queries
        </h2>
        
        <div className="space-y-3">
          {testQueries.map((query, index) => (
            <div key={index} className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => updateQuery(index, e.target.value)}
                placeholder="Enter a test query..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={() => removeQuery(index)}
                className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-md"
              >
                ×
              </button>
            </div>
          ))}
        </div>
        
        <div className="flex gap-3 mt-4">
          <button
            onClick={addQuery}
            className="px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-md border border-blue-200"
          >
            + Add Query
          </button>
          
          <button
            onClick={runEvaluation}
            disabled={isRunning || testQueries.length === 0 || !user}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isRunning ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {isRunning ? 'Running Evaluation...' : 'Run RAGAS Evaluation'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-800">
            <XCircle className="w-5 h-5" />
            <span className="font-medium">Evaluation Failed</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Overall Metrics */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Overall RAGAS Metrics
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Faithfulness */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Faithfulness</span>
                  {getScoreIcon(result.avg_faithfulness)}
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {result.avg_faithfulness.toFixed(3)}
                </div>
                <div className={`text-sm ${getScoreColor(result.avg_faithfulness)}`}>
                  {getScoreDescription(result.avg_faithfulness)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Factual consistency
                </div>
              </div>

              {/* Answer Relevancy */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Answer Relevancy</span>
                  {getScoreIcon(result.avg_answer_relevancy)}
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {result.avg_answer_relevancy.toFixed(3)}
                </div>
                <div className={`text-sm ${getScoreColor(result.avg_answer_relevancy)}`}>
                  {getScoreDescription(result.avg_answer_relevancy)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Query alignment
                </div>
              </div>

              {/* Context Precision */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Context Precision</span>
                  {getScoreIcon(result.avg_context_precision)}
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {result.avg_context_precision.toFixed(3)}
                </div>
                <div className={`text-sm ${getScoreColor(result.avg_context_precision)}`}>
                  {getScoreDescription(result.avg_context_precision)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Retrieval quality
                </div>
              </div>

              {/* Context Recall */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Context Recall</span>
                  {getScoreIcon(result.avg_context_recall)}
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {result.avg_context_recall.toFixed(3)}
                </div>
                <div className={`text-sm ${getScoreColor(result.avg_context_recall)}`}>
                  {getScoreDescription(result.avg_context_recall)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Information coverage
                </div>
              </div>
            </div>

            {/* Overall Score */}
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Overall RAGAS Score</h3>
                  <p className="text-sm text-gray-600">Average of all four metrics</p>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold text-gray-900">
                    {result.overall_ragas_score.toFixed(3)}
                  </div>
                  <div className={`text-sm font-medium ${getScoreColor(result.overall_ragas_score)}`}>
                    {getScoreDescription(result.overall_ragas_score)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Individual Query Results */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Search className="w-5 h-5" />
              Individual Query Results
            </h2>
            
            <div className="space-y-4">
              {result.individual_results.map((queryResult, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="mb-3">
                    <h4 className="font-medium text-gray-900 mb-1">
                      Query {index + 1}: {queryResult.query}
                    </h4>
                    <div className="text-sm text-gray-600">
                      Sources: {queryResult.context_count} | Overall: {queryResult.overall_score.toFixed(3)}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div className="text-center">
                      <div className="font-medium text-gray-700">Faithfulness</div>
                      <div className={`font-bold ${getScoreColor(queryResult.faithfulness)}`}>
                        {queryResult.faithfulness.toFixed(3)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-gray-700">Relevancy</div>
                      <div className={`font-bold ${getScoreColor(queryResult.answer_relevancy)}`}>
                        {queryResult.answer_relevancy.toFixed(3)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-gray-700">Precision</div>
                      <div className={`font-bold ${getScoreColor(queryResult.context_precision)}`}>
                        {queryResult.context_precision.toFixed(3)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-gray-700">Recall</div>
                      <div className={`font-bold ${getScoreColor(queryResult.context_recall)}`}>
                        {queryResult.context_recall.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Export Results */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Download className="w-5 h-5" />
              Export Results
            </h2>
            
            <button
              onClick={() => {
                const dataStr = JSON.stringify(result, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `ragas-evaluation-${result.evaluation_id}.json`;
                link.click();
              }}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download Results (JSON)
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGASEvaluation;