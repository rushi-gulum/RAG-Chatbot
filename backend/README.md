# RAG Chatbot Backend

A FastAPI-based backend API for the Agentic RAG Chatbot.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

- `GET /` - Health check and welcome message
- `GET /health` - Service health status  
- `POST /chat` - Chat endpoint for processing user messages

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

Copy `.env.example` to `.env` and configure your environment variables.