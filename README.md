# RAG Multi-Agent System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) multi-agent system using FastAPI, LangGraph, and Together AI. The system allows you to upload documents and query them using an intelligent agent that retrieves and reasons over the document contents.

## Features
- Multi-agent workflow for document retrieval and question answering
- FastAPI backend for easy API interactions
- Support for document ingestion and querying
- Flexible document loading (currently supports .txt files)
- Logging and error handling

## Prerequisites
- Python 3.12
- Docker (optional, but recommended)
- Together AI API Key

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/langraph_agent.git
cd langraph_agent
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```
TOGETHER_API_KEY=your_together_api_key_here
```

## Running the Application

### Option 1: Local Development
```bash
# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Interacting with the API

### 1. CLI Testing
Run the CLI interface:
```bash
python agents.py
```
- Choose to add documents first
- Then ask questions about the uploaded documents

### 2. Postman / API Testing

#### Health Check Endpoint
- **GET** `http://localhost:8000/`
- **GET** `http://localhost:8000/health`

#### Document Upload Endpoint
- **POST** `http://localhost:8000/upload`
- Content-Type: `multipart/form-data`
- Key: `file` (File)
- Value: Select your .txt file
- **Response**: 
  ```json
  {
    "message": "Successfully uploaded and processed filename.txt",
    "document_chunks": 5
  }
  ```

#### Query Endpoint
- **POST** `http://localhost:8000/ask`
- Content-Type: `application/json`
- Body:
```json
{
    "query": "What is the main topic of the document?",
}
```

## Supported Document Types
Currently supports:
- `.txt` files


## Document Ingestion Tips
- Place documents in the `data/` directory
- Supports text files with automatic chunking
- Large documents are split into manageable chunks

## Customization
- Modify `agents.py` to change LLM, embedding models
- Adjust prompts in the `reasoning_agent` function
- Configure logging levels in the logging setup

## Troubleshooting
- Ensure `TOGETHER_API_KEY` is correctly set
- Check `faiss_index/` and `data/` directories exist
- Review logs for detailed error information

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Example Workflow
1. Upload a document using the `/upload` endpoint
2. Query the uploaded document using the `/ask` endpoint
3. Receive AI-generated responses with source references

## API Workflow Example
```bash
# Upload a document
curl -X POST -F "file=@/path/to/your/document.txt" http://localhost:8000/upload

# Ask a question about the uploaded document
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic?", "max_docs": 5}'
```

## Deployed API Access
You can interact with the RAG Agent API using the deployed endpoint at: https://langraph-agent.onrender.com/

### API Endpoints
- **Health Check**: `GET` https://langraph-agent.onrender.com/health
- **Document Upload**: `POST` https://langraph-agent.onrender.com/upload
- **Query Endpoint**: `POST` https://langraph-agent.onrender.com/ask

### Example Curl Commands for Deployed API
```bash
# Health Check
curl https://langraph-agent.onrender.com/health

# Upload a document
curl -X POST -F "file=@/path/to/your/document.txt" https://langraph-agent.onrender.com/upload

# Ask a question about the uploaded document
curl -X POST https://langraph-agent.onrender.com/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the CNICA ?"}'
```

I have uploaded a document which is available for testing purposes you can upload any txt file and test it with new files.

### Using with Postman or API Clients
1. Base URL: `https://langraph-agent.onrender.com/`
2. Follow the same endpoint patterns as local development
3. Ensure your API client supports multipart form-data for document uploads


