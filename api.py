from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Optional, Dict, Any
import os
import shutil

# Import the workflow and document-related functions from agents.py
from agents import workflow, AgentState, add_documents, load_documents_from_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent API", description="API for interacting with a multi-agent RAG system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_docs: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Agent API. Use /ask endpoint to query the system."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Generate a unique filename
        file_path = os.path.join("data", file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded file: {file.filename}")
        
        # Load documents from the uploaded file
        docs = load_documents_from_file(file_path)
        
        if docs:
            # Add documents to vector store
            add_documents(docs)
            
            return {
                "message": f"Successfully uploaded and processed {file.filename}",
                "document_chunks": len(docs)
            }
        else:
            # If no documents could be loaded, remove the file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Could not process the uploaded file")
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        
        # Create initial state for the workflow
        state = AgentState(
            query=request.query,
            max_docs=request.max_docs,
            retrieved_docs=[],
            response=""
        )
        
        # Execute the agent workflow
        result = workflow.invoke(state)
        
        # Extract the response from the result
        response = result.get("response", "No response generated")
        
        logger.info(f"Generated response for query: {request.query}")
        
        return {
            "response": response
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
