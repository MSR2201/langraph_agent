from langgraph.graph import StateGraph
from typing import List, Dict, Any, TypedDict, Optional
import logging
import os
from pydantic import BaseModel, Field
from langchain_together import ChatTogether
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment variable
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.warning("TOGETHER_API_KEY not found in environment variables. Using default value.")
    TOGETHER_API_KEY = "your_api_key_here"  # Replace with your key if not using env vars
else:
    logger.info("Successfully loaded TOGETHER_API_KEY from environment variables")

# Initialize LLM
llm = ChatTogether(
    together_api_key=TOGETHER_API_KEY,
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.3,
)

# Initialize embeddings
embeddings = TogetherEmbeddings(
    together_api_key=TOGETHER_API_KEY,
    model="BAAI/bge-large-en-v1.5",
)

# Load or create vector store
os.makedirs("faiss_index", exist_ok=True)  # Ensure directory exists

try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("Loaded existing FAISS index")
except Exception as e:
    logger.warning(f"Could not load FAISS index: {str(e)}. Will create new index when documents are added.")
    
    # We need at least one document to initialize FAISS properly
    dummy_text = "This is a placeholder document to initialize the vector store."
    dummy_doc = Document(page_content=dummy_text, metadata={"source": "initialization"})
    
    # Create vector store with the dummy document
    vector_store = FAISS.from_documents([dummy_doc], embeddings)
    vector_store.save_local("faiss_index")
    logger.info("Created new FAISS index with placeholder document")

# Define state for the workflow
class AgentState(BaseModel):
    query: str
    max_docs: int = 5
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    response: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)

# Define agent functions
def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant documents based on the query."""
    logger.info(f"Retrieving documents for query: {state.query}")
    
    try:
        # Get documents from vector store
        docs = vector_store.similarity_search(state.query, k=state.max_docs)
        
        # Format retrieved documents for state
        retrieved_docs = []
        sources = []
        
        for i, doc in enumerate(docs):
            doc_info = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            retrieved_docs.append(doc_info)
            
            # Add to sources with simplified metadata
            source_info = {
                "id": i,
                "title": doc.metadata.get("title", f"Document {i}"),
                "source": doc.metadata.get("source", "Unknown")
            }
            sources.append(source_info)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return {"retrieved_docs": retrieved_docs, "sources": sources}
    
    except Exception as e:
        logger.error(f"Error in retrieval agent: {str(e)}")
        return {"retrieved_docs": [], "sources": []}

def reasoning_agent(state: AgentState) -> Dict[str, Any]:
    """Process the query and retrieved documents to generate a response."""
    logger.info("Generating response based on retrieved documents")
    
    try:
        # Format context from retrieved documents
        context = ""
        for i, doc in enumerate(state.retrieved_docs):
            context += f"\nDocument {i+1}:\n{doc['content']}\n"
        
        # Create prompt for the model
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.
        
        CONTEXT:
        {context}
        
        USER QUERY:
        {query}
        
        INSTRUCTIONS:
        1. Answer the user's query based ONLY on the provided context.
        2. If the context doesn't contain relevant information to answer the query, say "I don't have enough information to answer this question."
        3. Do not make up or hallucinate information that is not in the context.
        4. Provide a clear, concise, and informative response.
        5. Use bullet points or numbered lists when appropriate.
        
        YOUR RESPONSE:
        """)
        
        # Generate response using LLM
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": state.query})
        
        logger.info("Successfully generated response")
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Error in reasoning agent: {str(e)}")
        return {"response": f"I encountered an error while processing your request: {str(e)}"}

def format_agent(state: AgentState) -> Dict[str, Any]:
    """Format the final response with sources and additional information."""
    logger.info("Formatting final response")
    
    # The response is already formatted by the reasoning agent
    # Just ensuring the structure is consistent with what the API expects
    return {"response": state.response}

# Create the workflow graph
def create_workflow():
    """Create and return the agent workflow."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retriever", retrieval_agent)
    workflow.add_node("reasoner", reasoning_agent)
    workflow.add_node("formatter", format_agent)
    
    # Add edges
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", "formatter")
    
    # Set entry and end points
    workflow.set_entry_point("retriever")
    workflow.set_finish_point("formatter")
    
    return workflow.compile()

# Initialize the workflow
workflow = create_workflow()

# Helper function to add documents to the vector store
def add_documents(docs: List[Document]):
    """Add documents to the vector store."""
    global vector_store
    vector_store.add_documents(docs)
    vector_store.save_local("faiss_index")
    logger.info(f"Added {len(docs)} documents to vector store")

# Function to load documents from a file
def load_documents_from_file(file_path: str) -> List[Document]:
    """Load documents from a file and convert to Document objects."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Extract file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            # Simple text file handling
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Create metadata
            metadata = {
                "source": file_path,
                "title": os.path.basename(file_path)
            }
            
            # Split into smaller chunks (simple approach)
            chunks = []
            max_chunk_size = 1000
            content_parts = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            for i, part in enumerate(content_parts):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = i
                chunks.append(Document(page_content=part, metadata=doc_metadata))
            
            logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
        
        # Add support for more file types here (PDF, CSV, etc.)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading documents from {file_path}: {str(e)}")
        return []

def query_rag_system(query: str, max_docs: int = 5):
    """Query the RAG system and return the response."""
    try:
        # Create state for the workflow
        state = AgentState(
            query=query,
            max_docs=max_docs,
            retrieved_docs=[],
            response="",
            sources=[]
        )
        
        # Execute the workflow
        result = workflow.invoke(state)
        
        # Extract the response and sources from the result
        # The result from LangGraph is an AddableValuesDict, not an AgentState
        response = result.get("response", "No response generated")
        sources = result.get("sources", [])
        
        # Create a simple object to return that has the expected attributes
        class Result:
            def __init__(self, response, sources):
                self.response = response
                self.sources = sources
                
        return Result(response, sources)
    
    except Exception as e:
        logger.error(f"Error querying RAG system: {str(e)}")
        return None

# Main function for CLI testing
def main():
    print("\n=== RAG System Tester ===\n")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    while True:
        # Ask if the user wants to add documents
        add_docs = input("Do you want to add documents? (y/n): ").strip().lower()
        
        if add_docs == 'y':
            # Ask for file location
            file_path = input("Enter the file location: ").strip()
            
            if file_path:
                # Load and process documents
                print(f"Loading documents from {file_path}...")
                docs = load_documents_from_file(file_path)
                
                if docs:
                    # Add documents to vector store
                    print(f"Adding {len(docs)} documents to vector store...")
                    add_documents(docs)
                    print("Documents added successfully!")
                else:
                    print("No documents were loaded.")
        elif add_docs == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    
    print("\nYou can now ask questions about the documents.")
    
    while True:
        # Ask the question
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        print("\nProcessing your query...")
        result = query_rag_system(query)
        
        if result:
            print("\n=== Answer ===\n")
            print(result.response)
            
            if result.sources:
                print("\n=== Sources ===\n")
                for source in result.sources:
                    print(f"- {source['title']} ({source['source']})")
        else:
            print("Error: Could not process your query.")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
