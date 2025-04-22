from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

from app.data.ingestion import DocumentIngestion
from app.tools.qa_tool import QATool, QAToolInput
from app.tools.summary_tool import SummaryTool, SummaryToolInput
from app.agents.router_agent import RouterAgent, RouterInput, ToolType

app = FastAPI(
    title="Internal AI Assistant",
    description="AI assistant for product and engineering teams",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize components
document_ingestion = DocumentIngestion()
vectorstore = document_ingestion.get_or_create_vectorstore()
qa_tool = QATool(vectorstore)
summary_tool = SummaryTool()
router_agent = RouterAgent()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: Dict[str, Any]
    tool_used: str
    reasoning: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the AI assistant."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Route the query
    router_input = RouterInput(query=request.query)
    router_output = router_agent.route(router_input)
    
    # Process with the appropriate tool
    result = None
    if router_output.tool == ToolType.QA:
        qa_input = QAToolInput(query=router_output.reformulated_query)
        result = qa_tool.run(qa_input)
    elif router_output.tool == ToolType.SUMMARY:
        summary_input = SummaryToolInput(issue_text=router_output.reformulated_query)
        result = summary_tool.run(summary_input)
    else:
        result = {"message": "I'm not sure how to process this query. Could you rephrase it?"}
    
    return QueryResponse(
        result=result.dict() if hasattr(result, "dict") else result,
        tool_used=router_output.tool,
        reasoning=router_output.reasoning
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Add document reload endpoint for admin use
@app.post("/admin/reload-documents")
async def reload_documents():
    """Force reload of documents into the vector store."""
    global vectorstore
    vectorstore = document_ingestion.get_or_create_vectorstore(force_reload=True)
    return {"status": "Documents reloaded successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)