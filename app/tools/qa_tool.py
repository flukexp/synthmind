import os
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class QAToolInput(BaseModel):
    query: str = Field(description="The question to answer about internal documents")

class QAToolOutput(BaseModel):
    answer: str = Field(description="The answer to the question")
    source_documents: list = Field(description="The source documents used to answer")
    
class QATool:
    def __init__(self, vectorstore):
        OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        
        self.vectorstore = vectorstore
        self.llm = Ollama(
            base_url=OLLAMA_API_BASE_URL,
            model="mistral:7b",  
            temperature=0
        )
        
        # Craft a specialized prompt template for accurate and context-aware responses
        self.prompt_template = """
        You are an AI assistant for a product and engineering team.
        Your task is to answer questions accurately based on the internal documents provided.
        
        Question: {question}
        
        Context information from relevant documents:
        {context}
        
        Instructions:
        1. Answer based ONLY on the context provided
        2. If the answer is not in the context, say "I don't have enough information to answer this"
        3. Be concise and direct in your answer
        4. Format the answer in a structured way if appropriate
        
        Answer:
        """
        
        self.qa_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["question", "context"]
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve 3 most relevant documents
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
    def run(self, input_data: QAToolInput) -> QAToolOutput:
        """Run the QA tool on the given input."""
        query = input_data.query
        result = self.qa_chain({"query": query})
        
        # Extract source document information
        source_docs = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_docs.append({
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "source": doc.metadata.get("source", "Unknown")
                })
                
        return QAToolOutput(
            answer=result["result"],
            source_documents=source_docs
        )