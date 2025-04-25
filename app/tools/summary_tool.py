import os
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from app.utils.prompts import SUMMARY_PROMPT

class SummaryToolInput(BaseModel):
    issue_text: str = Field(description="The issue text to summarize")

class SummaryToolOutput(BaseModel):
    reported_issues: list = Field(description="List of reported issues")
    affected_components: list = Field(description="List of affected features/components")
    severity: str = Field(description="Severity assessment of the issue")
    
class SummaryTool:
    def __init__(self):
        OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        
        self.llm = Ollama(
            base_url=OLLAMA_API_BASE_URL,
            model="mistral:7b",  
            temperature=0
        )
        
        # Initialize prompt for issue summarization
        self.summary_prompt = SUMMARY_PROMPT
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt
        )
        
    def run(self, input_data: SummaryToolInput) -> SummaryToolOutput:
        """Run the summary tool on the given input."""
        issue_text = input_data.issue_text
        result = self.summary_chain.run(issue_text=issue_text)
        
        # The LLM should return JSON-formatted text, parse it
        # For robustness, adding error handling
        import json
        try:
            summary_data = json.loads(result)
            return SummaryToolOutput(
                reported_issues=summary_data["reported_issues"],
                affected_components=summary_data["affected_components"],
                severity=summary_data["severity"]
            )
        except json.JSONDecodeError:
            # Fallback response if JSON parsing fails
            return SummaryToolOutput(
                reported_issues=["Error parsing summary"],
                affected_components=["Unknown"],
                severity="Unknown"
            )