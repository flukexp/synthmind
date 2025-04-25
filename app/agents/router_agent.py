import os
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from enum import Enum
from app.utils.prompts import ROUTER_PROMPT

class ToolType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    UNKNOWN = "unknown"

class RouterInput(BaseModel):
    query: str = Field(description="The user query to route")

class RouterOutput(BaseModel):
    tool: ToolType = Field(description="The tool to use")
    reasoning: str = Field(description="The reasoning for choosing this tool")
    reformulated_query: str = Field(description="The query reformulated for the chosen tool")
    
class RouterAgent:
    def __init__(self):
        OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        
        self.llm = Ollama(
            base_url=OLLAMA_API_BASE_URL,
            model="mistral:7b",  
            temperature=0
        )
        
        # Initialize the prompt template
        self.router_prompt = ROUTER_PROMPT
        
        self.router_chain = LLMChain(
            llm=self.llm,
            prompt=self.router_prompt
        )
        
    def route(self, input_data: RouterInput) -> RouterOutput:
        """Route the query to the appropriate tool."""
        query = input_data.query
        result = self.router_chain.run(query=query)
        
        # Parse the result
        import json
        try:
            parsed_result = json.loads(result)
            tool = parsed_result.get("tool", "unknown")
            # Validate the tool type
            if tool not in [t.value for t in ToolType]:
                tool = "unknown"
                
            return RouterOutput(
                tool=tool,
                reasoning=parsed_result.get("reasoning", "No reasoning provided"),
                reformulated_query=parsed_result.get("reformulated_query", query)
            )
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return RouterOutput(
                tool="unknown",
                reasoning="Failed to parse router response",
                reformulated_query=query
            )