import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from enum import Enum

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
        
        # Create a specialized prompt for intelligent routing
        self.prompt_template = """
        You are a query router for an internal AI assistant. Your job is to determine which tool should handle a given query.

        Available tools:
        1. QA Tool ("qa"): Answers factual questions about bugs, features, or user feedback using internal documentation.
        2. Summary Tool ("summary"): Takes issue text and summarizes the reported issues, affected features, and severity.
        3. Unknown Tool ("unknown"): For anything that doesn't fit.

        USER QUERY: {query}

        ### EXAMPLES

        User Query: "What are the issues reported on email notifications?"
        Response:
        {{"tool": "qa", "reasoning": "The user is asking about issues mentioned in the documentation.", "reformulated_query": "List all reported issues related to email notifications."}}

        User Query: "Users say the dashboard doesn't update on mobile."
        Response:
        {{"tool": "summary", "reasoning": "This is an issue report needing summarization of problem and affected component.", "reformulated_query": "Summarize: Users say the dashboard doesn't update on mobile."}}

        User Query: "Hello, can you help me?"
        Response:
        {{"tool": "unknown", "reasoning": "This query doesn't match QA or summarization tasks.", "reformulated_query": "Hello, can you help me?"}}

        INSTRUCTIONS:
        Analyze the query and determine which tool should handle it.
        If the query doesn't match any tool, respond with "unknown".

        Think step-by-step about what the user is asking:
        1. What is the user trying to accomplish?
        2. Which tool best addresses this need?
        3. How should I reformulate the query for the selected tool?

        Respond with JSON containing:
        - tool: "qa", "summary", or "unknown"
        - reasoning: Brief explanation of why you chose this tool
        - reformulated_query: The query reformulated for the chosen tool

        RESPONSE:
        """

        
        self.router_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["query"]
        )
        
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