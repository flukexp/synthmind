from langchain.prompts import PromptTemplate

# QA Tool Prompts
QA_PROMPT_TEMPLATE = """
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

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["question", "context"]
)

# Summary Tool Prompts
SUMMARY_PROMPT_TEMPLATE = """
You are a technical analyst responsible for categorizing and summarizing product issues.

ISSUE TEXT:
{issue_text}

INSTRUCTIONS:
Analyze the issue text and provide a structured summary with the following components:
1. Extract all distinct reported issues
2. Identify all affected features or components
3. Assess the severity (Critical, High, Medium, Low)

Format your response as JSON with these keys:
- reported_issues: [list of issues]
- affected_components: [list of components]
- severity: "severity level"

SUMMARY:
"""

SUMMARY_PROMPT = PromptTemplate(
    template=SUMMARY_PROMPT_TEMPLATE,
    input_variables=["issue_text"]
)

# Router Agent Prompts
ROUTER_PROMPT_TEMPLATE = """
You are a query router for an internal AI assistant. Your job is to determine which tool should handle a given query.

Available tools:
1. QA Tool (qa): Answers questions by searching internal documents. Use for factual questions about bugs, features, and user feedback.
2. Summary Tool (summary): Summarizes issue texts with reported issues, affected components, and severity. Use when an issue needs summarization.

USER QUERY: {query}

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

ROUTER_PROMPT = PromptTemplate(
    template=ROUTER_PROMPT_TEMPLATE,
    input_variables=["query"]
)