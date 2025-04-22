import unittest
from unittest.mock import MagicMock, patch
from app.agents.router_agent import RouterAgent, RouterInput, RouterOutput, ToolType

class TestRouterAgent(unittest.TestCase):

    @patch("app.agents.router_agent.Ollama")
    @patch("app.agents.router_agent.LLMChain")
    def setUp(self, mock_llmchain_class, mock_ollama_class):
        # Mock LLMChain and Ollama
        self.mock_llm = MagicMock()
        self.mock_chain = MagicMock()
        mock_llmchain_class.return_value = self.mock_chain
        mock_ollama_class.return_value = self.mock_llm

        # Instantiate router agent
        self.router_agent = RouterAgent()

    def test_route_qa_tool(self):
        mock_response = """
        {
            "tool": "qa",
            "reasoning": "The user is asking for issue details available in documentation.",
            "reformulated_query": "List all known issues related to login flow."
        }
        """
        self.mock_chain.run.return_value = mock_response

        input_data = RouterInput(query="What are the known login issues?")
        output = self.router_agent.route(input_data)

        self.assertIsInstance(output, RouterOutput)
        self.assertEqual(output.tool, ToolType.QA)
        self.assertIn("documentation", output.reasoning)

    def test_route_summary_tool(self):
        mock_response = """
        {
            "tool": "summary",
            "reasoning": "This appears to be a user-submitted issue report.",
            "reformulated_query": "Summarize: The dashboard fails to load on Safari."
        }
        """
        self.mock_chain.run.return_value = mock_response

        input_data = RouterInput(query="The dashboard fails to load on Safari.")
        output = self.router_agent.route(input_data)

        self.assertEqual(output.tool, ToolType.SUMMARY)
        self.assertIn("user-submitted", output.reasoning)

    def test_route_unknown_tool(self):
        mock_response = """
        {
            "tool": "unknown",
            "reasoning": "The query is not actionable for QA or Summary.",
            "reformulated_query": "Hi, are you there?"
        }
        """
        self.mock_chain.run.return_value = mock_response

        input_data = RouterInput(query="Hi, are you there?")
        output = self.router_agent.route(input_data)

        self.assertEqual(output.tool, ToolType.UNKNOWN)
        self.assertIn("not actionable", output.reasoning)

    def test_route_invalid_json(self):
        self.mock_chain.run.return_value = "this is not JSON"

        input_data = RouterInput(query="Something strange")
        output = self.router_agent.route(input_data)

        self.assertEqual(output.tool, ToolType.UNKNOWN)
        self.assertEqual(output.reasoning, "Failed to parse router response")
        self.assertEqual(output.reformulated_query, "Something strange")

    def test_route_invalid_tool_enum(self):
        # Simulates a tool value not in ToolType
        mock_response = """
        {
            "tool": "invalid_tool",
            "reasoning": "Should default to unknown",
            "reformulated_query": "Check this query"
        }
        """
        self.mock_chain.run.return_value = mock_response

        input_data = RouterInput(query="What tool is this?")
        output = self.router_agent.route(input_data)

        self.assertEqual(output.tool, ToolType.UNKNOWN)
        self.assertEqual(output.reformulated_query, "Check this query")

if __name__ == "__main__":
    unittest.main()
