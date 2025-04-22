import unittest
from unittest.mock import MagicMock, patch
from app.tools.summary_tool import SummaryTool, SummaryToolInput, SummaryToolOutput

class TestSummaryTool(unittest.TestCase):

    @patch("app.tools.summary_tool.Ollama")
    @patch("app.tools.summary_tool.LLMChain")
    def setUp(self, mock_llmchain_class, mock_ollama_class):
        # Mock the LLM and LLMChain
        self.mock_llm = MagicMock()
        self.mock_chain = MagicMock()
        mock_llmchain_class.return_value = self.mock_chain
        mock_ollama_class.return_value = self.mock_llm

        # Initialize the tool
        self.summary_tool = SummaryTool()

    def test_run_valid_json_output(self):
        mock_result = """
        {
            "reported_issues": ["Login fails on mobile", "Unexpected logout"],
            "affected_components": ["Authentication", "Session Management"],
            "severity": "High"
        }
        """
        self.mock_chain.run.return_value = mock_result

        input_data = SummaryToolInput(issue_text="Users can't login on mobile and they get logged out unexpectedly.")
        output = self.summary_tool.run(input_data)

        self.assertIsInstance(output, SummaryToolOutput)
        self.assertEqual(output.severity, "High")
        self.assertIn("Login fails on mobile", output.reported_issues)
        self.assertIn("Authentication", output.affected_components)

    def test_run_invalid_json_output(self):
        # Simulate LLM returning bad JSON
        self.mock_chain.run.return_value = "This is not JSON"

        input_data = SummaryToolInput(issue_text="Something is wrong.")
        output = self.summary_tool.run(input_data)

        self.assertIsInstance(output, SummaryToolOutput)
        self.assertEqual(output.severity, "Unknown")
        self.assertEqual(output.reported_issues, ["Error parsing summary"])
        self.assertEqual(output.affected_components, ["Unknown"])

if __name__ == "__main__":
    unittest.main()
