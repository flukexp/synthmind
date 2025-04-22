import unittest
from unittest.mock import MagicMock, patch
from app.tools.qa_tool import QATool, QAToolInput, QAToolOutput

class MockRetriever:
    def __call__(self, *args, **kwargs):
        return []

    def get_relevant_documents(self, query):
        return []

class TestQATool(unittest.TestCase):
    @patch("app.tools.qa_tool.Ollama")
    @patch("app.tools.qa_tool.RetrievalQA")
    def setUp(self, mock_retrievalqa_class, mock_ollama_class):
        # Mock the vectorstore and its retriever
        self.mock_vectorstore = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever

        # Mock Ollama LLM and RetrievalQA
        self.mock_qa_chain = MagicMock()
        mock_retrievalqa_class.from_chain_type.return_value = self.mock_qa_chain
        mock_ollama_class.return_value = MagicMock()

        # Initialize the tool
        self.qa_tool = QATool(self.mock_vectorstore)

    def test_run_returns_expected_output(self):
        # Prepare mock return from qa_chain
        self.mock_qa_chain.return_value = {
            "result": "This is a test answer.",
            "source_documents": [
                MagicMock(page_content="This is the content of the doc.", metadata={"source": "Doc1"}),
                MagicMock(page_content="Another doc content.", metadata={"source": "Doc2"})
            ]
        }

        input_data = QAToolInput(query="What is the system design?")
        output = self.qa_tool.run(input_data)

        # Assertions
        self.assertIsInstance(output, QAToolOutput)
        self.assertEqual(output.answer, "This is a test answer.")
        self.assertEqual(len(output.source_documents), 2)
        self.assertIn("content", output.source_documents[0])
        self.assertIn("source", output.source_documents[0])
        self.assertEqual(output.source_documents[0]["source"], "Doc1")
        self.assertTrue(output.source_documents[0]["content"].startswith("This is the content"))

if __name__ == '__main__':
    unittest.main()
