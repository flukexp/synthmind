# Internal AI Assistant

This project provides an AI-powered assistant designed to help product and engineering teams efficiently extract actionable insights from internal documents and team reports. By leveraging advanced natural language processing (NLP) techniques, the assistant offers seamless querying, summarization, and intelligent query routing.

## Features

- **Query Answering**: Efficiently search through internal documents to retrieve answers to specific queries.
- **Issue Summarization**: Automatically summarize reported issues, detailing affected components and their severity.
- **Intelligent Query Routing**: Automatically route queries to the appropriate tool for precise processing.

## Getting Started

### Prerequisites

- Python 3.8+ (or higher)
- Docker (optional, for containerized setup)
- **Ollama** (Required to run Mistral:7b model)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/flukexp/synthmind.git
   cd synthmind
   ```

2. **Install Ollama**:
   You need to have **Ollama** installed to run the Mistral:7b model. You can download and install Ollama by following these instructions:

   - **For macOS**:
     - Download the [Ollama installer](https://ollama.com) and follow the installation instructions.
   
   - **For Windows**:
     - Visit the [Ollama website](https://ollama.com) and download the installer for Windows.
   
   - **For Linux**:
     - You can install Ollama by following the instructions provided on their [official site](https://ollama.com).

3. **Install Mistral:7b model**:
   Once Ollama is installed, you need to download the **Mistral:7b** model. You can do so by running the following command:

   ```bash
   ollama pull mistral:7b
   ```

   This will download and prepare the model for use in your project.

4. **Run with Docker**:
   ```bash
   docker compose up --build
   ```

   Alternatively, **run locally** without Docker:

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   .\venv\Scripts\activate  # For Windows
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000 & open index.html
   
   ```

5. **Access the API** at [http://localhost:8000](http://localhost:8000).

6. **Access the Frontend**:
   - If running with Docker, the frontend will be served at [http://localhost:8080](http://localhost:8080).
   - If running locally, the frontend will automatically open in your browser.

## API Documentation

Once the application is up and running, you can access the Swagger documentation for the API at [http://localhost:8000/docs](http://localhost:8000/docs). You can also test the API via the frontend interface.

### API Endpoints

- **`POST /query`**: Processes user queries.
- **`GET /health`**: Checks the health status of the application.
- **`POST /admin/reload-documents`**: Reloads the document embeddings for updating internal content.

## Project Structure

- **`app/`**: Contains the core application logic.
- **`agents/`**: Implementations for various AI agents responsible for query processing and routing.
- **`tools/`**: Implementations for different tools used by the assistant.
- **`data/`**: Handles data processing and document storage.
- **`utils/`**: Contains utility functions to support the application.
- **`tests/`**: Unit and integration tests to ensure application reliability and correctness.
- **`Dockerfile`, `Dockerfile-frontend`, `docker-compose.yml`**: Docker configuration files to facilitate containerized deployment.

## Design Decisions

- **Modular Architecture**: The project is designed with a modular approach to separate concerns and enhance maintainability and scalability.
- **Vector Search**: FAISS (Facebook AI Similarity Search) is used for efficient semantic search, improving the accuracy of query results.
- **Agent-Based Query Routing**: Queries are intelligently routed to the appropriate tool or agent for accurate and efficient processing.
- **Optimized API Usage**: Efficient prompt design tailored for the Mistral:7b:Q4_0 model on Ollama, minimizing costs and optimizing performance.

## Tools & Technologies

- **Model**: Mistral:7b:Q4_0 is used for AI-powered query processing. The model is integrated via Ollama setup.
- **Vector Store**: FAISS is used as the vector store for storing and retrieving document embeddings.
- **Embeddings**: The HuggingFace embedding model `sentence-transformers/all-MiniLM-L6-v2` is used to generate embeddings for the document contents.

## Testing

The project includes a set of unit and integration tests to ensure the proper functionality of the various components. The tests can be found in the `tests/` folder and include the following:

- **`test_qa_tool.py`**: Unit tests for the query answering tool.
- **`test_router_agent.py`**: Tests for routing queries to the appropriate agent or tool.
- **`test_summary_tool.py`**: Unit tests for the issue summarization tool.

## 📽️ Demo Videos

Watch how to run the AI Assistant in two ways:

### ▶️ Run Locally

[![Local Setup Video](https://img.youtube.com/vi/6qplU-bP-dg/0.jpg)](https://www.youtube.com/watch?v=6qplU-bP-dg)

### 🐳 Run with Docker

[![Docker Setup Video](https://img.youtube.com/vi/Ai5NxZsfX7U/0.jpg)](https://youtu.be/Ai5NxZsfX7U)
