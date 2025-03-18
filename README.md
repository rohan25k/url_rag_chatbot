# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that processes URLs and allows users to ask questions about the content.

## Features

- Process multiple URLs to extract and clean content
- Build a in-memory vector store for efficient similarity search
- Interactive chat interface for asking questions
- Context-aware responses using a RAG architecture
- "No relevant information" response when no appropriate context found

## Prerequisites

- Miniconda/Anaconda installed locally to setup the environment (see [Miniconda documentation](https://www.anaconda.com/docs/getting-started/miniconda/install) for installation instructions)
- Python 3.10+
- Ollama installed locally with the `deepseek-r1:1.5b` model (see [Ollama documentation](https://github.com/ollama/ollama) for installation instructions)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rohan25k/url_rag_chatbot.git
   cd url_rag_chatbot
   ```

2. Create a virtual environment:
   ```bash
   conda create -n my_env python=3.10
   conda activate my_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Ollama is running with the required model:
   ```bash
   ollama pull deepseek-r1:1.5b
   ollama serve  # If not already running as a service
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser to the URL shown in the terminal (typically http://localhost:8501)

3. Enter URLs to process in the text area (comma-separated)

4. Click "Process URLs" and wait for the processing to complete

5. Ask questions about the content in the chat interface

## Project Structure

- `app.py`: Main Streamlit application
- `src/data_processing.py`: Functions for loading and processing URL content
- `src/rag_model.py`: RAG model implementation
- `src/utils.py`: Utility functions

## How It Works

1. **URL Processing**:
   - The application fetches content from specified URLs
   - Content is cleaned to remove unnecessary HTML elements
   - Text is split into manageable chunks
   - Chunks are embedded and stored in a vector database

2. **Query Processing**:
   - User questions are embedded and compared to stored chunks
   - Most relevant chunks are retrieved
   - LLM generates an answer based on the retrieved context
   - If no relevant information is found, a specific message is shown

## License

MIT