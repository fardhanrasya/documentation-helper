# LangChain Documentation Web Application

This application provides a web interface for searching and interacting with LangChain documentation.

## Overview

The application consists of:
- A document ingestion pipeline for loading LangChain documentation 
- A vector store for semantic search and retrieval
- A chat interface for interacting with the documentation using LLMs
- Utilities for managing the document collection

## Getting Started

### Prerequisites
- Python 3.8+ 
- Pipenv for dependency management

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pipenv install
```

### Configuration

Configure the following environment variables:
- Required LLM API keys (details in `.env.example`)
- Vector store settings
- Other service credentials as needed

## Usage

### Document Ingestion

The `ingestion.py` script handles loading documentation into the vector store:

```bash
python ingestion.py
```

This will:
- Crawl and process the documentation files
- Split documents into chunks
- Generate embeddings
- Store in the vector database

### Running the Web Interface

Start the web application:

```bash
python main.py
```

This will launch the interface where you can:
- Search documentation
- Ask questions about LangChain
- Get context-aware responses

### Tools

- `list_models.py` - Utility for listing available models
- `backend/core.py` - Core chat and retrieval functionality

## Project Structure

- `.editorconfig` - Editor configuration
- `.gitignore` - Git ignore rules
- `Pipfile` - Python dependencies
- `ingestion.py` - Documentation ingestion pipeline
- `list_models.py` - Model listing utility 
- `main.py` - Main web application
- `backend/` - Backend logic and utilities
- `langchain-docs/` - Documentation source files

## Features

- Semantic search across LangChain documentation
- Natural language Q&A about LangChain
- Source citations and references
- Chat history and context tracking
- Multiple model support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.
