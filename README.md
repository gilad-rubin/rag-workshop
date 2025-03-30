# RAG Workshop

This repository contains a Retrieval-Augmented Generation (RAG) pipeline implementation using Haystack. The pipeline demonstrates how to build a question-answering system that combines document retrieval with language model generation.

# Project Setup

## Environment Setup

Choose one of the following methods to set up your environment:

**Using UV**

1.  `uv sync`
2.  `uv run jupyter lab notebooks/rag_pipeline.ipynb`

**Using Pip**

1.  Install a Python 3.12 environment using venv or conda (e.g., `conda create -n environment-name python=3.12`).
2.  Activate the environment (e.g., `conda activate environment-name`).
3.  `pip install -r requirements.txt`
4.  `jupyter lab notebooks/rag_pipeline.ipynb`

## API Key Setup

1.  Rename `.env.example` to `.env`.
2.  Add your OpenAI API key to the `.env` file in the following format:
    ```
    OPENAI_API_KEY=sk-...
    ```
    
## Features

- PDF document processing and text extraction
- Document chunking with sentence-level splitting
- Document embedding using Sentence Transformers
- In-memory document storage and retrieval
- Integration with OpenAI's language models
- Visualization of document embeddings
- Interactive question-answering pipeline

## Prerequisites

- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone git@github.com:gilad-rubin/rag-workshop.git
cd rag-workshop
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Then edit `.env` and add your OpenAI API key.

## Project Structure

- `notebooks/`: Jupyter notebooks demonstrating the RAG pipeline
  - `rag_pipeline.ipynb`: Main notebook with the complete pipeline implementation
- `src/`: Source code
  - `utils.py`: Utility functions for pipeline visualization and document embedding
- `data/raw/`: Directory for storing PDF documents to be processed

## Pipeline Components

The project implements two main pipelines:

### 1. Indexing Pipeline
- **PDF Conversion**: Converts PDF documents to Haystack Document objects
- **Document Splitting**: Splits documents into smaller chunks by sentences
- **Document Embedding**: Generates embeddings using Sentence Transformers
- **Document Storage**: Stores processed documents in an in-memory database

### 2. RAG Pipeline
- **Text Embedding**: Embeds user queries using the same model
- **Document Retrieval**: Retrieves relevant document chunks
- **Prompt Building**: Constructs prompts combining user questions with retrieved context
- **Answer Generation**: Generates answers using OpenAI's language models

## Usage

1. Place your PDF documents in the `data/raw/` directory

2. Open and run the `notebooks/rag_pipeline.ipynb` notebook:
```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

3. Follow the notebook cells to:
   - Process and index your documents
   - Visualize document embeddings
   - Ask questions and get AI-generated answers

## Models Used

- Document/Text Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Language Model: OpenAI GPT-4