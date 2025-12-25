
PDF-based RAG System using LangChain, FAISS & Groq


# Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions over PDF documents.
The system retrieves relevant chunks from the PDF using FAISS vector search and generates accurate answers using Groq-hosted LLMs via LangChain.
This project is designed to demonstrate end-to-end RAG architecture and is suitable for showcasing to recruiters as a practical LLM application.

# Architecture

1.PDF Loader – Loads PDF documents from the data/ directory

2.Text Splitter – Splits documents into overlapping chunks

3.Embeddings – Uses HuggingFace sentence-transformers (local embeddings)

4.Vector Store – FAISS index for fast similarity search

5.Retriever – Fetches relevant chunks for a user query

6.LLM (Groq) – Generates final answers using retrieved context

# Tech Stack

-Python

-LangChain

-FAISS (Vector Database)

-Groq API (LLM inference)

-HuggingFace Embeddings (all-MiniLM-L6-v2)

# Project Structure

rag-langchain-groq/

- app.py # Main RAG application
- data/ # PDF files for ingestion
- .gitignore # Git ignore rules
- README.md # Project documentation

# Setup Instructions

1️. Clone the repository

git clone https://github.com/suchithra-25/rag-langchain-groq.git

cd rag-langchain-groq

2️. Create and activate virtual environment

python -m venv venv
venv\Scripts\activate # Windows

3️. Install dependencies

pip install langchain langchain-community langchain-groq faiss-cpu pypdf huggingface-hub sentence-transformers

4️. Set Groq API Key

Create an environment variable:

setx GROQ_API_KEY "your_groq_api_key"

Restart terminal after this.

# Running the Application

python app.py

You will be prompted to enter a query. The system will:
Retrieve relevant content from the PDF
Generate an answer using Groq LLM

# Sample Output

The document explains the recruitment process of Federal Bank including eligibility criteria, test structure, and service agreements.


# Author

Suchithra B

Aspiring AI / LLM Engineer

