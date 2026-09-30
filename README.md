# Clinical Copilot

A local Retrieval-Augmented Generation (RAG) application with a lightweight agentic workflow for answering clinical research questions using a controlled knowledge base.

The project was built as a practical GenAI project to explore semantic retrieval, local LLM inference, tool use, and evidence-grounded generation without relying on paid API services.

## Overview

Clinical Copilot combines a semantic retrieval pipeline with a local large language model to generate answers grounded in retrieved documents.

The system can:

- Retrieve relevant clinical knowledge using semantic search
- Search a controlled knowledge base through a tool-based interface
- Extract metadata from scientific papers
- Generate answers using a local LLM
- Restrict responses to the available evidence
- Refuse to answer when the knowledge base does not contain sufficient information
- Evaluate retrieval performance using a set of predefined questions

The application runs locally using Python, Sentence Transformers, and Ollama.

## Architecture

```text
                         User question
                              |
                              v
                    +-------------------+
                    |   Agent workflow  |
                    +-------------------+
                       |             |
                decides action       |
                       |             |
                       v             v
                +----------+   +------------+
                |  Search  |   |  Metadata  |
                |   tool   |   |    tool    |
                +----------+   +------------+
                       |
                       v
              Semantic retrieval
                       |
                       v
               Top-k relevant
                 documents
                       |
                       v
              +----------------+
              | Context +      |
              | user question  |
              +----------------+
                       |
                       v
                Local LLM
                (Gemma 3)
                       |
                       v
              Evidence-grounded
                   answer