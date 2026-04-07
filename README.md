# PharmaGuide-AI

PharmaGuide-AI is a simple Retrieval-Augmented Generation (RAG) project built to answer questions about medication leaflets in a more structured and contextual way.

I created this project mainly to consolidate my knowledge of RAG systems in practice. Building it helped me work with essential concepts such as document loading, chunking strategies, metadata enrichment, embeddings, vector databases, retrieval, and LLM-based answer generation.

Instead of relying only on a language model’s memory, this project retrieves relevant information directly from official PDF leaflets and uses that context to produce grounded answers.

---

## Project Goal

The main goal of this project was not only to build a functional AI assistant for medication leaflets, but also to deepen my understanding of the RAG pipeline as a whole.

To make this project work, some concepts were especially important:

- **Chunks**: splitting long PDF documents into smaller pieces so the model can retrieve more relevant information
- **Metadata**: adding structured information such as medication name and content category to improve organization and retrieval
- **Embeddings**: transforming text into vectors so semantic search becomes possible
- **Vector Store**: storing those embeddings in Chroma for efficient retrieval
- **LLM**: using Gemini (Free tier) to generate final answers based on the retrieved context
- **Retriever + RAG Chain**: connecting everything into a pipeline that retrieves relevant chunks and answers the user’s question

This project was a practical way to move from theory to implementation and better understand how each part of a RAG architecture fits together.

---

## How It Works

The pipeline follows these steps:

1. Load medication leaflets in PDF format
2. Split the documents into chunks
3. Add metadata to each chunk
4. Generate embeddings using Google's embedding model
5. Store the vectors in Chroma
6. Retrieve the most relevant chunks for a given question
7. Use Gemini to generate an answer based on that retrieved context

---

## Tech Stack

- **Python**
- **LangChain**
- **Google Gemini API**
- **Chroma**
- **PyPDFLoader**
- **python-dotenv**

---

## Project Structure

```bash
PharmaGuide-AI/
│
├── main.py
├── .env
├── requirements.txt
├── bulaDipirona.pdf
├── bulaParacetamol.pdf
└── chroma_bulas/
```
## Final Note

PharmaGuide-AI was created as a study project to strengthen my understanding of Retrieval-Augmented Generation in practice.

More than just getting an answer from an LLM, this project helped me understand the importance of document preprocessing, chunking, metadata, embeddings, retrieval quality, and contextual generation. It was an important step in turning RAG concepts into something concrete and functional.
