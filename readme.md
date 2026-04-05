Tata Nexon BS6 — Multimodal RAG System
Retrieval-Augmented Generation system for Tata Nexon BS6 service manual intelligence. Supports text, table, and image-based queries via FastAPI.

Problem Statement
Domain Identification
This project operates within the automotive and mechanical engineering domain, specifically targeting the workflows of service engineers, workshop technicians, and vehicle owners working with Tata Motors vehicles — particularly the Tata Nexon BS6 SUV. These professionals and owners rely on the official Tata Nexon BS6 Owner's Manual to perform diagnostics, scheduled maintenance, fault identification, and component-level repairs.

Problem Description
The Tata Nexon BS6 Owner's Manual is a comprehensive, multimodal document containing dense procedural text, specification tables, warning diagrams, component layouts, and maintenance schedules. A service technician or an informed vehicle owner attempting to resolve a fault or perform maintenance must manually navigate this document — often across multiple unrelated sections — to locate the precise information needed.

The core problem is retrieval inefficiency in a safety-critical environment. Consider a technician needing to verify the tyre inflation pressure for the Nexon BS6 under full load, or a vehicle owner trying to understand what a specific dashboard warning light means. The answer may exist in a specifications table in one chapter and cross-referenced by a warning indicator diagram in another. A keyword search returns too many hits. The table of contents offers only coarse navigation. Neither approach surfaces the precise, contextually correct answer quickly.

This problem is compounded by the multimodal nature of the manual. Critical information is frequently split across modalities — a maintenance procedure described in text references a component layout diagram for physical location, and a specification table for tolerance values. Traditional search tools treat these modalities as disconnected, failing to retrieve the complete answer the technician or owner actually needs.

Why This Problem Is Unique
Unlike a generic document Q&A system, the Tata Nexon BS6 Owner's Manual presents specific challenges that make retrieval non-trivial:

Specialized terminology: Terms like TPMS (Tyre Pressure Monitoring System), ESP (Electronic Stability Programme), HVAC controls, BS6 emission norms, and Kryotec diesel engine specifications are Nexon-specific and context-dependent.
Precision requirements: Answers are safety-critical. An incorrect tyre pressure value, wrong engine oil grade (e.g. 15W-40 vs 0W-30), or missed service interval for the Nexon's Kryotec 2.0L diesel engine can cause vehicle damage or void the warranty.
Cross-modal dependency: Questions about dashboard warning indicators require image summaries. Fluid specification questions require table extraction. Step-by-step tyre change or jack usage procedures require sequential text retrieval. No single modality contains the complete answer.
Owner accessibility: Unlike dealership technicians, vehicle owners are not trained to navigate dense technical manuals. A natural language interface dramatically lowers the barrier to accessing safety-critical information.
Why RAG Is the Right Approach
Fine-tuning a language model on the Tata Nexon BS6 manual is impractical — it would require retraining for every new manual edition or variant, cannot cite the source page, and risks hallucinating safety-critical values such as torque limits or fluid capacities. Manual keyword search does not scale and cannot reason across text, tables, and diagrams simultaneously.

Retrieval-Augmented Generation addresses these limitations directly. By embedding document chunks — text paragraphs, specification tables, and VLM-generated descriptions of warning diagrams and component layouts — into a local vector index, the system retrieves only the most relevant chunks for a given query. The LLM then generates a grounded answer using exclusively retrieved context, with full source attribution (filename, page number, chunk type). This ensures every answer is traceable and verifiable against the official Tata Nexon BS6 Owner's Manual.

Expected Outcomes
A successful system will enable the following query types:

Text retrieval: "What is the procedure to use the scissor jack safely on the Nexon?" → returns step-by-step procedure with page reference.
Table retrieval: "What engine oil grade and capacity is recommended for the Kryotec 2.0L diesel engine?" → returns specification table data.
Image retrieval: "What does the orange TPMS warning light on the dashboard mean?" → returns VLM-generated description of the warning diagram.
The system reduces manual lookup time from 10–20 minutes to under 30 seconds, empowers vehicle owners to safely self-diagnose common issues, and provides a scalable foundation for other Tata Motors vehicle manuals.

Architecture Overview
                   ┌─── Text chunks ──────────────────────────┐
PDF → PyMuPDF ─────┼─── Table chunks ─────────────────────────┼──► Sentence-Transformers ──► ChromaDB
                   └─── Image chunks ──► VLM (OpenRouter) ─────┘
                                        (image → text description)

Query ──► Embed ──► ChromaDB (similarity search) ──► Top-K Chunks ──► LLM (OpenRouter) ──► Answer + Source
Ingestion Pipeline: PDF → PyMuPDF parser → 3 chunk types (text, table, image) → Images processed by VLM → All chunks embedded → Stored in ChromaDB

Query Pipeline: User question → Embedded → ChromaDB similarity search → Top-K chunks retrieved → LLM generates grounded answer → Response with sources

Technology Choices
Component	Choice	Justification
PDF Parser	PyMuPDF	Fast, reliable, native table detection, no external API needed
Embedding Model	sentence-transformers (all-MiniLM-L6-v2)	Runs locally, free, strong semantic similarity for technical text
Vector Store	ChromaDB	Local persistent storage, metadata filtering by chunk type, no cloud setup
LLM	OpenRouter (google/gemma-3-27b-it:free)	Free tier access to capable models, easy model switching via .env
VLM	OpenRouter (nvidia/nemotron-nano-12b-v2-vl:free)	Free vision model, handles technical diagrams from PDF pages
Framework	FastAPI	Native Pydantic validation, auto Swagger UI, production-ready
Setup Instructions
Prerequisites
Python 3.11
Git
OpenRouter API key (free at openrouter.ai)
1. Clone the repository
git clone https://github.com/Maharshikant/Nexon-RAG-System.git
cd Nexon-RAG-System
cd automotive-rag-system
2. Create virtual environment
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key
5. Add sample PDF
Place your PDF in the sample_documents/ folder:

sample_documents/Nexon-bs6-owners-manual.pdf
6. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
7. Ingest the PDF
Open browser at http://localhost:8000/docs and use POST /ingest to upload your PDF.

API Documentation
GET /health
Returns system status, model info, indexed chunk count, and uptime.

Response:

{
  "status": "ok",
  "model_llm": "google/gemma-3-27b-it:free",
  "model_vlm": "nvidia/nemotron-nano-12b-v2-vl:free",
  "model_embedding": "all-MiniLM-L6-v2",
  "chunks_indexed": 2378,
  "uptime_seconds": 53.07,
  "timestamp": "2026-04-05T08:49:38"
}
POST /ingest
Accepts PDF file upload, parses text + tables + images, embeds and indexes chunks.

Request: multipart/form-data with PDF file

Response:

{
  "message": "PDF ingested successfully.",
  "filename": "Nexon-bs6-owners-manual.pdf",
  "total_chunks": 2378,
  "text_chunks": 1724,
  "table_chunks": 352,
  "image_chunks": 302,
  "chunks_indexed": 2378,
  "total_indexed": 2378,
  "processing_time_seconds": 45.2
}
POST /query
Accepts natural language question, returns grounded answer with source references.

Request:

{
  "question": "What engine oil grade is recommended for the Kryotec diesel engine?",
  "top_k": 5
}
Response:

{
  "question": "What engine oil grade is recommended for the Kryotec diesel engine?",
  "answer": "0W20 ACEA C2 is the recommended engine oil grade. (Nexon-bs6-owners-manual.pdf, page 242, text)",
  "sources": [
    {
      "filename": "Nexon-bs6-owners-manual.pdf",
      "page_number": 242,
      "chunk_type": "text",
      "similarity": 0.5088
    }
  ],
  "chunks_retrieved": 5
}
GET /docs
Auto-generated Swagger UI — available at http://localhost:8000/docs

Screenshots
Swagger UI
Swagger UI

Health Endpoint
Health

Successful Ingestion
Ingest

Text Query Result
Text Query

Table Query Result
Table Query

Image Query Result
Image Query

Limitations & Future Work
Current Limitations
Free tier API rate limits (429 errors) may require retries between queries so LLM_Models are changed for different output ex. google/gemma-3-27b-it:free or gpt-oss-20b or openai/gpt-oss-20b:free
VLM processing applied to first 20 images per document due to free tier constraints — remaining image chunks use placeholder descriptions
Embedding model all-MiniLM-L6-v2 is optimized for general text — a domain-specific automotive embedding model would improve retrieval accuracy
No authentication on API endpoints — not production-ready as-is
Future Work
Enable full VLM processing for all images with a paid API tier
Add /documents endpoint to list all indexed files
Add /delete endpoint to remove specific documents from the index
Implement re-ranking of retrieved chunks for improved accuracy
Add support for multi-document cross-referencing queries
Deploy on cloud (AWS/GCP) with Docker for production use
