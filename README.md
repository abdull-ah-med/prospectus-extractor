# Prospectus Extraction Service

An AI-powered document processing pipeline for extracting structured data from university prospectuses. Built with Python, FastAPI, and local LLMs via Ollama.

## Overview

This service automates the extraction of academic information from PDF prospectuses, including:

- **Departments and Programs** - Academic units, degree offerings, and curricula
- **Facilities** - Laboratories, libraries, hostels, and campus amenities
- **Fee Structures** - Tuition, admission fees, and payment schedules
- **Admission Information** - Eligibility criteria, deadlines, and requirements

Extracted data is stored in PostgreSQL with vector embeddings for semantic search via pgvector.

## Architecture

```
prospectus-extractor/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── worker.py               # Service Bus queue consumer
│   ├── config.py               # Environment configuration
│   ├── models/
│   │   ├── schema.py           # Pydantic extraction schemas
│   │   └── db.py               # SQLAlchemy database models
│   └── services/
│       ├── document_parser.py  # PDF text extraction
│       ├── chunker.py          # Semantic text chunking
│       ├── classifier.py       # Section classification
│       ├── llm_client.py       # LLM-based data extraction
│       ├── embedder.py         # Vector embedding generation
│       ├── persister.py        # Database persistence
│       └── blob_storage.py     # Azure Blob Storage client
└── requirements.txt
```

## Processing Pipeline

1. **Document Ingestion** - PDF retrieved from Azure Blob Storage
2. **Text Extraction** - Content parsed with layout preservation using pdfplumber
3. **Chunking** - Document split into semantic chunks (headings, paragraphs, tables)
4. **Classification** - Chunks categorized by section type (programs, fees, facilities, etc.)
5. **LLM Extraction** - Structured data extracted using schema-guided prompts
6. **Validation** - Output validated against Pydantic models
7. **Persistence** - Data stored in PostgreSQL with JSON extraction snapshots
8. **Embedding** - Vector embeddings generated and stored in pgvector

## Prerequisites

- Python 3.11+
- PostgreSQL 16+ with pgvector extension
- Ollama (for local LLM inference)
- Azure Storage Account (for PDF storage)
- Azure Service Bus (for job queuing)

## Installation

### Local Development

1. Clone the repository and create a virtual environment:

```bash
cd prospectus-extractor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment variables (create `.env` file):

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/oneuni
AZURE_STORAGE_CONNECTION_STRING=<your-connection-string>
AZURE_SERVICEBUS_CONNECTION_STRING=<your-connection-string>
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=llama3.1:8b
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL_NAME=mxbai-embed-large
```

3. Start required services with Docker Compose:

```bash
docker-compose up -d ollama postgres
```

4. Pull the required Ollama models:

```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

### Docker Deployment

Build and run all services:

```bash
docker-compose up --build
```

## Usage

### Running the API Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Running the Queue Worker

```bash
python -m src.worker
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/process/{ingestion_id}` | POST | Trigger processing for an ingestion |
| `/status/{ingestion_id}` | GET | Get processing status |
| `/extraction/{ingestion_id}` | GET | Retrieve extracted data |

### Example Request

```bash
# Check service health
curl http://localhost:8000/health

# Get extraction status
curl http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000

# Retrieve extracted data
curl http://localhost:8000/extraction/550e8400-e29b-41d4-a716-446655440000
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `AZURE_STORAGE_CONNECTION_STRING` | - | Azure Blob Storage connection |
| `AZURE_STORAGE_CONTAINER` | `prospectuses` | Blob container name |
| `AZURE_SERVICEBUS_CONNECTION_STRING` | - | Service Bus connection |
| `AZURE_SERVICEBUS_QUEUE_NAME` | `prospectus-extraction-jobs` | Queue name |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `LLM_MODEL_NAME` | `llama3.1:8b` | LLM model for extraction |
| `LLM_TEMPERATURE` | `0.1` | Model temperature |
| `LLM_MAX_TOKENS` | `4096` | Maximum output tokens |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Embedding API endpoint |
| `EMBEDDING_MODEL_NAME` | `mxbai-embed-large` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensions |
| `CHUNK_SIZE` | `1000` | Maximum chunk size (chars) |
| `CHUNK_OVERLAP` | `350` | Chunk overlap (chars) |

## Database Schema

The service uses the following PostgreSQL tables:

- `prospectus_ingestions` - Tracks document processing lifecycle
- `prospectus_extractions` - Stores extracted JSON with confidence scores
- `prospectus_chunks` - Individual document chunks with section labels
- `prospectus_vectors` - Vector embeddings for semantic search (pgvector)

## Output Schema

Extracted data follows a unified JSON structure:

```json
{
  "schema_version": "v1.0.0",
  "university_name": "Example University",
  "university_short_name": "EU",
  "location": "City, Country",
  "departments": [
    {
      "name": "Department of Computer Science",
      "programs": [
        {
          "name": "BS Computer Science",
          "degree_type": "BS",
          "duration_years": 4,
          "credit_hours": 130
        }
      ]
    }
  ],
  "facilities": [],
  "fee_structure": [],
  "admissions": {},
  "metadata": {
    "total_chunks_processed": 150,
    "total_pages": 45,
    "confidence_scores": {}
  }
}
```

## License

This project is part of the OneUni platform.
