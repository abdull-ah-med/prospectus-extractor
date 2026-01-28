import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .config import settings
from .models.db import get_engine, get_session_factory, IngestionStatus
from .services.persister import PersisterService
from .worker import ExtractionWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

engine = get_engine(settings.database_url)
SessionFactory = get_session_factory(engine)
worker = ExtractionWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting prospectus extraction service")
    yield
    logger.info("Shutting down prospectus extraction service")


app = FastAPI(
    title="Prospectus Extraction Service",
    description="AI-powered prospectus data extraction API",
    version="1.0.0",
    lifespan=lifespan
)


class ProcessRequest(BaseModel):
    ingestion_id: str


class StatusResponse(BaseModel):
    ingestion_id: str
    status: str
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class ExtractionResponse(BaseModel):
    ingestion_id: str
    extraction_id: str
    schema_version: str
    extracted_json: dict
    confidence_score: float | None
    total_entities: int


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "prospectus-extractor"}


@app.post("/process/{ingestion_id}")
async def trigger_processing(ingestion_id: str, background_tasks: BackgroundTasks):
    try:
        uid = UUID(ingestion_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ingestion_id format")

    session = SessionFactory()
    try:
        persister = PersisterService(session)
        ingestion = persister.get_ingestion(uid)
        if not ingestion:
            raise HTTPException(status_code=404, detail="Ingestion not found")

        if ingestion.status == IngestionStatus.PROCESSING.value:
            raise HTTPException(status_code=409, detail="Already processing")

        background_tasks.add_task(worker.process_ingestion, uid)
        return {"message": "Processing started", "ingestion_id": ingestion_id}
    finally:
        session.close()


@app.get("/status/{ingestion_id}", response_model=StatusResponse)
async def get_status(ingestion_id: str):
    try:
        uid = UUID(ingestion_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ingestion_id format")

    session = SessionFactory()
    try:
        persister = PersisterService(session)
        ingestion = persister.get_ingestion(uid)
        if not ingestion:
            raise HTTPException(status_code=404, detail="Ingestion not found")

        return StatusResponse(
            ingestion_id=str(ingestion.ingestion_id),
            status=ingestion.status,
            error_message=ingestion.error_message,
            created_at=ingestion.created_at.isoformat() if ingestion.created_at else None,
            completed_at=ingestion.completed_at.isoformat() if ingestion.completed_at else None
        )
    finally:
        session.close()


@app.get("/extraction/{ingestion_id}", response_model=ExtractionResponse)
async def get_extraction(ingestion_id: str):
    try:
        uid = UUID(ingestion_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ingestion_id format")

    session = SessionFactory()
    try:
        persister = PersisterService(session)
        extraction = persister.get_extraction(uid)
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")

        return ExtractionResponse(
            ingestion_id=str(extraction.ingestion_id),
            extraction_id=str(extraction.extraction_id),
            schema_version=extraction.schema_version,
            extracted_json=extraction.extracted_json,
            confidence_score=float(extraction.confidence_scores) if extraction.confidence_scores else None,
            total_entities=extraction.total_entities_extracted or 0
        )
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
