import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.db import (
    ProspectusIngestion, ProspectusExtraction, ProspectusChunk,
    IngestionStatus
)
from ..models.schema import UniversityExtraction
from .chunker import TextChunk

logger = logging.getLogger(__name__)


class PersisterService:
    def __init__(self, session: Session):
        self.session = session

    def update_ingestion_status(
        self,
        ingestion_id: UUID,
        status: IngestionStatus,
        error_message: Optional[str] = None
    ):
        ingestion = self.session.query(ProspectusIngestion).filter(
            ProspectusIngestion.ingestion_id == ingestion_id
        ).first()

        if not ingestion:
            raise ValueError(f"Ingestion {ingestion_id} not found")

        ingestion.status = status.value
        ingestion.updated_at = datetime.now()

        if status == IngestionStatus.PROCESSING:
            ingestion.processing_started_at = datetime.now()
        elif status == IngestionStatus.COMPLETED:
            ingestion.completed_at = datetime.now()
        elif status == IngestionStatus.FAILED:
            ingestion.error_message = error_message
            ingestion.retry_count = (ingestion.retry_count or 0) + 1

        self.session.commit()
        logger.info(f"Updated ingestion {ingestion_id} status to {status.value}")

    def save_chunks(
        self,
        ingestion_id: UUID,
        chunks: list[TextChunk]
    ) -> int:
        saved = 0
        for chunk in chunks:
            db_chunk = ProspectusChunk(
                chunk_id=UUID(chunk.chunk_id),
                ingestion_id=ingestion_id,
                chunk_type=chunk.chunk_type.value if chunk.chunk_type else None,
                chunk_text=chunk.text,
                page_number=chunk.page_number,
                position_in_doc=chunk.position_in_doc,
                section_label=chunk.section_label
            )
            self.session.add(db_chunk)
            saved += 1

        self.session.commit()
        logger.info(f"Saved {saved} chunks for ingestion {ingestion_id}")
        return saved

    def save_extraction(
        self,
        ingestion_id: UUID,
        extraction: UniversityExtraction
    ) -> UUID:
        total_entities = (
            len(extraction.departments) +
            len(extraction.facilities) +
            len(extraction.fee_structure) +
            sum(len(d.programs) for d in extraction.departments)
        )

        avg_confidence = 0.0
        if extraction.metadata.confidence_scores:
            scores = list(extraction.metadata.confidence_scores.values())
            avg_confidence = sum(scores) / len(scores) if scores else 0.0

        db_extraction = ProspectusExtraction(
            ingestion_id=ingestion_id,
            schema_version=extraction.schema_version,
            extracted_json=extraction.model_dump(mode="json"),
            confidence_scores=Decimal(str(round(avg_confidence, 2))),
            total_entities_extracted=total_entities
        )

        self.session.add(db_extraction)
        self.session.commit()

        logger.info(
            f"Saved extraction for ingestion {ingestion_id}: "
            f"{total_entities} entities, {avg_confidence:.2f} avg confidence"
        )
        return db_extraction.extraction_id

    def get_ingestion(self, ingestion_id: UUID) -> Optional[ProspectusIngestion]:
        return self.session.query(ProspectusIngestion).filter(
            ProspectusIngestion.ingestion_id == ingestion_id
        ).first()

    def get_extraction(self, ingestion_id: UUID) -> Optional[ProspectusExtraction]:
        return self.session.query(ProspectusExtraction).filter(
            ProspectusExtraction.ingestion_id == ingestion_id
        ).order_by(ProspectusExtraction.created_at.desc()).first()

    def get_chunks(self, ingestion_id: UUID) -> list[ProspectusChunk]:
        return self.session.query(ProspectusChunk).filter(
            ProspectusChunk.ingestion_id == ingestion_id
        ).order_by(ProspectusChunk.position_in_doc).all()

    def delete_ingestion_data(self, ingestion_id: UUID):
        self.session.query(ProspectusChunk).filter(
            ProspectusChunk.ingestion_id == ingestion_id
        ).delete()
        self.session.query(ProspectusExtraction).filter(
            ProspectusExtraction.ingestion_id == ingestion_id
        ).delete()
        self.session.commit()
        logger.info(f"Deleted all data for ingestion {ingestion_id}")
