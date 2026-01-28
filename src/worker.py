import asyncio
import json
import logging
from uuid import UUID

from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from sqlalchemy.orm import Session

from .config import settings
from .models.db import get_engine, get_session_factory, IngestionStatus
from .services.blob_storage import BlobStorageService
from .services.document_parser import DocumentParserService
from .services.chunker import ChunkerService
from .services.classifier import ClassifierService
from .services.llm_client import ExtractionService
from .services.embedder import EmbedderService
from .services.persister import PersisterService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExtractionWorker:
    def __init__(self):
        self.blob_service = BlobStorageService()
        self.parser = DocumentParserService()
        self.chunker = ChunkerService()
        self.classifier = ClassifierService()
        self.extractor = ExtractionService()
        self.embedder = EmbedderService()

        self.engine = get_engine(settings.database_url)
        self.SessionFactory = get_session_factory(self.engine)

    def get_session(self) -> Session:
        return self.SessionFactory()

    async def process_ingestion(self, ingestion_id: UUID):
        session = self.get_session()
        persister = PersisterService(session)

        try:
            logger.info(f"Processing ingestion {ingestion_id}")
            persister.update_ingestion_status(ingestion_id, IngestionStatus.PROCESSING)

            ingestion = persister.get_ingestion(ingestion_id)
            if not ingestion:
                raise ValueError(f"Ingestion {ingestion_id} not found")

            logger.info(f"Downloading PDF from {ingestion.blob_url}")
            pdf_bytes = await self.blob_service.download_blob(ingestion.blob_url)

            logger.info("Parsing PDF document")
            parsed_doc = self.parser.parse_pdf(pdf_bytes)

            logger.info("Chunking document")
            chunks = self.chunker.chunk_document(parsed_doc)
            logger.info(f"Created {len(chunks)} chunks")

            logger.info("Classifying chunks")
            classified = self.classifier.classify_chunks(chunks)

            logger.info("Saving chunks to database")
            persister.save_chunks(ingestion_id, chunks)

            logger.info("Extracting structured data via LLM")
            extraction = await self.extractor.extract_all(chunks)

            logger.info("Saving extraction results")
            persister.save_extraction(ingestion_id, extraction)

            logger.info("Generating and storing embeddings")
            await self.embedder.embed_and_store(ingestion_id, chunks, session)

            persister.update_ingestion_status(ingestion_id, IngestionStatus.COMPLETED)
            logger.info(f"Completed processing ingestion {ingestion_id}")

        except Exception as e:
            logger.error(f"Failed to process ingestion {ingestion_id}: {e}", exc_info=True)
            persister.update_ingestion_status(
                ingestion_id,
                IngestionStatus.FAILED,
                str(e)
            )
            raise
        finally:
            session.close()
            await self.embedder.close()

    async def process_message(self, message: ServiceBusMessage):
        try:
            body = json.loads(str(message))
            ingestion_id = UUID(body["ingestion_id"])
            await self.process_ingestion(ingestion_id)
        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            raise

    async def run(self):
        logger.info("Starting extraction worker")
        logger.info(f"Connecting to Service Bus queue: {settings.azure_servicebus_queue_name}")

        async with ServiceBusClient.from_connection_string(
            settings.azure_servicebus_connection_string
        ) as client:
            receiver = client.get_queue_receiver(
                queue_name=settings.azure_servicebus_queue_name
            )

            async with receiver:
                logger.info("Worker ready, waiting for messages...")
                async for message in receiver:
                    try:
                        await self.process_message(message)
                        await receiver.complete_message(message)
                        logger.info("Message processed successfully")
                    except Exception as e:
                        logger.error(f"Message processing failed: {e}")
                        await receiver.abandon_message(message)


async def main():
    worker = ExtractionWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
