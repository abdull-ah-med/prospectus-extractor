import logging
import asyncio
import httpx
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ..config import settings
from ..models.db import ProspectusChunk, ProspectusVector, get_engine, get_session_factory
from .chunker import TextChunk

logger = logging.getLogger(__name__)


class EmbedderService:
    def __init__(self):
        self.base_url = settings.embedding_base_url
        self.model_name = settings.embedding_model_name
        self.dimensions = settings.embedding_dimensions
        self._client: Optional[httpx.AsyncClient] = None
        self.semaphore = asyncio.Semaphore(3)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate_embedding(self, text: str) -> List[float]:
        client = await self._get_client()
        url = f"{self.base_url}/api/embed"

        async with self.semaphore:
            response = await client.post(
                url,
                json={
                    "model": self.model_name,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()

            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            elif "embedding" in data:
                return data["embedding"]
            else:
                raise ValueError(f"Unexpected response format: {data.keys()}")

    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 10
    ) -> List[List[float]]:
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks = [self.generate_embedding(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Embedding failed for text {i + j}: {result}")
                    results.append([0.0] * self.dimensions)
                else:
                    results.append(result)

            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return results

    async def embed_chunks(self, chunks: List[TextChunk]) -> List[tuple[TextChunk, List[float]]]:
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.generate_embeddings_batch(texts)
        return list(zip(chunks, embeddings))

    async def embed_and_store(
        self, 
        ingestion_id: UUID,
        chunks: List[TextChunk],
        session: Session
    ) -> int:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        chunk_embeddings = await self.embed_chunks(chunks)
        stored_count = 0

        for chunk, embedding in chunk_embeddings:
            if all(v == 0.0 for v in embedding):
                logger.warning(f"Skipping zero embedding for chunk {chunk.chunk_id}")
                continue

            db_chunk = session.query(ProspectusChunk).filter(
                ProspectusChunk.chunk_id == UUID(chunk.chunk_id)
            ).first()

            if not db_chunk:
                db_chunk = ProspectusChunk(
                    chunk_id=UUID(chunk.chunk_id),
                    ingestion_id=ingestion_id,
                    chunk_type=chunk.chunk_type.value if chunk.chunk_type else None,
                    chunk_text=chunk.text,
                    page_number=chunk.page_number,
                    position_in_doc=chunk.position_in_doc,
                    section_label=chunk.section_label
                )
                session.add(db_chunk)
                session.flush()

            existing_vector = session.query(ProspectusVector).filter(
                ProspectusVector.chunk_id == db_chunk.chunk_id
            ).first()

            if existing_vector:
                existing_vector.embedding = embedding
            else:
                vector = ProspectusVector(
                    chunk_id=db_chunk.chunk_id,
                    embedding=embedding
                )
                session.add(vector)

            stored_count += 1

        session.commit()
        logger.info(f"Stored {stored_count} embeddings for ingestion {ingestion_id}")
        return stored_count

    async def similarity_search(
        self,
        query: str,
        session: Session,
        ingestion_id: Optional[UUID] = None,
        limit: int = 5
    ) -> List[tuple[ProspectusChunk, float]]:
        query_embedding = await self.generate_embedding(query)

        from sqlalchemy import text

        params = {"embedding": str(query_embedding), "limit": limit}
        
        if ingestion_id:
            sql = text("""
                SELECT c.*, v.embedding <=> :embedding::vector AS distance
                FROM prospectus_chunks c
                JOIN prospectus_vectors v ON c.chunk_id = v.chunk_id
                WHERE c.ingestion_id = :ingestion_id
                ORDER BY distance
                LIMIT :limit
            """)
            params["ingestion_id"] = str(ingestion_id)
        else:
            sql = text("""
                SELECT c.*, v.embedding <=> :embedding::vector AS distance
                FROM prospectus_chunks c
                JOIN prospectus_vectors v ON c.chunk_id = v.chunk_id
                ORDER BY distance
                LIMIT :limit
            """)

        result = session.execute(sql, params)
        rows = result.fetchall()

        chunks_with_scores = []
        for row in rows:
            chunk = session.query(ProspectusChunk).filter(
                ProspectusChunk.chunk_id == row.chunk_id
            ).first()
            if chunk:
                similarity = 1 - row.distance
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores


embedder_service = EmbedderService()
