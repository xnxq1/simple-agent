import asyncio
import json
import logging
import re
from asyncio import Semaphore

from langchain_core.messages import HumanMessage

from app.domain.topics import Topic
from app.infra.db.repos.topics import TopicsRepo
from app.infra.llm.client import LLMClient
from app.logic.nodes.ingest.base import (
    BaseIngestNode,
    Chunk,
    IngestState,
    SemanticChunk,
)

logger = logging.getLogger(__name__)


class MetadataFillingNode(BaseIngestNode):
    def __init__(self, topics_repo: TopicsRepo, llm_client: LLMClient):
        self.topics_repo = topics_repo
        self.llm_client = llm_client

    async def execute(self, state: IngestState) -> dict:
        topics: list[Topic] = await self.topics_repo.search(
            is_active=True, archived=False
        )
        topic_names = [topic.name for topic in topics]

        if not topic_names:
            logger.warning("No active topics found in DB, skipping metadata filling")
            updated_chunks = [
                Chunk(id=c.id, text=c.text, metadata={**c.metadata, "topic": []})
                for c in state.chunks
            ]
            return {"chunks": updated_chunks}

        updated_chunks = await self._execute_in_parallel(
            state=state, topics=topic_names
        )
        logger.debug(f"Metadata filled for {len(updated_chunks)} chunks")
        return {"chunks": updated_chunks}

    @staticmethod
    def _parse_topics(content: str) -> list[str]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
        try:
            result = json.loads(content)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse topics from LLM response: {content!r}")
            return []

    async def _execute_in_parallel(
        self, state: IngestState, topics: list[str]
    ) -> list[Chunk]:
        sem = Semaphore(5)
        results = {}

        async def inner(chunk: SemanticChunk):
            async with sem:
                topic_metadata = await self.llm_client.completions_create(
                    system_prompt=f"""You are an AI assistant that classifies document content into predefined topics.

Rules:
1. You are given an excerpt from a web document or article.
2. Select the topic(s) from the available list that best describe this content.
3. Only choose from the provided list; do not invent new topics.
4. You may choose more than one topic if multiple apply.
5. Be inclusive: assign a topic if the content is at least tangentially related.
6. Output only a JSON array of topic names, nothing else.
7. If truly no topic applies, output: []
8. Available topics: {json.dumps(topics)}""",
                    messages=[HumanMessage(content=chunk.text)],
                )
                topics_list = self._parse_topics(topic_metadata.content)

                for chunk_id in chunk.chunk_ids:
                    results[chunk_id] = topics_list

        gather_results = await asyncio.gather(
            *(inner(chunk) for chunk in state.semantic_chunks), return_exceptions=True
        )
        for i, exc in enumerate(gather_results):
            if isinstance(exc, BaseException):
                logger.warning(f"Metadata fill failed for semantic chunk {i}: {exc}")

        # Create new chunks with updated metadata (immutable update pattern)
        updated_chunks = []
        for chunk in state.chunks:
            updated_metadata = chunk.metadata.copy()
            updated_metadata["topic"] = results.get(chunk.id, [])
            updated_chunks.append(
                Chunk(id=chunk.id, text=chunk.text, metadata=updated_metadata)
            )

        return updated_chunks
