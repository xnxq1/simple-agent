import asyncio
import json
import logging
from asyncio import Semaphore

from langchain_core.messages import HumanMessage

from app.domain.topics import Topic
from app.infra.db.repos.topics import TopicsRepo
from app.infra.llm.client import LLMClient
from app.logic.nodes.ingest.base import BaseIngestNode, Chunk, IngestState, SemanticChunk

logger = logging.getLogger(__name__)


class MetadataFillingNode(BaseIngestNode):

    def __init__(self, topics_repo: TopicsRepo, llm_client: LLMClient):
        self.topics_repo = topics_repo
        self.llm_client = llm_client


    async def execute(self, state: IngestState) -> dict:
        topics: list[Topic] = await self.topics_repo.search(is_active=True, archived=False)
        updated_chunks = await self._execute_in_parallel(state=state, topics=[topic.name for topic in topics])
        logger.debug(f"Metadata filled for {len(updated_chunks)} chunks")
        return {"chunks": updated_chunks}


    async def _execute_in_parallel(self, state: IngestState, topics: list[str]) -> list[Chunk]:
        sem = Semaphore(5)
        chunk_map = {chunk.id: chunk for chunk in state.chunks}
        results = {}

        async def inner(chunk: SemanticChunk):
            async with sem:
                topic_metadata = await self.llm_client.completions_create(
                    system_prompt=f"""
                    You are an AI assistant responsible for determining the most relevant topic(s) for a given user query.
                    Rules:

                    1. You are provided with a list of available topics from the system.
                    2. Your task is to select the topic(s) that best match the user's query.
                    3. Only choose from the provided list; do not invent new topics.
                    4. You may choose more than one topic if multiple are relevant.
                    5. Always be concise and output only the topic names in a JSON array format.
                    6. If no topic is clearly relevant, output an empty array: [].
                    7. Do not include explanations or additional text.
                    8. Available topic: {topics}

                    Example:

                    User query: "I want to learn how transformers work in NLP."

                    Available topics: ["finance", "technology", "health", "education", "AI"]

                    Output:
                    ["technology", "AI"]

                    User query: "Benefits of meditation for mental health."

                    Available topics: ["finance", "technology", "health", "education", "AI"]

                    Output:
                    ["health"]
                    """,
                    messages=[HumanMessage(content=chunk.text)]
                    )
                try:
                    topics_list = json.loads(topic_metadata.content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse topics JSON for chunk: {e}. Using empty list.")
                    topics_list = []

                for chunk_id in chunk.chunk_ids:
                    results[chunk_id] = topics_list

        await asyncio.gather(
            *(
                inner(chunk)
                for chunk in state.semantic_chunks
            ))

        # Create new chunks with updated metadata (immutable update pattern)
        updated_chunks = []
        for chunk in state.chunks:
            updated_metadata = chunk.metadata.copy()
            if chunk.id in results:
                updated_metadata['topic'] = results[chunk.id]
            updated_chunks.append(Chunk(id=chunk.id, text=chunk.text, metadata=updated_metadata))

        return updated_chunks

