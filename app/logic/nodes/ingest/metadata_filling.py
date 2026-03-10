import asyncio
import json
from asyncio import Semaphore

from langchain_core.messages import ChatMessage, HumanMessage

from app.domain.topics import Topic
from app.infra.db.repos.topics import TopicsRepo
from app.infra.llm.client import LLMClient
from app.logic.nodes.ingest.base import BaseIngestNode, Chunk, IngestState, SemanticChunk


class MetadataFillingNode(BaseIngestNode):

    def __init__(self, topics_repo: TopicsRepo, llm_client: LLMClient):
        self.topics_repo = topics_repo
        self.llm_client = llm_client


    async def execute(self, state: IngestState) -> None:
        topics: list[Topic] = await self.topics_repo.search(is_active=True, archived=False)
        await self._execute_in_parallel(state=state, topics=[topic.name for topic in topics])
        print(state.chunks)


    async def _execute_in_parallel(self, state: IngestState, topics: list[str]):
        sem = Semaphore(5)
        chunk_map = {chunk.id: chunk for chunk in state.chunks}

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
                for chunk_id in chunk.chunk_ids:
                    chunk_map[chunk_id].metadata.update({'topic': json.loads(topic_metadata.content)})

        await asyncio.gather(
            *(
                inner(chunk)
                for chunk in state.semantic_chunks
            ))

