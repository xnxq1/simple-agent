from typing import NewType

from dishka import Provider, Scope, make_container, provide
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from llama_index.readers.web import TrafilaturaWebReader
from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.application.agent.router import AgentRouter
from app.application.ingest.router import IngestRouter
from app.application.topics.router import TopicRouter
from app.application.users.router import UserRouter
from app.infra.config import Settings
from app.infra.db.repos.topics import TopicsRepo
from app.infra.db.repos.users import UsersRepo
from app.infra.llm.client import LLMClient
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.handlers.topic import (
    CreateTopicHandler,
    GetTopicHandler,
    UpdateTopicHandler,
)
from app.logic.handlers.user import CreateUserHandler, GetUsersHandler
from app.logic.nodes.evaluator import Evaluator
from app.logic.nodes.ingest.base import IngestState
from app.logic.nodes.ingest.chunking import SemanticChunkingNode
from app.logic.nodes.ingest.embeddings import EmbeddingNode
from app.logic.nodes.ingest.loaders import WebLoaderNode
from app.logic.nodes.ingest.metadata_filling import MetadataFillingNode
from app.logic.nodes.ingest.qdrant import QdrantIngestNode
from app.logic.nodes.llm_node import LLMNode
from app.logic.nodes.state import MessagesState
from app.logic.nodes.tool_node import ToolNode
from app.logic.services.chunking import ChunkingService
from app.logic.services.embedding import EmbeddingService
from app.logic.services.evaluation import EvaluationService
from app.logic.services.metadata_filling import MetadataFillingService
from app.logic.services.vector_store import VectorStoreService
from app.logic.services.web_loader import WebLoaderService
from app.logic.tools.db import DBTools
from app.logic.tools.rag import RAGTools
from app.main import AppBuilder
from app.infra.db.repos.user_threads import UserThreadsRepo

AgentGraph = NewType("AgentGraph", CompiledStateGraph)
IngestGraph = NewType("IngestGraph", CompiledStateGraph)
RAGToolsType = NewType("RAGToolsType", list)
DBToolsType = NewType("DBToolsType", list)
ToolsType = NewType("ToolsType", list)
LLMWithoutToolsType = NewType("LLMWithoutToolsType", LLMClient)


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def settings(self) -> Settings:
        return Settings()

    @provide(scope=Scope.APP)
    def app_builder(
        self,
        settings: Settings,
        agent_graph: AgentGraph,
        ingest_graph: IngestGraph,
        topics_repo: TopicsRepo,
        users_repo: UsersRepo,
        user_threads_repo: UserThreadsRepo,
    ) -> AppBuilder:
        topic_router = TopicRouter(
            create_topic_handler=CreateTopicHandler(topic_repo=topics_repo),
            get_topic_handler=GetTopicHandler(topic_repo=topics_repo),
            update_topic_handler=UpdateTopicHandler(topic_repo=topics_repo),
        )
        user_router = UserRouter(
            create_user_handler=CreateUserHandler(users_repo=users_repo),
            get_users_handler=GetUsersHandler(users_repo=users_repo),
        )
        agent_router = AgentRouter(graph_agent=agent_graph, user_threads_repo=user_threads_repo)
        ingest_router = IngestRouter(ingest_graph=ingest_graph)
        return AppBuilder(
            routers=[agent_router.router, ingest_router.router, topic_router.router, user_router.router],
            settings=settings,
        )


class DBProvider(Provider):
    @provide(scope=Scope.APP)
    def async_engine(self, settings: Settings) -> AsyncEngine:
        db_url = settings.db_url.replace("postgresql://", "postgresql+asyncpg://")
        return create_async_engine(
            db_url,
            echo=settings.debug,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    @provide(scope=Scope.APP)
    def topics_repo(self, engine: AsyncEngine) -> TopicsRepo:
        return TopicsRepo(engine=engine)

    @provide(scope=Scope.APP)
    def users_repo(self, engine: AsyncEngine) -> UsersRepo:
        return UsersRepo(engine=engine)

    @provide(scope=Scope.APP)
    def postgres_checkpointer(self, settings: Settings) -> AsyncPostgresSaver:
        return AsyncPostgresSaver.from_conn_string(settings.db_url)

    @provide(scope=Scope.APP)
    def user_threads_repo(self, engine: AsyncEngine) -> UserThreadsRepo:
        return UserThreadsRepo(engine=engine)


class EmbeddingsProvider(Provider):
    @provide(scope=Scope.APP)
    def embeddings_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @provide(scope=Scope.APP)
    def cross_encoder_model(self) -> CrossEncoder:
        return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


class ToolsProvider(Provider):
    @provide(scope=Scope.APP)
    def rag_tools(
        self,
        qdrant_repo: QdrantRepo,
        embeddings_model: HuggingFaceEmbeddings,
        settings: Settings,
        cross_encoder_model: CrossEncoder,
    ) -> RAGToolsType:
        rag = RAGTools(
            qdrant_repo=qdrant_repo,
            embed_model=embeddings_model,
            settings=settings,
            cross_encoder_model=cross_encoder_model,
        )
        return [rag.search_docs]

    @provide(scope=Scope.APP)
    def db_tools(self, topics_repo: TopicsRepo) -> DBToolsType:
        tools = DBTools(topics_repo=topics_repo)
        return [tools.get_available_topics]

    @provide(scope=Scope.APP)
    def tools(
        self,
        rag_tools: RAGToolsType,
        db_tools: DBToolsType,
    ) -> ToolsType:
        return rag_tools + db_tools


class LLMProvider(Provider):
    @provide(scope=Scope.APP)
    def llm_client(self, settings: Settings, tools: ToolsType) -> LLMClient:
        model = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            api_key=settings.open_ai_api_key,
            base_url=settings.open_ai_base_url,
        )
        return LLMClient(model=model, settings=settings, tools=tools)

    @provide(scope=Scope.APP)
    def llm_client_no_tools(self, settings: Settings) -> LLMWithoutToolsType:
        """LLM client without tools, used for content analysis tasks."""
        model = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,
            api_key=settings.open_ai_api_key,
            base_url=settings.open_ai_base_url,
        )
        return LLMClient(model=model, settings=settings, tools=[])

    @provide(scope=Scope.APP)
    def llm_node(self, llm_client: LLMClient) -> LLMNode:
        return LLMNode(llm_client)

    @provide(scope=Scope.APP)
    def evaluate_node(
        self,
        service: EvaluationService,
    ) -> Evaluator:
        return Evaluator(service)

    @provide(scope=Scope.APP)
    async def graph_agent(
        self,
        llm_node: LLMNode,
        evaluator: Evaluator,
        tools: ToolsType,
        checkpointer: AsyncPostgresSaver,
    ) -> AgentGraph:
        graph = StateGraph(MessagesState)
        graph.add_node("llm_call", llm_node.execute)

        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node.execute)
        graph.add_node("evaluate", evaluator.execute)

        def should_continue(state: MessagesState) -> str:
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "set_answer"

        def set_answer(state: MessagesState) -> dict:
            last_message = state.messages[-1]
            return {"answer": last_message.content}

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", should_continue)
        graph.add_edge("tools", "llm_call")
        graph.add_node("set_answer", set_answer)
        graph.add_edge("set_answer", "evaluate")
        graph.add_edge("evaluate", END)

        app = graph.compile(checkpointer=checkpointer)
        return app


class ServicesProvider(Provider):
    @provide(scope=Scope.APP)
    def web_loader_service(self) -> WebLoaderService:
        return WebLoaderService(TrafilaturaWebReader())

    @provide(scope=Scope.APP)
    def chunking_service(self) -> ChunkingService:
        return ChunkingService()

    @provide(scope=Scope.APP)
    def embedding_service(
        self, embeddings_model: HuggingFaceEmbeddings
    ) -> EmbeddingService:
        return EmbeddingService(embeddings_model)

    @provide(scope=Scope.APP)
    def vector_store_service(
        self, qdrant_repo: QdrantRepo, settings: Settings
    ) -> VectorStoreService:
        return VectorStoreService(qdrant_repo=qdrant_repo, settings=settings)

    @provide(scope=Scope.APP)
    def metadata_filling_service(
        self,
        topics_repo: TopicsRepo,
        llm_client_no_tools: LLMWithoutToolsType,
    ) -> MetadataFillingService:
        return MetadataFillingService(
            topics_repo=topics_repo, llm_client=llm_client_no_tools
        )

    @provide(scope=Scope.APP)
    def evaluation_service(
        self,
        llm_client_no_tools: LLMWithoutToolsType,
        embeddings_model: HuggingFaceEmbeddings,
        cross_encoder_model: CrossEncoder,
    ) -> EvaluationService:
        return EvaluationService(
            llm_client=llm_client_no_tools,
            embed_model=embeddings_model,
            cross_encoder_model=cross_encoder_model,
        )


class IngestProvider(Provider):
    @provide(scope=Scope.APP)
    def qdrant_repo(self, settings: Settings) -> QdrantRepo:
        return QdrantRepo(
            client=AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            ),
            settings=settings,
        )

    @provide(scope=Scope.APP)
    def web_loader_node(self, service: WebLoaderService) -> WebLoaderNode:
        return WebLoaderNode(service)

    @provide(scope=Scope.APP)
    def chunking_node(self, service: ChunkingService) -> SemanticChunkingNode:
        return SemanticChunkingNode(service)

    @provide(scope=Scope.APP)
    def embedding_node(self, service: EmbeddingService) -> EmbeddingNode:
        return EmbeddingNode(service)

    @provide(scope=Scope.APP)
    def qdrant_ingest_node(self, service: VectorStoreService) -> QdrantIngestNode:
        return QdrantIngestNode(service)

    @provide(scope=Scope.APP)
    def metadata_filling_node(
        self,
        service: MetadataFillingService,
    ) -> MetadataFillingNode:
        return MetadataFillingNode(service)

    @provide(scope=Scope.APP)
    def ingest_graph(
        self,
        web_loader_node: WebLoaderNode,
        chunking_node: SemanticChunkingNode,
        metadata_filling_node: MetadataFillingNode,
        embedding_node: EmbeddingNode,
        qdrant_ingest_node: QdrantIngestNode,
    ) -> IngestGraph:
        "loader -> chunking -> metadata filling -> embedding -> vector db"
        graph = StateGraph(IngestState)
        graph.add_node("loader", web_loader_node.execute)
        graph.add_node("chunking", chunking_node.execute)
        graph.add_node("embedding", embedding_node.execute)
        graph.add_node("qdrant", qdrant_ingest_node.execute)
        graph.add_node("metadata", metadata_filling_node.execute)

        graph.add_edge(START, "loader")
        graph.add_edge("loader", "chunking")
        graph.add_edge("chunking", "metadata")
        graph.add_edge("metadata", "embedding")
        graph.add_edge("embedding", "qdrant")
        graph.add_edge("qdrant", END)
        return graph.compile()


providers = (
    AppProvider(),
    DBProvider(),
    EmbeddingsProvider(),
    ServicesProvider(),
    LLMProvider(),
    IngestProvider(),
    ToolsProvider(),
)
container = make_container(*providers)
