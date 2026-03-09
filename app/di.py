from dishka import Provider, Scope, make_container, provide
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from llama_index.readers.web import TrafilaturaWebReader
from qdrant_client import QdrantClient, AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.application.agent.router import AgentRouter
from app.application.ingest.router import IngestRouter
from app.application.topics.router import TopicRouter
from app.infra.config import Settings
from app.infra.db.repos.topics import TopicsRepo
from app.infra.llm.client import LLMClient
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.handlers.topic import CreateTopicHandler
from app.logic.nodes.chunking import SemanticChunkingNode
from app.logic.nodes.embeddings import EmbeddingNode

from app.logic.nodes.llm_node import LLMNode
from app.logic.nodes.loaders import WebLoaderNode
from app.logic.nodes.qdrant import QdrantIngestNode
from app.logic.nodes.state import MessagesState, IngestState

from app.logic.tools.rag import RAGTools
from app.main import AppBuilder

from typing import NewType

AgentGraph = NewType("AgentGraph", CompiledStateGraph)
IngestGraph = NewType("IngestGraph", CompiledStateGraph)
RAGToolsType = NewType("RAGToolsType", list)
ToolsType = NewType("ToolsType", list)

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
    ) -> AppBuilder:
        topic_router = TopicRouter(create_topic_handler=CreateTopicHandler(topic_repo=topics_repo))
        agent_router = AgentRouter(graph_agent=agent_graph)
        ingest_router = IngestRouter(ingest_graph=ingest_graph)
        return AppBuilder(routers=[agent_router.router, ingest_router.router, topic_router.router], settings=settings)

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


class ToolsProvider(Provider):
    @provide(scope=Scope.APP)
    def rag_tools(self, qdrant_repo: QdrantRepo) -> RAGToolsType:
        rag = RAGTools(qdrant_repo=qdrant_repo, embed_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        return [rag.search_docs]

    @provide(scope=Scope.APP)
    def tools(
        self,
        rag_tools: RAGToolsType,
    ) -> ToolsType:
        return rag_tools

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
    def llm_node(self, llm_client: LLMClient) -> LLMNode:
        return LLMNode(llm_client)



    @provide(scope=Scope.APP)
    def graph_agent(
        self,
        llm_node: LLMNode,
        tools: ToolsType,
    ) -> AgentGraph:
        graph = StateGraph(MessagesState)
        graph.add_node("llm_call", llm_node.execute)

        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)

        def should_continue(state: MessagesState) -> str:
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", should_continue)
        graph.add_edge("tools", "llm_call")

        app = graph.compile()
        return app





class IngestProvider(Provider):

    @provide(scope=Scope.APP)
    def qdrant_repo(self, settings: Settings) -> QdrantRepo:
        return QdrantRepo(
            client=AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            ),
            settings=settings
        )

    @provide(scope=Scope.APP)
    def web_loader_node(self) -> WebLoaderNode:
        return WebLoaderNode(TrafilaturaWebReader())

    @provide(scope=Scope.APP)
    def chunking_node(self) -> SemanticChunkingNode:
        return SemanticChunkingNode()

    @provide(scope=Scope.APP)
    def embedding_node(self) -> EmbeddingNode:
        return EmbeddingNode(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    @provide(scope=Scope.APP)
    def qdrant_ingest_node(self, qdrant_repo: QdrantRepo) -> QdrantIngestNode:
        return QdrantIngestNode(qdrant_repo=qdrant_repo)

    @provide(scope=Scope.APP)
    def ingest_graph(
        self,
        web_loader_node: WebLoaderNode,
        chunking_node: SemanticChunkingNode,
        embedding_node: EmbeddingNode,
        qdrant_ingest_node: QdrantIngestNode,
    ) -> IngestGraph:
        "loader -> chunking -> embedding -> vector db"
        graph = StateGraph(IngestState)
        graph.add_node("loader", web_loader_node.execute)
        graph.add_node("chunking", chunking_node.execute)
        graph.add_node("embedding", embedding_node.execute)
        graph.add_node("qdrant", qdrant_ingest_node.execute)

        graph.add_edge(START, "loader")
        graph.add_edge("loader", "chunking")
        graph.add_edge("chunking", "embedding")
        graph.add_edge("embedding", "qdrant")
        graph.add_edge("qdrant", END)
        return graph.compile()



providers = (
    AppProvider(),
    DBProvider(),
    LLMProvider(),
    IngestProvider(),
    ToolsProvider(),
)
container = make_container(*providers)
