from dishka import Provider, Scope, make_container, provide
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from llama_index.readers.web import TrafilaturaWebReader
from qdrant_client import QdrantClient, AsyncQdrantClient

from app.application.agent.router import AgentRouter
from app.application.ingest.router import IngestRouter
from app.infra.config import Settings
from app.infra.llm.client import LLMClient
from app.infra.qdrant.repos.repos import QdrantRepo

from app.logic.nodes.llm_node import LLMNode
from app.logic.nodes.loaders import WebLoaderNode
from app.logic.nodes.state import MessagesState, IngestState
from app.logic.tools.test import tools
from app.main import AppBuilder

from typing import NewType

AgentGraph = NewType("AgentGraph", CompiledStateGraph)
IngestGraph = NewType("IngestGraph", CompiledStateGraph)

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
    ) -> AppBuilder:
        agent_router = AgentRouter(graph_agent=agent_graph)
        agent_router.register_routes()
        ingest_router = IngestRouter(ingest_graph=ingest_graph)
        ingest_router.register_routes()
        return AppBuilder(routers=[agent_router.router, ingest_router.router], settings=settings)


class LLMProvider(Provider):
    @provide(scope=Scope.APP)
    def llm_client(self, settings: Settings) -> LLMClient:
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
    def ingest_graph(
        self,
        web_loader_node: WebLoaderNode,
    ) -> IngestGraph:
        "loader -> chunking -> embedding -> vector db"
        graph = StateGraph(IngestState)
        graph.add_node("ingest", web_loader_node.execute)

        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", END)
        return graph.compile()



providers = (
    AppProvider(),
    LLMProvider(),
    IngestProvider(),
)
container = make_container(*providers)
