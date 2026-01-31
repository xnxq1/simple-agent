from dishka import Provider, Scope, make_container, provide
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI

from app.application.agent.router import AgentRouter
from app.infra.config import Settings
from app.infra.llm.client import LLMClient
from app.logic.nodes.agent import AgentNode
from app.logic.nodes.diagram import CreateDiagramNode, RecognizeDiagramNode
from app.logic.nodes.state import StateSchema
from app.main import AppBuilder


class AppProvider(Provider):
    @provide(scope=Scope.APP)
    def settings(self) -> Settings:
        return Settings()

    @provide(scope=Scope.APP)
    def app_builder(self, settings: Settings, graph: CompiledStateGraph) -> AppBuilder:
        agent_router = AgentRouter(graph_agent=graph)
        agent_router.register_routes()
        return AppBuilder(routers=[agent_router.router], settings=settings)


class LLMProvider(Provider):
    @provide(scope=Scope.APP)
    def create_openai_client(self, settings: Settings) -> AsyncOpenAI:
        client = AsyncOpenAI(
            api_key=settings.open_ai_api_key,
            base_url=settings.open_ai_base_url,
        )

        return client

    @provide(scope=Scope.APP)
    def llm_client(self, client: AsyncOpenAI, settings: Settings) -> LLMClient:
        return LLMClient(client=client, settings=settings)

    @provide(scope=Scope.APP)
    def agent_node(self, llm_client: LLMClient) -> AgentNode:
        return AgentNode(llm_client)

    @provide(scope=Scope.APP)
    def recognize_diagram_mode(self, llm_client: LLMClient) -> RecognizeDiagramNode:
        return RecognizeDiagramNode(llm_client)

    @provide(scope=Scope.APP)
    def create_diagram_mode(self) -> CreateDiagramNode:
        return CreateDiagramNode()

    @provide(scope=Scope.APP)
    def graph_agent(
        self,
        recognize_diagram_node: RecognizeDiagramNode,
        create_diagram_node: CreateDiagramNode,
    ) -> CompiledStateGraph:
        graph = StateGraph(StateSchema)
        graph.add_node("recognize_diagram", recognize_diagram_node.execute)
        graph.add_node("create_diagram", create_diagram_node.execute)

        graph.add_edge(START, "recognize_diagram")
        graph.add_edge("recognize_diagram", "create_diagram")
        graph.add_edge("create_diagram", END)
        app = graph.compile()

        return app


providers = (
    AppProvider(),
    LLMProvider(),
)
container = make_container(*providers)
