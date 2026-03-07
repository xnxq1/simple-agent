from dishka import Provider, Scope, make_container, provide
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from openai import AsyncOpenAI

from app.application.agent.router import AgentRouter
from app.infra.config import Settings
from app.infra.llm.client import LLMClient

from app.logic.nodes.llm_node import LLMNode
from app.logic.nodes.state import MessagesState
from app.logic.tools.test import tools
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
        return LLMClient(client=client, settings=settings, tools=tools)

    @provide(scope=Scope.APP)
    def llm_node(self, llm_client: LLMClient) -> LLMNode:
        return LLMNode(llm_client)



    @provide(scope=Scope.APP)
    def graph_agent(
        self,
        llm_node: LLMNode,
    ) -> CompiledStateGraph:
        graph = StateGraph(MessagesState)
        graph.add_node("llm_call", llm_node.execute)

        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)

        def should_continue(state: MessagesState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", should_continue)
        graph.add_edge("tools", "llm_call")

        app = graph.compile()
        return app


providers = (
    AppProvider(),
    LLMProvider(),
)
container = make_container(*providers)
