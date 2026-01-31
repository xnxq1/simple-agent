import os
import tempfile

from diagrams import Cluster
from diagrams import Diagram as DiagramCtx

from app.infra.llm.client import LLMClient
from app.infra.models import DIAGRAM_MAPPING, Diagram, diagram_response_format
from app.logic.nodes.state import StateSchema


class RecognizeDiagramNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.sys_prompt = """
            Convert this description into a JSON diagram with nodes, edges and clusters.
            Supported node types: {entities}.
            Only return valid JSON. id create in UUID format
            Example JSON:
            {{
              "nodes": [
                {{"id": "2a7acee7-6523-4f4d-a572-729ce86ce243", "type": "service", "name": "Auth Service"}},
                {{"id": "a573fb29-6e79-4188-8b56-dc69285a53e4", "type": "gateway", "name": "API Gateway"}},
                {{"id": "79760297-3f04-4bb7-acfb-683fd9c62ab2", "type": "queue", "name": "SQS Queue"}},
                {{"id": "8fad4bc1-4a47-4f24-83df-16393c32aeab", "type": "database", "name": "Shared RDS"}}
              ],
              "edges": [
                {{"from_": "2a7acee7-6523-4f4d-a572-729ce86ce243", "to_": "a573fb29-6e79-4188-8b56-dc69285a53e4"}},
                {{"from_": "79760297-3f04-4bb7-acfb-683fd9c62ab2", "to_": "a573fb29-6e79-4188-8b56-dc69285a53e4"}}
              ],
              "clusters: [
                {{"node_ids": ["2a7acee7-6523-4f4d-a572-729ce86ce243", "a573fb29-6e79-4188-8b56-dc69285a53e4"], "name": "Mega cluster"}}
              ]
            }}
            
            """.format(entities=list(DIAGRAM_MAPPING.keys()))
        self.user_prompt = """
        query: {query}
        answer: 
        """

    async def execute(self, state: StateSchema) -> dict:
        res: Diagram = await self.llm.completions_create(
            system_prompt=self.sys_prompt,
            user_query=self.user_prompt.format(query=state.query),
            response_format=diagram_response_format,
            response_class=Diagram,
        )
        return {"result": res}


class CreateDiagramNode:
    async def execute(self, state: StateSchema) -> dict:
        res: Diagram = state.result
        node_map = {node.id: node for node in res.nodes}
        diagram_map = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "diagram")

        with DiagramCtx("Generated Diagram", filename=path, show=False, direction="TB"):
            for cluster in res.clusters:
                with Cluster(cluster.name):
                    for node_id in cluster.node_ids:
                        node = node_map[node_id]
                        model = DIAGRAM_MAPPING[node.type]
                        diagram_map[node.id] = model(node.name)

            for node in res.nodes:
                if node.id not in diagram_map:
                    model = DIAGRAM_MAPPING[node.type]
                    diagram_map[node.id] = model(node.name)

            for edge in res.edges:
                from_ = diagram_map[edge.from_]
                to_ = diagram_map[edge.to_]
                from_ >> to_

        final_path = f"{path}.png"
        with open(final_path, "rb") as f:
            png_bytes = f.read()

        return {"image": png_bytes}
