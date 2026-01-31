from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.integration import SQS
from diagrams.aws.network import APIGateway
from pydantic import BaseModel


class Node(BaseModel):
    id: str
    type: str
    name: str


class Edge(BaseModel):
    from_: str
    to_: str


class Cluster(BaseModel):
    node_ids: list[str]
    name: str


class Diagram(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    clusters: list[Cluster]


DIAGRAM_MAPPING = {"service": EC2, "gateway": APIGateway, "queue": SQS, "database": RDS}

diagram_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagram_definition",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique node identifier (UUID)",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["service", "gateway", "queue", "database"],
                                "description": "Node type",
                            },
                            "name": {
                                "type": "string",
                                "description": "Human-readable node name",
                            },
                        },
                        "required": ["id", "type", "name"],
                        "additionalProperties": False,
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_": {
                                "type": "string",
                                "description": "Source node id",
                            },
                            "to_": {"type": "string", "description": "Target node id"},
                        },
                        "required": ["from_", "to_"],
                        "additionalProperties": False,
                    },
                },
                "clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "IDs of nodes inside the cluster",
                            },
                            "name": {"type": "string", "description": "Cluster name"},
                        },
                        "required": ["node_ids", "name"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["nodes", "edges", "clusters"],
            "additionalProperties": False,
        },
    },
}
