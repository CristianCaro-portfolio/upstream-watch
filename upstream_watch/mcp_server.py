"""MCP server (Python SDK v2, MCPServer) so any MCP client can query the gauge graph.

Run over stdio:  python -m upstream_watch mcp
Tools are read only. Brief approval stays in the LangGraph flow, where a person signs it off.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .agent import ask
from .config import get_settings
from .pipeline import agent_from_settings


def build_server(sample: bool = False, embedder_name: str | None = None) -> MCPServer:
    agent, kit, _, _ = agent_from_settings(get_settings(), sample=sample, embedder_name=embedder_name)
    server = MCPServer(
        "upstream-watch",
        instructions=(
            "River gauge knowledge graph for the Potomac basin above Washington DC. "
            "Gauges are NWS location ids such as BRKM2 (Little Falls) or PORM2 (Point of Rocks)."
        ),
    )

    def resolve(gauge: str) -> str:
        if kit.graph.gauge(gauge.upper()):
            return gauge.upper()
        linked = kit.linker.link(gauge)
        if not linked:
            raise ValueError(f"unknown gauge: {gauge}")
        return linked[0]

    @server.tool(description="Flood stages, record crest and location for a gauge (id or name).")
    def gauge_info(gauge: str) -> dict:
        info = kit.graph.gauge(resolve(gauge))
        return {k: v for k, v in info.items() if not k.startswith("_")}

    @server.tool(description="Gauges downstream of a gauge with typical crest travel time in hours.")
    def downstream_eta(gauge: str, max_hops: int = 4) -> list[dict]:
        hops = kit.graph.downstream(resolve(gauge), max_hops=max(1, min(max_hops, 12)))
        return [h.__dict__ for h in hops]

    @server.tool(description="NWS impact statements reached at a gauge for a given stage in feet.")
    def impacts_at(gauge: str, stage_ft: float, limit: int = 3) -> list[dict]:
        return kit.graph.impacts_at(resolve(gauge), stage_ft, limit=max(1, min(limit, 10)))

    @server.tool(description="Ask a question about the gauge network. Answers cite their sources.")
    def ask_question(question: str) -> dict:
        state = ask(agent, question, mode="graph", thread_id=f"mcp-{hash(question)}")
        return {"answer": state["answer"], "sources": state.get("hits", []), "grounded": state.get("grounded")}

    return server


def main() -> None:  # pragma: no cover - blocking stdio loop
    build_server().run("stdio")
