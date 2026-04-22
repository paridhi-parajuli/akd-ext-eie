"""EIE (Earth Information Explorer) Agent for earth science data exploration.

This module implements the EIE Agent that orchestrates earth science data
discovery, statistics, and visualization using tools deployed on FastMCP cloud.

Public API:
    EIEAgent, EIEAgentInputSchema, EIEAgentOutputSchema, EIEAgentConfig
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from agents import function_tool
from pydantic import Field

from akd_ext.tools.get_place import GetPlaceTool, GetPlaceToolInputSchema
from akd_ext.tools.set_datetime import SetDatetimeTool, SetDatetimeToolInputSchema
from akd_ext.tools.collections_rag import CollectionsRAGTool, CollectionsRAGToolInputSchema
from akd_ext.tools.stac_search import STACSearchTool, STACSearchToolInputSchema
from akd_ext.tools.stats import StatsTool, StatsToolInputSchema
from akd_ext.tools.viz import VizTool, VizToolInputSchema
from akd_ext.tools.utils import is_cmr_backed

from akd._base import (
    InputSchema,
    OutputSchema,
    TextOutput,
    RunContext,
)
from akd._base.streaming import StreamEvent
from akd_ext.agents._base import (
    OpenAIBaseAgent,
    OpenAIBaseAgentConfig,
)

from loguru import logger

# -----------------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------------

EIE_AGENT_SYSTEM_PROMPT = """
ROLE
You are Earth Information Explorer (EIE), a server-side backend Earth science dataset discovery and descriptive statistics agent for NASA's VEDA STAC catalog.

You can ONLY help with earth science DATA discovery and analysis using your tools. You must NOT answer general knowledge questions, explain scientific concepts, or provide background information — even if the topic is related to earth science. If the user asks anything that doesn't require calling your tools (e.g., "what are greenhouse gases?", "explain climate change"), politely say it's outside your scope and redirect them to dataset discovery.

You are:
- Strictly sequential (for analysis tasks)
- Human-confirmation gated at critical decision points
- Fail-fast, with one controlled exception at the stats stage (partial per-item tolerance)
- Deterministic, reproducible in regards to tool outputs, non-speculative
- Single-collection only (v1) and per-item independent analysis
- Not predictive and not interpretive (no causality, no policy advice, no severity labeling)

OBJECTIVE
The agent may support lightweight dataset discovery queries (e.g., "What datasets exist for NO2 over California?") without requiring datetime or AOI inputs. In such cases, the agent may return collection candidates before enforcing the full execution pipeline.

Given a user's natural-language Earth science question, help them:
1. Parse and confirm datetime range (ISO8601 interval).
2. Geocode and confirm AOI.
3. Retrieve Top-K (default 5) candidate VEDA collections via semantic similarity.
4. Require human dataset and variable selection (no auto-selection).
5. Run STAC item search for the chosen collection, confirmed datetime, confirmed AOI.
6. Compute per-item descriptive statistics (at least mean/min/max, plus any additional returned stats).
7. Generate mandatory visualization tile URLs for each successful item.
8. Return a response strictly conforming to the EIEAgentOutput output schema with auditability, provenance when available, and explicit unknowns.

CONTEXT & INPUTS

Primary users: Beginner (public), intermediate (students/decision makers), advanced (researchers). Adapt explanation depth, but never alter the execution rules.

The current date and time is {now}.

TOOLS (authoritative)
You must orchestrate tools in this mandatory order:

- set_datetime(value): Validate and set a datetime range.
  Input: ISO-8601 range string (YYYY-MM-DD/YYYY-MM-DD). LLM converts user temporal phrasing before calling (e.g., "Oct to Dec 2021" → "2021-10-01/2021-12-31"). Use current date for relative time parsing ("last month", "summer 2022").
  The tool validates the format:
    - If invalid → returns error message; LLM should adjust and retry
    - If valid → triggers user confirmation prompt
  Output: Validation result and pending confirmation state.
  A valid interval is required for STAC item retrieval and downstream analysis.
  If the user intent is dataset discovery only, datetime may be omitted until analysis begins.

- get_place(query): Resolve a place name to bbox AND GeoJSON geometry via GeoDini.
  Input: place query string
  Output: PlaceResult with fields: place, bbox, geometry, error.
  Call this when spatial filtering is required for STAC search.
  For dataset discovery queries, AOI may be deferred.
  The user will be prompted to confirm the location automatically.

- collections_rag(query): Search top-K STAC collections via semantic similarity (RAG).
  Input: query (data description like 'NO2 air quality')
  Output: Top-K (default 5) collections with:
    - cosine_similarity
    - cosine_distance
    - spatial_overlap (bool)
    - temporal_overlap (bool)
    - is_cmr_backed (bool)
    - available_variables (list, for CMR collections)
  The user will be prompted to select a collection automatically (and a variable, if the collection is CMR-backed with multiple variables).

- stac_search(): Search STAC catalog for COG items.
  Reads selected_collection_id from state. No arguments needed.
  Skip for CMR-backed collections (is_cmr_backed=true). Only required for VEDA COG collections before stats/viz.

- stats(): Fetch raster statistics.
  Per-band stats (b1, b2, …). Do not assume band meaning.
  For VEDA COG, requires stac_search first. For CMR, call directly after collection and variable selection.
  Reads selected_collection_id and selected_variable from state. No arguments needed.

- viz(): Get raster tile URLs.
  Uses STAC render metadata OR stats min/max. Warn if missing metadata.
  For VEDA COG, requires stac_search first. For CMR, call directly after collection and variable selection.
  Reads selected_collection_id and selected_variable from state. No arguments needed.

INTERNAL STATE
- datetime_range
- place_result
- collections_result
- selected_collection_id
- selected_variable
- stac_result
- stats_result
- viz_result
- pending_confirmation (used for confirmation flow)

CONSTRAINTS & STYLE RULES

Human-controlled decisions (must confirm):
- Datetime (when doing analysis)
- AOI (when used)
- Dataset selection
- Variable selection (for CMR collections with multiple variables)

Tool Batching Rules:
- set_datetime, get_place, and collections_rag each trigger an automatic user confirmation step.
- Call only ONE of these confirmation-triggering tools per turn — do not batch them together.
- After each one returns, wait for the next turn before calling the next tool.
- stats and viz do NOT trigger confirmations, so they CAN be called together in the same turn.

Output Suppression Rules:
- After receiving stats results, DO NOT print any statistics values, summaries, interpretations, or analysis. Instead, simply say "Statistics retrieved." and STOP.
- After receiving viz results, DO NOT print the tile URLs. Instead, simply say "Visualization layers generated." and STOP.
- When both stats and viz results are returned in the same turn, say "Statistics retrieved. Visualization layers generated." and STOP.
- Do not use background knowledge, only use the tools to answer questions.

Non-goals (you must never):
- Perform forecasting
- Do multi-dataset comparisons
- Infer missing metadata
- Provide causal/policy conclusions

Allowed:
- Surfacing forecast datasets (e.g., CMIP)

Scope:
- Only datasets hosted in NASA VEDA STAC and CMR-backed collections exposed through VEDA

Statistical Guardrails:
- Flag nodata values
- Warn on low valid_percent

Failure Handling:
- If datetime fails: Analysis → halt; Discovery → continue
- If geocoding fails: Analysis → halt; Discovery → continue

Communication Style:
- Use "semantic similarity" not "relevance" when describing collection matches
- Use "match strength" (High/Moderate/Weak) based on cosine_similarity
- Be transparent about limitations and unknowns

PROCESS
You must follow this canonical execution flow for analysis tasks.
However, for dataset discovery queries (when the user is exploring datasets without requesting statistics), you may begin at the Collection Discovery step before enforcing datetime or AOI steps.
Once the user proceeds to analysis, all required prior steps must be completed before continuing.

0. Query interpretation
   Extract: topic, time (if any), place (if any), collection_id (if any)
   If the user provides a collection_id directly, call select_collection(collection_id=...) immediately to record it.
   If discovery intent → skip to Collection Discovery

1. Datetime gate (analysis only)
   Call set_datetime()
   User confirms

2. AOI gate (analysis only)
   Call get_place() if needed
   User confirms

3. Collection discovery
   Call collections_rag()
   Return Top-K with: id, title, description, cosine_similarity, cosine_distance, match strength, spatial_overlap, temporal_overlap
   After collections_rag returns, check whether any results are actually relevant to the user's request. If none of the returned collections relate to what the user asked for (e.g. the user asked about "social vulnerability" but only precipitation or air quality data was found), STOP and explain that no relevant datasets were found. Do NOT proceed to stats or viz with an irrelevant collection.
   If no collections have both spatial and temporal overlap, explain which matched thematically but why they don't cover the requested extent.

4. Dataset selection
   User must choose. When the user selects a collection, you MUST call select_collection(collection_id=...) to record it.
   For CMR collections with multiple variables, also pass selected_variable to select_collection.

5. STAC search
   If 0 items → halt and explain no data available
   Skip for CMR-backed collections (is_cmr_backed=true).

6. Stats
   Per item and band (if multiple)
   Handle partial failures

7. Visualization
   Generate tile URLs

8. Output assembly
  Return compiled output meeting the EIEAgentOutput schema


EXTENSIBILITY NOTE
New tools must:
- Not bypass confirmation gates
- Preserve auditability
- Follow schema
"""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _default_tool_state() -> dict[str, Any]:
    """Return a fresh tool state dict mirroring EIEState from eie-llm-backend."""
    return {
        "datetime_range": None,
        "place_result": None,           # GetPlaceToolOutputSchema (serialized as dict when persisted)
        "collections_result": None,     # CollectionsRAGToolOutputSchema (serialized as dict when persisted)
        "stac_result": None,            # STACSearchToolOutputSchema (serialized as dict when persisted)
        "stats_result": None,           # StatsToolOutputSchema
        "viz_result": None,             # VizToolOutputSchema
        "selected_collection_id": None,
        "selected_variable": None,
        "collection_metadata": None,    # full STAC JSON
    }


def _make_local_tools(state: dict[str, Any]) -> list:
    """Create all 6 EIE tools locally with shared state.

    Mirrors the eie-llm-backend's LangGraph pattern: tools read/write to
    shared state so the LLM only passes minimal arguments (query, place name,
    collection selection).  Large data (geometry, items, collection metadata)
    flows through state and never bloats the LLM context.

    Args:
        state: Mutable dict shared across all tools. Passed in by the agent
               so it can be serialized/restored between stateless API requests.
    """
    _state = state

    # Reuse existing tool instances
    _datetime_tool = SetDatetimeTool()
    _place_tool = GetPlaceTool()
    _rag_tool = CollectionsRAGTool()
    _stac_tool = STACSearchTool()
    _stats_tool = StatsTool()
    _viz_tool = VizTool()

    # ── set_datetime_tool ───────────────────────────────────────────

    @function_tool(name_override="set_datetime_tool")
    async def set_datetime_tool(value: str) -> str:
        """Validate and normalize an ISO-8601 datetime range.

        Args:
            value: ISO-8601 range 'YYYY-MM-DD/YYYY-MM-DD'
        """
        result = await _datetime_tool._arun(SetDatetimeToolInputSchema(value=value))
        if result.datetime:
            _state["datetime_range"] = result.datetime
        return result.model_dump_json()

    # ── get_place_tool ──────────────────────────────────────────────

    @function_tool(name_override="get_place_tool")
    async def get_place_tool(query: str) -> str:
        """Resolve a place name to a bounding box and geometry via geocoding.

        Args:
            query: A place name or location (e.g. 'California', 'Houston TX')
        """
        result = await _place_tool._arun(GetPlaceToolInputSchema(query=query))

        # Store as dict so state is always plain dicts (works after restore from DB)
        _state["place_result"] = result.model_dump()

        # Replace full geometry with bbox polygon for the LLM (~100 tokens instead of 10K+)
        bbox_geom = None
        if result.bbox:
            w, s, e, n = result.bbox
            bbox_geom = {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            }

        return json.dumps({
            "place": result.place,
            "bbox": result.bbox,
            "geometry": bbox_geom,
            "error": result.error,
        })

    # ── collections_rag_tool ────────────────────────────────────────

    @function_tool(name_override="collections_rag_tool")
    async def collections_rag_tool(query: str, top_k: int = 5) -> str:
        """Search for relevant STAC collections using semantic search.
        Reads bbox and datetime_range from previous tool calls automatically.

        Args:
            query: Data description (e.g. 'NO2 air quality', 'methane emissions')
            top_k: Number of results to return (default 5)
        """
        place_result = _state.get("place_result") or {}
        result = await _rag_tool._arun(CollectionsRAGToolInputSchema(
            query=query,
            top_k=top_k,
            bbox=place_result.get("bbox"),
            datetime_range=_state.get("datetime_range"),
        ))
        _state["collections_result"] = result.model_dump()
        return result.model_dump_json()

    # ── select_collection ───────────────────────────────────────────

    @function_tool(name_override="select_collection")
    def select_collection(collection_id: str, selected_variable: str | None = None) -> str:
        """Record the user's collection and (for CMR) variable selection.
        Call this when the user picks a collection from collections_rag results or explictly mentions it.

        Args:
            collection_id: The collection ID chosen by the user
            selected_variable: Variable name for CMR collections with multiple variables (for CMR)
        """
        # Validate collection_id against known matches
        collections_result = _state.get("collections_result") or {}
        matches = collections_result.get("matches", [])
        valid_ids = [m.get("id") for m in matches]

        if collection_id not in valid_ids:
            return json.dumps({
                "error": f"Invalid collection '{collection_id}'. Valid options: {valid_ids}",
                "selected_collection_id": None,
                "selected_variable": None,
            })

        _state["selected_collection_id"] = collection_id
        if selected_variable:
            _state["selected_variable"] = selected_variable

        # Look up collection_metadata from rag results
        for m in matches:
            if m.get("id") == collection_id:
                _state["collection_metadata"] = m.get("collection_metadata")
                # Validate selected_variable for CMR collections
                available_vars = m.get("available_variables") or []
                if selected_variable and available_vars and selected_variable not in available_vars:
                    return json.dumps({
                        "error": f"Invalid variable '{selected_variable}'. Valid options: {available_vars}",
                        "selected_collection_id": collection_id,
                        "selected_variable": None,
                    })
                break

        return json.dumps({
            "selected_collection_id": collection_id,
            "selected_variable": selected_variable,
        })

    # ── stac_search_tool ────────────────────────────────────────────

    @function_tool(name_override="stac_search_tool")
    async def stac_search_tool(collection_id: str, limit: int = 15) -> str:
        """Search STAC catalog for COG items. Reads bbox and datetime from state.

        Args:
            collection_id: The collection ID to search (from collections_rag results)
            limit: Maximum number of items to return (default 15)
        """
        place_result = _state.get("place_result") or {}
        datetime_range = _state.get("datetime_range")
        if not place_result.get("bbox"):
            return json.dumps({"error": "No bbox — run get_place_tool first"})
        if not datetime_range:
            return json.dumps({"error": "No datetime — run set_datetime_tool first"})

        result = await _stac_tool._arun(STACSearchToolInputSchema(
            collections=[collection_id],
            bbox=place_result["bbox"],
            datetime=datetime_range,
            limit=limit,
        ))

        _state["stac_result"] = result.model_dump()
        return result.model_dump_json()

    # ── stats_tool ──────────────────────────────────────────────────

    @function_tool(name_override="stats_tool")
    async def stats_tool() -> str:
        """Fetch raster zonal statistics. Reads geometry, items, and collection info from state.
        For CMR collections, selected_variable must be set via select_collection first.
        """
        place_result = _state.get("place_result") or {}
        geometry = place_result.get("geometry")
        if not geometry:
            return json.dumps({"error": "No geometry — run get_place_tool first"})

        stac_result = _state.get("stac_result") or {}
        raw_items = stac_result.get("items", [])
        items = (
            [{"url": item.get("asset_url"), "id": item.get("id"), "datetime": item.get("datetime")}
             for item in raw_items if item.get("asset_url")]
            if raw_items else None
        )

        # Only pass collection_metadata for CMR collections — VEDA uses items path
        collection_metadata = _state.get("collection_metadata")
        if collection_metadata and not is_cmr_backed(collection_metadata):
            collection_metadata = None

        result = await _stats_tool._arun(StatsToolInputSchema(
            geometry=geometry,
            items=items,
            collection_id=_state.get("selected_collection_id"),
            collection_metadata=collection_metadata,
            datetime_range=_state.get("datetime_range"),
            selected_variable=_state.get("selected_variable"),
        ))
        _state["stats_result"] = result.model_dump()
        return result.model_dump_json()

    # ── viz_tool ────────────────────────────────────────────────────

    @function_tool(name_override="viz_tool")
    async def viz_tool() -> str:
        """Build raster tile URLs for visualization. Reads items and collection info from state.
        For CMR collections, selected_variable must be set via select_collection first.
        """
        stac_result = _state.get("stac_result") or {}
        raw_items = stac_result.get("items", [])
        items = (
            [{"url": item.get("asset_url"), "id": item.get("id"), "datetime": item.get("datetime")}
             for item in raw_items if item.get("asset_url")]
            if raw_items else None
        )

        # Only pass collection_metadata for CMR collections
        collection_metadata = _state.get("collection_metadata")
        if collection_metadata and not is_cmr_backed(collection_metadata):
            collection_metadata = None

        result = await _viz_tool._arun(VizToolInputSchema(
            items=items,
            collection_id=_state.get("selected_collection_id"),
            collection_metadata=collection_metadata,
            datetime_range=_state.get("datetime_range"),
            selected_variable=_state.get("selected_variable"),
        ))
        _state["viz_result"] = result.model_dump()
        return result.model_dump_json()

    return [set_datetime_tool, get_place_tool, collections_rag_tool, select_collection, stac_search_tool, stats_tool, viz_tool]


class EIEAgentConfig(OpenAIBaseAgentConfig):
    """Configuration for EIE Agent."""

    description: str = Field(
        default=(
            "Earth science data exploration agent that helps users discover STAC collections, "
            "fetch raster statistics, and generate visualization tile URLs using VEDA and CMR data."
        )
    )
    system_prompt: str = Field(default=EIE_AGENT_SYSTEM_PROMPT)
    model_name: str = Field(default="gpt-5.2")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(default=None)
    tools: list[Any] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Input / Output Schemas
# -----------------------------------------------------------------------------


class EIEAgentInputSchema(InputSchema):
    """Input schema for EIE Agent."""

    query: str = Field(..., description="Earth science data question or task")


class EIEAgentOutputSchema(OutputSchema):
    """Output schema for EIE Agent."""

    __response_field__ = "result"
    result: str = Field(..., description="Response with data discovery results, statistics, or visualization info")


# -----------------------------------------------------------------------------
# EIE Agent
# -----------------------------------------------------------------------------


class EIEAgent(OpenAIBaseAgent[EIEAgentInputSchema, TextOutput]):
    """Earth Information Explorer Agent for earth science data exploration.

    Orchestrates STAC collection discovery, raster statistics, and
    visualization using tools deployed on FastMCP cloud.

    Tool state (geometry, items, collection metadata, etc.) is stored on
    ``run_context.tool_state`` — an extra field that piggybacks on RunContext
    persistence but never reaches the LLM (only ``messages`` are sent).
    The caller just saves/restores RunContext as usual; tool state comes along
    automatically.
    """

    input_schema = EIEAgentInputSchema
    output_schema = TextOutput #| EIEAgentOutputSchema 
    config_schema = EIEAgentConfig

    def __init__(
        self,
        config: EIEAgentConfig | None = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self._tool_state: dict[str, Any] = _default_tool_state()
        config = config or EIEAgentConfig()
        if not config.tools:
            config.tools = _make_local_tools(self._tool_state)
        super().__init__(config=config, debug=debug, **kwargs)

    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream with auto-sync of tool state to/from run_context.tool_state."""
        # Restore tool state from run_context (if resuming a session)
        if run_context and hasattr(run_context, "tool_state") and run_context.tool_state:
            self._tool_state.update(run_context.tool_state)

        async for event in super().astream(params=params, run_context=run_context, **kwargs):
            # Save tool state back to run_context after every event
            event.run_context.tool_state = dict(self._tool_state)
            yield event


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = EIEAgent(EIEAgentConfig(debug=True))
        logger.info(f"Agent description: {agent.description}")
        question = "Show me NO2 air quality data for Washington DC from January to June 2020"

        async for event in agent.astream(EIEAgentInputSchema(query=question)):
            logger.info(event)

    asyncio.run(main())
