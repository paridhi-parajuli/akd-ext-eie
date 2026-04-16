"""EIE (Earth Intelligence Engine) Agent for earth science data exploration.

This module implements the EIE Agent that orchestrates earth science data
discovery, statistics, and visualization using tools deployed on FastMCP cloud.

Public API:
    EIEAgent, EIEAgentInputSchema, EIEAgentOutputSchema, EIEAgentConfig
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from agents import HostedMCPTool, function_tool
from pydantic import Field

from akd_ext._types import OpenAITool
from akd_ext.tools.get_place import GetPlaceTool, GetPlaceToolInputSchema
from akd_ext.tools.set_datetime import SetDatetimeTool, SetDatetimeToolInputSchema
from akd_ext.tools.collections_rag import CollectionsRAGTool, CollectionsRAGToolInputSchema, CollectionsResult
from akd_ext.tools.stac_search import STACSearchTool, STACSearchToolInputSchema
from akd_ext.tools.stats import StatsTool, StatsToolInputSchema
from akd_ext.tools.viz import VizTool, VizToolInputSchema
from akd_ext.tools.utils import is_cmr_backed

from akd._base import (
    InputSchema,
    OutputSchema,
    TextOutput,
)
from akd_ext.agents._base import (
    OpenAIBaseAgent,
    OpenAIBaseAgentConfig,
)

from loguru import logger

# -----------------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------------

EIE_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and help with tasks relating to earth science data. Any topics unrelated to earth science data should be considered out of scope and explained as such.

You have the following tools available to you. Each tool is stateless — you must pass all required parameters explicitly on every call.

TOOLS:

- set_datetime_tool: Validate and normalize an ISO-8601 datetime range.
  Args: value (ISO-8601 range 'YYYY-MM-DD/YYYY-MM-DD')
  Returns: validated datetime range or error message.

- get_place_tool: Resolve a place name to a bounding box and GeoJSON geometry via geocoding.
  Args: query (place name, e.g. 'California', 'Houston TX')
  Returns: place name, bbox [west, south, east, north], and GeoJSON geometry.

- collections_rag_tool: Search for relevant STAC collections using semantic search.
  Args: query (data description, e.g. 'NO2 air quality'), top_k (default 5), bbox (optional), datetime_range (optional)
  Returns: ranked collection matches with spatial/temporal overlap flags, source (veda or cmr), and similarity scores.

- stac_search_tool: Search STAC catalog for COG items within a collection.
  Args: collections (list of collection IDs), bbox [west, south, east, north], datetime (ISO range), limit (default 15)
  Returns: list of items with id, collection, datetime, and asset_url (COG URL).

- stats_tool: Fetch raster zonal statistics. Two modes:
  VEDA COG mode: Pass items (list with url, id, datetime) + geometry (GeoJSON Polygon).
  CMR mode: Pass collection_metadata (dict with collection_concept_id, renders, etc.) + datetime_range + geometry + selected_variable.
  Returns: per-band statistics (min, max, mean, etc.) for each item or timestep.

- viz_tool: Build raster tile URLs for map visualization. Two modes:
  VEDA COG mode: Pass items (list with url, id, datetime) + collection_id.
  CMR mode: Pass collection_metadata (dict with collection_concept_id, renders, etc.) + datetime_range + selected_variable.
  Returns: tile URL templates with colormap, rescale, and collection metadata.

CRITICAL INSTRUCTIONS:

- When the user mentions a time range, convert it to ISO-8601 format yourself (e.g., "Oct to Dec 2021" → "2021-10-01/2021-12-31").
- Always validate the datetime with set_datetime_tool before using it in other tools.
- Always resolve the place with get_place_tool before using bbox/geometry in other tools.
- Pass bbox and datetime_range to collections_rag_tool so it can check spatial/temporal overlap.
- For VEDA COG collections: run stac_search_tool first to get items with asset URLs, then pass those items to stats_tool and viz_tool.
- For CMR-backed collections (identified by collection_concept_id in the match): skip stac_search_tool, pass the collection metadata directly to stats_tool and viz_tool with the selected variable.
- stats_tool and viz_tool can be called in the same turn since they are independent.

MANDATORY CONFIRMATION FLOW — YOU MUST FOLLOW THIS EXACTLY:

RULE: You may call AT MOST ONE tool per turn. After calling a tool, you MUST respond to the user and STOP. Wait for the user's next message before calling another tool. The ONLY exception is stats_tool + viz_tool which may be called together at the final step.

These three tools each REQUIRE a dedicated turn with user confirmation before you proceed:

1. set_datetime_tool → After it returns, tell the user: "I've set the datetime to [value]. Does this look correct?" Then STOP. Do NOT call get_place_tool or any other tool in this turn.

2. get_place_tool → After it returns, tell the user: "I found [place name] with bounding box [bbox]. Is this the right location?" Then STOP. Do NOT call collections_rag_tool or any other tool in this turn.

3. collections_rag_tool → After it returns, present the matching collections as a numbered list with title and description. Ask: "Which collection would you like to use? (enter the number)" Then STOP. Do NOT call stac_search_tool or any other tool in this turn.

VIOLATIONS (never do these):
- Calling set_datetime_tool AND get_place_tool in the same turn
- Calling get_place_tool AND collections_rag_tool in the same turn
- Calling collections_rag_tool AND stac_search_tool in the same turn
- Skipping the confirmation question after any of the three tools above
- Proceeding to the next step without the user explicitly confirming

Only after the user selects a collection may you call stac_search_tool and then stats_tool + viz_tool.

TOOL SEQUENCES:

For requests with place AND time range:
  1) set_datetime_tool(value="YYYY-MM-DD/YYYY-MM-DD") — present result, ask to confirm, STOP
  2) get_place_tool(query="place name") — present result, ask to confirm, STOP
  3) collections_rag_tool(query="data description", bbox=..., datetime_range=...) — present collections, ask user to select, STOP
  4) For VEDA COG: stac_search_tool(collections=[id], bbox=..., datetime=...) then stats_tool + viz_tool with the items
     For CMR: stats_tool + viz_tool directly with collection_metadata, datetime_range, geometry, selected_variable

For requests with place only (no time range):
  1) get_place_tool(query="place name") — present result, ask to confirm, STOP
  2) collections_rag_tool(query="data description", bbox=...) — present results, mention stats/viz need a time range

For requests with time range only (no place):
  1) set_datetime_tool(value="YYYY-MM-DD/YYYY-MM-DD") — present result, ask to confirm, STOP
  2) collections_rag_tool(query="data description", datetime_range=...) — present results, mention stats/viz need a place

After receiving stats results, simply say "Statistics retrieved." and stop — do not print statistics values or analysis.
After receiving viz results, simply say "Visualization layers generated." and stop — do not print tile URLs.

Do not use background knowledge, only use the tools above to answer questions.

The current date is {now}.
"""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _make_local_tools() -> list:
    """Create all 6 EIE tools locally with shared state.

    Mirrors the eie-llm-backend's LangGraph pattern: tools read/write to
    shared state so the LLM only passes minimal arguments (query, place name,
    collection selection).  Large data (geometry, items, collection metadata)
    flows through state and never bloats the LLM context.

    Tool logic is reused from the existing tool classes — these wrappers
    only handle state management and geometry simplification.
    """
    _state: dict[str, Any] = {
        # Mirrors EIEState from eie-llm-backend
        "datetime_range": None,
        "place_result": None,           # PlaceResult: {place, bbox, geometry}
        "collections_result": None,     # CollectionsResult: {collections, matches}
        "stac_result": None,            # StacSearchResult: {items, item_ids}
        "stats_result": None,           # StatsToolOutputSchema
        "viz_result": None,             # VizToolOutputSchema
        "selected_collection_id": None,
        "selected_variable": None,
        "collection_metadata": None,    # full STAC JSON (not in backend state, but needed for stateless tools)
    }

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
        return result.model_dump()

    # ── get_place_tool ──────────────────────────────────────────────

    @function_tool(name_override="get_place_tool")
    async def get_place_tool(query: str) -> str:
        """Resolve a place name to a bounding box and geometry via geocoding.

        Args:
            query: A place name or location (e.g. 'California', 'Houston TX')
        """
        result = await _place_tool._arun(GetPlaceToolInputSchema(query=query))

        # Store full place_result in state (geometry never shown to LLM)
        _state["place_result"] = result

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
        place_result = _state.get("place_result")
        result = await _rag_tool._arun(CollectionsRAGToolInputSchema(
            query=query,
            top_k=top_k,
            bbox=place_result.bbox if place_result else None,
            datetime_range=_state.get("datetime_range"),
        ))
        _state["collections_result"] = result
        return result.model_dump()

    # ── stac_search_tool ────────────────────────────────────────────

    @function_tool(name_override="stac_search_tool")
    async def stac_search_tool(collection_id: str, limit: int = 15) -> str:
        """Search STAC catalog for COG items. Reads bbox and datetime from state.

        Args:
            collection_id: The collection ID to search (from collections_rag results)
            limit: Maximum number of items to return (default 15)
        """
        place_result = _state.get("place_result")
        datetime_range = _state.get("datetime_range")
        if not place_result or not place_result.bbox:
            return json.dumps({"error": "No bbox — run get_place_tool first"})
        if not datetime_range:
            return json.dumps({"error": "No datetime — run set_datetime_tool first"})

        _state["selected_collection_id"] = collection_id

        # Look up full collection metadata from collections_rag enrichment (no extra HTTP call)
        collections_result = _state.get("collections_result")
        if collections_result:
            match = next((m for m in collections_result.matches if m.id == collection_id), None)
            if match:
                _state["collection_metadata"] = match.collection_metadata

        result = await _stac_tool._arun(STACSearchToolInputSchema(
            collections=[collection_id],
            bbox=place_result.bbox,
            datetime=datetime_range,
            limit=limit,
        ))

        _state["stac_result"] = result
        return result.model_dump()

    # ── stats_tool ──────────────────────────────────────────────────

    @function_tool(name_override="stats_tool")
    async def stats_tool(selected_variable: str | None = None) -> str:
        """Fetch raster zonal statistics. Reads geometry, items, and collection info from state.

        Args:
            selected_variable: Variable name for CMR collections (optional, from available_variables)
        """
        if selected_variable:
            _state["selected_variable"] = selected_variable

        place_result = _state.get("place_result")
        if not place_result or not place_result.geometry:
            return json.dumps({"error": "No geometry — run get_place_tool first"})

        stac_result = _state.get("stac_result")
        items = (
            [{"url": item.asset_url, "id": item.id, "datetime": item.datetime}
             for item in stac_result.items if item.asset_url]
            if stac_result else None
        )

        logger.info(f"Geometry passed to stats tool: {place_result.geometry}")

        result = await _stats_tool._arun(StatsToolInputSchema(
            geometry=place_result.geometry,
            items=items,
            collection_id=_state.get("selected_collection_id"),
            collection_metadata=_state.get("collection_metadata"),
            datetime_range=_state.get("datetime_range"),
            selected_variable=_state.get("selected_variable"),
        ))
        _state["stats_result"] = result
        return result.model_dump()

    # ── viz_tool ────────────────────────────────────────────────────

    @function_tool(name_override="viz_tool")
    async def viz_tool(selected_variable: str | None = None) -> str:
        """Build raster tile URLs for visualization. Reads items and collection info from state.

        Args:
            selected_variable: Variable name for CMR collections (optional, from available_variables)
        """
        if selected_variable:
            _state["selected_variable"] = selected_variable

        stac_result = _state.get("stac_result")
        items = (
            [{"url": item.asset_url, "id": item.id, "datetime": item.datetime}
             for item in stac_result.items if item.asset_url]
            if stac_result else None
        )

        result = await _viz_tool._arun(VizToolInputSchema(
            items=items,
            collection_id=_state.get("selected_collection_id"),
            collection_metadata=_state.get("collection_metadata"),
            datetime_range=_state.get("datetime_range"),
            selected_variable=_state.get("selected_variable"),
        ))
        _state["viz_result"] = result
        return result.model_dump()

    return [set_datetime_tool, get_place_tool, collections_rag_tool, stac_search_tool, stats_tool, viz_tool]


def get_default_eie_tools() -> list[OpenAITool]:
    """All 6 EIE tools running locally with shared state. No MCP server needed."""
    return list(_make_local_tools())


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
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(default="low")
    tools: list[Any] = Field(default_factory=get_default_eie_tools)


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
    """Earth Intelligence Engine Agent for earth science data exploration.

    Orchestrates STAC collection discovery, raster statistics, and
    visualization using tools deployed on FastMCP cloud.
    """

    input_schema = EIEAgentInputSchema
    output_schema = TextOutput # EIEAgentOutputSchema |
    config_schema = EIEAgentConfig


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = EIEAgent(EIEAgentConfig(debug=True))
        logger.info(f"Agent description: {agent.description}")
        question = "Show me NO2 air quality data for Washington DC from January to June 2020"

        async for event in agent.astream(EIEAgentInputSchema(query=question)):
            logger.info(event)

    asyncio.run(main())
