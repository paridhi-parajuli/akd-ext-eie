"""Collections RAG tool: search STAC collections via the external RAG service."""
from __future__ import annotations

import os

import httpx
from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool, BaseToolConfig
from loguru import logger
from pydantic import Field

from akd_ext.mcp import mcp_tool
from akd_ext.tools.utils import fetch_collection_metadata, is_cmr_backed


class CollectionsRAGToolConfig(BaseToolConfig):
    """Configuration for the Collections RAG Search Tool."""

    base_url: str = Field(
        default=os.getenv(
            "COLLECTIONS_RAG_URL",
            "https://k8s-eiellm-eiellmng-3bfff7cd13-5b11596fb1756b2e.elb.us-west-2.amazonaws.com",
        ),
        description="Base URL for the collections RAG service",
    )
    veda_api_root: str = Field(
        default=os.getenv("VEDA_API_ROOT", "https://dev.openveda.cloud/api"),
        description="VEDA API root (used to fetch full collection metadata for enrichment)",
    )

    @property
    def stac_url(self) -> str:
        """STAC API URL derived from veda_api_root."""
        return f"{self.veda_api_root.rstrip('/')}/stac"


class CollectionMatchInfo(OutputSchema):
    """A single collection match with enriched metadata."""

    id: str = Field(..., description="Collection ID")
    title: str | None = Field(None, description="Collection title")
    description: str | None = Field(None, description="Collection description")
    spatial_overlap: bool = Field(default=False, description="Whether the collection spatially overlaps the query bbox")
    temporal_overlap: bool = Field(default=False, description="Whether the collection temporally overlaps the query range")
    cosine_distance: float | None = Field(None, description="Cosine distance from query (None for CMR results)")
    cosine_similarity: float | None = Field(None, description="Cosine similarity to query (None for CMR results)")
    source: str = Field(default="veda", description="Result source: 'veda' or 'cmr'")
    cmr_rank: int | None = Field(None, description="Position in CMR results (None for VEDA)")
    time_density: str | None = Field(None, description="Temporal density: 'day', 'month', 'year', or None")
    is_cmr_backed: bool = Field(False, description="True if collection data is accessed via CMR (has collection_concept_id)")
    concept_id: str | None = Field(None, description="CMR collection_concept_id if is_cmr_backed=True")
    available_variables: list[str] = Field(default_factory=list, description="Available renderable variables for CMR collections (from renders keys)")
    collection_metadata: dict | None = Field(None, exclude=True, description="Full STAC collection JSON (excluded from serialization)")


class CollectionsRAGToolInputSchema(InputSchema):
    """Input schema for collections RAG search."""

    query: str = Field(..., description="Semantic search query for finding relevant collections")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    bbox: list[float] | None = Field(
        None, description="Optional bounding box [west, south, east, north] for spatial overlap check"
    )
    datetime_range: str | None = Field(
        None, description="Optional ISO-8601 range 'start/end' for temporal overlap check"
    )


class CollectionsResult(OutputSchema):
    """Result from collections search — matches eie-llm-backend's CollectionsResult."""

    collections: list[str] = Field(default_factory=list, description="Matched collection IDs")
    matches: list[CollectionMatchInfo] = Field(default_factory=list, description="Detailed match info with coverage")
    error: str | None = Field(default=None, description="Error message if search failed")


@mcp_tool
class CollectionsRAGTool(BaseTool[CollectionsRAGToolInputSchema, CollectionsResult]):
    """
    Search for relevant STAC collections using semantic similarity.

    Calls the external collections RAG service which bundles LanceDB +
    embedding model + CMR fallback. Returns ranked collection matches
    with spatial/temporal overlap flags.

    Input parameters (query-time, LLM-controllable):
    - query: Semantic search query (e.g. "methane plumes")
    - top_k: Number of results (1-50, default 5)
    - bbox: Optional [west, south, east, north] for spatial overlap checking
    - datetime_range: Optional 'start/end' ISO-8601 range for temporal overlap checking

    Configuration parameters (instance-time, user-controlled):
    - base_url: Base URL for the collections RAG service

    Returns matches with:
    - id, title, description: Collection metadata
    - spatial_overlap / temporal_overlap: Whether the collection overlaps the query area/time
    - cosine_distance / cosine_similarity: Embedding similarity (VEDA results only)
    - source: 'veda' or 'cmr'
    - cmr_rank: CMR relevance position (CMR results only)
    - time_density: Temporal resolution
    """

    input_schema = CollectionsRAGToolInputSchema
    output_schema = CollectionsResult
    config_schema = CollectionsRAGToolConfig

    async def _arun(self, params: CollectionsRAGToolInputSchema) -> CollectionsResult:
        """Execute collections search via the external RAG service."""
        url = f"{self.config.base_url.rstrip('/')}/agent/search/collections"

        request_body: dict = {
            "query": params.query,
            "top_k": params.top_k,
        }
        if params.bbox is not None:
            request_body["bbox"] = params.bbox
        if params.datetime_range is not None:
            request_body["datetime_range"] = params.datetime_range

        logger.debug(f"Collections RAG request: {request_body}")

        async with httpx.AsyncClient(timeout=30, verify=False) as client:
            try:
                response = await client.post(url, json=request_body)
                response.raise_for_status()
                data = response.json()
            except httpx.TimeoutException as e:
                msg = f"Collections RAG service timed out after 30s"
                raise TimeoutError(msg) from e
            except httpx.HTTPStatusError as e:
                msg = f"Collections RAG service returned {e.response.status_code}: {e.response.text}"
                raise RuntimeError(msg) from e

        # Enrich each match with full metadata (is_cmr_backed, concept_id, available_variables)
        # Mirrors eie-llm-backend's injected_tools.py enrichment step.
        auxiliary_suffixes = ("_cnt", "_cond", "Error", "_error")
        enriched_matches = []
        for item in data:
            coll_metadata = fetch_collection_metadata(item["id"], self.config.stac_url)
            cmr_backed = is_cmr_backed(coll_metadata)
            concept_id = coll_metadata.get("collection_concept_id") if coll_metadata and cmr_backed else None

            available_variables: list[str] = []
            if cmr_backed and coll_metadata:
                renders = coll_metadata.get("renders", {})
                available_variables = [
                    v for v in renders.keys()
                    if not any(s in v for s in auxiliary_suffixes)
                ]

            enriched_matches.append(
                CollectionMatchInfo(
                    **item,
                    is_cmr_backed=cmr_backed,
                    concept_id=concept_id,
                    available_variables=available_variables,
                    collection_metadata=coll_metadata,
                )
            )

        logger.debug(f"Collections RAG returned {len(enriched_matches)} matches (enriched)")

        return CollectionsResult(
            collections=[m.id for m in enriched_matches],
            matches=enriched_matches,
        )
