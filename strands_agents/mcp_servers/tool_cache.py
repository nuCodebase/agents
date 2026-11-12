"""Redis-backed caching for deterministic MCP tool results."""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

from redis.asyncio import Redis
from redis.exceptions import RedisError

LOGGER = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


class RedisToolCache:
    """Cache deterministic tool results using canonical, versioned keys."""

    def __init__(
        self,
        url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.url = url if url is not None else os.getenv("REDIS_URL")
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("TOOL_CACHE_TTL_SECONDS", "3600")
        )
        self.client = (
            Redis.from_url(self.url, encoding="utf-8", decode_responses=True)
            if self.url
            else None
        )

    @staticmethod
    def key(tool_name: str, arguments: dict[str, Any]) -> str:
        """Return a stable key for one tool name and exact argument set."""
        canonical = json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"math-tool:v1:{tool_name}:{digest}"

    async def get_or_compute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: Callable[[], ResultT],
    ) -> ResultT:
        """Return a cached result, or compute and cache it on a miss.

        Redis failures are deliberately fail-open: the deterministic operation
        still runs, and the tool call remains available during cache outages.
        """
        if self.client is None:
            return operation()

        try:
            cache_key = self.key(tool_name, arguments)
            cached = await self.client.get(cache_key)
            if cached is not None:
                LOGGER.info("tool_cache_hit tool=%s key=%s", tool_name, cache_key)
                return json.loads(cached)
        except (RedisError, ValueError, TypeError) as error:
            LOGGER.warning("tool_cache_read_failed tool=%s error=%s", tool_name, error)
            return operation()

        result = operation()
        try:
            serialized = json.dumps(result, allow_nan=False)
            await self.client.set(cache_key, serialized, ex=self.ttl_seconds)
            LOGGER.info("tool_cache_miss tool=%s key=%s", tool_name, cache_key)
        except (RedisError, ValueError, TypeError) as error:
            LOGGER.warning("tool_cache_write_failed tool=%s error=%s", tool_name, error)
        return result

    async def close(self) -> None:
        """Close the Redis connection pool when caching is enabled."""
        if self.client is not None:
            await self.client.aclose()


tool_cache = RedisToolCache()
