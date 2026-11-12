"""Unit tests for deterministic Redis tool-result caching."""

import asyncio

from redis.exceptions import ConnectionError

from mcp_servers.tool_cache import RedisToolCache


class FakeRedis:
    """Small async Redis stand-in supporting cache reads and writes."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.writes: list[tuple[str, str, int]] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = value
        self.writes.append((key, value, ex))


class FailingRedis:
    """Redis stand-in representing an unavailable cache."""

    async def get(self, key: str) -> None:
        raise ConnectionError("redis unavailable")


def test_key_is_independent_of_argument_order() -> None:
    first = RedisToolCache.key("add", {"a": 2.0, "b": 3.0})
    second = RedisToolCache.key("add", {"b": 3.0, "a": 2.0})

    assert first == second
    assert first.startswith("math-tool:v1:add:")


def test_cache_miss_then_hit_computes_only_once() -> None:
    cache = RedisToolCache(url=None, ttl_seconds=120)
    fake_redis = FakeRedis()
    cache.client = fake_redis  # type: ignore[assignment]
    calls = 0

    def operation() -> float:
        nonlocal calls
        calls += 1
        return 42.0

    async def exercise() -> tuple[float, float]:
        first = await cache.get_or_compute(
            "multiply", {"a": 6.0, "b": 7.0}, operation
        )
        second = await cache.get_or_compute(
            "multiply", {"a": 6.0, "b": 7.0}, operation
        )
        return first, second

    assert asyncio.run(exercise()) == (42.0, 42.0)
    assert calls == 1
    assert len(fake_redis.writes) == 1
    assert fake_redis.writes[0][2] == 120


def test_cache_failure_falls_back_to_operation() -> None:
    cache = RedisToolCache(url=None, ttl_seconds=120)
    cache.client = FailingRedis()  # type: ignore[assignment]

    result = asyncio.run(
        cache.get_or_compute("add", {"a": 2.0, "b": 3.0}, lambda: 5.0)
    )

    assert result == 5.0
