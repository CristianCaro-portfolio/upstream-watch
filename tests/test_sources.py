import httpx
import pytest

from upstream_watch import sources
from upstream_watch.sources import CachedClient, SourceError


def test_retries_then_caches(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = CachedClient(tmp_path, transport=httpx.MockTransport(handler), backoff_s=0.0)
    assert client.get_json("https://example.org/a", {"x": 1}) == {"ok": True}
    assert client.get_json("https://example.org/a", {"x": 1}) == {"ok": True}
    assert calls["n"] == 2  # one retry, then the second call came from the cache


def test_client_errors_are_not_retried(tmp_path):
    client = CachedClient(tmp_path, transport=httpx.MockTransport(lambda r: httpx.Response(404)), backoff_s=0.0)
    with pytest.raises(SourceError, match="404"):
        client.get_json("https://example.org/missing")


def test_offline_without_cache_fails_fast(tmp_path):
    with pytest.raises(SourceError, match="offline"):
        CachedClient(tmp_path, offline=True).get_json("https://example.org/a")


def test_ogc_paging_follows_offset(tmp_path):
    seen = []

    def handler(request):
        offset = int(request.url.params["offset"])
        seen.append(offset)
        n = 2 if offset < 4 else 1
        return httpx.Response(200, json={"features": [{"properties": {"i": offset + i}} for i in range(n)]})

    client = CachedClient(tmp_path, transport=httpx.MockTransport(handler), backoff_s=0.0)
    items = list(sources._paged_items(client, "continuous", {"limit": 2}))
    assert seen == [0, 2, 4] and len(items) == 5
