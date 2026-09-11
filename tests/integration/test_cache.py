"""Cache headers: no-cache (revalidate) by default, no-store in debug.

Root-caused a real bug: with no Cache-Control header at all, Chrome
heuristic-cached app.js (and JSON API responses), so a restarted server
kept serving a stale UI until the cache expired or devtools'
"Disable cache" was enabled.
"""

from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def test_default_revalidates_everything(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    with TestClient(app) as c:
        for path in ["/", "/app.js", "/style.css", "/api/ui-state",
                     "/api/health", "/api/models"]:
            r = c.get(path)
            assert r.headers.get("cache-control") == "no-cache", path


def test_default_static_files_still_etag(tmp_path):
    """Revalidation must stay cheap: StaticFiles' ETag/last-modified are
    kept, so an unchanged file is a 304, not a re-upload."""
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    with TestClient(app) as c:
        r = c.get("/app.js")
        etag = r.headers.get("etag")
        assert etag
        again = c.get("/app.js", headers={"if-none-match": etag})
        assert again.status_code == 304


def test_debug_forbids_caching(tmp_path):
    app = create_app(config=Config(root=tmp_path, debug=True),
                     backend=MockBackend())
    with TestClient(app) as c:
        for path in ["/", "/app.js", "/api/ui-state", "/api/health"]:
            r = c.get(path)
            assert r.headers.get("cache-control") == "no-store", path
            assert r.headers.get("pragma") == "no-cache", path
            assert r.headers.get("expires") == "0", path


def test_debug_does_not_affect_content(tmp_path):
    app = create_app(config=Config(root=tmp_path, debug=True),
                     backend=MockBackend())
    with TestClient(app) as c:
        assert "imagagen" in c.get("/").text
        assert c.get("/api/ui-state").json()["state"]["prompt"] == ""
