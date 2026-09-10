import httpx

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_health_and_dirs(tmp_path):
    async with _client(tmp_path) as c:
        r = await c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["backend"] == "Mock Model"
    # directory layout created
    for d in ("models", "models/segmentation", "loras", "brushes", "presets",
              "model-configs", "output"):
        assert (tmp_path / d).is_dir(), d


async def test_index_served(tmp_path):
    async with _client(tmp_path) as c:
        r = await c.get("/")
        assert r.status_code == 200
        assert "imagagen" in r.text


async def test_generate_writes_output(tmp_path):
    async with _client(tmp_path) as c:
        r = await c.post(
            "/api/generate",
            json={
                "prompt": "Photo of a turtle",
                "width": 32,
                "height": 32,
                "steps": 2,
                "seed": 123,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["output"] == "1-photo-of-a-turtle.png"
        assert body["seed"] == 123
        assert (tmp_path / "output" / body["output"]).is_file()

        # second generation increments the leading number
        r2 = await c.post(
            "/api/generate",
            json={"prompt": "photo of a frog", "width": 32, "height": 32, "steps": 2, "seed": 1},
        )
        assert r2.json()["output"] == "2-photo-of-a-frog.png"

        listing = (await c.get("/api/output")).json()
        assert [f["name"] for f in listing] == ["1-photo-of-a-turtle.png", "2-photo-of-a-frog.png"]


async def test_generate_request_validation(tmp_path):
    # FastAPI rejects payloads with wrong types for dataclass fields
    async with _client(tmp_path) as c:
        r = await c.post("/api/generate", json={"prompt": "x", "width": "abc"})
        assert r.status_code == 422
