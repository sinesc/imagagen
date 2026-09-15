"""API tests for styles: CRUD, temporary styles, generation + PNG metadata."""

import io
import json

import pytest
from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config

CSV_HEADER = "name,prompt,negative_prompt,optgroup"


class RecordingBackend(MockBackend):
    """MockBackend that remembers the prompts the backend actually saw."""

    def __init__(self):
        super().__init__(steps_delay=0.0)
        self.requests = []

    def generate(self, request, progress=None):
        self.requests.append((request.prompt, request.negative_prompt))
        return super().generate(request, progress)


@pytest.fixture
def client(tmp_path):
    backend = RecordingBackend()
    app = create_app(config=Config(root=tmp_path), backend=backend)
    with TestClient(app) as c:
        c.backend = backend
        c.root = tmp_path
        yield c


def _seed_csv(client):
    (client.root / "styles.csv").write_text(
        "\n".join(
            [
                CSV_HEADER,
                'Cinematic,"cinematic film still {prompt} . moody",,"Photo Styles"',
                'Dunes,"in the dunes, white sand",,"Background"',
                'Negs,,"digital art, cartoon",Negatives',
            ]
        )
        + "\n",
        "utf-8",
    )


def test_styles_empty_default(client):
    r = client.get("/api/styles")
    assert r.status_code == 200
    assert r.json() == {"styles": []}


def test_styles_crud(client):
    _seed_csv(client)
    names = [s["name"] for s in client.get("/api/styles").json()["styles"]]
    assert names == ["Cinematic", "Dunes", "Negs"]

    # create
    r = client.post("/api/styles", json={
        "name": "Analog Film", "prompt": "analog film photo {prompt}",
        "negative_prompt": "clean", "optgroup": "Photo Styles",
    })
    assert r.status_code == 200
    # upsert by name
    r = client.post("/api/styles", json={
        "name": "Analog Film", "prompt": "vintage {prompt}",
    })
    assert r.status_code == 200
    styles = {s["name"]: s for s in client.get("/api/styles").json()["styles"]}
    assert styles["Analog Film"]["prompt"] == "vintage {prompt}"
    assert styles["Analog Film"]["optgroup"] == ""

    # validation: empty name -> 422
    assert client.post("/api/styles", json={"name": ""}).status_code == 422

    # delete
    r = client.post("/api/styles/delete", json={"name": "Dunes"})
    assert r.status_code == 200
    assert client.post("/api/styles/delete", json={"name": "Dunes"}).status_code == 404
    # disk file reflects the changes
    text = (client.root / "styles.csv").read_text("utf-8")
    assert "Dunes" not in text and "Analog Film" in text


def test_styles_temporary(client):
    _seed_csv(client)
    r = client.post("/api/styles/temp", json={
        "styles": [
            {"name": "Cinematic", "prompt": "changed?"},  # exists: ignored
            {"name": "Lost", "prompt": "lost {prompt}", "negative_prompt": "x"},
        ]
    })
    assert r.status_code == 200
    styles = {s["name"]: s for s in r.json()["styles"]}
    assert styles["Cinematic"]["temporary"] is False
    assert styles["Cinematic"]["prompt"].startswith("cinematic film still")
    assert styles["Lost"]["temporary"] is True
    assert styles["Lost"]["optgroup"] == "Session"
    # never written to the CSV
    assert "Lost" not in (client.root / "styles.csv").read_text("utf-8")
    # a real style with the same name supersedes the temporary
    client.post("/api/styles", json={"name": "Lost", "prompt": "real now"})
    styles = {s["name"]: s for s in client.get("/api/styles").json()["styles"]}
    assert styles["Lost"]["temporary"] is False
    assert styles["Lost"]["prompt"] == "real now"


def _wait_thumb(client, ws, timeout: float = 15.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break
        if msg.get("type") == "image":
            return msg
    raise TimeoutError("no image event")


def test_generate_applies_styles_and_metadata(client):
    _seed_csv(client)
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "a frog",
            "negative_prompt": "ugly",
            "width": 64, "height": 64, "steps": 2, "seed": 7,
            "styles": ["Cinematic", "Negs"],
        })
        assert r.status_code == 200
        _wait_thumb(client, ws)
    # the backend saw the APPLIED prompt (order: Cinematic, then Negs)
    prompt, negative = client.backend.requests[-1]
    assert prompt == "cinematic film still a frog . moody"
    assert negative == "ugly, digital art, cartoon"
    # the PNG metadata carries the RAW prompt + the style contents
    png = Image.open(io.BytesIO(client.get("/api/thumb/1/0.png").content))
    meta = png.info
    assert meta["parameters"].splitlines()[0] == "a frog"
    assert meta["Negative prompt"] == "ugly"
    styles = json.loads(meta["parameters"].splitlines()[-1].removeprefix("Styles:"))
    assert styles == [
        {"name": "Cinematic", "prompt": "cinematic film still {prompt} . moody",
         "negative_prompt": ""},
        {"name": "Negs", "prompt": "", "negative_prompt": "digital art, cartoon"},
    ]
    # /api/png-meta round-trips it (what the UI import consumes)
    r = client.post("/api/png-meta", content=client.get("/api/thumb/1/0.png").content)
    out = r.json()["meta"]
    assert out["prompt"] == "a frog"
    assert out["styles"][0]["name"] == "Cinematic"


def test_generate_skips_deleted_style(client):
    _seed_csv(client)
    client.post("/api/styles/delete", json={"name": "Cinematic"})
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "a frog", "width": 64, "height": 64, "steps": 2, "seed": 7,
            "styles": ["Cinematic", "Negs"],
        })
        assert r.status_code == 200
        _wait_thumb(client, ws)
    prompt, negative = client.backend.requests[-1]
    assert prompt == "a frog"  # deleted style skipped
    assert negative == "digital art, cartoon"


def _forge_png(client, prompt, negative, w=16, h=12) -> bytes:
    """A Forge/A1111 PNG: the fully EXPANDED prompt as the meta prompt, no
    dedicated style list (that is what Forge writes)."""
    from imagagen import metadata

    buf = io.BytesIO()
    img, info = metadata.attach_metadata(
        Image.new("RGB", (w, h), (5, 5, 5)),
        metadata.build_parameters(
            prompt, negative, 24, "DPM++ 2M SDE", "Polyexponential", 7.5,
            1261832509, w, h, "some_model", "abc123"),
    )
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def test_png_meta_restores_forge_styles(client):
    _seed_csv(client)
    # what Forge stored: base "a frog"/"ugly" with [Negs, Dunes, Cinematic]
    # applied, i.e. this exact expanded prompt and negative
    expanded_prompt = "cinematic film still a frog, in the dunes, white sand . moody"
    expanded_negative = "ugly, digital art, cartoon"
    data = _forge_png(client, expanded_prompt, expanded_negative)
    assert b"Styles:" not in data  # genuinely foreign metadata

    r = client.post("/api/png-meta", content=data)
    m = r.json()["meta"]
    # the styles are resolved out of the expanded prompt, and the prompts
    # shown are the original (unapplied) ones
    assert [s["name"] for s in m["styles"]] == ["Negs", "Dunes", "Cinematic"]
    assert m["prompt"] == "a frog"
    assert m["negative_prompt"] == "ugly"


def test_png_meta_forge_unknown_styles_untouched(client):
    _seed_csv(client)
    # expanded by styles that are NOT in the CSV: nothing matches, so the
    # expanded prompt is left as-is (nothing to restore)
    data = _forge_png(client, "a totally alien prompt", "some negative")
    m = client.post("/api/png-meta", content=data).json()["meta"]
    assert "styles" not in m
    assert m["prompt"] == "a totally alien prompt"
    assert m["negative_prompt"] == "some negative"


def test_generate_no_styles_keeps_old_metadata(client):
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "plain", "negative_prompt": "n",
            "width": 64, "height": 64, "steps": 2, "seed": 7,
        })
        assert r.status_code == 200
        _wait_thumb(client, ws)
    png = Image.open(io.BytesIO(client.get("/api/thumb/1/0.png").content))
    assert "Styles:" not in png.info["parameters"]
