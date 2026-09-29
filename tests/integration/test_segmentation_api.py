"""Person mask generation API (Phase 5): model listing + mask endpoint.

The segmentation service is injected with a fake factory (no mediapipe,
no real .tflite needed): the endpoint wiring, canvas interaction
(set_mask, undo) and error paths are what's under test here.
"""

import numpy as np
import pytest
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config
from imagagen.segmentation import SegmentationService


class FakeM:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def numpy_view(self):
        return self.arr


class FakeSegmenter:
    def __init__(self, labels, confs):
        self.labels = list(labels)
        self.confs = confs

    def segment(self, image):
        return type("Out", (), {"confidence_masks": [FakeM(c) for c in self.confs]})

    def close(self):
        pass


LABELS = ["background", "hair", "face"]


@pytest.fixture
def client(tmp_path):
    models_dir = tmp_path / "models" / "segmentation"
    models_dir.mkdir(parents=True)
    (models_dir / "selfie_multiclass_256x256.tflite").write_bytes(b"tflite")

    # 8x8 image: hair = top-left quarter at 0.5, face = bottom-right
    # quarter at 1.0 (no overlap); background = everything else at 0.3
    H = W = 8
    hair = np.zeros((H, W), np.float32)
    hair[:4, :4] = 0.5
    face = np.zeros((H, W), np.float32)
    face[4:, 4:] = 1.0
    bg = np.zeros((H, W), np.float32)
    bg[:] = 0.3
    svc = SegmentationService(
        models_dir,
        segmenter_factory=lambda p: FakeSegmenter(
            LABELS, [bg, hair, face]),
    )
    app = create_app(
        config=Config(root=tmp_path), backend=MockBackend(), segmentation=svc
    )
    with TestClient(app) as c:
        yield c


def set_canvas(client, w=8, h=8):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    r = client.post("/api/canvas/upload", content=buf.getvalue())
    assert r.status_code == 200, r.text


def mask_png(client) -> np.ndarray | None:
    from io import BytesIO

    from PIL import Image

    r = client.get("/api/mask.png")
    if r.status_code == 404:
        return None
    return np.asarray(Image.open(BytesIO(r.content)))


def test_segmentation_models_endpoint(client):
    body = client.get("/api/segmentation/models").json()
    assert body == {"models": [{
        "id": "selfie_multiclass_256x256.tflite",
        "name": "selfie_multiclass_256x256",
        "labels": LABELS,
    }]}


def test_segmentation_mask_endpoint_writes_mask(client):
    set_canvas(client)
    r = client.post("/api/segmentation/mask", json={
        "model": "selfie_multiclass_256x256.tflite",
        "classes": ["hair", "face"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["changed"] is True
    # union of hair (4x4) and face (4x4) support: 32 pixels total
    assert body["pixels"] == 32

    mask = mask_png(client)
    assert mask.shape == (8, 8)
    assert mask[0, 0] == 127      # hair: 0.5 * 255 = 127.5, truncated on cast
    assert mask[7, 7] == 255      # face: full confidence
    assert mask[0, 7] == 0        # neither class


def test_segmentation_unions_with_existing_mask(client):
    set_canvas(client)
    # pre-paint a mask pixel at (3, 5): outside hair (cols/rows 0..3 x 0..3)
    # and outside face (4..) -> must survive a face-only segmentation
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (8, 8), (200, 30, 30))
    mask = Image.new("RGBA", (8, 8), (128, 128, 128, 0))
    mask.load()[3, 5] = (128, 128, 128, 255)
    buf_i, buf_m = BytesIO(), BytesIO()
    img.save(buf_i, format="PNG")
    mask.save(buf_m, format="PNG")
    r = client.post("/api/canvas/sync",
                    files={"image": ("canvas.png", buf_i.getvalue(), "image/png"),
                           "mask": ("mask.png", buf_m.getvalue(), "image/png")})
    assert r.status_code == 200
    before = mask_png(client)
    painted = np.argwhere(before > 0)
    assert painted.size > 0

    r = client.post("/api/segmentation/mask", json={
        "model": "selfie_multiclass_256x256.tflite", "classes": ["face"],
    })
    assert r.status_code == 200
    after = mask_png(client)
    assert after[7, 7] == 255  # face region segmented
    for y, x in painted:
        if not (y < 4 and x < 4):  # not inside the hair block
            assert after[y, x] == before[y, x] > 0  # hand-paint kept (union)
    # the painted pixel also counts in the reported total
    assert r.json()["pixels"] >= int((after > 0).sum())


def test_segmentation_errors(client):
    r = client.post("/api/segmentation/mask", json={
        "model": "selfie_multiclass_256x256.tflite", "classes": ["hair"],
    })
    assert r.status_code == 400 and "canvas is empty" in r.json()["error"]

    set_canvas(client)
    r = client.post("/api/segmentation/mask", json={
        "model": "nope.tflite", "classes": ["hair"],
    })
    assert r.status_code == 400 and "unknown segmentation model" in r.json()["error"]
    r = client.post("/api/segmentation/mask", json={
        "model": "selfie_multiclass_256x256.tflite", "classes": ["cats"],
    })
    assert r.status_code == 400 and "unknown class" in r.json()["error"]
    r = client.post("/api/segmentation/mask", json={
        "model": "selfie_multiclass_256x256.tflite", "classes": [],
    })
    assert r.status_code == 400 and "no classes selected" in r.json()["error"]


def test_no_models_dir_content_lists_empty(tmp_path):
    # the service is constructed in create_app when none is injected:
    # with an empty models/segmentation/ the endpoint returns []
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    with TestClient(app) as c:
        assert c.get("/api/segmentation/models").json() == {"models": []}
