"""Sri's six-parameter stopping head and its exact public lexical features.

The float32 values below were exported from the public, pretrained checkpoints
with ``torch.load(..., weights_only=True)``. Loading this module needs no Torch,
network, model weights, or old Yash experiment artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import string
import urllib.request

SRI_REVISION = "5503cabeb2ec0094413b7d491ccb08ec9c991c61"
SRI_ROOT = f"https://raw.githubusercontent.com/acm-research-f26/Halt-IQ/{SRI_REVISION}/implementation-3"
FEATURES = ("distance", "round_fraction", "distance_delta", "critic_approved", "has_previous_draft")
QUESTIONS_SHA256 = "c82fb64d47e75c1b50938b3d8be2d20c364273cb4404171283397c5efaa81dc2"
PROTOCOL_SHA256 = "386fb3667fff2ac810149d5e327fe3a0cf55089477560d8f2ddeca512a8e4ccc"

# Every entry is a trained checkpoint, not a seeded random initialization.
_EXPORTED = {
    0: ([0.41224560141563416, 0.5887176990509033, -0.3605964183807373, 0.08626075088977814, 0.08794842660427094], 0.5353063941001892, "edd6822a38b75f112e09fd924d74d6965b351ec25820ab99aeadf17a26b3b453"),
    1: ([0.5618273615837097, 0.09025855362415314, -0.08669748902320862, 0.5480368137359619, -0.28265145421028137], 0.6063304543495178, "8ae5810a7a680f7fd50b72d985c30208ae46b0420d060a46be3a6486031dd076"),
    2: ([0.4870452880859375, 0.1575474888086319, 0.22772742807865143, 0.33644354343414307, 0.26236340403556824], 0.4657556414604187, "dd11a94c64ddc79ac4bbcb899cb63424d916545301464b4dca873dad5adddfaa"),
    3: ([0.3502989113330841, 0.25042691826820374, -0.19666515290737152, 0.35265618562698364, 0.4774613380432129], 0.3823167085647583, "9ef9c8c003da230e263c8a6841e7acb05fc8a89e11dac9189b3fda7362ec74f3"),
    4: ([0.41125932335853577, 0.4200232923030853, -0.3926645517349243, 0.14630477130413055, -0.08746660500764847], -0.00667919497936964, "fa68ea9052b9413a437d3174f9027aff8028e13fb0f432427bb2e2e00ba0c4fe"),
}


def normalize(text: str) -> str:
    """Exact Sri normalizer; lexical sets, not semantic embeddings."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def lexical_distance(a: str, b: str) -> float:
    aa, bb = set(normalize(a).split()), set(normalize(b).split())
    return 1.0 - len(aa & bb) / max(1, len(aa | bb))


def make_features(draft: str, previous: str | None, round_index: int,
                  max_rounds: int, approved: bool, previous_distance: float | None = 0) -> list[float]:
    """Return Sri's features; ``round_index`` is ZERO based.

    Sri stores the first distance as None, maps it to 1 for the head, and sets
    distance_delta to zero in BOTH the first and second rounds. Pass the previous
    round's actual lexical distance from the third round onward. The denominator
    is the predeclared round cap, never the eventual stopping round.
    """
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise ValueError("round_index must be an integer")
    if not isinstance(max_rounds, int) or max_rounds < 1 or not 0 <= round_index < max_rounds:
        raise ValueError("round_index must lie within the declared round cap")
    if not isinstance(draft, str) or (round_index and not isinstance(previous, str)):
        raise ValueError("A draft and previous draft are required after the first round")
    if not isinstance(approved, bool):
        raise ValueError("approved must be a boolean")
    if previous_distance is not None and (not math.isfinite(previous_distance) or not 0 <= previous_distance <= 1):
        raise ValueError("previous_distance must be finite in [0,1] or None")
    distance = 1.0 if round_index == 0 else lexical_distance(previous, draft)
    delta = 0.0 if round_index < 2 or previous_distance is None else distance - previous_distance
    return [distance, (round_index + 1) / max_rounds, delta, float(approved), float(round_index > 0)]


@dataclass(frozen=True)
class LinearHead:
    weights: tuple[float, ...]
    bias: float
    seed: int = 0
    checkpoint_sha256: str = ""

    def __post_init__(self):
        if len(self.weights) != 5 or not all(math.isfinite(v) for v in (*self.weights, self.bias)):
            raise ValueError("A finite, five-weight trained head is required")

    @classmethod
    def pretrained(cls, seed: int = 0) -> "LinearHead":
        if seed not in _EXPORTED:
            raise ValueError("The pinned study has seeds 0 through 4")
        weights, bias, checksum = _EXPORTED[seed]
        return cls(tuple(weights), bias, seed, checksum)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearHead":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = cls.pretrained(data["seed"])
        if data != expected.as_dict():
            raise ValueError("JSON differs from the pinned pretrained checkpoint export")
        return expected

    def probability(self, features: list[float]) -> float:
        if len(features) != 5 or not all(math.isfinite(v) for v in features):
            raise ValueError("Exactly five finite features are required")
        z = self.bias + sum(w * float(x) for w, x in zip(self.weights, features))
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    @property
    def provenance(self) -> dict:
        return self.as_dict()

    def as_dict(self) -> dict:
        return {"schema": "sri-linear-stop-head-v1", "source_revision": SRI_REVISION,
                "source_url": f"{SRI_ROOT}/study150/results/seed{self.seed}/stop_head.pt",
                "checkpoint_sha256": self.checkpoint_sha256, "checkpoint_bytes": 5301,
                "seed": self.seed, "weights": list(self.weights), "bias": self.bias,
                "features": list(FEATURES), "stop_threshold": 0.5, "trained": True,
                "training_questions": 50, "validation_questions": 20,
                "questions_sha256": QUESTIONS_SHA256, "protocol_sha256": PROTOCOL_SHA256,
                "original_approval_feature": "Qwen reviewer output exactly APPROVED",
                "original_generator": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"}


def download_checkpoint(cache_dir: str | Path, seed: int = 0) -> Path:
    """Verify original bytes BEFORE safely parsing; export portable JSON weights.

    Torch is imported only here. The original pickle is never loaded with
    weights_only=False. Existing mismatching files fail rather than being reused.
    """
    head = LinearHead.pretrained(seed)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"sri-seed{seed}-stop_head.pt"
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(head.as_dict()["source_url"], timeout=60).read()
    if len(data) != 5301 or hashlib.sha256(data).hexdigest() != head.checkpoint_sha256:
        raise ValueError("Sri checkpoint size or SHA256 differs from the pinned artifact")
    import io
    import torch
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (tuple(state["features"]) != FEATURES or state["seed"] != seed or state["source"] != "live"
            or state["cache_manifest"]["dataset_sha256"] != QUESTIONS_SHA256
            or state["cache_manifest"]["protocol_sha256"] != PROTOCOL_SHA256
            or len(state["train_ids"]) != 50 or len(state["dev_ids"]) != 20
            or set(state["train_ids"]) & set(state["dev_ids"])
            or state["weights"]["head.weight"].tolist() != [list(head.weights)]
            or state["weights"]["head.bias"].tolist() != [head.bias]):
        raise ValueError("Sri checkpoint contents differ from the verified export")
    if not path.exists():
        path.write_bytes(data)
    exported = root / f"sri-seed{seed}-stop_head.json"
    exported.write_text(json.dumps(head.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return exported
