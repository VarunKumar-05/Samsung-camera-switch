"""All the look-defining knobs in one place. Presets are plain dataclasses; override via JSON or CLI."""
import json
from dataclasses import asdict, dataclass, field, replace


@dataclass
class MergeParams:
    tile: int = 16
    levels: int = 4
    strictness: float = 1.0      # lower = rejects mismatched pixels more aggressively (less ghosting, less denoise)
    window: int = 5


@dataclass
class DenoiseParams:
    luma: float = 1.0            # strength multiplier on the noise-model sigma (0 = off)
    chroma: float = 2.0          # px of Gaussian blur applied to Cb/Cr
    spatial: int = 5             # bilateral window (px) used on the variance-stabilised raw planes


@dataclass
class SharpenParams:
    amount: float = 0.6          # unsharp strength on luma
    radius: float = 1.2          # px
    threshold: float = 0.004     # gradients below this (gamma-domain luma) are treated as noise and left alone
    halo: float = 0.02           # max overshoot beyond local min/max (suppresses halos)


@dataclass
class ToneParams:
    exposure_ev: float = 0.0     # added to auto exposure
    auto_expose: bool = True
    compress: float = 0.75       # <1 compresses the low-frequency (base) log-luminance range
    detail: float = 1.05         # >1 boosts local contrast
    shadows_ev: float = 0.0      # lift applied to dark base regions
    saturation: float = 1.05
    contrast: float = 0.15       # gentle S-curve strength (0 = none)
    white: float = 4.0           # shoulder white point (linear, relative to mid-grey 0.18 scaling)
    base_sigma_frac: float = 0.02  # base-layer blur as a fraction of image size


@dataclass
class Params:
    merge: MergeParams = field(default_factory=MergeParams)
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    sharpen: SharpenParams = field(default_factory=SharpenParams)
    tone: ToneParams = field(default_factory=ToneParams)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Params":
        p = Params()
        for name, cls in (("merge", MergeParams), ("denoise", DenoiseParams), ("sharpen", SharpenParams), ("tone", ToneParams)):
            if name in d:
                setattr(p, name, cls(**d[name]))
        return p


PRESETS = {
    # Faithful: minimal sharpening, modest tone compression. The "less aggressive than stock" look.
    "natural": Params(),
    # Almost untouched: great for judging the raw merge itself.
    "neutral": Params(denoise=DenoiseParams(luma=0.7, chroma=1.5), sharpen=SharpenParams(amount=0.25),
                      tone=ToneParams(compress=0.9, detail=1.0, saturation=1.0, contrast=0.05)),
    # Phone-ish: strong local contrast, shadows lifted, crisper.
    "punchy": Params(denoise=DenoiseParams(luma=1.3, chroma=2.5), sharpen=SharpenParams(amount=1.1, radius=1.0, threshold=0.003),
                     tone=ToneParams(compress=0.6, detail=1.3, shadows_ev=0.5, saturation=1.2, contrast=0.3)),
}


def preset(name: str) -> Params:
    return Params.from_dict(json.loads(json.dumps(asdict(PRESETS[name]))))  # deep copy
