"""
Evaluate a trained checkpoint, calibrate it, and export it to ONNX.

    python export_face.py

Split out from train_face.py so the expensive part does not have to be
repeated to redo the cheap part. Training writes
`best-vit-emotion-v3/`; everything after that - held-out scoring,
temperature fitting, ONNX export - runs from those weights in a couple
of minutes.

It also avoids a Windows-specific failure: DataLoader workers spawn
fresh processes that each re-import torch and map its CUDA DLLs, and
with enough workers that exhausts the paging file
("[WinError 1455] The paging file is too small"). Evaluation is not
augmentation-bound, so it defaults to `--workers 0` and sidesteps the
problem entirely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# torch's ONNX exporter logs progress with emoji. The Windows console
# defaults to cp1252, which cannot encode them, and the export dies on
# a UnicodeEncodeError in a print statement rather than on anything to
# do with the model. Force UTF-8 before torch is imported.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR.parent / "vit-emotion-v2-deployment"

sys.path.insert(0, str(DEPLOY_DIR))

from labels import NUM_SHARED_LABELS, SHARED_LABELS  # noqa: E402

from train_face import (  # noqa: E402
    IMAGE_SIZE,
    MEAN,
    STD,
    SEED,
    FaceFolder,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    index_folder,
    softmax_np,
)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=BASE_DIR / "best-vit-emotion-v3")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "vit-emotion-v3.onnx")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 avoids the Windows paging-file limit; see the docstring.")
    parser.add_argument("--cpu", action="store_true")

    arguments = parser.parse_args()

    if not arguments.checkpoint.is_dir():
        raise SystemExit(
            f"{arguments.checkpoint} not found. Run train_face.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu"
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"device: {device}")

    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        arguments.checkpoint
    ).to(device)

    # The head width is the contract with the deployment. A checkpoint
    # with a different number of outputs would export cleanly and then
    # mislabel everything downstream, so it is checked here rather than
    # discovered later.
    assert model.config.num_labels == NUM_SHARED_LABELS, (
        f"checkpoint has {model.config.num_labels} classes, "
        f"expected {NUM_SHARED_LABELS}"
    )

    # ---------------- validation, for the temperature ----------------

    paths, labels = index_folder(arguments.data / "train")

    rng = np.random.default_rng(SEED)
    val_idx = []
    for c in range(NUM_SHARED_LABELS):
        members = np.where(labels == c)[0]
        rng.shuffle(members)
        val_idx.extend(members[: max(1, int(0.08 * len(members)))])
    val_idx = np.array(val_idx)

    loader_kwargs = dict(num_workers=arguments.workers,
                         pin_memory=device.type == "cuda")

    val_loader = DataLoader(
        FaceFolder(paths[val_idx], labels[val_idx], False),
        arguments.batch_size, shuffle=False, **loader_kwargs,
    )

    print(f"validation {len(val_idx)} images ...", flush=True)
    validation = evaluate(model, val_loader, device, amp_dtype)

    temperature = fit_temperature(validation["logits"], validation["labels"])

    # ---------------- held-out test ----------------

    test_paths, test_labels = index_folder(arguments.data / "fer2013_test")

    test_loader = DataLoader(
        FaceFolder(test_paths, test_labels, False),
        arguments.batch_size, shuffle=False, **loader_kwargs,
    )

    print(f"held-out FER2013 test {len(test_labels)} images ...", flush=True)
    test = evaluate(model, test_loader, device, amp_dtype)

    ece_raw = expected_calibration_error(softmax_np(test["logits"]), test["labels"])
    ece_cal = expected_calibration_error(
        softmax_np(test["logits"] / temperature), test["labels"]
    )

    print("\n" + "=" * 68)
    print("HELD-OUT FER2013 TEST  (never seen in training)")
    print("=" * 68)
    print(f"  accuracy    {test['accuracy'] * 100:.2f}%")
    print(f"  macro-F1    {test['macro_f1'] * 100:.2f}%")
    print(f"  temperature {temperature:.3f}   ECE {ece_raw:.4f} -> {ece_cal:.4f}")
    print("\n  per-class F1 (%)")
    for name, value in test["per_class_f1"].items():
        print(f"    {name:<9} {value * 100:6.2f}")
    print("\n  deployed v2 on these same images: 74.40% acc / 73.72% macro-F1")
    print("=" * 68)

    # ---------------- export ----------------

    model.eval().cpu()

    class Wrapper(nn.Module):
        """Unwrap the HF output so ONNX sees a plain logits tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        Wrapper(model), dummy, str(arguments.out),
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18, do_constant_folding=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(arguments.out),
                                   providers=["CPUExecutionProvider"])

    with torch.no_grad():
        torch_out = Wrapper(model)(dummy).numpy()
    onnx_out = session.run(None, {"pixel_values": dummy.numpy()})[0]

    drift = float(np.abs(torch_out - onnx_out).max())

    print(f"\nONNX export: max |torch - onnx| = {drift:.2e}")
    assert drift < 1e-3, "ONNX export diverged from the torch model"
    assert onnx_out.shape == (2, NUM_SHARED_LABELS), onnx_out.shape

    # Batch 2 is exercised deliberately: flip-TTA sends two views in one
    # call, so a static batch axis would break it at runtime.
    print("signature: pixel_values [N,3,224,224] -> logits [N,7], dynamic batch")

    summary = {}
    summary_path = arguments.checkpoint.parent / "train.log"

    config = {
        "image_size": IMAGE_SIZE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "temperature": round(temperature, 4),
        "id2label": {str(i): n for i, n in enumerate(SHARED_LABELS)},
        "_provenance": {
            "datasets": "FER2013 train + AffectNet + RAF-DB (77,003 images)",
            "heldout_test": "FER2013 test (7,178 images)",
            "test_accuracy": round(test["accuracy"], 4),
            "test_macro_f1": round(test["macro_f1"], 4),
            "per_class_f1": {k: round(v, 4)
                             for k, v in test["per_class_f1"].items()},
            "ece_raw": round(ece_raw, 4),
            "ece_calibrated": round(ece_cal, 4),
            "baseline_v2": {"test_accuracy": 0.7440, "test_macro_f1": 0.7372},
        },
    }

    config_path = arguments.out.with_name("emotion_config.json")
    config_path.write_text(json.dumps(config, indent=4))

    print(f"\nWrote {arguments.out}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
