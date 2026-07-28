"""Fetch the slice cache and the shared-segmenter checkpoint from Hugging Face.

The repo `douyeszn/tumor-aware-sr` holds the rebuilt enhancing-tumor cache and
the checkpoints trained against one frozen segmenter. Nothing here is generated
locally, so a fresh machine can go from clone to demo without a GPU-hours detour.
"""
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "douyeszn/tumor-aware-sr"
ROOT = os.environ.get("TRUSTMRI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__)))))
DEST = os.path.join(ROOT, "safety", "assets")

WANT = [
    "data/et_full.npz",          # 288 MB, 17,233 slices from 468 cases
    "shared/sh_w40_sl0.0.pt",    # SR pair + the shared frozen segmenter
    "shared/sh_w40_sl0.5.pt",
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for f in WANT:
        try:
            p = hf_hub_download(REPO, f, local_dir=DEST)
            print(f"  OK   {f:28s} {os.path.getsize(p)/1e6:8.1f} MB")
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {f}: {type(e).__name__} {str(e)[:120]}")
            print("       (if this is a 401/GatedRepoError the repo is gated again;"
                  " ungate it or pass HF_TOKEN)")
            return 1
    print(f"\nassets in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
