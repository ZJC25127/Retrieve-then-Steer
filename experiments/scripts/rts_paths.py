"""Paths follow the original OpenPI/experiments layout, relative to this file."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent
VLAC_ROOT = ROOT / "vlac"
OPENPI = Path(os.environ.get("RTS_OPENPI_ROOT", str(RELEASE / "openpi"))).expanduser().resolve()


def configure_video_saving():
    """Keep the original episode-end callback while suppressing video I/O."""
    if os.environ.get("RTS_SAVE_VIDEOS", "0") != "1":
        import imageio

        imageio.mimwrite = lambda *args, **kwargs: None
