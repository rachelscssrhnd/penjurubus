import os
from typing import Optional

try:
    from .config import ROOT
except Exception:
    ROOT = os.environ.get("PENJURUBUS_ROOT", r"D:/penjurubus")

RAW_OSM_DIR = os.path.join(ROOT, "data", "raw", "osm")
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")


def _find_in_tree(tree_root: str, basename: str) -> Optional[str]:
    if not os.path.isdir(tree_root):
        return None
    for dirpath, _, files in os.walk(tree_root):
        if basename in files:
            return os.path.join(dirpath, basename)
    return None


def resolve_path(path: str) -> str:
    """Resolve a data file path to the raw/osm copy if present, otherwise fall back to the parquet backup.

    Behavior:
    - If `path` exists as given, return it.
    - If basename exists under `data/raw/osm` (recursively), return that path.
    - If basename exists under `data/parquet` (recursively), return that path.
    - Otherwise return the original `path`.
    """
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path

    basename = os.path.basename(path)
    # prefer raw/osm
    p = _find_in_tree(RAW_OSM_DIR, basename)
    if p:
        return p
    # fallback to parquet folder
    p = _find_in_tree(PARQUET_DIR, basename)
    if p:
        return p
    # last-resort: return original path (may be relative)
    return path


def resolve_join(*parts: str) -> str:
    return resolve_path(os.path.join(*parts))
