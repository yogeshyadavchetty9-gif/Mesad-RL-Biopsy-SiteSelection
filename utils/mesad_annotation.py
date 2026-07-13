"""
MESAD Annotation Parser
Reads the paired annotation files produced by the MESAD dataset.

MESAD stores annotations in TWO files per image:

    <basename>.bboxes.tsv
        No header.  Each line is one bounding box:
            x1 <TAB> y1 <TAB> x2 <TAB> y2
        Coordinates are in pixel space relative to the original image.

    <basename>.bboxes.labels.tsv   (optional but usually present)
        No header.  One label per line, matching the rows in .bboxes.tsv.

build_index() pairs every image with its .bboxes.tsv file (and the matching
.bboxes.labels.tsv if present) and returns a list of dicts ready for MESADEnv.
"""

import os


# ────────────────────────────────────────────────────────────────────────────
# Low-level readers
# ────────────────────────────────────────────────────────────────────────────

def parse_bboxes(bboxes_path: str) -> list[tuple[float, float, float, float]]:
    """
    Read a .bboxes.tsv file.

    Returns a list of (x1, y1, x2, y2) tuples in pixel coordinates.
    Skips blank lines and lines that cannot be parsed as four numbers.
    """
    boxes = []
    if not os.path.exists(bboxes_path):
        return boxes
    with open(bboxes_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                boxes.append((x1, y1, x2, y2))
            except ValueError:
                continue
    return boxes


def parse_labels(labels_path: str) -> list[str]:
    """
    Read a .bboxes.labels.tsv file.

    Returns a list of label strings, one per line.
    """
    labels = []
    if not os.path.exists(labels_path):
        return labels
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                labels.append(stripped)
    return labels


def parse_mesad_annotations(base_path: str) -> list[dict]:
    """
    Parse the MESAD annotation pair for one image.

    Parameters
    ----------
    base_path : str
        Full path *without* the extension suffix, e.g.
        ``/data/val/annotations/real3_frame_5``.
        The function appends ``.bboxes.tsv`` and ``.bboxes.labels.tsv``.

    Returns
    -------
    list of dicts, one per bounding box::

        {
            "label": str | None,
            "x1": float,
            "y1": float,
            "x2": float,
            "y2": float,
        }
    """
    bboxes_path = base_path + ".bboxes.tsv"
    labels_path = base_path + ".bboxes.labels.tsv"

    boxes  = parse_bboxes(bboxes_path)
    labels = parse_labels(labels_path)

    records = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        records.append({
            "label": labels[i] if i < len(labels) else None,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return records


# Keep a legacy shim so any code that calls parse_tsv() still works.
# It tries the MESAD paired format first, then falls back to a single TSV.
def parse_tsv(tsv_path: str) -> list[dict]:
    """
    Compatibility shim.  Prefer parse_mesad_annotations() for new code.

    If *tsv_path* ends in ``.bboxes.tsv`` the MESAD paired parser is used.
    Otherwise the file is read as a plain headerless bbox TSV.
    """
    if tsv_path.endswith(".bboxes.tsv"):
        base = tsv_path[: -len(".bboxes.tsv")]
        return parse_mesad_annotations(base)

    # Plain fallback: headerless x1 y1 x2 y2
    records = []
    if not os.path.exists(tsv_path):
        return records
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                records.append({"label": None, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            except ValueError:
                continue
    return records


# ────────────────────────────────────────────────────────────────────────────
# Dataset index builder
# ────────────────────────────────────────────────────────────────────────────

def build_index(image_dir: str, annotation_dir: str) -> list[dict]:
    """
    Pair every image in *image_dir* with its MESAD annotation files.

    For each image ``<basename>.jpg`` the function looks for::

        <annotation_dir>/<basename>.bboxes.tsv          ← required for scoring
        <annotation_dir>/<basename>.bboxes.labels.tsv   ← optional labels

    Images without a matching ``.bboxes.tsv`` are included with an empty
    annotation list (the env treats them as unannotated, reward = 0).

    Returns
    -------
    list of dicts::

        {
            "image_path":  str,
            "annotations": list[dict],   # as returned by parse_mesad_annotations()
        }
    """
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    img_map: dict[str, str] = {}
    for fname in sorted(os.listdir(image_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() in img_exts:
            img_map[base] = os.path.join(image_dir, fname)

    index = []
    for base, img_path in img_map.items():
        base_annot = os.path.join(annotation_dir, base)
        annots = parse_mesad_annotations(base_annot)
        index.append({"image_path": img_path, "annotations": annots})

    return index


# ────────────────────────────────────────────────────────────────────────────
# Coordinate utilities
# ────────────────────────────────────────────────────────────────────────────

def get_grid_cell(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    grid_cols: int = 4, grid_rows: int = 4,
) -> int:
    """
    Given an annotation bounding box in pixel space, return the grid cell
    index (row-major) that best covers the box centre.

    Cell numbering::

        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15    (for a 4×4 grid)
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    col = int(cx / img_w * grid_cols)
    row = int(cy / img_h * grid_rows)
    col = min(col, grid_cols - 1)
    row = min(row, grid_rows - 1)
    return row * grid_cols + col


def cell_to_bbox(
    cell: int, img_w: int, img_h: int,
    grid_cols: int = 4, grid_rows: int = 4,
) -> tuple[int, int, int, int]:
    """Return the pixel bounding box (x1, y1, x2, y2) for a grid cell."""
    row = cell // grid_cols
    col = cell  % grid_cols
    cw  = img_w // grid_cols
    ch  = img_h // grid_rows
    x1  = col * cw
    y1  = row * ch
    x2  = x1 + cw
    y2  = y1 + ch
    return x1, y1, x2, y2


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union of two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
