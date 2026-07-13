"""
MESADEnv — HistoGym-style Gymnasium environment for the MESAD dataset.

Adapts HistoGym's spatial-navigation paradigm to MESAD's JPG image patches
and TSV annotations.  The agent observes the full image (resized to
OBS_H × OBS_W grayscale) and selects one of GRID_COLS × GRID_ROWS cells as
the biopsy site in a single step.

Action Space  (Discrete 16 for the default 4×4 grid):
    Integer in [0, grid_cols * grid_rows).
    Cell index is row-major: action = row * grid_cols + col.

Observation Space:
    Box(0, 255, (OBS_H, OBS_W, 1), np.uint8)
    Full image resized to OBS_H × OBS_W, converted to grayscale, channel-last.
    SB3's VecTransposeImage converts this to (1, OBS_H, OBS_W) at load time.

Reward:
    +10.0   if selected cell overlaps an annotated lesion (IoU ≥ threshold)
    -2.0    if selected cell misses every annotated region
     0.0    when no annotation exists for this image (unannotated patch)

Episode:
    Always terminates after exactly one step (single-step MDP).

Compatibility:
    Stable-Baselines3 / standard Gymnasium (gymnasium ≥ 0.26)
"""

import os
import random
from typing import Optional

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.mesad_annotation import (
    build_index,
    cell_to_bbox,
    iou,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (override via constructor kwargs)
# ─────────────────────────────────────────────────────────────────────────────
GRID_COLS   = 4      # columns in the selection grid
GRID_ROWS   = 4      # rows    in the selection grid
OBS_W       = 128    # observation image width  (pixels)
OBS_H       = 128    # observation image height (pixels)
IOU_THRESH  = 0.3    # IoU threshold to count a selection as a "hit"


class MESADEnv(gym.Env):
    """
    HistoGym-style RL environment for the MESAD histopathology patch dataset.

    Single-step MDP: the agent sees the full image and picks one of
    grid_cols × grid_rows cells as the biopsy site.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing MESAD JPG images.
    annotation_dir : str
        Path to the directory containing matching TSV annotation files.
    grid_cols : int
        Number of grid columns.  Default 4.
    grid_rows : int
        Number of grid rows.  Default 4.
    obs_w : int
        Width  (pixels) of the observation image.  Default 128.
    obs_h : int
        Height (pixels) of the observation image.  Default 128.
    iou_thresh : float
        Minimum IoU to count a selection as hitting an annotated region.
    seed : int | None
        Optional RNG seed for reproducibility.

    Example
    -------
    >>> from mesad_env import MESADEnv
    >>> env = MESADEnv(image_dir="mesad-real/train/images",
    ...                annotation_dir="mesad-real/train/annotations")
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(3)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        grid_cols: int    = GRID_COLS,
        grid_rows: int    = GRID_ROWS,
        obs_w: int        = OBS_W,
        obs_h: int        = OBS_H,
        iou_thresh: float = IOU_THRESH,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.image_dir       = image_dir
        self.annotation_dir  = annotation_dir
        self.grid_cols       = grid_cols
        self.grid_rows       = grid_rows
        self.obs_w           = obs_w
        self.obs_h           = obs_h
        self.iou_thresh      = iou_thresh

        # Build dataset index: list of {image_path, annotations}
        self.index = build_index(image_dir, annotation_dir)
        if not self.index:
            raise FileNotFoundError(
                f"No images found in '{image_dir}'.  "
                "Check that the directory contains .jpg/.png files."
            )

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Action: direct cell selection (row-major index)
        self.action_space = spaces.Discrete(grid_cols * grid_rows)

        # Observation: full image resized to (obs_h, obs_w, 1) grayscale.
        # SB3's VecTransposeImage converts (H, W, 1) → (1, H, W).
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs_h, obs_w, 1),
            dtype=np.uint8,
        )

        # Internal state (initialised in reset())
        self._image:          Optional[np.ndarray] = None
        self._img_h:          int  = 0
        self._img_w:          int  = 0
        self._annotations:    list = []
        self._selected_cell:  int  = -1   # cell chosen in last step
        self._current_entry:  Optional[dict] = None

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── Core Gymnasium API ─────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample a random entry from the index
        entry = random.choice(self.index)
        self._current_entry = entry
        self._selected_cell = -1

        # Load image (BGR → RGB)
        img_bgr = cv2.imread(entry["image_path"])
        if img_bgr is None:
            raise IOError(f"Could not read image: {entry['image_path']}")
        self._image  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._img_h, self._img_w = self._image.shape[:2]

        self._annotations = entry["annotations"]

        obs  = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._selected_cell = int(action)
        reward = self._compute_reward(self._selected_cell)

        # Always terminate after one step
        obs  = self._get_observation()
        info = self._get_info()
        return obs, reward, True, False, info

    def render(self, mode: str = "human"):
        """
        Render current state.

        mode='human'      → display via OpenCV window
        mode='rgb_array'  → return an RGB numpy array
        """
        if self._image is None:
            return None

        canvas = self._image.copy()

        # Draw all annotated boxes (green)
        for annot in self._annotations:
            if None not in (annot["x1"], annot["y1"], annot["x2"], annot["y2"]):
                pt1 = (int(annot["x1"]), int(annot["y1"]))
                pt2 = (int(annot["x2"]), int(annot["y2"]))
                cv2.rectangle(canvas, pt1, pt2, (0, 200, 0), 2)

        # Draw grid (light blue)
        cw = self._img_w // self.grid_cols
        ch = self._img_h // self.grid_rows
        for c in range(1, self.grid_cols):
            cv2.line(canvas, (c * cw, 0), (c * cw, self._img_h), (100, 180, 255), 1)
        for r in range(1, self.grid_rows):
            cv2.line(canvas, (0, r * ch), (self._img_w, r * ch), (100, 180, 255), 1)

        # Highlight selected cell (yellow), if a step has been taken
        if self._selected_cell >= 0:
            x1, y1, x2, y2 = cell_to_bbox(
                self._selected_cell,
                self._img_w, self._img_h, self.grid_cols, self.grid_rows,
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)

        if mode == "human":
            bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imshow("MESADEnv", bgr)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return canvas

    def close(self):
        cv2.destroyAllWindows()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Return the full image resized to (obs_h, obs_w, 1) grayscale."""
        if self._image is None:
            return np.zeros((self.obs_h, self.obs_w, 1), dtype=np.uint8)
        resized = cv2.resize(self._image, (self.obs_w, self.obs_h))
        gray    = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]

    def _compute_reward(self, cell_idx: int) -> float:
        """
        Reward for selecting cell_idx.

        +10.0  if cell overlaps any annotation box with IoU ≥ iou_thresh.
        -2.0   if cell misses all annotation boxes.
         0.0   if the image has no spatial annotations (unannotated patch).
        """
        spatial = [
            a for a in self._annotations
            if None not in (a["x1"], a["y1"], a["x2"], a["y2"])
        ]
        if not spatial:
            return 0.0

        cursor_box = cell_to_bbox(
            cell_idx, self._img_w, self._img_h, self.grid_cols, self.grid_rows
        )
        for a in spatial:
            if iou(cursor_box, (a["x1"], a["y1"], a["x2"], a["y2"])) >= self.iou_thresh:
                return 10.0

        return -2.0

    def _get_info(self) -> dict:
        """Return diagnostic info dict (not used for training, useful for debugging)."""
        return {
            "image_path":    self._current_entry["image_path"] if self._current_entry else "",
            "selected_cell": self._selected_cell,
            "n_annotations": len(self._annotations),
        }

    # ── Convenience ───────────────────────────────────────────────────────

    def cell_to_rowcol(self, cell_idx: int) -> tuple[int, int]:
        """Convert flat cell index to (row, col)."""
        return divmod(cell_idx, self.grid_cols)
