from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FreewayExtractorResult:
    state: np.ndarray
    debug: dict
    overlay_data: dict


@dataclass
class FreewayFeatureExtractor:
    """Extract compact lane-local hazard features from ALE Freeway RGB frames.

    The constants below describe the NTSC Freeway playfield used by
    ``ALE/Freeway-v5``. They are intentionally named and centralized because
    the validation script is responsible for checking them against installed
    ALE frames.
    """

    playable_y_min: int = 30
    playable_y_max: int = 188
    chicken_x_min: int = 30
    chicken_x_max: int = 62
    chicken_spawn_y_min: int = 180
    chicken_spawn_y_max: int = 200
    chicken_spawn_center: tuple[int, int] = (47, 191)
    collision_x_margin: int = 12
    lane_count: int = 10
    y_bins: int = 12
    gap_bins: int = 16
    nvec: tuple[int, ...] = (11, 12, 16, 16, 16, 2, 2, 2)
    last_valid_state: np.ndarray | None = None
    failure_count: int = 0
    fallback_count: int = 0
    last_chicken_center: tuple[int, int] | None = None
    hidden_chicken_frames: int = 0
    max_hidden_chicken_frames: int = 30
    last_debug: dict = field(default_factory=dict)
    last_overlay_data: dict = field(default_factory=dict)

    def reset_tracking(self) -> None:
        self.last_valid_state = None
        self.last_chicken_center = None
        self.hidden_chicken_frames = 0
        self.last_debug = {}
        self.last_overlay_data = {}

    def _is_collision_or_respawn_reacquire(self, candidate: dict) -> bool:
        if self.last_chicken_center is None:
            return False

        cx, cy = candidate["center"]
        px, py = self.last_chicken_center

        dx = abs(cx - px)
        dy_down = cy - py

        # Freeway collision can knock the chicken downward by a large amount,
        # especially with frameskip=4. This is not an extractor failure.
        return (
            dx <= 20
            and dy_down >= 25
            and self.playable_y_min <= cy <= self.chicken_spawn_y_max
            and self.chicken_x_min <= cx <= self.chicken_x_max
        )

    def extract(
        self,
        frame: np.ndarray,
        *,
        strict: bool = True,
        allow_fallback: bool = False,
    ) -> FreewayExtractorResult:
        debug: dict = {
            "fallback_used": False,
            "failure_count": int(self.failure_count),
            "fallback_count": int(self.fallback_count),
        }
        try:
            frame = self._validate_frame(frame)
            lane_bands = self.lane_bands
            chicken = self._detect_chicken(frame)
            lane = self._lane_for_y(chicken["center"][1])
            y_bin = self._y_bin(chicken["center"][1])

            lane_indices = [
                lane,
                self._neighbor_lane(lane, -1),
                self._neighbor_lane(lane, 1),
            ]
            features = [self._lane_hazard(frame, idx, chicken["center"][0]) for idx in lane_indices]
            state = np.asarray(
                [
                    lane,
                    y_bin,
                    features[0]["gap_bin"],
                    features[1]["gap_bin"],
                    features[2]["gap_bin"],
                    features[0]["blocked"],
                    features[1]["blocked"],
                    features[2]["blocked"],
                ],
                dtype=np.int64,
            )
            self._validate_state(state)

            overlay_data = {
                "chicken": chicken,
                "lane_bands": lane_bands,
                "selected_lanes": {
                    "current": lane_indices[0],
                    "next": lane_indices[1],
                    "previous": lane_indices[2],
                },
                "lane_features": features,
                "state": state.copy(),
            }
            debug.update(
                {
                    "chicken_detected": True,
                    "chicken_center": tuple(int(v) for v in chicken["center"]),
                    "chicken_bbox": tuple(int(v) for v in chicken["bbox"]),
                    "chicken_detection_source": chicken["source"],
                    "chicken_candidates": chicken["candidate_count"],
                    "chicken_lane": int(lane),
                    "state": state.tolist(),
                    "failure_count": int(self.failure_count),
                    "fallback_count": int(self.fallback_count),
                }
            )
            self.last_valid_state = state.copy()
            self.last_chicken_center = tuple(int(v) for v in chicken["center"])
            self.last_debug = dict(debug)
            self.last_overlay_data = dict(overlay_data)
            return FreewayExtractorResult(state=state, debug=debug, overlay_data=overlay_data)
        except Exception as exc:
            self.failure_count += 1
            debug.update(
                {
                    "chicken_detected": False,
                    "error": str(exc),
                    "failure_count": int(self.failure_count),
                    "fallback_count": int(self.fallback_count),
                }
            )
            if allow_fallback and self.last_valid_state is not None:
                self.fallback_count += 1
                debug["fallback_used"] = True
                debug["fallback_count"] = int(self.fallback_count)
                overlay_data = dict(self.last_overlay_data)
                overlay_data["fallback_used"] = True
                return FreewayExtractorResult(
                    state=self.last_valid_state.copy(),
                    debug=debug,
                    overlay_data=overlay_data,
                )
            if strict:
                raise RuntimeError(f"Freeway feature extraction failed: {debug}") from exc
            raise

    @property
    def lane_bands(self) -> list[tuple[int, int]]:
        height = self.playable_y_max - self.playable_y_min
        bands: list[tuple[int, int]] = []
        for idx in range(self.lane_count):
            y0 = self.playable_y_min + int(round(idx * height / self.lane_count))
            y1 = self.playable_y_min + int(round((idx + 1) * height / self.lane_count))
            bands.append((y0, y1))
        return bands

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.shape != (210, 160, 3):
            raise ValueError(f"Expected ALE Freeway RGB frame shape (210, 160, 3), received {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Expected integer RGB frame, received dtype {arr.dtype}")
        return arr.astype(np.uint8, copy=False)

    def _detect_chicken(self, frame: np.ndarray) -> dict:
        candidates = self._chicken_candidates(frame)
        if self.last_chicken_center is None:
            spawn_candidates = [
                candidate
                for candidate in candidates
                if self.chicken_spawn_y_min <= candidate["center"][1] <= self.chicken_spawn_y_max
                and self.chicken_x_min <= candidate["center"][0] <= self.chicken_x_max
            ]
            if spawn_candidates:
                best = min(
                    spawn_candidates,
                    key=lambda c: self._squared_distance(c["center"], self.chicken_spawn_center),
                )
                best["source"] = "bottom_spawn_candidate"
                best["candidate_count"] = len(candidates)
                return best

            # Some ALE versions expose no controlled chicken in the reset frame.
            # Seed the tracker from the documented left-player spawn so the first
            # stepped frame selects the nearby controlled sprite, not road cars.
            x, y = self.chicken_spawn_center
            return {
                "bbox": (x - 4, y - 5, x + 4, y + 5),
                "center": self.chicken_spawn_center,
                "area": 0,
                "source": "spawn_prior_no_visible_candidate",
                "candidate_count": len(candidates),
            }

        if not candidates:
            if self.last_chicken_center[1] >= self.chicken_spawn_y_min:
                self.hidden_chicken_frames += 1
                x, y = self.chicken_spawn_center
                return {
                    "bbox": (x - 4, y - 5, x + 4, y + 5),
                    "center": self.chicken_spawn_center,
                    "area": 0,
                    "source": "spawn_hidden_prior",
                    "candidate_count": 0,
                }
            if self.last_chicken_center[1] <= self.playable_y_min:
                self.hidden_chicken_frames += 1
                x, y = self.chicken_spawn_center
                return {
                    "bbox": (x - 4, y - 5, x + 4, y + 5),
                    "center": self.chicken_spawn_center,
                    "area": 0,
                    "source": "top_crossing_spawn_prior",
                    "candidate_count": 0,
                }
            self.hidden_chicken_frames += 1
            if self.hidden_chicken_frames <= self.max_hidden_chicken_frames:
                x, y = self.last_chicken_center
                return {
                    "bbox": (x - 4, y - 5, x + 4, y + 5),
                    "center": self.last_chicken_center,
                    "area": 0,
                    "source": "occluded_previous_center_prior",
                    "candidate_count": 0,
                }
            raise RuntimeError("No chicken-shaped controlled-player candidates found")
        self.hidden_chicken_frames = 0

        plausible = [candidate for candidate in candidates if self._is_temporally_plausible_chicken(candidate)]
        if not plausible:
            reacquire_candidates = [
                c for c in candidates
                if self._is_collision_or_respawn_reacquire(c)
            ]
            if reacquire_candidates:
                best = min(
                    reacquire_candidates,
                    key=lambda c: (
                        abs(c["center"][0] - self.chicken_spawn_center[0]),
                        abs(c["center"][1] - self.last_chicken_center[1]),
                    ),
                )
                best["source"] = "collision_or_respawn_reacquire"
                best["candidate_count"] = len(candidates)
                return best

            best = min(candidates, key=lambda c: self._squared_distance(c["center"], self.last_chicken_center))
            if self._squared_distance(best["center"], self.last_chicken_center) > 55**2:
                raise RuntimeError(
                    "Closest chicken candidate is too far from previous controlled center: "
                    f"previous={self.last_chicken_center}, best={best}"
                )
        else:
            best = min(plausible, key=lambda c: self._squared_distance(c["center"], self.last_chicken_center))

        best["source"] = "nearest_previous_center"
        best["candidate_count"] = len(candidates)
        return best

    def _chicken_candidates(self, frame: np.ndarray) -> list[dict]:
        mask = self._chicken_mask(frame)
        components = self._connected_components(mask)
        candidates: list[dict] = []
        for component in components:
            x0, y0, x1, y1 = component["bbox"]
            width = x1 - x0
            height = y1 - y0
            area = int(component["area"])
            aspect = width / max(1, height)

            # Controlled chicken sprites are compact vertical yellow bodies.
            # Yellow cars and lane dividers are wider, thinner, or too large.
            if not (10 <= area <= 32):
                continue
            if not (5 <= width <= 7 and 6 <= height <= 14):
                continue
            if not (0.45 <= aspect <= 1.35):
                continue
            if not (self.chicken_x_min <= (x0 + x1) // 2 <= self.chicken_x_max):
                continue

            center = ((x0 + x1) // 2, (y0 + y1) // 2)
            candidates.append(
                {
                    "bbox": (int(x0), int(y0), int(x1), int(y1)),
                    "center": (int(center[0]), int(center[1])),
                    "area": area,
                    "aspect": float(aspect),
                }
            )
        candidates.sort(key=lambda c: (c["center"][1], c["center"][0]))
        return candidates

    def _chicken_mask(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.int16)
        green = frame[:, :, 1].astype(np.int16)
        blue = frame[:, :, 2].astype(np.int16)
        yellow = (red > 180) & (green > 180) & (blue < 140)
        white = (red > 210) & (green > 210) & (blue > 180)
        mask = yellow | white
        roi_mask = np.zeros(mask.shape, dtype=bool)
        roi_mask[0:self.chicken_spawn_y_max + 1, self.chicken_x_min:self.chicken_x_max] = True
        return mask & roi_mask

    @staticmethod
    def _squared_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return int((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _is_temporally_plausible_chicken(self, candidate: dict) -> bool:
        if self.last_chicken_center is None:
            return True
        dx = abs(candidate["center"][0] - self.last_chicken_center[0])
        dy = abs(candidate["center"][1] - self.last_chicken_center[1])
        if dx > 18 or dy > 24:
            return False
        # Lane cars move horizontally while staying at nearly fixed y. The
        # controlled chicken stays in the left x column and changes mostly in y.
        return not (dx > 8 and dy <= 2)

    def _connected_components(self, mask: np.ndarray) -> list[dict]:
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[dict] = []
        height, width = mask.shape
        for start_y, start_x in np.argwhere(mask):
            if visited[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            components.append(
                {
                    "area": len(pixels),
                    "bbox": (min(xs), min(ys), max(xs) + 1, max(ys) + 1),
                }
            )
        return components

    def _lane_for_y(self, y: int) -> int:
        for idx, (y0, y1) in enumerate(self.lane_bands):
            if y0 <= y < y1:
                return int(idx)
        return 10

    def _neighbor_lane(self, lane: int, delta: int) -> int:
        if lane < 0 or lane >= self.lane_count:
            return 10
        value = lane + delta
        return int(value) if 0 <= value < self.lane_count else 10

    def _y_bin(self, y: int) -> int:
        clipped = np.clip(int(y), self.playable_y_min, self.playable_y_max - 1)
        frac = (clipped - self.playable_y_min) / max(1, self.playable_y_max - self.playable_y_min)
        return int(np.clip(np.floor(frac * self.y_bins), 0, self.y_bins - 1))

    def _lane_hazard(self, frame: np.ndarray, lane: int, chicken_x: int) -> dict:
        if lane == 10:
            return {"lane": int(lane), "blocked": 0, "gap_bin": self.gap_bins - 1, "car_segments": []}
        y0, y1 = self.lane_bands[lane]
        lane_img = frame[y0:y1]
        mask = self._car_mask(lane_img)
        col_counts = mask.sum(axis=0)
        occupied_cols = col_counts > max(2, (y1 - y0) // 5)
        segments = self._segments_from_mask(occupied_cols)

        x_min = max(0, int(chicken_x) - self.collision_x_margin)
        x_max = min(frame.shape[1], int(chicken_x) + self.collision_x_margin + 1)
        blocked = int(bool(occupied_cols[x_min:x_max].any()))

        nearest_distance = frame.shape[1]
        for sx0, sx1 in segments:
            if sx1 < chicken_x:
                nearest_distance = min(nearest_distance, chicken_x - sx1)
            elif sx0 > chicken_x:
                nearest_distance = min(nearest_distance, sx0 - chicken_x)
            else:
                nearest_distance = 0
        gap_bin = int(np.clip(np.floor(nearest_distance / (frame.shape[1] / self.gap_bins)), 0, self.gap_bins - 1))
        return {
            "lane": int(lane),
            "blocked": blocked,
            "gap_bin": gap_bin,
            "car_segments": [(int(a), int(b), int(y0), int(y1)) for a, b in segments],
            "collision_band": (int(x_min), int(x_max), int(y0), int(y1)),
        }

    def _car_mask(self, lane_img: np.ndarray) -> np.ndarray:
        red = lane_img[:, :, 0].astype(np.int16)
        green = lane_img[:, :, 1].astype(np.int16)
        blue = lane_img[:, :, 2].astype(np.int16)
        maxc = np.maximum.reduce([red, green, blue])
        minc = np.minimum.reduce([red, green, blue])
        saturated = (maxc > 70) & ((maxc - minc) > 35)
        chicken_like = (red > 180) & (green > 180) & (blue < 140)
        road_divider = (red > 170) & (green > 170) & (blue < 80)
        return saturated & ~chicken_like & ~road_divider

    def _segments_from_mask(self, occupied: np.ndarray) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for idx, value in enumerate(occupied):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start >= 3:
                    segments.append((start, idx))
                start = None
        if start is not None and len(occupied) - start >= 3:
            segments.append((start, len(occupied)))
        return segments

    def _validate_state(self, state: np.ndarray) -> None:
        if state.shape != (8,) or state.dtype != np.int64:
            raise RuntimeError(f"Invalid Freeway symbolic state shape/dtype: {state.shape}, {state.dtype}")
        nvec = np.asarray(self.nvec, dtype=np.int64)
        if not bool(np.all(state >= 0) and np.all(state < nvec)):
            raise RuntimeError(f"Freeway symbolic state outside nvec {self.nvec}: {state!r}")
