"""Vispy + PyQt6 GPU point-cloud viewer for the DESI catalog.

Renders up to ~18M points with a fly-through camera. Supports runtime
toggles for point size (luminosity vs apparent flux) and color (SPECTYPE,
redshift, absolute magnitude). LOD swaps to a 15% random subset during
fast motion so interactivity stays above 30 fps on large catalogs.
"""

from __future__ import annotations

import collections
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QLabel, QTextBrowser
from scipy.spatial import cKDTree
from vispy import scene
from vispy.color import Colormap, get_colormap
from vispy.util.quaternion import Quaternion as VispyQuat
from vispy.visuals.filters import Filter

from process import (
    PointCloud, SPECTYPE_GALAXY, SPECTYPE_QSO, SPECTYPE_OTHER,
)


# --- depth fog filter --------------------------------------------------------


class _FogFilter(Filter):
    """Depth fog using gl_Position.w (linear eye distance).

    Dims distant points linearly: full brightness at camera, floor
    brightness at fog_end Mpc.
    """

    _VERT = """
    varying float v_fog_w;
    void fog_vert() {
        v_fog_w = gl_Position.w;
    }
    """

    _FRAG = """
    varying float v_fog_w;
    void fog_frag() {
        if (v_fog_w > $render_dist) discard;
        float fog = clamp(1.0 - v_fog_w / $fog_end, $fog_floor, 1.0);
        gl_FragColor.a *= fog;
    }
    """

    def __init__(self, fog_end: float = 8000.0, floor: float = 0.15,
                 render_dist: float = 1e9):
        super().__init__(vcode=self._VERT, fcode=self._FRAG)
        self.fog_end = fog_end
        self.floor = floor
        self.render_dist = render_dist

    @property
    def fog_end(self) -> float:
        return self._fog_end

    @fog_end.setter
    def fog_end(self, v: float) -> None:
        self._fog_end = v
        self.fshader['fog_end'] = float(v)

    @property
    def floor(self) -> float:
        return self._floor

    @floor.setter
    def floor(self, v: float) -> None:
        self._floor = v
        self.fshader['fog_floor'] = float(v)

    @property
    def render_dist(self) -> float:
        return self._render_dist

    @render_dist.setter
    def render_dist(self, v: float) -> None:
        self._render_dist = v
        self.fshader['render_dist'] = float(v)


# --- visual tuning constants -------------------------------------------------

POINT_SIZE_MIN = 1.0
POINT_SIZE_MAX = 4.0
POINT_ALPHA = 0.7                        # semi-transparent for depth layering
LOD_FRACTION = 0.5                       # subset size when LOD active
LOD_IDLE_MS = 180                        # restore full set after this much idle
LOD_AUTO_THRESHOLD = 5_000_000           # default LOD on only when n >= this
FPS_WINDOW = 60                          # rolling frames for fps average
BOOST_FACTOR = 5.0                       # Shift-held speed multiplier
SPEED_WHEEL_STEP = 1.25                  # scroll-wheel speed multiplier
BASE_SPEED_MUL = 0.03                    # baseline _speed_mul — chosen so a
                                         # ~7 Gpc scene takes ~30s at cruise
BACKGROUND_COLOR = (0.0, 0.0, 0.015, 1.0)

PICK_RAY_SAMPLES = 2000
PICK_RADIUS_PX = 8.0
HIGHLIGHT_SIZE = 14.0
HIGHLIGHT_COLOR = np.array([[1.0, 1.0, 0.4, 0.9]], dtype=np.float32)

_MOVEMENT_KEYS = frozenset({
    "W", "A", "S", "D", "F", "C",
    "SHIFT", "LSHIFT", "RSHIFT",
    "UP", "DOWN", "LEFT", "RIGHT",
})


class _FlyCameraNoWheel(scene.cameras.FlyCamera):
    """FlyCamera with mouse-wheel disabled and a clean speed multiplier.

    The stock FlyCamera uses `scale_factor` for two unrelated things:
      1. Movement speed (velocity = speed * scale_factor * dt in on_timer).
      2. Perspective projection scale (fx = fy = scale_factor in
         _update_transform — this is what makes the scene look 'zoomed').

    So adjusting scale_factor to change speed visibly zooms/clips the scene,
    which is confusing. Here we freeze scale_factor (so the projection is
    stable) and instead post-multiply the position delta each tick by a
    separate `_speed_mul` that the Viewer owns.

    When `_lock_target` is set, the camera continuously re-aims at that
    world-space position every tick (mouse-look is suppressed).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._speed_mul = 1.0
        self._lock_target: np.ndarray | None = None
        self._lock_ref_dist: float = 1.0
        self._drag_active: bool = False
        self._drag_last_pos = np.zeros(2)

    def viewbox_mouse_event(self, event):    # type: ignore[override]
        if event.type == "mouse_wheel":
            return

        me = event.mouse_event

        if event.type == "mouse_press" and me._button == 2:
            self._drag_active = True
            self._drag_last_pos = np.array(me.pos[:2], dtype=float)
            self._viewbox.canvas.native.setCursor(
                Qt.CursorShape.BlankCursor)
            event.handled = True
            if not self._timer.running:
                self._timer.start()
            return

        if event.type == "mouse_release" and self._drag_active:
            self._drag_active = False
            self._viewbox.canvas.native.unsetCursor()
            self._rotation1 = (
                self._rotation2 * self._rotation1).normalize()
            self._rotation2 = VispyQuat()
            self._update_from_mouse = True
            event.handled = True
            return

        if event.type == "mouse_move":
            if self._lock_target is not None:
                return
            if not self._drag_active:
                return

            pos = np.array(me.pos[:2], dtype=float)
            delta = pos - self._drag_last_pos
            self._drag_last_pos = pos.copy()

            w, h = self._viewbox.size
            if abs(delta[0]) > w * 0.1 or abs(delta[1]) > h * 0.1:
                return

            d_az = float(delta[0] / w) * 0.5 * np.pi
            d_el = float(delta[1] / h) * 0.5 * np.pi

            q_az = VispyQuat.create_from_axis_angle(d_az, 0, 1, 0)
            q_el = VispyQuat.create_from_axis_angle(d_el, 1, 0, 0)
            self._rotation2 = (
                (q_el * q_az).normalize() * self._rotation2
            ).normalize()

            margin = min(w, h) * 0.05
            if (pos[0] < margin or pos[0] > w - margin
                    or pos[1] < margin or pos[1] > h - margin):
                native = self._viewbox.canvas.native
                QCursor.setPos(
                    native.mapToGlobal(native.rect().center()))

            self._update_from_mouse = True
            event.handled = True

    def on_timer(self, event):               # type: ignore[override]
        pre_eye = np.array(self._center, dtype=np.float64)

        if self._speed_mul == 1.0:
            super().on_timer(event)
        else:
            super().on_timer(event)
            new_center = np.array(self._center, dtype=np.float64)
            delta = new_center - pre_eye
            if np.any(delta):
                scaled = pre_eye + delta * self._speed_mul
                self.center = tuple(scaled)

        if self._lock_target is not None:
            eye = np.array(self._center, dtype=np.float64)
            raw_delta = eye - pre_eye
            pre_dir = self._lock_target - pre_eye
            pre_norm = float(np.linalg.norm(pre_dir))

            if np.any(raw_delta) and self._lock_ref_dist > 0:
                dist_scale = min(pre_norm / self._lock_ref_dist, 1.0)
                eye = pre_eye + raw_delta * dist_scale
                self._center = tuple(eye)

            to_tgt = self._lock_target - eye
            dist = float(np.linalg.norm(to_tgt))
            crossed = pre_norm > 1e-10 and np.dot(to_tgt, pre_dir) < 0
            min_dist = 0.1
            if dist < min_dist or crossed:
                approach = pre_dir / pre_norm if pre_norm > 1e-10 else np.array([0., 0., 1.])
                eye = self._lock_target - approach * min_dist
                self._center = tuple(eye)
            current_up = np.array(
                self._rotation1.inverse().rotate_point((0, 1, 0)),
                dtype=np.float64,
            )
            quat = _look_at_quat(eye, self._lock_target, up_hint=current_up)
            self._rotation1 = VispyQuat(*quat)
            self.view_changed()

# SPECTYPE palette: warm cream for galaxies, bright cyan for quasars (point sources)
PALETTE_SPECTYPE = np.array([
    [1.00, 0.92, 0.75, POINT_ALPHA],     # GALAXY
    [0.45, 0.85, 1.00, POINT_ALPHA],     # QSO
    [0.63, 0.63, 0.63, POINT_ALPHA],     # other
], dtype=np.float32)

LRG_BIT = np.int64(1 << 0)
ELG_BIT = np.int64(1 << 1)
QSO_BIT = np.int64(1 << 2)
BGS_ANY_BIT = np.int64(1 << 60)
SCND_ANY_BIT = np.int64(1 << 62)
MWS_ANY_BIT = np.int64(1 << 61)

PALETTE_SUBTYPE = np.array([
    [0.90, 0.25, 0.20, POINT_ALPHA],   # 0 LRG: red
    [0.30, 0.75, 1.00, POINT_ALPHA],   # 1 ELG: cyan
    [0.30, 0.90, 0.40, POINT_ALPHA],   # 2 BGS: green
    [1.00, 0.85, 0.20, POINT_ALPHA],   # 3 QSO: gold
    [0.70, 0.50, 0.85, POINT_ALPHA],   # 4 Secondary: purple
    [0.40, 0.40, 0.40, POINT_ALPHA],   # 5 Unclassified: dark gray
], dtype=np.float32)


# --- precomputed size / color arrays -----------------------------------------

def _normalize_percentile(values: np.ndarray,
                          lo_pct: float, hi_pct: float) -> np.ndarray:
    """Linearly map values into [0, 1] using the given percentile bounds."""
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return t


def _size_from_log_L(log_L: np.ndarray) -> np.ndarray:
    """Absolute luminosity sizing. Brighter (larger log_L) → bigger point."""
    t = _normalize_percentile(log_L, 5.0, 99.5)
    return (POINT_SIZE_MIN + t * (POINT_SIZE_MAX - POINT_SIZE_MIN)).astype(np.float32)


def _size_from_flux(flux_r: np.ndarray) -> np.ndarray:
    """Apparent flux sizing (legacy comparison mode)."""
    log_f = np.log10(np.clip(flux_r, 1e-3, None))
    t = _normalize_percentile(log_f, 5.0, 99.5)
    return (POINT_SIZE_MIN + t * (POINT_SIZE_MAX - POINT_SIZE_MIN)).astype(np.float32)


def _color_by_spectype(spectype: np.ndarray) -> np.ndarray:
    return PALETTE_SPECTYPE[spectype]


def _colormap_rgba(cmap_name: str, t: np.ndarray) -> np.ndarray:
    """Apply a Vispy colormap, return (N, 4) float32 RGBA."""
    cm = get_colormap(cmap_name)
    rgba = cm.map(t.reshape(-1, 1)).astype(np.float32)
    return rgba.reshape(-1, 4)


# Custom bright redshift gradient: cyan (nearby / low z) -> white -> red (far /
# high z). All three stops are bright, so nothing disappears against a black
# background — unlike viridis, whose low end is dark purple.
_REDSHIFT_CMAP = Colormap([
    (0.40, 0.85, 1.00, 1.0),   # z~0  : cyan
    (1.00, 1.00, 0.90, 1.0),   # z~1  : warm white
    (1.00, 0.40, 0.25, 1.0),   # z>=3 : red
])


def _color_by_redshift(z: np.ndarray) -> np.ndarray:
    t = np.clip((z - 0.001) / (4.0 - 0.001), 0.0, 1.0).astype(np.float32)
    rgba = _REDSHIFT_CMAP.map(t.reshape(-1, 1)).astype(np.float32).reshape(-1, 4)
    rgba[:, 3] = POINT_ALPHA
    return rgba


def _color_by_absmag(M_r: np.ndarray) -> np.ndarray:
    # Brighter (more negative) → hotter end of the fire colormap.
    # Robust bounds so one outlier doesn't collapse contrast.
    hi, lo = np.percentile(M_r, [99.5, 5.0])   # hi = least bright end
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((hi - M_r) / (hi - lo), 0.0, 1.0).astype(np.float32)
    # Floor the mapping at 0.25 so even the faintest galaxies aren't pure black.
    t = 0.25 + 0.75 * t
    rgba = _colormap_rgba("fire", t)
    rgba[:, 3] = POINT_ALPHA
    return rgba


def _color_by_subtype(desi_target: np.ndarray) -> np.ndarray:
    codes = np.full(len(desi_target), 5, dtype=np.uint8)  # unclassified
    codes[(desi_target & (SCND_ANY_BIT | MWS_ANY_BIT)) != 0] = 4  # secondary
    codes[(desi_target & BGS_ANY_BIT) != 0] = 2
    codes[(desi_target & ELG_BIT) != 0] = 1
    codes[(desi_target & LRG_BIT) != 0] = 0
    codes[(desi_target & QSO_BIT) != 0] = 3
    return PALETTE_SUBTYPE[codes]


_GR_CMAP = Colormap([
    (0.30, 0.60, 1.00, 1.0),
    (0.95, 0.95, 0.90, 1.0),
    (1.00, 0.35, 0.15, 1.0),
])


def _color_by_gr(flux_g: np.ndarray, flux_r: np.ndarray) -> np.ndarray:
    valid = (flux_g > 0) & (flux_r > 0)
    ratio = np.where(valid, flux_g / flux_r, 1.0)
    g_r = -2.5 * np.log10(ratio)
    t = np.clip((g_r - 0.0) / 1.5, 0.0, 1.0).astype(np.float32)
    rgba = _GR_CMAP.map(t.reshape(-1, 1)).astype(np.float32).reshape(-1, 4)
    rgba[:, 3] = POINT_ALPHA
    rgba[~valid] = [0.4, 0.4, 0.4, POINT_ALPHA * 0.5]
    return rgba


_LOOKBACK_CMAP = Colormap([
    (1.00, 0.95, 0.50, 1.0),
    (0.85, 0.50, 0.85, 1.0),
    (0.20, 0.30, 0.80, 1.0),
])


def _color_by_lookback(lookback_gyr: np.ndarray) -> np.ndarray:
    t = np.clip(lookback_gyr / 13.0, 0.0, 1.0).astype(np.float32)
    rgba = _LOOKBACK_CMAP.map(t.reshape(-1, 1)).astype(np.float32).reshape(-1, 4)
    rgba[:, 3] = POINT_ALPHA
    return rgba


# --- picking / look-at helpers -----------------------------------------------


def _mat3_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion (Shepperd)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _look_at_quat(eye: np.ndarray, target: np.ndarray,
                   up_hint: np.ndarray | None = None) -> np.ndarray:
    """World-to-camera quaternion [w,x,y,z] for camera at *eye* facing *target*."""
    fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d = np.linalg.norm(fwd)
    if d < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    fwd /= d
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    else:
        up_hint = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(fwd, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R_c2w = np.column_stack([right, up, -fwd])
    q = _mat3_to_quat(R_c2w)
    q[1:] *= -1  # conjugate → world-to-camera
    return q


# --- modes -------------------------------------------------------------------

@dataclass
class ViewerState:
    size_mode: str = "lum"              # "lum" | "flux"
    color_mode: str = "gr"              # spectype|redshift|absmag|subtype|gr|lookback
    camera_mode: str = "fly"            # "fly" | "turntable"
    lod_enabled: bool = True            # user can force-disable with K
    lod_active: bool = False
    base_speed_mul: float = 1.0
    selected_idx: int | None = None


# --- main viewer -------------------------------------------------------------

class Viewer:
    def __init__(self, pc: PointCloud, dataset: str = "DR1"):
        self.pc = pc
        self.state = ViewerState()

        print(f"[viewer] precomputing size arrays...")
        self.size_lum = _size_from_log_L(pc.log_L)
        self.size_flux = _size_from_flux(pc.flux_r)

        print(f"[viewer] precomputing color arrays (6 modes)...")
        self.color_spectype = _color_by_spectype(pc.spectype)
        self.color_redshift = _color_by_redshift(pc.z)
        self.color_absmag = _color_by_absmag(pc.M_r)
        self.color_subtype = _color_by_subtype(pc.desi_target)
        self.color_gr = _color_by_gr(pc.flux_g, pc.flux_r)
        self.color_lookback = _color_by_lookback(pc.lookback_gyr)

        n_subset = max(1, int(pc.n * LOD_FRACTION))
        perm = np.random.default_rng(42).permutation(pc.n)
        self.subset_idx = np.sort(perm[:n_subset])    # sorted = better memory locality

        self._kdtree: cKDTree | None = None
        self._kdtree_thread = threading.Thread(
            target=self._build_kdtree, daemon=True,
        )
        self._kdtree_thread.start()
        self._scene_extent = float(np.linalg.norm(pc.xyz, axis=1).max())

        # Default LOD on only for catalogs large enough to actually need it.
        # At ~1M points modern GPUs render full interactively; the flicker of
        # swapping to a subset during motion is more distracting than helpful.
        self.state.lod_enabled = False

        self.canvas = scene.SceneCanvas(
            title=f"DESI {dataset.upper()} Map Explorer",
            keys="interactive",
            size=(1400, 900),
            bgcolor=BACKGROUND_COLOR,
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()

        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.antialias = 1
        # Additive blending: overlapping points accumulate brightness, giving
        # bright regions a natural glow — closer to what a real star field
        # looks like than opaque discs.
        self.markers.set_gl_state(
            "additive", depth_test=False, blend=True,
            blend_func=("src_alpha", "one"),
        )
        self._fog = _FogFilter(floor=0.15)
        self.markers.attach(self._fog)

        self._highlight = scene.visuals.Markers(parent=self.view.scene)
        self._highlight.set_data(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=np.array([1.0]),
            face_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
            edge_width=0.0, symbol="disc",
        )
        self._highlight.set_gl_state(
            "translucent", depth_test=False, blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self._highlight.visible = False

        self._grid_visuals: list = []
        self._grid_visible = True
        self._build_grid()

        self._held_keys: set[str] = set()
        self._last_input_time = time.monotonic()
        self._frame_times: collections.deque[float] = collections.deque(maxlen=FPS_WINDOW)

        # Upload point data before attaching the camera — Vispy's camera
        # assignment triggers view_changed() -> set_range() which queries the
        # markers' bounds, and the visual must already have data.
        self._apply_data()

        self._fly = self._make_fly_camera()
        self._turn = scene.cameras.TurntableCamera(
            parent=self.view.scene, fov=60.0,
        )
        self.view.camera = self._fly

        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.key_release.connect(self._on_key_release)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self.canvas.events.draw.connect(self._on_draw)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)

        self._lod_timer = QTimer()
        self._lod_timer.setInterval(50)
        self._lod_timer.timeout.connect(self._tick_lod)
        self._lod_timer.start()

        self._help_label = QLabel(parent=self.canvas.native)
        self._help_label.setTextFormat(Qt.TextFormat.PlainText)
        self._help_label.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(5, 5, 20, 210);"
            "  color: rgba(255, 255, 255, 220);"
            "  border: 1px solid rgba(80, 80, 200, 120);"
            "  border-radius: 6px;"
            "  padding: 10px;"
            "  font-family: Consolas, monospace;"
            "  font-size: 9pt;"
            "}"
        )
        self._help_label.setText(self._help_content())
        self._help_label.adjustSize()
        self._help_label.move(10, 10)
        self._help_label.show()

        self._status_text = scene.visuals.Text(
            text="",
            parent=self.canvas.scene,
            color=(0.85, 0.85, 1.0, 0.9),
            anchor_x="right", anchor_y="bottom",
            font_size=9,
            pos=(self.canvas.size[0] - 10, self.canvas.size[1] - 10),
        )
        self._info_panel = QTextBrowser(parent=self.canvas.native)
        self._info_panel.setOpenExternalLinks(True)
        self._info_panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._info_panel.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_panel.setStyleSheet(
            "QTextBrowser {"
            "  background-color: rgba(5, 5, 20, 210);"
            "  color: #eeeedd;"
            "  border: 1px solid rgba(80, 80, 200, 120);"
            "  border-radius: 6px;"
            "  padding: 10px;"
            "  font-size: 9pt;"
            "}"
            "a { color: #77aaff; text-decoration: none; }"
        )
        self._info_panel.setFixedWidth(380)
        self._info_panel.hide()

        self._legend_label = QLabel(parent=self.canvas.native)
        self._legend_label.setTextFormat(Qt.TextFormat.RichText)
        self._legend_label.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(5, 5, 20, 210);"
            "  color: rgba(255, 255, 255, 220);"
            "  border: 1px solid rgba(80, 80, 200, 120);"
            "  border-radius: 6px;"
            "  padding: 10px;"
            "  font-family: Consolas, monospace;"
            "  font-size: 9pt;"
            "}"
        )
        self.canvas.events.resize.connect(self._on_resize)

        self._update_status()
        self._update_legend()

    # -- camera setup ---------------------------------------------------------

    def _make_fly_camera(self) -> scene.cameras.FlyCamera:
        cam = _FlyCameraNoWheel(parent=self.view.scene, fov=60.0)
        cam.auto_roll = False
        # Strip out keyboard yaw/pitch/roll bindings (mouse does looking).
        # Also free up I, J, K, L for our UI controls.
        to_drop = {"I", "J", "K", "L"}
        cam._keymap = {k: v for k, v in cam._keymap.items()
                       if k not in to_drop}
        return cam

    def _build_grid(self) -> None:
        radii = [1000, 2000, 4000, 6000]
        grid_alpha = 0.12
        n_seg = 80

        for r in radii:
            for gen in (self._sphere_ring_xy, self._sphere_ring_xz,
                        self._sphere_ring_yz):
                pts = gen(r, n_seg)
                line = scene.visuals.Line(
                    pos=pts, color=(0.5, 0.5, 0.7, grid_alpha),
                    parent=self.view.scene, width=1.0,
                )
                self._grid_visuals.append(line)

            label = scene.visuals.Text(
                text=f"{r} Mpc",
                pos=(r, 0, 0), color=(0.6, 0.6, 0.8, 0.35),
                font_size=8, parent=self.view.scene,
                anchor_x="left", anchor_y="bottom",
            )
            self._grid_visuals.append(label)

        axis_len = max(radii) * 1.1
        axis_colors = [
            (1.0, 0.3, 0.3, grid_alpha * 1.5),
            (0.3, 1.0, 0.3, grid_alpha * 1.5),
            (0.3, 0.3, 1.0, grid_alpha * 1.5),
        ]
        for i, color in enumerate(axis_colors):
            pts = np.zeros((2, 3), dtype=np.float32)
            pts[0, i] = -axis_len
            pts[1, i] = axis_len
            line = scene.visuals.Line(
                pos=pts, color=color,
                parent=self.view.scene, width=1.0,
            )
            self._grid_visuals.append(line)

    @staticmethod
    def _sphere_ring_xy(r: float, n: int) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n + 1, dtype=np.float32)
        pts = np.zeros((n + 1, 3), dtype=np.float32)
        pts[:, 0] = r * np.cos(t)
        pts[:, 1] = r * np.sin(t)
        return pts

    @staticmethod
    def _sphere_ring_xz(r: float, n: int) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n + 1, dtype=np.float32)
        pts = np.zeros((n + 1, 3), dtype=np.float32)
        pts[:, 0] = r * np.cos(t)
        pts[:, 2] = r * np.sin(t)
        return pts

    @staticmethod
    def _sphere_ring_yz(r: float, n: int) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n + 1, dtype=np.float32)
        pts = np.zeros((n + 1, 3), dtype=np.float32)
        pts[:, 1] = r * np.cos(t)
        pts[:, 2] = r * np.sin(t)
        return pts

    def _adjust_render_dist(self, key: str) -> None:
        factor = 1.5 if key == "]" else 1.0 / 1.5
        new_dist = self._fog.render_dist * factor
        new_dist = float(np.clip(new_dist, 10.0, self._scene_extent * 2.0))
        self._fog.render_dist = new_dist
        self._fog.fog_end = new_dist * 0.9
        print(f"[viewer] render distance: {new_dist:.0f} Mpc")
        self._update_status()
        self.canvas.update()

    def _toggle_grid(self) -> None:
        self._grid_visible = not self._grid_visible
        for v in self._grid_visuals:
            v.visible = self._grid_visible
        self.canvas.update()
        print(f"[viewer] grid {'on' if self._grid_visible else 'off'}")

    def _build_kdtree(self) -> None:
        t0 = time.monotonic()
        tree = cKDTree(self.pc.xyz)
        self._kdtree = tree
        print(f"[viewer] spatial index ready ({time.monotonic() - t0:.1f}s)")

    # -- data upload ----------------------------------------------------------

    def _current_size(self) -> np.ndarray:
        return self.size_lum if self.state.size_mode == "lum" else self.size_flux

    def _current_color(self) -> np.ndarray:
        m = self.state.color_mode
        if m == "spectype":
            return self.color_spectype
        if m == "redshift":
            return self.color_redshift
        if m == "absmag":
            return self.color_absmag
        if m == "subtype":
            return self.color_subtype
        if m == "gr":
            return self.color_gr
        return self.color_lookback

    def _apply_data(self) -> None:
        size = self._current_size()
        color = self._current_color()
        xyz = self.pc.xyz

        if self.state.lod_active and self.state.lod_enabled:
            idx = self.subset_idx
            self.markers.set_data(
                pos=xyz[idx], size=size[idx], face_color=color[idx],
                edge_width=0.0, symbol="disc",
            )
        else:
            self.markers.set_data(
                pos=xyz, size=size, face_color=color,
                edge_width=0.0, symbol="disc",
            )

    # -- status overlay -------------------------------------------------------

    def _help_content(self) -> str:
        return (
            "DESI Map Explorer — controls:\n"
            "  W/A/S/D  : move\n"
            "  F / C    : up / down\n"
            "  Q / E    : roll left / right\n"
            "  Shift    : 5x speed boost (hold)\n"
            "  RMB drag : look around\n"
            "  Wheel    : adjust speed\n"
            "  LMB      : select / deselect point\n"
            "  T        : fly <-> turntable camera\n"
            "  V        : cycle color mode\n"
            "               redshift / abs-mag / spectype /\n"
            "               subtype / g-r color / lookback time\n"
            "  [ / ]    : decrease / increase render distance\n"
            "  J        : toggle distance grid\n"
            "  G        : toggle color legend\n"
            "  L        : toggle size (luminosity <-> apparent flux)\n"
            "  K        : toggle LOD on/off\n"
            "  H        : toggle this help\n"
            "  Esc      : quit\n"
        )

    def _update_status(self) -> None:
        lod_str = "ON" if self.state.lod_enabled else "OFF"
        lod_active = " (subset)" if self.state.lod_active and self.state.lod_enabled else ""
        fps = self._current_fps()
        mem = _rss_mb()
        sel = f"  sel={self.state.selected_idx}" if self.state.selected_idx is not None else ""
        self._status_text.text = (
            f"n={self.pc.n:,}  size={self.state.size_mode}  "
            f"color={self.state.color_mode}  cam={self.state.camera_mode}  "
            f"spd={self.state.base_speed_mul:.1e}  "
            f"LOD={lod_str}{lod_active}  dist={self._fog.render_dist:.0f}Mpc  "
            f"fps={fps:5.1f}  rss={mem:.0f}MB{sel}  H=help"
        )

    def _on_resize(self, event) -> None:
        w, h = self.canvas.size
        self._status_text.pos = (w - 10, h - 10)
        self._info_panel.move(w - self._info_panel.width() - 10, 10)
        if self._legend_label.isVisible():
            self._legend_label.move(
                10, h - self._legend_label.height() - 30)

    # -- input ---------------------------------------------------------------

    def _key_str(self, event) -> str:
        """Return a canonical uppercase key string (or the special Key name)."""
        k = event.key
        if k is None:
            return ""
        name = k.name if hasattr(k, "name") else str(k)
        return name.upper() if len(name) == 1 else name

    def _on_key_press(self, event) -> None:
        key = self._key_str(event)
        self._held_keys.add(key)

        if key in ("SHIFT", "LSHIFT", "RSHIFT") and self.state.camera_mode == "fly":
            self._apply_speed()
            return

        if key == "L":
            self._toggle_size_mode()
        elif key == "V":
            self._cycle_color_mode()
        elif key == "T":
            self._toggle_camera()
        elif key == "K":
            self._toggle_lod_enabled()
        elif key in ("[", "]"):
            self._adjust_render_dist(key)
        elif key == "J":
            self._toggle_grid()
        elif key == "G":
            self._toggle_legend()
        elif key == "H":
            self._help_label.setVisible(not self._help_label.isVisible())
        elif key == "ESCAPE":
            self.canvas.close()

    def _on_key_release(self, event) -> None:
        key = self._key_str(event)
        self._held_keys.discard(key)
        # Only movement-key releases count as input activity for the LOD timer,
        # so UI toggles (L/V/T/K/H) don't flip LOD on for 180 ms each press.
        if key in _MOVEMENT_KEYS:
            self._last_input_time = time.monotonic()
        if key in ("SHIFT", "LSHIFT", "RSHIFT") and self.state.camera_mode == "fly":
            self._apply_speed()

    def _on_mouse_move(self, event) -> None:
        if event.is_dragging:
            self._last_input_time = time.monotonic()

    def _on_mouse_wheel(self, event) -> None:
        # Wheel only adjusts speed — do NOT mark this as motion input,
        # otherwise LOD flips on while adjusting speed and the scene flickers.
        delta = event.delta[1]
        factor = SPEED_WHEEL_STEP if delta > 0 else 1.0 / SPEED_WHEEL_STEP
        self.state.base_speed_mul *= factor
        self.state.base_speed_mul = float(np.clip(self.state.base_speed_mul, 1e-5, 100.0))
        self._apply_speed()
        self._update_status()

    def _on_mouse_press(self, event) -> None:
        if event.button != 1:
            return
        if self.state.selected_idx is not None:
            self._deselect()
            return
        if self.state.camera_mode != "fly":
            print("[viewer] point selection requires fly camera (press T)")
            return
        idx = self._pick_nearest(event.pos[0], event.pos[1])
        if idx is not None:
            self._select(idx)

    # -- action handlers ------------------------------------------------------

    def _apply_speed(self) -> None:
        if self.state.camera_mode != "fly":
            return
        # scale_factor is frozen at the scene-fit value set in center_camera()
        # so the projection doesn't move. Speed lives on _speed_mul, which
        # post-multiplies the per-tick position delta.
        boosted = BOOST_FACTOR if ("SHIFT" in self._held_keys
                                    or "LSHIFT" in self._held_keys
                                    or "RSHIFT" in self._held_keys) else 1.0
        self._fly._speed_mul = BASE_SPEED_MUL * self.state.base_speed_mul * boosted

    def _apply_boost(self, on: bool) -> None:
        self._apply_speed()

    def _toggle_size_mode(self) -> None:
        self.state.size_mode = "flux" if self.state.size_mode == "lum" else "lum"
        print(f"[viewer] size mode: {self.state.size_mode}")
        self._apply_data()
        self._update_status()

    def _cycle_color_mode(self) -> None:
        order = ["spectype", "redshift", "absmag", "subtype", "gr", "lookback"]
        i = order.index(self.state.color_mode)
        self.state.color_mode = order[(i + 1) % len(order)]
        print(f"[viewer] color mode: {self.state.color_mode}")
        self._apply_data()
        self._update_status()
        if self._legend_label.isVisible():
            self._update_legend()

    def _toggle_camera(self) -> None:
        if self.state.camera_mode == "fly":
            self.state.camera_mode = "turntable"
            self.view.camera = self._turn
        else:
            self.state.camera_mode = "fly"
            self.view.camera = self._fly
            self._apply_speed()
        print(f"[viewer] camera: {self.state.camera_mode}")
        self._update_status()

    def _toggle_lod_enabled(self) -> None:
        self.state.lod_enabled = not self.state.lod_enabled
        if not self.state.lod_enabled and self.state.lod_active:
            self.state.lod_active = False
            self._apply_data()
        print(f"[viewer] LOD {'enabled' if self.state.lod_enabled else 'disabled'}")
        self._update_status()

    # -- color legend ---------------------------------------------------------

    def _toggle_legend(self) -> None:
        if self._legend_label.isVisible():
            self._legend_label.hide()
        else:
            self._update_legend()
            self._legend_label.show()

    def _update_legend(self) -> None:
        self._legend_label.setText(self._build_legend_html())
        self._legend_label.adjustSize()
        _, h = self.canvas.size
        self._legend_label.move(
            10, h - self._legend_label.height() - 30)

    def _build_legend_html(self) -> str:
        mode = self.state.color_mode
        if mode == "spectype":
            return self._legend_categorical("Spectral Type", [
                (PALETTE_SPECTYPE[0], "GALAXY"),
                (PALETTE_SPECTYPE[1], "QSO"),
                (PALETTE_SPECTYPE[2], "Other"),
            ])
        if mode == "redshift":
            return self._legend_gradient(
                "Redshift (z)", _REDSHIFT_CMAP, "z = 0", "z = 4")
        if mode == "absmag":
            return self._legend_gradient(
                "Absolute Magnitude (M_r)", get_colormap("fire"),
                "bright", "faint", t_min=0.25)
        if mode == "subtype":
            return self._legend_categorical("DESI Target Class", [
                (PALETTE_SUBTYPE[0], "LRG — Luminous Red Galaxy"),
                (PALETTE_SUBTYPE[1], "ELG — Emission Line Galaxy"),
                (PALETTE_SUBTYPE[2], "BGS — Bright Galaxy Survey"),
                (PALETTE_SUBTYPE[3], "QSO — Quasar"),
                (PALETTE_SUBTYPE[4], "Secondary / MWS"),
                (PALETTE_SUBTYPE[5], "Unclassified"),
            ])
        if mode == "gr":
            return self._legend_gradient(
                "Rest-frame g − r", _GR_CMAP,
                "blue (star-forming)", "red (elliptical)")
        return self._legend_gradient(
            "Lookback Time", _LOOKBACK_CMAP,
            "0 Gyr (now)", "13 Gyr (ancient)")

    @staticmethod
    def _legend_gradient(title: str, cmap, lo: str, hi: str,
                         t_min: float = 0.0) -> str:
        n = 32
        t = np.linspace(t_min, 1.0, n, dtype=np.float32)
        colors = cmap.map(t.reshape(-1, 1)).reshape(-1, 4)
        blocks = "".join(
            f"<span style='background:rgb({int(c[0]*255)},"
            f"{int(c[1]*255)},{int(c[2]*255)})'>"
            f"&nbsp;&nbsp;</span>"
            for c in colors
        )
        return (
            f"<div style='line-height:1.6'>"
            f"<b>{title}</b><br>{blocks}<br>"
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
            f"<td style='color:#999' align='left'>{lo}</td>"
            f"<td style='color:#999' align='right'>{hi}</td>"
            f"</tr></table></div>"
        )

    @staticmethod
    def _legend_categorical(title: str,
                            items: list[tuple[np.ndarray, str]]) -> str:
        rows = ""
        for color, label in items:
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            rows += (f"<span style='color:rgb({r},{g},{b})'>"
                     f"██</span> {label}<br>")
        return (
            f"<div style='line-height:1.8'>"
            f"<b>{title}</b><br>{rows}</div>"
        )

    # -- selection / picking --------------------------------------------------

    def _screen_to_ray(self, sx: float, sy: float
                       ) -> tuple[np.ndarray, np.ndarray]:
        cam = self._fly
        origin = np.array(cam.center, dtype=np.float64)
        w, h = self.canvas.size
        aspect = w / h
        half_h = np.tan(np.radians(cam.fov) / 2.0)
        half_w = half_h * aspect
        nx = (2.0 * sx / w - 1.0) * half_w
        ny = (1.0 - 2.0 * sy / h) * half_h
        ray_cam = np.array([nx, ny, -1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = np.array(
            cam._rotation1.inverse().rotate_point(ray_cam), dtype=np.float64,
        )
        ray_world /= np.linalg.norm(ray_world)
        return origin, ray_world

    def _pick_nearest(self, sx: float, sy: float) -> int | None:
        tree = self._kdtree
        if tree is None:
            print("[viewer] spatial index still building, try again")
            return None
        origin, ray_dir = self._screen_to_ray(sx, sy)
        max_dist = min(self._fog.render_dist, self._scene_extent * 2.0)
        t_vals = np.unique(np.concatenate([
            np.geomspace(0.01, 100.0, PICK_RAY_SAMPLES // 4),
            np.linspace(100.0, max_dist, PICK_RAY_SAMPLES * 3 // 4),
        ]))
        samples = origin + ray_dir * t_vals[:, None]
        _, indices = tree.query(samples, k=5)
        unique_idx = np.unique(indices.ravel())
        candidates = self.pc.xyz[unique_idx].astype(np.float64)
        v = candidates - origin
        proj_len = v @ ray_dir
        perp = v - proj_len[:, None] * ray_dir
        angular = np.arctan2(
            np.linalg.norm(perp, axis=1), np.abs(proj_len),
        )
        angular[proj_len <= 0] = np.inf
        cam_dist = np.linalg.norm(v, axis=1)
        angular[cam_dist > self._fog.render_dist] = np.inf
        h_px = self.canvas.size[1]
        max_angle = (np.radians(self._fly.fov) / h_px) * PICK_RADIUS_PX
        within = angular <= max_angle
        if not np.any(within):
            return None
        cam_dist_pick = cam_dist.copy()
        cam_dist_pick[~within] = np.inf
        best = int(np.argmin(cam_dist_pick))
        return int(unique_idx[best])

    def _select(self, idx: int) -> None:
        self.state.selected_idx = idx
        pos = self.pc.xyz[idx:idx + 1]
        self._highlight.set_data(
            pos=pos, size=np.array([HIGHLIGHT_SIZE]),
            face_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
            edge_color=HIGHLIGHT_COLOR, edge_width=2.0, symbol="disc",
        )
        self._highlight.visible = True
        self._info_panel.setHtml(self._build_info_html(idx))
        self._info_panel.document().adjustSize()
        h = int(self._info_panel.document().size().height()) + 24
        self._info_panel.setFixedHeight(min(h, 400))
        w_canvas = self.canvas.size[0]
        self._info_panel.move(w_canvas - self._info_panel.width() - 10, 10)
        self._info_panel.show()
        if self.state.camera_mode == "fly":
            target = self.pc.xyz[idx].astype(np.float64)
            ref = float(np.linalg.norm(target - np.array(self._fly.center)))
            self._fly._lock_target = target
            self._fly._lock_ref_dist = max(ref, 0.1)
            self._look_at_point(self.pc.xyz[idx])
        self._update_status()
        self.canvas.update()
        print(f"[viewer] selected point {idx}")

    def _deselect(self) -> None:
        self.state.selected_idx = None
        self._fly._lock_target = None
        self._highlight.visible = False
        self._info_panel.hide()
        self._update_status()
        self.canvas.update()
        print("[viewer] deselected")

    def _build_info_html(self, idx: int) -> str:
        pc = self.pc
        names = {SPECTYPE_GALAXY: "GALAXY", SPECTYPE_QSO: "QSO",
                 SPECTYPE_OTHER: "OTHER"}
        stype = names.get(int(pc.spectype[idx]), "?")
        dt = int(pc.desi_target[idx])
        sub_parts = []
        if dt & int(LRG_BIT): sub_parts.append("LRG")
        if dt & int(ELG_BIT): sub_parts.append("ELG")
        if dt & int(QSO_BIT): sub_parts.append("QSO")
        if dt & int(BGS_ANY_BIT): sub_parts.append("BGS")
        if not sub_parts:
            if dt & (int(SCND_ANY_BIT) | int(MWS_ANY_BIT)):
                sub_parts.append("Secondary")
            else:
                sub_parts.append("Unclassified")
        sub_label = " / ".join(sub_parts)
        z_val = float(pc.z[idx])
        d_comov = float(np.linalg.norm(pc.xyz[idx]))
        d_lum = d_comov * (1.0 + z_val)
        M_r_val = float(pc.M_r[idx])
        L_solar = 10.0 ** float(pc.log_L[idx])
        m_r = 22.5 - 2.5 * np.log10(max(float(pc.flux_r[idx]), 1e-30))
        t_lb = f"{float(pc.lookback_gyr[idx]):.2f} Gyr"
        fg = float(pc.flux_g[idx])
        fr = float(pc.flux_r[idx])
        g_r_str = f"{-2.5 * np.log10(fg / fr):.2f}" if fg > 0 and fr > 0 else "N/A"
        ra = float(pc.target_ra[idx])
        dec = float(pc.target_dec[idx])
        tid = int(pc.target_id[idx])

        img_url = (f"https://www.legacysurvey.org/viewer"
                   f"?ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr10&zoom=16")
        wide_url = (f"https://www.legacysurvey.org/viewer"
                    f"?ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr10&zoom=12")
        ned_url = (f"https://ned.ipac.caltech.edu/cgi-bin/nph-objsearch"
                   f"?search_type=Near+Position+Search"
                   f"&in_csys=Equatorial&in_equinox=J2000.0"
                   f"&lon={ra:.6f}d&lat={dec:.6f}d&radius=1.0"
                   f"&out_csys=Equatorial&out_equinox=J2000.0"
                   f"&of=pre_text")
        simbad_url = (f"https://simbad.u-strasbg.fr/simbad/sim-coo"
                      f"?Coord={ra:.6f}+{dec:.6f}"
                      f"&Radius=10&Radius.unit=arcsec")

        g = "#999999"
        return (
            f"<div style='font-family:Consolas,monospace;line-height:1.6'>"
            f"<span style='color:#ffdd88;font-weight:bold'>[{stype}]</span> "
            f"<span style='color:#aaa'>TARGETID</span> {tid}<br>"
            f"<span style='color:{g}'>Target</span> {sub_label}<br>"
            f"<br>"
            f"<span style='color:{g}'>Redshift</span> "
            f"z&nbsp;=&nbsp;{z_val:.4f}<br>"
            f"<span style='color:{g}'>Lookback</span> {t_lb}<br>"
            f"<span style='color:{g}'>Distance</span> "
            f"{d_comov:.1f}&nbsp;Mpc (comoving)<br>"
            f"<span style='color:{g}'>Lum dist</span> "
            f"{d_lum:.1f}&nbsp;Mpc<br>"
            f"<span style='color:{g}'>Abs mag</span> "
            f"M<sub>r</sub>&nbsp;=&nbsp;{M_r_val:.2f}<br>"
            f"<span style='color:{g}'>App mag</span> "
            f"m<sub>r</sub>&nbsp;=&nbsp;{m_r:.2f}<br>"
            f"<span style='color:{g}'>Luminosity</span> "
            f"{L_solar:.2e}&nbsp;L<sub>&#9737;</sub><br>"
            f"<span style='color:{g}'>g &minus; r</span> "
            f"{g_r_str}<br>"
            f"<span style='color:{g}'>RA / Dec</span> "
            f"{ra:.4f}&deg; / {dec:.4f}&deg;<br>"
            f"<br>"
            f"<a href='{img_url}'>Sky Close-up</a>"
            f" &nbsp;&#8226;&nbsp; "
            f"<a href='{wide_url}'>Wide Field</a><br>"
            f"<a href='{ned_url}'>NED</a>"
            f" &nbsp;&#8226;&nbsp; "
            f"<a href='{simbad_url}'>SIMBAD</a>"
            f"</div>"
        )

    def _look_at_point(self, target_xyz: np.ndarray) -> None:
        cam = self._fly
        eye = np.array(cam.center, dtype=np.float64)
        quat = _look_at_quat(eye, target_xyz)
        cam._rotation1 = VispyQuat(*quat)
        cam.view_changed()

    # -- frame + LOD tick -----------------------------------------------------

    def _on_draw(self, event) -> None:
        self._frame_times.append(time.monotonic())

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def _tick_lod(self) -> None:
        # Tie LOD to input activity, not fps. Previously we watched fps and
        # flipped back to the full set once it recovered — but if the full
        # set can't hit 30 fps at rest, that creates a feedback loop where
        # LOD cycles on and off forever and the view blinks.
        want_lod = self._is_moving() and self.state.lod_enabled
        if want_lod != self.state.lod_active:
            self.state.lod_active = want_lod
            self._apply_data()
        self._update_status()

    def _is_moving(self) -> bool:
        if self._held_keys & _MOVEMENT_KEYS:
            return True
        idle_ms = (time.monotonic() - self._last_input_time) * 1000.0
        return idle_ms < LOD_IDLE_MS

    # -- public ---------------------------------------------------------------

    def center_camera(self) -> None:
        """Center view on the scene so the data is visible on startup."""
        extent = float(np.linalg.norm(self.pc.xyz, axis=1).max())
        self._fly.center = (0.0, 0.0, 0.0)
        self._fly.rotation1 = VispyQuat(1.0, 0.0, 0.0, 0.0)     # identity quat
        # Freeze scale_factor at something proportional to the scene so the
        # perspective projection + near/far clip planes encompass the data.
        # Do NOT change this at runtime — movement speed is handled separately
        # via FlyCamera._speed_mul (see _apply_speed).
        self._fly.scale_factor = extent
        # Widen the near/far ratio. Vispy's default depth_value=1e6 puts the
        # near plane at ~6 Mpc at this scale — anything closer clips. 1e12
        # gives near≈6 kpc / far≈6 Tpc so the user can fly right up to an
        # individual galaxy. Z-buffer precision is not a concern for a pure
        # point cloud (no overlapping opaque surfaces).
        self._fly.depth_value = 1e12
        self._fog.fog_end = extent * 1.1
        self._fog.render_dist = 256.0
        self._turn.scale_factor = extent
        self._turn.depth_value = 1e12
        self._turn.center = (0.0, 0.0, 0.0)
        self._turn.distance = extent * 1.3
        self._apply_speed()
        nearest = int(np.argmin(np.linalg.norm(self.pc.xyz, axis=1)))
        self._look_at_point(self.pc.xyz[nearest])

    def show(self) -> None:
        self.canvas.show()


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def run_viewer(pc: PointCloud, dataset: str = "DR1") -> int:
    from vispy import app as vispy_app

    app = vispy_app.use_app("pyqt6")
    app.create()

    mem0 = _rss_mb()
    print(f"[viewer] starting | points={pc.n:,} | xyz={pc.xyz.nbytes/1e6:.1f} MB "
          f"| rss={mem0:.0f} MB | flux_column={pc.flux_column}")

    viewer = Viewer(pc, dataset=dataset)
    viewer.center_camera()
    viewer.show()

    app.run()
    return 0


if __name__ == "__main__":
    from pathlib import Path
    from process import load_or_build

    pc = load_or_build(Path("data/zall-pix-fuji.fits"),
                       Path("data/points_v2.npz"))
    sys.exit(run_viewer(pc))
