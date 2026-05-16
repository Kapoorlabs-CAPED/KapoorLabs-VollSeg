"""QTabWidget-based dock widget for the KapoorLabs VollSeg napari plugin.

Design mirrors the original ``vollseg-napari`` plugin:

- **magicgui** for every path / file picker / model-type chooser
  (``RadioButtons`` + ``ComboBox`` + ``FileEdit``, hide-on-toggle).
- **plain Qt** form layouts inside each parameter tab for the numeric
  knobs (tile sizes, thresholds, n_rays, chunk shape).

Tabs::

    [logo]  KapoorLabs VollSeg
    ──────────────────────────────────────────────────────────
    Input │ Models │ Inference │ Postproc │ Output
    ──────────────────────────────────────────────────────────
            (active tab contents)
    ──────────────────────────────────────────────────────────
                                                   [ Run ▶ ]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from magicgui.widgets import CheckBox, ComboBox, Container, FileEdit, RadioButtons
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._model_catalog import MODEL_CATALOG, ROLE_LABELS
from ._runner import RoleChoice, RunSpec, run

_LOGO = Path(__file__).parent / "resources" / "kapoorlogo.png"

# RadioButton choices, mirroring the original "(Pretrained X / Custom X / None)" pattern.
_MODE_CHOICES = [
    ("Pretrained (HuggingFace)", "pretrained"),
    ("Custom checkpoint folder", "custom"),
    ("None", "none"),
]


def _spin(default: int, lo: int = 0, hi: int = 10_000) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(default)
    return s


def _dspin(
    default: float, lo: float = 0.0, hi: float = 1e6, step: float = 0.1
) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setDecimals(4)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(default)
    return s


# ============================================================ Input tab


class _InputTab(QWidget):
    """Image layer + voxel spacing."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.layer_box = QComboBox()
        self.dz = _dspin(1.0, step=0.05)
        self.dy = _dspin(1.0, step=0.05)
        self.dx = _dspin(1.0, step=0.05)
        self.has_time = QCheckBox("Leading axis is time (T)")

        form = QFormLayout(self)
        form.addRow("Image layer", self.layer_box)
        form.addRow("Voxel dz (μm)", self.dz)
        form.addRow("Voxel dy (μm)", self.dy)
        form.addRow("Voxel dx (μm)", self.dx)
        form.addRow(self.has_time)

        self._refresh_layers()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

    def _refresh_layers(self, *_):
        from napari.layers import Image

        current = self.layer_box.currentText()
        self.layer_box.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.layer_box.addItem(layer.name)
        if current:
            idx = self.layer_box.findText(current)
            if idx >= 0:
                self.layer_box.setCurrentIndex(idx)

    def selected_array(self) -> Optional[np.ndarray]:
        name = self.layer_box.currentText()
        if not name:
            return None
        return np.asarray(self.viewer.layers[name].data)


# ============================================================ Models tab


class _RoleBlock:
    """One role's UI block: RadioButtons + pretrained ComboBox + custom FileEdit.

    The pretrained combo and the custom file-edit start hidden; the
    radio's ``changed`` signal toggles which one is visible.
    """

    def __init__(self, role: str):
        self.role = role
        self.radio = RadioButtons(
            label=ROLE_LABELS[role] + " — model type",
            choices=_MODE_CHOICES,
            value="none",
            orientation="horizontal",
        )
        self.pretrained = ComboBox(
            label=f"Pretrained {ROLE_LABELS[role]}",
            choices=MODEL_CATALOG.get(role, []) or [""],
            visible=False,
        )
        self.custom = FileEdit(
            label=f"Custom {ROLE_LABELS[role]} folder",
            mode="d",
            visible=False,
        )
        self.container = Container(
            widgets=[self.radio, self.pretrained, self.custom],
            labels=True,
        )
        self.radio.changed.connect(self._on_mode_changed)

    def _on_mode_changed(self, value):
        self.pretrained.visible = value == "pretrained"
        self.custom.visible = value == "custom"

    def to_choice(self) -> RoleChoice:
        mode = str(self.radio.value)
        if mode == "pretrained":
            name = self.pretrained.value or ""
            return RoleChoice(mode="pretrained", pretrained_name=str(name))
        if mode == "custom":
            path = self.custom.value
            return RoleChoice(
                mode="custom",
                custom_path=Path(str(path)) if path else None,
            )
        return RoleChoice(mode="none")


class _ModelsTab(QWidget):
    """Five role blocks + membrane-mode + local cache directory (all magicgui)."""

    def __init__(self):
        super().__init__()
        self.blocks: dict[str, _RoleBlock] = {
            role: _RoleBlock(role)
            for role in ("care", "maskunet", "unet", "stardist", "cellpose")
        }

        self.membrane = CheckBox(
            text="Membrane mode (use CellPose as the outer segmenter)",
            value=False,
        )
        self.cache_dir = FileEdit(
            label="Local model cache",
            mode="d",
            value=str(Path.home() / ".cache" / "kapoorlabs-vollseg"),
        )

        # One Container per group; both go into a QVBoxLayout via .native.
        models_container = Container(
            widgets=[block.container for block in self.blocks.values()],
            labels=False,
        )
        bottom_container = Container(
            widgets=[self.membrane, self.cache_dir],
            labels=True,
        )

        layout = QVBoxLayout(self)
        models_group = QGroupBox("Per-role model selection")
        mg_layout = QVBoxLayout(models_group)
        mg_layout.addWidget(models_container.native)
        layout.addWidget(models_group)

        bottom_group = QGroupBox("Pipeline shape & local cache")
        bg_layout = QVBoxLayout(bottom_group)
        bg_layout.addWidget(bottom_container.native)
        layout.addWidget(bottom_group)
        layout.addStretch(1)

    def choice(self, role: str) -> RoleChoice:
        return self.blocks[role].to_choice()


# ============================================================ Inference tab


class _InferenceTab(QWidget):
    """Tiling, StarDist rays, NMS thresholds (numeric parameters)."""

    def __init__(self):
        super().__init__()
        self.n_rays = _spin(96, 4, 1024)
        self.prob_thresh = _dspin(0.5, 0.0, 1.0, step=0.05)
        self.nms_thresh = _dspin(0.4, 0.0, 1.0, step=0.05)
        self.use_prob = QCheckBox("Override default prob_thresh")
        self.use_nms = QCheckBox("Override default nms_thresh")
        self.tile_z = _spin(1, 1, 64)
        self.tile_y = _spin(1, 1, 64)
        self.tile_x = _spin(1, 1, 64)
        self.seedpool = QCheckBox("Seedpool fusion (requires both U-Net + StarDist)")

        form = QFormLayout(self)
        form.addRow("StarDist rays", self.n_rays)
        form.addRow("prob_thresh", self.prob_thresh)
        form.addRow("", self.use_prob)
        form.addRow("nms_thresh", self.nms_thresh)
        form.addRow("", self.use_nms)
        tiles = QHBoxLayout()
        tiles.addWidget(QLabel("Z"))
        tiles.addWidget(self.tile_z)
        tiles.addWidget(QLabel("Y"))
        tiles.addWidget(self.tile_y)
        tiles.addWidget(QLabel("X"))
        tiles.addWidget(self.tile_x)
        form.addRow("n_tiles", tiles)
        form.addRow(self.seedpool)


# ============================================================ Postproc tab


class _PostprocTab(QWidget):
    """Chunked-prediction overlap settings + tile sizes (numeric parameters)."""

    def __init__(self):
        super().__init__()
        self.enable_chunk = QCheckBox(
            "Chunked prediction (for volumes too big to fit GPU)"
        )
        self.chunk_z = _spin(64, 8, 4096)
        self.chunk_y = _spin(256, 8, 4096)
        self.chunk_x = _spin(256, 8, 4096)
        self.overlap_z = _spin(8, 0, 512)
        self.overlap_y = _spin(32, 0, 512)
        self.overlap_x = _spin(32, 0, 512)

        form = QFormLayout(self)
        form.addRow(self.enable_chunk)
        chunk_row = QHBoxLayout()
        for w in (self.chunk_z, self.chunk_y, self.chunk_x):
            chunk_row.addWidget(w)
        form.addRow("chunk (Z, Y, X)", chunk_row)
        ov_row = QHBoxLayout()
        for w in (self.overlap_z, self.overlap_y, self.overlap_x):
            ov_row.addWidget(w)
        form.addRow("overlap (Z, Y, X)", ov_row)

    def chunk(self) -> Optional[tuple[int, int, int]]:
        if not self.enable_chunk.isChecked():
            return None
        return (self.chunk_z.value(), self.chunk_y.value(), self.chunk_x.value())

    def overlap(self) -> tuple[int, int, int]:
        return (self.overlap_z.value(), self.overlap_y.value(), self.overlap_x.value())


# ============================================================ Output tab


class _OutputTab(QWidget):
    """Output layer naming + optional TIFF dump (magicgui FileEdit for the dir)."""

    def __init__(self):
        super().__init__()
        self.prefix = QLineEdit("vollseg")
        self.save_to_disk = QCheckBox("Also write results as TIFFs")
        self.out_dir = FileEdit(label="Output directory", mode="d", value="")

        out_group = QGroupBox("Output")
        form = QFormLayout(out_group)
        form.addRow("Layer prefix", self.prefix)
        form.addRow(self.save_to_disk)

        out_container = Container(widgets=[self.out_dir], labels=True)
        form.addRow(out_container.native)

        layout = QVBoxLayout(self)
        layout.addWidget(out_group)
        layout.addStretch(1)


# ============================================================ main widget


class VollSegWidget(QWidget):
    """Main dock widget — header + tabs + run button."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Header
        header = QHBoxLayout()
        if _LOGO.exists():
            logo = QLabel()
            logo.setPixmap(
                QPixmap(str(_LOGO)).scaledToHeight(48, Qt.SmoothTransformation)
            )
            header.addWidget(logo)
        title = QLabel("<h2>KapoorLabs VollSeg</h2>")
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title, 1)

        # Tabs
        self.tabs = QTabWidget()
        self.input_tab = _InputTab(self.viewer)
        self.models_tab = _ModelsTab()
        self.inference_tab = _InferenceTab()
        self.postproc_tab = _PostprocTab()
        self.output_tab = _OutputTab()
        self.tabs.addTab(self.input_tab, "Input")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.postproc_tab, "Postproc")
        self.tabs.addTab(self.output_tab, "Output")

        # Footer — keep the status line short so a 5000-char traceback
        # can never blow up the dock layout. Full text always goes to
        # stderr; the label shows only a one-line summary.
        self.run_btn = QPushButton("Run ▶")
        self.run_btn.clicked.connect(self._on_run)
        self.status = QLabel("")
        self.status.setWordWrap(False)
        self.status.setMaximumHeight(48)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        root = QVBoxLayout(self)
        root.addLayout(header)
        root.addWidget(self.tabs, 1)
        root.addWidget(self.status)
        root.addWidget(self.run_btn)

    # -------------------------------------------------- run handler

    def _collect_spec(self) -> Optional[RunSpec]:
        image = self.input_tab.selected_array()
        if image is None:
            self.status.setText("⚠ Select an Image layer in the Input tab.")
            return None
        return RunSpec(
            image=image,
            voxel_spacing=(
                self.input_tab.dz.value(),
                self.input_tab.dy.value(),
                self.input_tab.dx.value(),
            ),
            care=self.models_tab.choice("care"),
            unet=self.models_tab.choice("unet"),
            maskunet=self.models_tab.choice("maskunet"),
            stardist=self.models_tab.choice("stardist"),
            cellpose=self.models_tab.choice("cellpose"),
            model_dir=Path(str(self.models_tab.cache_dir.value)),
            n_rays=self.inference_tab.n_rays.value(),
            prob_thresh=(
                self.inference_tab.prob_thresh.value()
                if self.inference_tab.use_prob.isChecked()
                else None
            ),
            nms_thresh=(
                self.inference_tab.nms_thresh.value()
                if self.inference_tab.use_nms.isChecked()
                else None
            ),
            n_tiles=(
                self.inference_tab.tile_z.value(),
                self.inference_tab.tile_y.value(),
                self.inference_tab.tile_x.value(),
            ),
            seedpool=self.inference_tab.seedpool.isChecked(),
            chunk=self.postproc_tab.chunk(),
            overlap=self.postproc_tab.overlap(),
            membrane_mode=bool(self.models_tab.membrane.value),
        )

    def _on_run(self):
        spec = self._collect_spec()
        if spec is None:
            return
        self.run_btn.setEnabled(False)
        self.status.setText("Running…")

        # Lazy import — keeps `import kapoorlabs_vollseg_napari` cheap
        # for CI tests that don't spin up a Qt event loop.
        from napari.qt import thread_worker

        @thread_worker
        def _do_run():
            return run(spec)

        worker = _do_run()
        worker.returned.connect(lambda layers: self._on_done(spec, layers))
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_done(self, spec: RunSpec, layers: dict[str, np.ndarray]):
        prefix = self.output_tab.prefix.text() or "vollseg"
        for kind, arr in layers.items():
            name = f"{prefix}_{kind}"
            if kind in ("labels", "semantic"):
                self.viewer.add_labels(arr.astype(np.uint32), name=name)
            else:
                self.viewer.add_image(arr, name=name)
        if self.output_tab.save_to_disk.isChecked():
            self._save_tiffs(layers)
        self.status.setText(f"Done. Wrote layers: {', '.join(layers.keys())}")
        self.run_btn.setEnabled(True)

    def _save_tiffs(self, layers: dict[str, np.ndarray]):
        import tifffile

        out_dir = Path(str(self.output_tab.out_dir.value) or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_tab.prefix.text() or "vollseg"
        for kind, arr in layers.items():
            tifffile.imwrite(out_dir / f"{prefix}_{kind}.tif", arr)

    def _on_error(self, exc: Exception):
        import traceback

        # Full traceback to stderr / terminal so the developer can debug.
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        # One-line summary for the dock label (truncated hard so a
        # 5000-char nested-repr message can't break the GUI layout).
        msg = f"{type(exc).__name__}: {exc}".splitlines()[0]
        if len(msg) > 160:
            msg = msg[:159] + "…"
        self.status.setText(f"✗ {msg}  (full traceback in terminal)")
        self.run_btn.setEnabled(True)
