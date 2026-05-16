"""QTabWidget dock for KapoorLabs Curvature.

Layout mirrors the VollSeg plugin (header + tabs + footer):

    [logo]  KapoorLabs Curvature
    ──────────────────────────────────────────────────────────
    Input │ Lines │ Plot │ Record
    ──────────────────────────────────────────────────────────
            (active tab contents)
    ──────────────────────────────────────────────────────────
    status                                       [ Record ▶ ]

- **Input** (magicgui ``FileEdit``s) — pick raw image / curvature image
  (file or folder), Load button loads them into napari layers.
- **Lines** — Shapes-layer picker + "Add line layer" button + table of
  current polylines / their lengths.
- **Plot** — embedded matplotlib canvas. Updates on viewer-dims change
  to show curvature (+ intensity on a twin axis) along the selected
  line at the current time.
- **Record** — output folder, ``samples_per_line`` slider, Record
  button → walks the T axis, builds ``(T, L)`` kymographs for every
  selected line × channel, writes TIFFs + a per-line CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from magicgui.widgets import Container, FileEdit
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._plot import ProfileCanvas
from ._profile import (
    LineProfile,
    TimelapseRecording,
    record_timelapse_kymograph,
    sample_polyline,
)


_LOGO = Path(__file__).parent / "resources" / "kapoorlogo.png"


# =============================================== small loader helper


def _load_image_or_stack(path: Path) -> np.ndarray:
    """Load a single TIFF or a folder of TIFFs (alphabetical) stacked on axis 0."""
    p = Path(path)
    if p.is_file():
        return tifffile.imread(p)
    if p.is_dir():
        files = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        if not files:
            raise FileNotFoundError(f"No .tif files in {p}")
        # Stack along a new leading axis (treated as T downstream).
        return np.stack([tifffile.imread(f) for f in files], axis=0)
    raise FileNotFoundError(f"Not a file or directory: {p}")


# =============================================== Input tab


class _InputTab(QWidget):
    """Pick raw + curvature paths via magicgui FileEdit, load into napari."""

    def __init__(self, viewer, parent: CurvatureWidget):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        # magicgui — file *or* directory (FileEdit mode 'r' for files; users
        # who want a folder pass a folder path and we detect at load time).
        self.raw_path = FileEdit(label="Raw image / folder", mode="r", value="")
        self.curv_path = FileEdit(label="Curvature image / folder", mode="r", value="")
        load_btn = QPushButton("Load both into napari")
        load_btn.clicked.connect(self._on_load)

        self.raw_layer_label = QLabel("(not loaded)")
        self.curv_layer_label = QLabel("(not loaded)")

        layout = QVBoxLayout(self)
        paths_group = QGroupBox("Source paths")
        paths_box = Container(widgets=[self.raw_path, self.curv_path], labels=True)
        v = QVBoxLayout(paths_group)
        v.addWidget(paths_box.native)
        v.addWidget(load_btn)
        layout.addWidget(paths_group)

        status_group = QGroupBox("Loaded layers")
        form = QFormLayout(status_group)
        form.addRow("Raw:", self.raw_layer_label)
        form.addRow("Curvature:", self.curv_layer_label)
        layout.addWidget(status_group)
        layout.addStretch(1)

    def _on_load(self):
        raw = self._load_one(self.raw_path.value, "raw")
        curv = self._load_one(self.curv_path.value, "curvature")
        self.raw_layer_label.setText(raw or "(none)")
        self.curv_layer_label.setText(curv or "(none)")
        self.parent_widget.set_loaded_layer_names(raw, curv)

    def _load_one(self, path_value, label: str) -> Optional[str]:
        if not path_value or str(path_value) in ("", "."):
            return None
        try:
            arr = _load_image_or_stack(Path(str(path_value)))
        except Exception as e:  # show in status
            self.parent_widget.show_status(f"✗ {label}: {e}")
            return None
        name = f"curvature_{label}" if label == "curvature" else label
        self.viewer.add_image(arr, name=name)
        return name


# =============================================== Lines tab


class _LinesTab(QWidget):
    """Choose the Shapes layer holding line profiles + show a summary table."""

    def __init__(self, viewer, parent: CurvatureWidget):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        self.shapes_combo = QComboBox()
        add_btn = QPushButton("Add new Shapes layer for lines")
        add_btn.clicked.connect(self._on_add_shapes)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Line #", "Vertices", "Length (px)"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Shapes layer"))
        picker_row.addWidget(self.shapes_combo, 1)
        picker_row.addWidget(refresh_btn)
        picker_row.addWidget(add_btn)
        layout.addLayout(picker_row)
        layout.addWidget(self.table, 1)

        self.shapes_combo.currentIndexChanged.connect(self._refresh_table)
        self.viewer.layers.events.inserted.connect(self._refresh)
        self.viewer.layers.events.removed.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_):
        from napari.layers import Shapes

        current = self.shapes_combo.currentText()
        self.shapes_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                self.shapes_combo.addItem(layer.name)
        if current:
            idx = self.shapes_combo.findText(current)
            if idx >= 0:
                self.shapes_combo.setCurrentIndex(idx)
        self._refresh_table()

    def _refresh_table(self):
        lines = self.current_lines()
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            self.table.setItem(row, 0, QTableWidgetItem(str(line.line_id)))
            self.table.setItem(row, 1, QTableWidgetItem(str(len(line.points))))
            self.table.setItem(row, 2, QTableWidgetItem(f"{line.length():.2f}"))
        self.parent_widget.on_lines_changed()

    def _on_add_shapes(self):
        layer = self.viewer.add_shapes(name="curvature_lines", shape_type="line")
        # Refresh + select it; tell the user to draw with napari's line tool.
        self._refresh()
        idx = self.shapes_combo.findText(layer.name)
        if idx >= 0:
            self.shapes_combo.setCurrentIndex(idx)
        self.parent_widget.show_status(
            "Draw lines on the new layer (napari Shapes tools → 'Add lines')."
        )

    def current_shapes_layer(self):
        name = self.shapes_combo.currentText()
        if not name or name not in self.viewer.layers:
            return None
        return self.viewer.layers[name]

    def current_lines(self) -> list[LineProfile]:
        layer = self.current_shapes_layer()
        if layer is None:
            return []
        out: list[LineProfile] = []
        for i, (pts, kind) in enumerate(zip(layer.data, layer.shape_type)):
            # Accept lines and (open) polylines / paths.
            if kind not in ("line", "path"):
                continue
            arr = np.asarray(pts, dtype=np.float64)
            # Drop any leading non-YX axes (T, Z) so we end up with (N, 2).
            if arr.shape[1] > 2:
                arr = arr[:, -2:]
            if arr.shape[0] < 2:
                continue
            out.append(LineProfile(line_id=i, points=arr))
        return out


# =============================================== Plot tab


class _PlotTab(QWidget):
    """Embedded matplotlib canvas + a line picker."""

    def __init__(self, viewer, parent: CurvatureWidget):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        self.line_combo = QComboBox()
        self.canvas = ProfileCanvas()

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show line"))
        row.addWidget(self.line_combo, 1)
        layout.addLayout(row)
        layout.addWidget(self.canvas, 1)

        self.line_combo.currentIndexChanged.connect(
            lambda *_: self.parent_widget.refresh_plot()
        )

    def selected_line_id(self) -> Optional[int]:
        text = self.line_combo.currentText()
        return int(text.split()[1]) if text.startswith("Line ") else None

    def refresh_combo(self, lines: list[LineProfile]):
        current = self.line_combo.currentText()
        self.line_combo.clear()
        for ln in lines:
            self.line_combo.addItem(f"Line {ln.line_id}")
        if current:
            idx = self.line_combo.findText(current)
            if idx >= 0:
                self.line_combo.setCurrentIndex(idx)


# =============================================== Record tab


class _RecordTab(QWidget):
    """Sweep the T axis, record kymographs to disk."""

    def __init__(self, parent: CurvatureWidget):
        super().__init__()
        self.parent_widget = parent
        self.out_dir = FileEdit(label="Output folder", mode="d", value="")
        self.samples = QSpinBox()
        self.samples.setRange(16, 8192)
        self.samples.setValue(256)
        self.linewidth = QSpinBox()
        self.linewidth.setRange(1, 32)
        self.linewidth.setValue(1)
        self.stem = QComboBox()
        self.stem.setEditable(True)
        self.stem.addItems(["recording"])

        layout = QVBoxLayout(self)
        form_group = QGroupBox("Output")
        form = QFormLayout(form_group)
        form.addRow(Container(widgets=[self.out_dir]).native)
        form.addRow("Samples per line", self.samples)
        form.addRow("Line width (px)", self.linewidth)
        form.addRow("Filename stem", self.stem)
        layout.addWidget(form_group)
        layout.addStretch(1)


# =============================================== main widget


class CurvatureWidget(QWidget):
    """Top-level dock widget."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._raw_name: Optional[str] = None
        self._curv_name: Optional[str] = None

        # Header
        header = QHBoxLayout()
        if _LOGO.exists():
            logo = QLabel()
            logo.setPixmap(
                QPixmap(str(_LOGO)).scaledToHeight(48, Qt.SmoothTransformation)
            )
            header.addWidget(logo)
        title = QLabel("<h2>KapoorLabs Curvature</h2>")
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title, 1)

        # Tabs
        self.tabs = QTabWidget()
        self.input_tab = _InputTab(self.viewer, self)
        self.lines_tab = _LinesTab(self.viewer, self)
        self.plot_tab = _PlotTab(self.viewer, self)
        self.record_tab = _RecordTab(self)
        self.tabs.addTab(self.input_tab, "Input")
        self.tabs.addTab(self.lines_tab, "Lines")
        self.tabs.addTab(self.plot_tab, "Plot")
        self.tabs.addTab(self.record_tab, "Record")

        # Footer
        self.record_btn = QPushButton("Record ▶")
        self.record_btn.clicked.connect(self._on_record)
        self.status = QLabel("")
        self.status.setWordWrap(False)
        self.status.setMaximumHeight(48)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        root = QVBoxLayout(self)
        root.addLayout(header)
        root.addWidget(self.tabs, 1)
        root.addWidget(self.status)
        root.addWidget(self.record_btn)

        # Live plot updates when the napari T slider moves.
        self.viewer.dims.events.current_step.connect(lambda *_: self.refresh_plot())

    # ------------------------------------------------ shared state

    def set_loaded_layer_names(self, raw: Optional[str], curv: Optional[str]):
        self._raw_name = raw
        self._curv_name = curv
        self.refresh_plot()

    def show_status(self, msg: str):
        if len(msg) > 160:
            msg = msg[:159] + "…"
        self.status.setText(msg)

    def on_lines_changed(self):
        # _LinesTab fires this from its constructor (during the
        # initial layer-scan), before plot_tab / record_tab exist.
        # Bail out until the parent finishes wiring.
        if not hasattr(self, "plot_tab"):
            return
        self.plot_tab.refresh_combo(self.lines_tab.current_lines())
        self.refresh_plot()

    # ------------------------------------------------ plot

    def _layer_array(self, name: Optional[str]) -> Optional[np.ndarray]:
        if not name or name not in self.viewer.layers:
            return None
        return np.asarray(self.viewer.layers[name].data)

    def _current_frame(self, vol: np.ndarray) -> np.ndarray:
        """Slice ``vol`` to the YX plane napari is currently displaying."""
        steps = list(self.viewer.dims.current_step)
        if vol.ndim == 2:
            return vol
        if vol.ndim == 3:
            return vol[steps[0] if steps else 0]
        if vol.ndim == 4:
            return vol[
                steps[0] if len(steps) > 0 else 0, steps[1] if len(steps) > 1 else 0
            ]
        raise ValueError(f"Unsupported volume ndim={vol.ndim}")

    def refresh_plot(self):
        # Same guard — viewer.dims.events.current_step also wires to
        # this slot before all tabs exist on the very first emit.
        if not all(hasattr(self, n) for n in ("lines_tab", "plot_tab", "record_tab")):
            return
        lines = self.lines_tab.current_lines()
        if not lines:
            return
        line_id = self.plot_tab.selected_line_id()
        if line_id is None:
            return
        line = next((ln for ln in lines if ln.line_id == line_id), None)
        if line is None:
            return

        curv = self._layer_array(self._curv_name)
        raw = self._layer_array(self._raw_name)
        try:
            curv_prof = (
                sample_polyline(
                    self._current_frame(curv),
                    line.points,
                    linewidth=int(self.record_tab.linewidth.value()),
                    num_samples=int(self.record_tab.samples.value()),
                )
                if curv is not None
                else None
            )
            raw_prof = (
                sample_polyline(
                    self._current_frame(raw),
                    line.points,
                    linewidth=int(self.record_tab.linewidth.value()),
                    num_samples=int(self.record_tab.samples.value()),
                )
                if raw is not None
                else None
            )
        except Exception as e:
            self.show_status(f"✗ {type(e).__name__}: {e}")
            return

        if curv_prof is None and raw_prof is None:
            return
        # Plot curvature; if only intensity is present put it on the
        # primary axis instead.
        if curv_prof is None:
            curv_prof = raw_prof
            raw_prof = None
            title = f"Line {line.line_id} — intensity only"
        else:
            title = (
                f"Line {line.line_id} — frame "
                f"{tuple(self.viewer.dims.current_step)[:2]}"
            )
        self.plot_tab.canvas.draw_profile(curv_prof, intensity=raw_prof, title=title)

    # ------------------------------------------------ record

    def _on_record(self):
        lines = self.lines_tab.current_lines()
        if not lines:
            self.show_status("⚠ Draw at least one line first.")
            return
        out_dir = str(self.record_tab.out_dir.value)
        if not out_dir:
            self.show_status("⚠ Pick an output folder in the Record tab.")
            return

        curv = self._layer_array(self._curv_name)
        raw = self._layer_array(self._raw_name)
        if curv is None and raw is None:
            self.show_status("⚠ Load raw and/or curvature in the Input tab first.")
            return

        # Figure out how many frames to sweep. Prefer T from a 3-or-4D
        # volume; fall back to 1 for static 2D inputs.
        ref = curv if curv is not None else raw
        if ref.ndim == 2:
            n_frames = 1
            z = None
        elif ref.ndim == 3:
            n_frames = ref.shape[0]
            z = None
        else:  # 4D = TZYX
            n_frames = ref.shape[0]
            steps = list(self.viewer.dims.current_step)
            z = int(steps[1]) if len(steps) > 1 else 0

        volumes: dict[str, np.ndarray] = {}
        if curv is not None:
            volumes["curvature"] = curv
        if raw is not None:
            volumes["intensity"] = raw

        try:
            recording = record_timelapse_kymograph(
                volumes,
                lines,
                n_frames=n_frames,
                z=z,
                samples_per_line=int(self.record_tab.samples.value()),
            )
        except Exception as e:
            import traceback

            traceback.print_exception(type(e), e, e.__traceback__)
            self.show_status(f"✗ {type(e).__name__}: {e}  (full traceback in terminal)")
            return

        self._write_recording(Path(out_dir), recording)
        # Update the plot to show the just-recorded kymograph.
        sel = self.plot_tab.selected_line_id()
        if sel is not None and "curvature" in recording.kymographs:
            ky = recording.kymographs["curvature"].get(sel)
            if ky is not None:
                self.plot_tab.canvas.draw_kymograph(
                    ky,
                    title=f"Line {sel} κ kymograph (T={n_frames})",
                )
        self.show_status(
            f"✓ Recorded {n_frames} frames × {len(lines)} lines × "
            f"{len(volumes)} channels → {out_dir}"
        )

    def _write_recording(
        self,
        out_dir: Path,
        rec: TimelapseRecording,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.record_tab.stem.currentText() or "recording"

        # 1) Per-(channel, line) TIFF kymographs.
        for channel, by_line in rec.kymographs.items():
            for line_id, ky in by_line.items():
                tifffile.imwrite(
                    out_dir / f"{stem}_{channel}_line{line_id}.tif",
                    ky.astype(np.float32),
                )

        # 2) One CSV per line with all channels side by side.
        line_ids = (
            sorted(next(iter(rec.kymographs.values())).keys()) if rec.kymographs else []
        )
        for line_id in line_ids:
            path = out_dir / f"{stem}_line{line_id}.csv"
            with path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                header = ["frame", "sample_idx"] + list(rec.kymographs.keys())
                writer.writerow(header)
                T = rec.n_frames
                L = rec.samples_per_line
                for t in range(T):
                    for s in range(L):
                        row = [t, s] + [
                            float(rec.kymographs[ch][line_id][t, s])
                            for ch in rec.kymographs
                        ]
                        writer.writerow(row)
