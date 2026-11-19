from __future__ import annotations

import numpy as np
import napari
from napari.layers import Image, Labels
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

WIDGET_TITLE = "Layer math"
NEW_LAYER = "— new layer —"


# =============================================================================
# Core computations (pure numpy, no Qt)
# =============================================================================

# name -> (short symbol for auto naming, function)
IMAGE_OPS = {
    "A + B  (add)": ("+", lambda a, b: a + b),
    "A − B  (subtract)": ("-", lambda a, b: a - b),
    "A × B  (multiply)": ("*", lambda a, b: a * b),
    "A ÷ B  (divide)": ("/", lambda a, b: a / b),
    "|A − B|  (abs. difference)": ("absdiff", lambda a, b: np.abs(a - b)),
    "min(A, B)": ("min", np.minimum),
    "max(A, B)": ("max", np.maximum),
    "mean(A, B)": ("mean", lambda a, b: (a + b) * 0.5),
    "(A − B) / (A + B)  (normalised diff.)": ("nd", lambda a, b: (a - b) / (a + b)),
    "log2(A ÷ B)  (log ratio)": ("log2ratio", lambda a, b: np.log2(a / b)),
}


def image_math(a, b, op: str, dtype=np.float32, invalid_value=0.0):
    """Element-wise math between two arrays (or array and scalar).

    Non-finite results (x/0, 0/0, log(0) ...) are replaced by ``invalid_value``
    (0 or NaN).
    """
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    if a.shape != b.shape:
        np.broadcast_shapes(a.shape, b.shape)  # raises a clear ValueError
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = IMAGE_OPS[op][1](a, b)
    res = np.asarray(res, dtype=dtype)
    bad = ~np.isfinite(res)
    if bad.any():
        res[bad] = invalid_value
    return res


# ---------------------------------------------------------------- single image
# Parameter spec: (key, kind, label, default, extra)
#   kind: "float" -> extra = (min, max, decimals)   "int" -> extra = (min, max)
#         "bool"  -> extra = None                    "choice" -> extra = [options]

THRESH_METHODS = ["Otsu", "Li", "Triangle", "Yen", "Mean", "Manual"]


def _sigma_vec(value, ctx):
    """Per-axis sigma in pixels (0 on the RGB channel axis)."""
    n = ctx["n_spatial"]
    s = np.full(n, float(value))
    if ctx.get("spacing") is not None:
        s = s / np.asarray(ctx["spacing"], float)[-n:]
    if ctx.get("rgb"):
        s = np.append(s, 0.0)
    return s


def _size_vec(value, ctx):
    """Per-axis odd-ish integer window size (1 on the RGB channel axis)."""
    s = _sigma_vec(value, ctx)
    s = np.maximum(1, np.rint(s)).astype(int)
    if ctx.get("rgb"):
        s[-1] = 1
    return tuple(s)


def _invert(a, p, ctx):
    if p["range"].startswith("data"):
        return np.nanmin(a) + np.nanmax(a) - a
    dt = ctx["dtype_in"]
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.max) + float(info.min) - a
    return 1.0 - a  # float images: assume 0..1


def _norm_minmax(a, p, ctx):
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _norm_percentile(a, p, ctx):
    lo, hi = np.nanpercentile(a, [p["low"], p["high"]])
    out = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return np.clip(out, 0, 1) if p["clip"] else out


def _zscore(a, p, ctx):
    sd = np.nanstd(a)
    return (a - np.nanmean(a)) / sd if sd > 0 else np.zeros_like(a)


def _gauss(a, p, ctx):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, _sigma_vec(p["sigma"], ctx))


def _median(a, p, ctx):
    from scipy import ndimage
    return ndimage.median_filter(a, size=_size_vec(p["size"], ctx))


def _bg_sub(a, p, ctx):
    from scipy import ndimage
    out = a - ndimage.gaussian_filter(a, _sigma_vec(p["sigma"], ctx))
    return np.maximum(out, 0) if p["clip"] else out


def _tophat(a, p, ctx):
    from scipy import ndimage
    return ndimage.white_tophat(a, size=_size_vec(p["size"], ctx))


def _dog(a, p, ctx):
    from scipy import ndimage
    s1, s2 = sorted((p["sigma1"], p["sigma2"]))
    return (ndimage.gaussian_filter(a, _sigma_vec(s1, ctx))
            - ndimage.gaussian_filter(a, _sigma_vec(s2, ctx)))


def _gradmag(a, p, ctx):
    from scipy import ndimage
    if ctx.get("rgb"):
        raise ValueError("Gradient magnitude does not support RGB images.")
    return ndimage.gaussian_gradient_magnitude(a, _sigma_vec(p["sigma"], ctx))


def _threshold(a, p, ctx):
    if ctx.get("rgb"):
        raise ValueError("Thresholding an RGB image is not supported "
                         "(split the channels first).")
    method = p["method"]
    finite = a[np.isfinite(a)]
    if method == "Manual":
        t = p["value"]
    elif method == "Mean":
        t = float(finite.mean())
    else:
        try:
            from skimage import filters
        except ImportError as e:
            raise ValueError(f"'{method}' needs scikit-image.") from e
        t = float(getattr(filters, f"threshold_{method.lower()}")(finite))
    ctx["thresholds"].append(t)
    m = a < t if p["below"] else a > t
    return m.astype(np.uint8)


# name -> (short name, output kind, params, function, help)
SINGLE_OPS = {
    "Invert": ("inv", "image",
        [("range", "choice", "Range:",
          "data (min + max − x)",
          ["data (min + max − x)", "dtype (e.g. 255 − x; 1 − x for float)"])],
        _invert, "Flips intensities. 'dtype' with output 'same as input' "
                 "gives the classic 8/16-bit inversion."),
    "Normalise min–max → [0, 1]": ("norm", "image", [], _norm_minmax,
        "Linear stretch of the full data range to 0..1."),
    "Normalise percentiles → [0, 1]": ("pnorm", "image",
        [("low", "float", "Low %:", 1.0, (0, 100, 2)),
         ("high", "float", "High %:", 99.8, (0, 100, 2)),
         ("clip", "bool", "clip to [0, 1]", True, None)],
        _norm_percentile,
        "Robust contrast normalisation, ignores hot pixels / outliers."),
    "Standardise (z-score)": ("zscore", "image", [], _zscore,
        "(x − mean) / std."),
    "Clip": ("clip", "image",
        [("low", "float", "Min:", 0.0, (-1e12, 1e12, 4)),
         ("high", "float", "Max:", 255.0, (-1e12, 1e12, 4))],
        lambda a, p, c: np.clip(a, p["low"], p["high"]),
        "Values below min / above max are set to min / max."),
    "Scale & offset (a·x + b)": ("lin", "image",
        [("a", "float", "a (scale):", 1.0, (-1e12, 1e12, 6)),
         ("b", "float", "b (offset):", 0.0, (-1e12, 1e12, 6))],
        lambda a, p, c: a * p["a"] + p["b"], "Linear transform."),
    "Absolute value": ("abs", "image", [], lambda a, p, c: np.abs(a),
        "Handy after a subtraction."),
    "Log  log(1 + x)": ("log1p", "image", [], lambda a, p, c: np.log1p(a),
        "Compresses bright values. x < −1 is invalid."),
    "Square root": ("sqrt", "image", [], lambda a, p, c: np.sqrt(a),
        "Mild compression / variance stabilisation. Negative x is invalid."),
    "Power  x^p  (gamma)": ("pow", "image",
        [("p", "float", "p:", 0.5, (-100, 100, 4))],
        lambda a, p, c: np.power(a, p["p"]),
        "Gamma correction; best on data normalised to 0..1."),
    "---1": None,
    "Gaussian blur": ("gauss", "image",
        [("sigma", "float", "σ:", 1.0, (0, 1000, 2))], _gauss,
        "Smoothing. σ in pixels (or layer units, see option below)."),
    "Median filter": ("median", "image",
        [("size", "float", "Size:", 3, (1, 1000, 0))], _median,
        "Removes salt-and-pepper noise, keeps edges. Slow for big 3D sizes."),
    "Background subtract (Gaussian)": ("bgsub", "image",
        [("sigma", "float", "σ (large):", 20.0, (0, 10000, 2)),
         ("clip", "bool", "clip negatives to 0", True, None)], _bg_sub,
        "x − gaussian(x, σ). Removes uneven illumination / haze."),
    "White top-hat": ("tophat", "image",
        [("size", "float", "Size:", 15, (1, 10000, 0))], _tophat,
        "Keeps bright structures smaller than 'size' (rolling-ball-like)."),
    "Difference of Gaussians": ("dog", "image",
        [("sigma1", "float", "σ small:", 1.0, (0, 1000, 2)),
         ("sigma2", "float", "σ large:", 3.0, (0, 1000, 2))], _dog,
        "Band-pass; enhances blobs/spots of about σ size."),
    "Gradient magnitude": ("grad", "image",
        [("sigma", "float", "σ:", 1.0, (0, 1000, 2))], _gradmag,
        "Edge strength (Gaussian-derivative)."),
    "---2": None,
    "Threshold → labels": ("thr", "labels",
        [("method", "choice", "Method:", "Otsu", THRESH_METHODS),
         ("value", "float", "Manual value:", 0.0, (-1e12, 1e12, 4)),
         ("below", "bool", "keep values below threshold", False, None)],
        _threshold, "Binary labels layer (1 = foreground). The threshold "
                    "used is shown in the status line."),
}


def single_image_op(a, op, params=None, *, rgb=False, per_plane=False,
                    spacing=None, dtype=np.float32, invalid_value=0.0):
    """Apply a SINGLE_OPS operation. Returns (result, info_dict)."""
    _, kind, spec, fn, _ = SINGLE_OPS[op]
    p = {k: d for k, _, _, d, _ in spec}
    p.update(params or {})
    a = np.asarray(a)
    thresholds = []
    ctx = {"rgb": rgb, "dtype_in": a.dtype, "thresholds": thresholds}
    work = a.astype(dtype, copy=False)
    n_spatial = work.ndim - (1 if rgb else 0)

    def run(x, n):
        c = dict(ctx, n_spatial=n, spacing=None)
        if spacing is not None:
            c["spacing"] = np.asarray(spacing, float)[-n:]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return fn(x, p, c)

    if per_plane and n_spatial > 2:
        lead = work.shape[: n_spatial - 2]
        out = None
        for idx in np.ndindex(lead):
            r = run(work[idx], 2)
            if out is None:
                out = np.empty(lead + r.shape, r.dtype)
            out[idx] = r
    else:
        out = run(work, n_spatial)

    if kind == "image":
        out = np.asarray(out, dtype=dtype)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = invalid_value
    info = {"kind": kind}
    if thresholds:
        th = thresholds
        info["threshold"] = th[0] if len(th) == 1 else (min(th), max(th))
    return out, info


def cast_like(arr, dtype):
    """Round/clip a float result back into an integer (or float) dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.nan_to_num(arr, nan=0.0)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    if dtype == bool:
        return arr != 0
    return arr.astype(dtype)


# ---------------------------------------------------------------- labels
def _fg(d):
    return d != 0


def fit_label_dtype(arr):
    """Cast to the smallest integer dtype that can hold all values."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.astype(np.uint8)
    mn, mx = int(arr.min()), int(arr.max())
    cands = (
        (np.uint8, np.uint16, np.uint32, np.uint64)
        if mn >= 0
        else (np.int8, np.int16, np.int32, np.int64)
    )
    for dt in cands:
        info = np.iinfo(dt)
        if info.min <= mn and mx <= info.max:
            return arr.astype(dt, copy=False)
    return arr.astype(np.int64, copy=False)


def _sequential(d):
    """Relabel non-zero values to 1..n (order preserved). Returns (array, n)."""
    vals = np.unique(d)
    vals = vals[vals != 0]
    if vals.size == 0:
        return np.zeros(d.shape, np.int64), 0
    out = np.searchsorted(vals, d).astype(np.int64) + 1
    out[d == 0] = 0
    return out, int(vals.size)


def _connected_components(d, full_connectivity: bool, binary: bool):
    from scipy import ndimage

    conn = d.ndim if full_connectivity else 1
    struct = ndimage.generate_binary_structure(d.ndim, conn)
    if binary:
        return ndimage.label(_fg(d), structure=struct)[0].astype(np.int64)
    try:  # split by value: touching objects with different labels stay apart
        from skimage.measure import label as sk_label

        return sk_label(d, background=0, connectivity=conn).astype(np.int64)
    except ImportError:  # slow fallback: one value at a time
        out = np.zeros(d.shape, np.int64)
        offset = 0
        for v in np.unique(d):
            if v == 0:
                continue
            lab, n = ndimage.label(d == v, structure=struct)
            m = lab > 0
            out[m] = lab[m] + offset
            offset += n
        return out


def relabel(d, full_connectivity=False, binary=False, per_plane=False):
    d = np.asarray(d)
    if per_plane and d.ndim > 2:
        out = np.zeros(d.shape, np.int64)
        offset = 0
        for idx in np.ndindex(d.shape[:-2]):
            lab = _connected_components(d[idx], full_connectivity, binary)
            m = lab > 0
            plane = out[idx]  # view
            plane[m] = lab[m] + offset
            offset += int(lab.max()) if lab.size else 0
        return out
    return _connected_components(d, full_connectivity, binary)


# name -> (min inputs, max inputs or None, short name, help text)
LABEL_OPS = {
    "Invert (binary)": (1, 1, "invert",
        "Exactly one layer. Background → 1, any label → 0."),
    "Add (sum values)": (1, None, "sum",
        "Sums label values pixel by pixel (1 + 1 = 2 where they overlap)."),
    "Binary add (union)": (1, None, "union",
        "Any label in any layer → 1. One layer = binarise."),
    "Intersect (binary)": (1, None, "intersect",
        "Pixels labelled in all layers → 1."),
    "Combine keep": (1, None, "combine",
        "Overlays layers keeping original values; on overlap the later "
        "(lower in the list) layer wins."),
    "Combine accumulate": (1, None, "accumulate",
        "Relabels each layer to 1..n and offsets it past the previous layers, "
        "so all labels are unique; later layer wins on overlap."),
    "Subtract (binary)": (2, None, "subtract",
        "First layer minus all the others. Binary output."),
    "Multiply": (1, None, "multiply",
        "Multiplies label values (labels × binary mask = masked labels)."),
    "Relabel (connected components)": (1, None, "relabel",
        "Connected-component relabelling. Several inputs are merged first "
        "(combine keep, or union in binary mode)."),
}


def labels_math(arrays, op: str, *, full_connectivity=False, binary=False,
                per_plane=False):
    """Combine a list of label arrays. Returns an integer array."""
    if not arrays:
        raise ValueError("No input layers.")
    lo, hi, _, _ = LABEL_OPS[op]
    n = len(arrays)
    if n < lo or (hi is not None and n > hi):
        need = f"exactly {lo}" if lo == hi else f"at least {lo}"
        raise ValueError(f"'{op}' needs {need} input layer(s), got {n}.")

    arrays = [np.asarray(a) for a in arrays]
    for a in arrays:
        if not np.issubdtype(a.dtype, np.integer) and a.dtype != bool:
            raise ValueError(f"Label data must be integer, got {a.dtype}.")
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    arrays = [np.broadcast_to(a, shape) for a in arrays]

    if op == "Invert (binary)":
        return (~_fg(arrays[0])).astype(np.uint8)

    if op == "Add (sum values)":
        out = np.zeros(shape, np.int64)
        for a in arrays:
            out += a
        return fit_label_dtype(out)

    if op == "Binary add (union)":
        out = np.zeros(shape, bool)
        for a in arrays:
            out |= _fg(a)
        return out.astype(np.uint8)

    if op == "Intersect (binary)":
        out = np.ones(shape, bool)
        for a in arrays:
            out &= _fg(a)
        return out.astype(np.uint8)

    if op == "Combine keep":
        out = np.zeros(shape, np.result_type(*arrays, np.uint8))
        for a in arrays:
            m = _fg(a)
            out[m] = a[m]
        return fit_label_dtype(out)

    if op == "Combine accumulate":
        out = np.zeros(shape, np.int64)
        offset = 0
        for a in arrays:
            seq, k = _sequential(a)
            m = seq > 0
            out[m] = seq[m] + offset
            offset += k
        return fit_label_dtype(out)

    if op == "Subtract (binary)":
        out = _fg(arrays[0]).copy()
        for a in arrays[1:]:
            out &= ~_fg(a)
        return out.astype(np.uint8)

    if op == "Multiply":
        out = np.ones(shape, np.int64)
        for a in arrays:
            out *= a
        return fit_label_dtype(out)

    if op == "Relabel (connected components)":
        if len(arrays) == 1:
            merged = arrays[0]
        elif binary:
            merged = labels_math(arrays, "Binary add (union)")
        else:
            merged = labels_math(arrays, "Combine keep")
        res = relabel(merged, full_connectivity, binary, per_plane)
        return fit_label_dtype(res)

    raise ValueError(f"Unknown operation {op!r}")


# =============================================================================
# Qt widget
# =============================================================================

def _layer_array(layer):
    if getattr(layer, "multiscale", False):
        raise ValueError(f"'{layer.name}' is multiscale – not supported.")
    return np.asarray(layer.data)


class LayerCombo(QComboBox):
    """Combo box listing layers of one type; refreshes itself when opened."""

    def __init__(self, viewer, layer_type, none_text=None, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._type = layer_type
        self._none_text = none_text
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(14)
        self.refresh()

    def showPopup(self):
        self.refresh()
        super().showPopup()

    def refresh(self):
        current = self.current_layer()
        self.blockSignals(True)
        self.clear()
        if self._none_text is not None:
            self.addItem(self._none_text, None)
        # same order as napari's layer list (top first)
        for layer in reversed(list(self._viewer.layers)):
            if isinstance(layer, self._type):
                self.addItem(layer.name, layer)
        for i in range(self.count()):
            if current is not None and self.itemData(i) is current:
                self.setCurrentIndex(i)
                break
        self.blockSignals(False)

    def current_layer(self):
        layer = self.currentData()
        if layer is None or layer not in self._viewer.layers:
            return None
        return layer

    def set_type(self, layer_type):
        if layer_type is not self._type:
            self._type = layer_type
            self.refresh()

    def set_layer(self, layer):
        self.refresh()
        for i in range(self.count()):
            if self.itemData(i) is layer:
                self.setCurrentIndex(i)
                return


def _output_box(viewer, layer_type):
    box = QGroupBox("Output")
    form = QFormLayout(box)
    target = LayerCombo(viewer, layer_type, none_text=NEW_LAYER)
    target.setToolTip("Create a new layer, or overwrite an existing one.")
    name = QLineEdit()
    name.setPlaceholderText("auto")
    form.addRow("Write to:", target)
    form.addRow("New layer name:", name)
    target.currentIndexChanged.connect(
        lambda *_: name.setEnabled(target.current_layer() is None)
    )
    return box, target, name


class LayerMathWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setMinimumWidth(340)

        tabs = QTabWidget()
        tabs.addTab(self._build_image_tab(), "Two images")
        tabs.addTab(self._build_single_tab(), "Single image")
        tabs.addTab(self._build_labels_tab(), "Labels")

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray;")

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(self.status)

        # reserve room for the (initially hidden) relabel options so nothing
        # gets squeezed when they appear
        self.cc_opts.setVisible(True)
        self.setMinimumHeight(self.sizeHint().height())
        self._on_label_op_changed(self.lbl_op.currentText())

        # keep in sync with the layer list
        ev = viewer.layers.events
        self._connections = [
            (ev.inserted, self._on_layers_changed),
            (ev.removed, self._on_layers_changed),
            (ev.moved, self._on_layers_changed),
        ]
        for emitter, cb in self._connections:
            emitter.connect(cb)
        for layer in viewer.layers:
            layer.events.name.connect(self._on_layers_changed)
        self.destroyed.connect(self._disconnect)

    # ------------------------------------------------------------------ images
    def _build_image_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        inp = QGroupBox("Inputs")
        form = QFormLayout(inp)
        self.img_a = LayerCombo(self.viewer, Image)
        self.img_b = LayerCombo(self.viewer, Image)
        if self.img_b.count() > 1:
            self.img_b.setCurrentIndex(1)
        swap = QToolButton()
        swap.setText("⇅")
        swap.setToolTip("Swap A and B")
        swap.clicked.connect(self._swap_ab)
        row_a = QHBoxLayout()
        row_a.addWidget(self.img_a, 1)
        row_a.addWidget(swap)
        form.addRow("A:", row_a)
        form.addRow("B:", self.img_b)

        self.use_const = QCheckBox("use constant for B")
        self.const = QDoubleSpinBox()
        self.const.setRange(-1e12, 1e12)
        self.const.setDecimals(6)
        self.const.setValue(1.0)
        self.const.setEnabled(False)
        self.use_const.toggled.connect(self.const.setEnabled)
        self.use_const.toggled.connect(lambda on: self.img_b.setEnabled(not on))
        row_c = QHBoxLayout()
        row_c.addWidget(self.use_const)
        row_c.addWidget(self.const, 1)
        form.addRow(row_c)
        v.addWidget(inp)

        opt = QGroupBox("Operation")
        f2 = QFormLayout(opt)
        self.img_op = QComboBox()
        self.img_op.addItems(list(IMAGE_OPS))
        f2.addRow(self.img_op)
        self.img_dtype = QComboBox()
        self.img_dtype.addItems(["float32", "float64"])
        f2.addRow("Output type:", self.img_dtype)
        self.img_invalid = QComboBox()
        self.img_invalid.addItems(["0", "NaN"])
        self.img_invalid.setToolTip("Value used where the result is not finite "
                                    "(division by zero, log of 0 ...)")
        f2.addRow("x/0, invalid →", self.img_invalid)
        v.addWidget(opt)

        box, self.img_target, self.img_name = _output_box(self.viewer, Image)
        v.addWidget(box)

        btn = QPushButton("Compute")
        btn.setMinimumHeight(30)
        btn.clicked.connect(lambda: self._run(self._compute_image))
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _swap_ab(self):
        a, b = self.img_a.current_layer(), self.img_b.current_layer()
        self.img_a.set_layer(b)
        self.img_b.set_layer(a)

    def _compute_image(self):
        A = self.img_a.current_layer()
        if A is None:
            raise ValueError("Choose image layer A.")
        a = _layer_array(A)
        op = self.img_op.currentText()
        sym = IMAGE_OPS[op][0]
        if self.use_const.isChecked():
            b = self.const.value()
            b_name = f"{b:g}"
            src = A
        else:
            B = self.img_b.current_layer()
            if B is None:
                raise ValueError("Choose image layer B (or use a constant).")
            b = _layer_array(B)
            b_name = B.name
            src = A if A.ndim >= B.ndim else B
            if getattr(A, "rgb", False) != getattr(B, "rgb", False):
                show_warning("Mixing an RGB and a non-RGB image.")

        dtype = np.dtype(self.img_dtype.currentText())
        invalid = 0.0 if self.img_invalid.currentText() == "0" else np.nan
        res = image_math(a, b, op, dtype=dtype, invalid_value=invalid)

        if sym in "+-*/":
            auto = f"{A.name} {sym} {b_name}"
        else:
            auto = f"{sym}({A.name}, {b_name})"
        name = self.img_name.text().strip() or auto
        return self._emit(res, Image, self.img_target.current_layer(), name, src)

    # ------------------------------------------------------------ single image
    def _build_single_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        inp = QGroupBox("Input")
        f = QFormLayout(inp)
        self.sgl_in = LayerCombo(self.viewer, Image)
        f.addRow("Image:", self.sgl_in)
        v.addWidget(inp)

        opt = QGroupBox("Operation")
        ov = QVBoxLayout(opt)
        self.sgl_op = QComboBox()
        for name, spec in SINGLE_OPS.items():
            if spec is None:
                self.sgl_op.insertSeparator(self.sgl_op.count())
            else:
                self.sgl_op.addItem(name)
        ov.addWidget(self.sgl_op)
        self.sgl_help = QLabel()
        self.sgl_help.setWordWrap(True)
        self.sgl_help.setStyleSheet("color: gray;")
        ov.addWidget(self.sgl_help)

        # one parameter page per operation
        self.sgl_stack = QStackedWidget()
        self.sgl_pages = {}   # op -> (page index, {key: widget})
        for name, spec in SINGLE_OPS.items():
            if spec is None:
                continue
            page = QWidget()
            pf = QFormLayout(page)
            pf.setContentsMargins(0, 0, 0, 0)
            widgets = {}
            for key, kind, label, default, extra in spec[2]:
                if kind == "float":
                    wd = QDoubleSpinBox()
                    wd.setRange(extra[0], extra[1])
                    wd.setDecimals(extra[2])
                    wd.setValue(default)
                    pf.addRow(label, wd)
                elif kind == "int":
                    wd = QSpinBox()
                    wd.setRange(extra[0], extra[1])
                    wd.setValue(default)
                    pf.addRow(label, wd)
                elif kind == "bool":
                    wd = QCheckBox(label)
                    wd.setChecked(default)
                    pf.addRow(wd)
                else:  # choice
                    wd = QComboBox()
                    wd.addItems(extra)
                    wd.setCurrentText(default)
                    pf.addRow(label, wd)
                widgets[key] = wd
            self.sgl_pages[name] = (self.sgl_stack.addWidget(page), widgets)
        ov.addWidget(self.sgl_stack)

        # manual threshold value only matters for method "Manual"
        thr = self.sgl_pages["Threshold → labels"][1]
        thr["method"].currentTextChanged.connect(
            lambda m: thr["value"].setEnabled(m == "Manual"))
        thr["value"].setEnabled(False)

        self.sgl_plane = QCheckBox("each 2D plane separately")
        self.sgl_plane.setToolTip(
            "For stacks / time-lapse: filters, normalisation and automatic "
            "thresholds are computed per 2D plane instead of on the whole volume.")
        ov.addWidget(self.sgl_plane)
        self.sgl_units = QCheckBox("σ / size in layer units (uses scale)")
        self.sgl_units.setToolTip(
            "Divide σ / size by the layer scale per axis, so an anisotropic "
            "z-stack (e.g. scale 2, 0.2, 0.2) gets a physically round kernel.")
        ov.addWidget(self.sgl_units)

        f3 = QFormLayout()
        self.sgl_dtype = QComboBox()
        self.sgl_dtype.addItems(["float32", "float64", "same as input"])
        f3.addRow("Output type:", self.sgl_dtype)
        self.sgl_invalid = QComboBox()
        self.sgl_invalid.addItems(["0", "NaN"])
        self.sgl_invalid.setToolTip("Value used where the result is not finite")
        f3.addRow("invalid →", self.sgl_invalid)
        ov.addLayout(f3)
        v.addWidget(opt)

        box, self.sgl_target, self.sgl_name = _output_box(self.viewer, Image)
        v.addWidget(box)

        btn = QPushButton("Compute")
        btn.setMinimumHeight(30)
        btn.clicked.connect(lambda: self._run(self._compute_single))
        v.addWidget(btn)
        v.addStretch(1)

        self.sgl_op.currentTextChanged.connect(self._on_single_op_changed)
        self._on_single_op_changed(self.sgl_op.currentText())
        return w

    def _on_single_op_changed(self, op):
        spec = SINGLE_OPS.get(op)
        if spec is None:
            return
        short, kind, params, _, help_ = spec
        self.sgl_help.setText(help_)
        self.sgl_stack.setCurrentIndex(self.sgl_pages[op][0])
        uses_size = any(k in ("sigma", "sigma1", "size") for k, *_ in params)
        self.sgl_units.setEnabled(uses_size)
        is_labels = kind == "labels"
        self.sgl_dtype.setEnabled(not is_labels)
        self.sgl_invalid.setEnabled(not is_labels)
        self.sgl_target.set_type(Labels if is_labels else Image)

    def _compute_single(self):
        A = self.sgl_in.current_layer()
        if A is None:
            raise ValueError("Choose an image layer.")
        a = _layer_array(A)
        op = self.sgl_op.currentText()
        short, kind, spec, _, _ = SINGLE_OPS[op]
        params = {}
        for key, wd in self.sgl_pages[op][1].items():
            if isinstance(wd, QCheckBox):
                params[key] = wd.isChecked()
            elif isinstance(wd, QComboBox):
                params[key] = wd.currentText()
            else:
                params[key] = wd.value()

        rgb = bool(getattr(A, "rgb", False))
        spacing = None
        if self.sgl_units.isChecked() and self.sgl_units.isEnabled():
            sc = np.asarray(A.scale, float)
            spacing = sc  # scale has one entry per non-RGB axis
        out_dt = self.sgl_dtype.currentText()
        work_dt = np.float64 if out_dt == "float64" else np.float32
        invalid = 0.0 if self.sgl_invalid.currentText() == "0" else np.nan

        res, info = single_image_op(
            a, op, params, rgb=rgb, per_plane=self.sgl_plane.isChecked(),
            spacing=spacing, dtype=work_dt, invalid_value=invalid)
        if kind == "image" and out_dt == "same as input":
            res = cast_like(res, a.dtype)

        name = self.sgl_name.text().strip() or f"{short}({A.name})"
        layer_type = Labels if kind == "labels" else Image
        msg = self._emit(res, layer_type, self.sgl_target.current_layer(),
                         name, A)
        th = info.get("threshold")
        if th is not None:
            msg += (f"  – threshold {th:.6g}" if np.isscalar(th) else
                    f"  – thresholds {th[0]:.6g}…{th[1]:.6g}")
        return msg

    # ------------------------------------------------------------------ labels
    def _build_labels_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        inp = QGroupBox("Inputs  (bottom = most recent, wins on overlap)")
        iv = QVBoxLayout(inp)
        self.lbl_list = QListWidget()
        self.lbl_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lbl_list.setMinimumHeight(110)
        iv.addWidget(self.lbl_list)

        row = QHBoxLayout()
        add_btn = QPushButton("Add selected")
        add_btn.setToolTip("Add the labels layers selected in napari's layer list")
        add_btn.clicked.connect(self._add_selected_labels)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_label_inputs)
        clr_btn = QPushButton("Clear")
        clr_btn.clicked.connect(self.lbl_list.clear)
        up_btn = QToolButton()
        up_btn.setText("↑")
        up_btn.clicked.connect(lambda: self._move_label_input(-1))
        dn_btn = QToolButton()
        dn_btn.setText("↓")
        dn_btn.clicked.connect(lambda: self._move_label_input(+1))
        for b_ in (add_btn, rm_btn, clr_btn):
            row.addWidget(b_)
        row.addStretch(1)
        row.addWidget(up_btn)
        row.addWidget(dn_btn)
        iv.addLayout(row)
        v.addWidget(inp)

        opt = QGroupBox("Operation")
        ov = QVBoxLayout(opt)
        self.lbl_op = QComboBox()
        self.lbl_op.addItems(list(LABEL_OPS))
        ov.addWidget(self.lbl_op)
        self.lbl_help = QLabel()
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setStyleSheet("color: gray;")
        ov.addWidget(self.lbl_help)

        self.cc_opts = QWidget()
        cf = QFormLayout(self.cc_opts)
        cf.setContentsMargins(0, 0, 0, 0)
        self.cc_conn = QComboBox()
        self.cc_conn.addItems(["faces only (4 / 6-connected)",
                               "full (8 / 26-connected)"])
        cf.addRow("Connectivity:", self.cc_conn)
        self.cc_binary = QCheckBox("treat as binary (merge touching labels)")
        self.cc_binary.setToolTip(
            "Off: touching objects with different values stay separate.\n"
            "On: everything non-zero is one foreground mask.")
        cf.addRow(self.cc_binary)
        self.cc_plane = QCheckBox("each 2D plane separately (e.g. time-lapse)")
        cf.addRow(self.cc_plane)
        ov.addWidget(self.cc_opts)
        v.addWidget(opt)

        self.lbl_op.currentTextChanged.connect(self._on_label_op_changed)
        self._on_label_op_changed(self.lbl_op.currentText())

        box, self.lbl_target, self.lbl_name = _output_box(self.viewer, Labels)
        v.addWidget(box)

        btn = QPushButton("Compute")
        btn.setMinimumHeight(30)
        btn.clicked.connect(lambda: self._run(self._compute_labels))
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _on_label_op_changed(self, op):
        self.lbl_help.setText(LABEL_OPS[op][3])
        self.cc_opts.setVisible(op.startswith("Relabel"))
        # a floating dock is its own window: grow it if the options don't fit
        win = self.window()
        if win is not self and win.isWindow():
            need = win.sizeHint().expandedTo(win.minimumSizeHint())
            if need.height() > win.height() or need.width() > win.width():
                win.resize(win.size().expandedTo(need))

    def _label_inputs(self):
        layers = []
        for i in range(self.lbl_list.count()):
            layer = self.lbl_list.item(i).data(Qt.UserRole)
            if layer in self.viewer.layers:
                layers.append(layer)
        return layers

    def _add_selected_labels(self):
        present = self._label_inputs()
        # napari layer-list order (bottom -> top), so the topmost ends up last
        sel = [l for l in self.viewer.layers
               if l in self.viewer.layers.selection and isinstance(l, Labels)]
        if not sel:
            self.status.setText("No labels layers selected in the layer list.")
            return
        added = 0
        for layer in sel:
            if any(layer is p for p in present):
                continue
            item = QListWidgetItem(layer.name)
            item.setData(Qt.UserRole, layer)
            self.lbl_list.addItem(item)
            added += 1
        self.status.setText(f"Added {added} layer(s).")

    def _remove_label_inputs(self):
        for item in self.lbl_list.selectedItems():
            self.lbl_list.takeItem(self.lbl_list.row(item))

    def _move_label_input(self, step):
        rows = sorted(self.lbl_list.row(i) for i in self.lbl_list.selectedItems())
        if not rows:
            return
        if step > 0:
            rows = rows[::-1]
        n = self.lbl_list.count()
        moved = []
        for r in rows:
            nr = r + step
            if 0 <= nr < n and nr not in moved:
                item = self.lbl_list.takeItem(r)
                self.lbl_list.insertItem(nr, item)
                moved.append(nr)
            else:
                moved.append(r)
        self.lbl_list.clearSelection()
        for r in moved:
            self.lbl_list.item(r).setSelected(True)

    def _compute_labels(self):
        layers = self._label_inputs()
        if not layers:
            raise ValueError("Add some labels layers first (select them in "
                             "the layer list, then 'Add selected').")
        op = self.lbl_op.currentText()
        arrays = [_layer_array(l) for l in layers]
        res = labels_math(
            arrays, op,
            full_connectivity=self.cc_conn.currentIndex() == 1,
            binary=self.cc_binary.isChecked(),
            per_plane=self.cc_plane.isChecked(),
        )
        src = max(layers, key=lambda l: l.ndim)
        short = LABEL_OPS[op][2]
        names = ", ".join(l.name for l in layers)
        if len(names) > 40:
            names = names[:37] + "..."
        name = self.lbl_name.text().strip() or f"{short}({names})"
        return self._emit(res, Labels, self.lbl_target.current_layer(), name, src)

    # ------------------------------------------------------------------ common
    def _emit(self, data, layer_type, target, name, src):
        if target is not None:
            if target.ndim != data.ndim:
                raise ValueError(
                    f"Result is {data.ndim}D but '{target.name}' is "
                    f"{target.ndim}D – write to a new layer instead.")
            target.data = data
            if layer_type is Image:
                target.reset_contrast_limits()
            target.refresh()
            return f"Overwrote '{target.name}'  {data.shape} {data.dtype}"

        kw = dict(name=name, scale=src.scale, translate=src.translate)
        if layer_type is Image:
            if getattr(src, "rgb", False) and data.shape == src.data.shape:
                kw["rgb"] = True
            layer = self.viewer.add_image(data, **kw)
        else:
            layer = self.viewer.add_labels(data, **kw)
        return f"Created '{layer.name}'  {data.shape} {data.dtype}"

    def _run(self, fn):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            msg = fn()
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.status.setText(f"⚠ {e}")
            show_error(str(e))
            return
        QApplication.restoreOverrideCursor()
        self.status.setText(msg)
        show_info(msg)

    def _on_layers_changed(self, event=None):
        try:
            if event is not None and getattr(event, "type", "") == "inserted":
                event.value.events.name.connect(self._on_layers_changed)
            for combo in (self.img_a, self.img_b, self.img_target,
                          self.sgl_in, self.sgl_target, self.lbl_target):
                combo.refresh()
            for i in reversed(range(self.lbl_list.count())):
                item = self.lbl_list.item(i)
                layer = item.data(Qt.UserRole)
                if layer not in self.viewer.layers:
                    self.lbl_list.takeItem(i)
                else:
                    item.setText(layer.name)
        except RuntimeError:  # widget already deleted
            self._disconnect()

    def _disconnect(self, *_):
        for emitter, cb in getattr(self, "_connections", []):
            try:
                emitter.disconnect(cb)
            except Exception:  # noqa: BLE001
                pass
        self._connections = []


# =============================================================================
# Entry point
# =============================================================================

def open_layer_math(viewer=None, floating=True):
    viewer = viewer or napari.current_viewer() or napari.Viewer()
    qt_win = viewer.window._qt_window

    old = getattr(qt_win, "_layer_math_dock", None)
    if old is not None:
        try:
            old.show()
            old.raise_()
            return old
        except RuntimeError:  # was deleted
            pass

    widget = LayerMathWidget(viewer)
    dock = viewer.window.add_dock_widget(widget, name=WIDGET_TITLE, area="right")
    if floating:
        dock.setFloating(True)
        dock.resize(widget.sizeHint().expandedTo(widget.minimumSizeHint()))
    qt_win._layer_math_dock = dock
    return dock


if __name__ == "__main__":
    _standalone = napari.current_viewer() is None
    open_layer_math()
    if _standalone:
        napari.run()
