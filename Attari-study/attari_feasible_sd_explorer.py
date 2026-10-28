# =============================================================================
# Attari et al. - feasible (SD_pre, SD_post) pairs for a given r and within-group d
# -----------------------------------------------------------------------------
# Self-contained cell (does not depend on the cell above; names are distinct so the
# first app keeps working). Requires: numpy, matplotlib, ipywidgets.
#
#   Var(Post - Pre) = Var(Pre) + Var(Post) - 2 Cov(Pre, Post)
#   => SD_pre^2 + SD_post^2 - 2 r SD_pre SD_post = SD_change^2
#   within-group d = mean_change / SD_change  =>  SD_change = |mean_change| / |d|
#
# For fixed mean change, r and d, every (SD_pre, SD_post) on this curve is
# mathematically compatible. Purely arithmetic: no causal claim, and d here is a
# WITHIN-group standardized change, not the Fluoxetine-vs-Nortriptyline SMD.
# =============================================================================

import io
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


@dataclass(frozen=True)
class FeasGroup:
    name: str
    mean_pre: float
    mean_post: float
    mean_change: float       # reported Mean(Post) - Mean(Pre)
    sd_pre: float            # reported
    sd_post: float           # reported
    sd_change: float         # reported

    @property
    def r_attari(self):
        # r = (SD_pre^2 + SD_post^2 - SD_change^2) / (2 SD_pre SD_post)
        return (self.sd_pre**2 + self.sd_post**2 - self.sd_change**2) / (2 * self.sd_pre * self.sd_post)

    @property
    def d_attari(self):
        return self.mean_change / self.sd_change


FEAS_GROUPS = [
    FeasGroup("Fluoxetine", 28.9, 17.9, -10.9, 8.4, 8.1, 2.6),
    FeasGroup("Nortriptyline", 28.4, 25.8, -2.6, 8.7, 8.5, 0.8),
]

FEAS_R_RANGE = (-0.99, 0.999)
FEAS_R_STEP = 1e-5      # fine: near r ~ 0.996 a 0.001 step moves Nortriptyline's d by ~0.03
FEAS_D_RANGE = (-8.0, -0.1)
FEAS_D_STEP = 0.001
FEAS_TOL_D = 0.01       # "Compatible" if |selected d - d implied by Attari SDs| < 0.01
FEAS_AXIS_MAX = 15.0    # default shared axis limit (both panels, both axes)
FEAS_N_PARAM = 4000

FEAS_COLORS = {"curve": "#2a78d6", "attari": "#0b0b0b", "nearest": "#898781",
               "grid": "#e1e0d9", "axis": "#c3c2b7", "ink2": "#52514e", "surface": "#fcfcfb",
               "good": "#006300", "bad": "#b3261e"}


# -----------------------------------------------------------------------------
# Formulas
# -----------------------------------------------------------------------------
def feas_sd_change(mean_change, d):
    """SD_change = |mean_change| / |d|  (the curve depends on |d| only; sign = direction)."""
    return abs(mean_change) / max(abs(d), 1e-9)


def feas_curve(c, r, n=FEAS_N_PARAM):
    """Points (x, y) = (SD_pre, SD_post) on x^2 + y^2 - 2rxy = c^2 with x > 0, y > 0.

    Equivalent to the quadratic solution y = r x +/- sqrt(c^2 - (1 - r^2) x^2)
    (real for |x| <= c / sqrt(1 - r^2)), but drawn parametrically so both branches
    join smoothly at the vertical tangents. Rotating by 45 deg,
    u = (x + y)/sqrt2, v = (y - x)/sqrt2 gives (1 - r)u^2 + (1 + r)v^2 = c^2, so
    u = c cos t / sqrt(1 - r), v = c sin t / sqrt(1 + r).
    Returns a list of contiguous positive-quadrant segments (no NaNs)."""
    t = np.linspace(0, 2 * np.pi, n)
    u = c * np.cos(t) / np.sqrt(1 - r)
    v = c * np.sin(t) / np.sqrt(1 + r)
    x, y = (u - v) / np.sqrt(2), (u + v) / np.sqrt(2)
    keep = (x > 0) & (y > 0)
    # Split into runs of consecutive kept points (wrap-around joined when t=0 and t=2pi both kept).
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, breaks + 1)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    return [(x[run], y[run]) for run in runs if run.size > 1]


def feas_d_implied(mean_change, sd_pre, sd_post, r):
    """d implied by a given SD pair at correlation r: mean_change / sqrt(x^2 + y^2 - 2rxy)."""
    var = sd_pre**2 + sd_post**2 - 2 * r * sd_pre * sd_post
    return mean_change / np.sqrt(var) if var > 1e-12 else np.nan


def feas_r_needed(sd_pre, sd_post, c):
    """r at which the given SD pair lies on the curve for change SD c."""
    return (sd_pre**2 + sd_post**2 - c**2) / (2 * sd_pre * sd_post)


def feas_nearest(segments, x0, y0):
    """Closest sampled curve point to (x0, y0) and its Euclidean distance."""
    if not segments:
        return None
    xs = np.concatenate([s[0] for s in segments])
    ys = np.concatenate([s[1] for s in segments])
    k = np.argmin(np.hypot(xs - x0, ys - y0))
    return xs[k], ys[k], float(np.hypot(xs[k] - x0, ys[k] - y0))


# -----------------------------------------------------------------------------
# Plot + summary (widget-free)
# -----------------------------------------------------------------------------
def feas_draw(ax, g, r, d, axis_max=FEAS_AXIS_MAX):
    c = feas_sd_change(g.mean_change, d)
    segs = feas_curve(c, r)

    ax.set_facecolor(FEAS_COLORS["surface"])
    ax.plot([0, axis_max], [0, axis_max], color=FEAS_COLORS["axis"], lw=1, ls="--", zorder=1,
            label="SD pre = SD post")
    for i, (x, y) in enumerate(segs):
        ax.plot(x, y, color=FEAS_COLORS["curve"], lw=2.2, zorder=3,
                label=f"Feasible SDs (r = {r:.4f}, d = {d:.3f})" if i == 0 else None)

    near = feas_nearest(segs, g.sd_pre, g.sd_post)
    if near and near[2] > 0.05:  # subtle segment (hidden when the point is effectively on the curve) from the Attari point to the nearest curve point
        ax.plot([g.sd_pre, near[0]], [g.sd_post, near[1]], color=FEAS_COLORS["nearest"],
                lw=1.2, ls=":", zorder=2, label=f"Distance to curve ({near[2]:.2f})")

    ax.plot(g.sd_pre, g.sd_post, "D", ms=10, color=FEAS_COLORS["attari"], mec="white", mew=1.5,
            zorder=5, label="Attari et al. reported SDs")
    ax.annotate(f"({g.sd_pre}, {g.sd_post})", (g.sd_pre, g.sd_post), xytext=(10, -12),
                textcoords="offset points", fontsize=9, color=FEAS_COLORS["ink2"])

    beyond = any((s[0].max() > axis_max) or (s[1].max() > axis_max) for s in segs)
    if beyond:
        ax.text(0.98, 0.02, "Curve extends beyond the axes", transform=ax.transAxes,
                ha="right", fontsize=8, color=FEAS_COLORS["ink2"])
    if not segs:
        ax.text(0.5, 0.5, "No positive SD pair is feasible", transform=ax.transAxes,
                ha="center", color=FEAS_COLORS["bad"])

    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect("equal")
    ax.set_xlabel("SD pre")
    ax.set_ylabel("SD post")
    ax.set_title(f"{g.name}: feasible (SD pre, SD post)", fontsize=12, loc="left")
    ax.grid(True, color=FEAS_COLORS["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FEAS_COLORS["axis"])
    ax.tick_params(colors=FEAS_COLORS["ink2"])
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)


def feas_summary(g, r, d):
    c = feas_sd_change(g.mean_change, d)
    d_imp = feas_d_implied(g.mean_change, g.sd_pre, g.sd_post, r)
    diff = abs(d - d_imp) if np.isfinite(d_imp) else np.inf
    ok = diff < FEAS_TOL_D
    r_need = feas_r_needed(g.sd_pre, g.sd_post, c)
    r_need_txt = f"{r_need:.5f}" + ("" if -1 < r_need < 1 else " (outside (-1, 1): impossible)")
    status = (f"<b style='color:{FEAS_COLORS['good']}'>Compatible</b>" if ok else
              f"<b style='color:{FEAS_COLORS['bad']}'>Not compatible</b>")
    rows = [
        ("Fixed mean change (Post - Pre)", f"{g.mean_change:.1f}"),
        ("Selected r", f"{r:.5f} &nbsp;<span style='color:#898781'>(Attari implied {g.r_attari:.5f})</span>"),
        ("Selected within-group d", f"{d:.3f} &nbsp;<span style='color:#898781'>(Attari {g.d_attari:.3f})</span>"),
        ("Implied SD change = |mean change| / |d|", f"{c:.3f}"),
        ("Attari SD pre / SD post", f"{g.sd_pre} / {g.sd_post}"),
        ("d implied by Attari SDs at current r", "undefined" if not np.isfinite(d_imp) else f"{d_imp:.3f}"),
        ("Difference |selected d - implied d|", f"{diff:.3f}"),
        ("r that puts Attari SDs on this curve", r_need_txt),
        (f"Status (tolerance {FEAS_TOL_D})", status),
    ]
    body = "".join(f"<tr><td style='padding:2px 12px 2px 0;color:#52514e'>{k}</td>"
                   f"<td style='padding:2px 0;font-variant-numeric:tabular-nums'>{v}</td></tr>"
                   for k, v in rows)
    return f"<table style='font-size:13px;border-collapse:collapse'>{body}</table>"


def feas_fig_to_png(fig):
    """Render a figure to PNG bytes and close it (so the inline backend never shows it too)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Interactive panel (one class for both arms)
# -----------------------------------------------------------------------------
class FeasPanel:
    def __init__(self, g, get_axis_max):
        self.g, self.get_axis_max, self._suspend = g, get_axis_max, False
        style, layout = {"description_width": "70px"}, widgets.Layout(width="95%")
        self.r = widgets.FloatSlider(value=g.r_attari, min=FEAS_R_RANGE[0], max=FEAS_R_RANGE[1],
                                     step=FEAS_R_STEP, description="r", readout_format=".5f",
                                     continuous_update=True, style=style, layout=layout)
        self.d = widgets.FloatSlider(value=g.d_attari, min=FEAS_D_RANGE[0], max=FEAS_D_RANGE[1],
                                     step=FEAS_D_STEP, description="d (within)", readout_format=".3f",
                                     continuous_update=True, style=style, layout=layout)
        # PNG replaced in place on each update (no stacked plots in VS Code / JupyterLab).
        self.out = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
        self.summary = widgets.HTML()
        for w in (self.r, self.d):
            w.observe(self.update, names="value")
        self.widget = widgets.VBox(
            [widgets.HTML(f"<h3 style='margin:4px 0'>{g.name}</h3>"), self.r, self.d,
             self.out, self.summary], layout=widgets.Layout(width="50%", padding="0 10px"))
        self.update()

    def update(self, *_):
        if self._suspend:
            return
        fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
        fig.patch.set_facecolor(FEAS_COLORS["surface"])
        feas_draw(ax, self.g, self.r.value, self.d.value, self.get_axis_max())
        fig.tight_layout()
        self.out.value = feas_fig_to_png(fig)
        self.summary.value = feas_summary(self.g, self.r.value, self.d.value)

    def reset(self):
        self._suspend = True
        self.r.value, self.d.value = self.g.r_attari, self.g.d_attari
        self._suspend = False
        self.update()


def build_feasible_app():
    axis_max = widgets.FloatSlider(value=FEAS_AXIS_MAX, min=5, max=80, step=1,
                                   description="Shared axis max", readout_format=".0f",
                                   style={"description_width": "120px"},
                                   layout=widgets.Layout(width="400px"))
    panels = [FeasPanel(g, lambda: axis_max.value) for g in FEAS_GROUPS]

    def redraw_all(*_):
        for p in panels:
            p.update()

    def reset_both(_):
        axis_max.unobserve(redraw_all, names="value")
        axis_max.value = FEAS_AXIS_MAX
        axis_max.observe(redraw_all, names="value")
        for p in panels:
            p.reset()

    axis_max.observe(redraw_all, names="value")
    btn = widgets.Button(description="Reset Both", icon="undo", button_style="primary",
                         tooltip="Restore r and d implied by the Attari et al. values")
    btn.on_click(reset_both)

    intro = widgets.HTML(
        "<div style='font-size:13px;line-height:1.45;max-width:1100px'>"
        "<b>Which (SD pre, SD post) pairs are possible?</b> For a fixed mean change, a chosen pre-post "
        "correlation r and a chosen <i>within-group</i> standardized change d = mean change / SD<sub>change</sub>, "
        "the feasible pairs satisfy SD<sub>pre</sub>&sup2; + SD<sub>post</sub>&sup2; &minus; "
        "2r&middot;SD<sub>pre</sub>&middot;SD<sub>post</sub> = (mean change / d)&sup2;. "
        "If the blue curve passes through the black diamond, the chosen r and d are consistent with the "
        "reported SDs. This is arithmetic, not a causal estimate, and d is not the between-group "
        "(Fluoxetine vs Nortriptyline) standardized mean difference.</div>")

    return widgets.VBox([
        intro,
        widgets.HBox([btn, axis_max], layout=widgets.Layout(align_items="center", gap="16px")),
        widgets.HBox([p.widget for p in panels]),
    ])


feasible_app = build_feasible_app()
display(feasible_app)
