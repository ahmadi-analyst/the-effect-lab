# =============================================================================
# Attari et al. - pre/post SD, correlation, change-score SD and standardized change
# -----------------------------------------------------------------------------
# Run this single cell in Jupyter Notebook / JupyterLab.
# Requires: numpy, matplotlib, ipywidgets   (pip install numpy matplotlib ipywidgets)
#
# What the plot shows (within-group, per treatment arm):
#   SD_change(r) = sqrt(SD_pre^2 + SD_post^2 - 2 r SD_pre SD_post)
#   SMD(r)       = mean_change / SD_change(r),   mean_change = Mean(Post) - Mean(Pre)
#
# This is pure arithmetic on summary statistics: "given SD_pre, SD_post and an
# assumed pre-post correlation r, what change-score SD and standardized change
# are implied?"  It is NOT a between-group effect size (Fluoxetine vs Nortriptyline).
# =============================================================================

import io
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# -----------------------------------------------------------------------------
# Study inputs (Attari et al.) - also the values restored by "Reset Both"
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class StudyGroup:
    name: str
    mean_pre: float
    mean_post: float
    mean_change_reported: float  # as printed in the paper
    sd_pre: float
    sd_post: float
    sd_change_reported: float

    @property
    def mean_change_computed(self):
        return self.mean_post - self.mean_pre


FLUOXETINE = StudyGroup("Fluoxetine", mean_pre=28.9, mean_post=17.9, mean_change_reported=-10.9,
                        sd_pre=8.4, sd_post=8.1, sd_change_reported=2.6)
NORTRIPTYLINE = StudyGroup("Nortriptyline", mean_pre=28.4, mean_post=25.8, mean_change_reported=-2.6,
                           sd_pre=8.7, sd_post=8.5, sd_change_reported=0.8)

# Which mean change drives the curves. The paper reports -10.9 for Fluoxetine, while
# 17.9 - 28.9 = -11.0 (rounding in the published means). The reported value is used
# so the reference marker reproduces the paper's -10.9 / 2.6. Set to False to use
# Mean(Post) - Mean(Pre) instead.
USE_REPORTED_MEAN_CHANGE = True

# -----------------------------------------------------------------------------
# Display settings
# -----------------------------------------------------------------------------
SD_MIN, SD_MAX, SD_STEP = 0.5, 20.0, 0.1
R_GRID = np.linspace(-1.0, 1.0, 4001)  # dense grid over the full valid range
EPS_SD = 1e-6                          # SD_change below this is treated as zero (masked)
X_LIMITS = (-1.02, 1.02)
Y_LIMITS_DEFAULT = (-15.0, 0.5)        # shared by both panels so they are comparable

COLORS = {
    "curve": "#2a78d6",      # current-slider curve
    "attari": "#0b0b0b",     # fixed study reference point
    "required": "#eb6834",   # required r for reported change SD
    "probe": "#898781",      # probe r guide
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "ink2": "#52514e",
    "surface": "#fcfcfb",
}


# -----------------------------------------------------------------------------
# Core formulas
# -----------------------------------------------------------------------------
def change_sd(sd_pre, sd_post, r):
    """SD of (post - pre): sqrt(SD_pre^2 + SD_post^2 - 2 r SD_pre SD_post).
    Returns NaN where the variance is ~0 or negative (numerically), so callers never divide by 0."""
    var = sd_pre**2 + sd_post**2 - 2.0 * np.asarray(r, dtype=float) * sd_pre * sd_post
    sd = np.sqrt(np.clip(var, 0.0, None))
    return np.where(sd > EPS_SD, sd, np.nan)


def standardized_change(mean_change, sd_pre, sd_post, r):
    """Within-group SMD = mean_change / SD_change(r). Sign is preserved (negative = CDI fell)."""
    return mean_change / change_sd(sd_pre, sd_post, r)


def implied_r(sd_pre, sd_post, sd_change):
    """Correlation that reproduces a given change SD:
    r = (SD_pre^2 + SD_post^2 - SD_change^2) / (2 SD_pre SD_post)."""
    return (sd_pre**2 + sd_post**2 - sd_change**2) / (2.0 * sd_pre * sd_post)


def is_valid_r(r):
    return -1.0 <= r <= 1.0


def mean_change_of(group):
    return group.mean_change_reported if USE_REPORTED_MEAN_CHANGE else group.mean_change_computed


def reference_point(group):
    """Fixed Attari et al. point: (implied r from original SDs, mean change / reported change SD)."""
    return (implied_r(group.sd_pre, group.sd_post, group.sd_change_reported),
            mean_change_of(group) / group.sd_change_reported)


# -----------------------------------------------------------------------------
# Plotting (widget-free, so it can also be used for static figures)
# -----------------------------------------------------------------------------
def draw_panel(ax, group, sd_pre, sd_post, r_probe, y_limits=Y_LIMITS_DEFAULT):
    d = mean_change_of(group)
    ref_r, ref_smd = reference_point(group)
    r_req = implied_r(sd_pre, sd_post, group.sd_change_reported)

    ax.set_facecolor(COLORS["surface"])
    ax.axhline(0.0, color=COLORS["axis"], lw=1.2, zorder=1)  # SMD = 0 reference

    # Current curve; points with a ~zero denominator are NaN and therefore not drawn.
    smd_curve = standardized_change(d, sd_pre, sd_post, R_GRID)
    ax.plot(R_GRID, smd_curve, color=COLORS["curve"], lw=2.2, zorder=3,
            label=f"Current curve (SD pre {sd_pre:.1f}, SD post {sd_post:.1f})")

    # Probe r: vertical guide + the SMD it implies under the current SDs.
    smd_probe = float(standardized_change(d, sd_pre, sd_post, r_probe))
    ax.axvline(r_probe, color=COLORS["probe"], lw=1, ls=":", zorder=2)
    if np.isfinite(smd_probe) and y_limits[0] <= smd_probe <= y_limits[1]:
        ax.plot(r_probe, smd_probe, "o", ms=7, color=COLORS["probe"], mec="white", mew=1.5,
                zorder=4, label=f"Probe r = {r_probe:.3f}")

    # Required r to reproduce the reported change SD with the current SDs (optional marker).
    # By construction its y-value equals mean_change / reported change SD.
    if is_valid_r(r_req):
        # Hollow ring so it stays visible when it coincides with the Attari point (at reset values).
        ax.plot(r_req, d / group.sd_change_reported, "o", ms=16, mfc="none",
                mec=COLORS["required"], mew=2.2, zorder=7, label="Required r for reported change SD")

    # Fixed reference point from the original paper (never moves with the sliders).
    ax.plot(ref_r, ref_smd, "D", ms=10, color=COLORS["attari"], mec="white", mew=1.5,
            zorder=6, label="Attari et al. implied point")
    ax.annotate(f"r = {ref_r:.3f}\nSMD = {ref_smd:.2f}", (ref_r, ref_smd),
                xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, color=COLORS["ink2"])

    # Flag when the curve runs past the shared y-axis floor (it diverges as r -> 1 if SD_pre ~ SD_post).
    finite = smd_curve[np.isfinite(smd_curve)]
    if finite.size and finite.min() < y_limits[0]:
        ax.text(0.02, 0.03, f"Curve continues below {y_limits[0]:.0f} "
                f"(min shown grid value {finite.min():.1f})",
                transform=ax.transAxes, fontsize=8, color=COLORS["ink2"])

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Pre-post correlation (r)")
    ax.set_ylabel("Standardized change (SMD)")
    ax.set_title(f"{group.name}: within-group standardized change", fontsize=12, loc="left")
    ax.grid(True, color=COLORS["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["ink2"])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=8.5, frameon=False)


def summary_html(group, sd_pre, sd_post, r_probe):
    d = mean_change_of(group)
    ref_r, ref_smd = reference_point(group)
    r_req = implied_r(sd_pre, sd_post, group.sd_change_reported)
    sd_probe = float(change_sd(sd_pre, sd_post, r_probe))
    smd_probe = d / sd_probe if np.isfinite(sd_probe) else np.nan

    if is_valid_r(r_req):
        req_text = f"{r_req:.4f} <span style='color:#006300'>(valid)</span>"
    else:
        req_text = (f"{r_req:.4f} <span style='color:#b3261e'>(outside [-1, 1]: these SDs cannot "
                    f"reproduce the reported change SD of {group.sd_change_reported} for any valid r)</span>")
    fmt = lambda x, p=3: "undefined (SD change ~ 0)" if not np.isfinite(x) else f"{x:.{p}f}"
    source = "reported" if USE_REPORTED_MEAN_CHANGE else "Post - Pre"

    rows = [
        ("Current SD pre / SD post", f"{sd_pre:.1f} / {sd_post:.1f}"),
        (f"Mean change ({source})",
         f"{d:.1f} &nbsp;<span style='color:#898781'>(Post - Pre = {group.mean_change_computed:.1f})</span>"),
        ("Reported change SD", f"{group.sd_change_reported}"),
        ("Attari implied r (original SDs)", f"{ref_r:.4f}"),
        ("Attari standardized change", f"{ref_smd:.3f}"),
        ("Required r for reported change SD (current SDs)", req_text),
        (f"At probe r = {r_probe:.3f}: implied change SD", fmt(sd_probe)),
        (f"At probe r = {r_probe:.3f}: implied SMD", fmt(smd_probe)),
    ]
    body = "".join(f"<tr><td style='padding:2px 12px 2px 0;color:#52514e'>{k}</td>"
                   f"<td style='padding:2px 0;font-variant-numeric:tabular-nums'>{v}</td></tr>"
                   for k, v in rows)
    return f"<table style='font-size:13px;border-collapse:collapse'>{body}</table>"


def fig_to_png(fig):
    """Render a figure to PNG bytes and close it (so the inline backend never shows it too)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Interactive panel (one class, instantiated once per treatment arm)
# -----------------------------------------------------------------------------
class TreatmentPanel:
    def __init__(self, group, get_y_limits):
        self.group = group
        self.get_y_limits = get_y_limits
        self._suspend = False
        style = {"description_width": "70px"}
        layout = widgets.Layout(width="95%")
        slider = lambda desc, val: widgets.FloatSlider(
            value=val, min=SD_MIN, max=SD_MAX, step=SD_STEP, description=desc,
            readout=True, readout_format=".1f", continuous_update=True, style=style, layout=layout)
        self.sd_pre = slider("SD pre", group.sd_pre)
        self.sd_post = slider("SD post", group.sd_post)
        self.r_probe = widgets.FloatSlider(
            value=reference_point(group)[0], min=-1.0, max=1.0, step=0.001, description="Probe r",
            readout=True, readout_format=".3f", continuous_update=True, style=style, layout=layout)
        # An Image widget whose PNG is replaced on each update: avoids stacked plots in
        # front ends (e.g. VS Code) where clear_output() inside an Output widget is unreliable.
        self.plot_out = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
        self.summary = widgets.HTML()
        for w in (self.sd_pre, self.sd_post, self.r_probe):
            w.observe(self.update, names="value")
        self.widget = widgets.VBox(
            [widgets.HTML(f"<h3 style='margin:4px 0'>{group.name}</h3>"),
             self.sd_pre, self.sd_post, self.r_probe, self.plot_out, self.summary],
            layout=widgets.Layout(width="50%", padding="0 10px"))
        self.update()

    def update(self, *_):
        if self._suspend:
            return
        sd_pre, sd_post, r_probe = self.sd_pre.value, self.sd_post.value, self.r_probe.value
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=100)
        fig.patch.set_facecolor(COLORS["surface"])
        draw_panel(ax, self.group, sd_pre, sd_post, r_probe, self.get_y_limits())
        fig.tight_layout()
        self.plot_out.value = fig_to_png(fig)
        self.summary.value = summary_html(self.group, sd_pre, sd_post, r_probe)

    def reset(self):
        self._suspend = True  # set all values, then redraw once
        self.sd_pre.value = self.group.sd_pre
        self.sd_post.value = self.group.sd_post
        self.r_probe.value = reference_point(self.group)[0]
        self._suspend = False
        self.update()


# -----------------------------------------------------------------------------
# Assemble the app
# -----------------------------------------------------------------------------
def build_app():
    # One shared y-axis floor keeps both panels on the same scale (explicit, user-controlled).
    y_floor = widgets.FloatSlider(value=Y_LIMITS_DEFAULT[0], min=-60.0, max=-2.0, step=0.5,
                                  description="Shared y-axis floor", readout_format=".1f",
                                  style={"description_width": "140px"},
                                  layout=widgets.Layout(width="420px"))
    get_y_limits = lambda: (y_floor.value, Y_LIMITS_DEFAULT[1])

    panels = [TreatmentPanel(FLUOXETINE, get_y_limits), TreatmentPanel(NORTRIPTYLINE, get_y_limits)]

    def redraw_all(*_):
        for p in panels:
            p.update()

    def reset_both(_):
        y_floor.unobserve(redraw_all, names="value")
        y_floor.value = Y_LIMITS_DEFAULT[0]
        y_floor.observe(redraw_all, names="value")
        for p in panels:
            p.reset()

    y_floor.observe(redraw_all, names="value")
    reset_btn = widgets.Button(description="Reset Both", icon="undo", button_style="primary",
                               tooltip="Restore the original Attari et al. values in both panels")
    reset_btn.on_click(reset_both)

    intro = widgets.HTML(
        "<div style='font-size:13px;line-height:1.45;max-width:1100px'>"
        "<b>What this shows.</b> For each arm, the curve is "
        "SMD(r) = mean change / &radic;(SD<sub>pre</sub>&sup2; + SD<sub>post</sub>&sup2; "
        "&minus; 2r&middot;SD<sub>pre</sub>&middot;SD<sub>post</sub>). "
        "It answers: <i>given these pre/post SDs and an assumed pre-post correlation, what change-score "
        "SD and standardized change are mathematically implied?</i> The correlation does not cause the "
        "effect; it only determines how the reported summary statistics translate into a standardized value. "
        "<br><b>Not a between-group effect size.</b> These are <i>within-group</i> standardized changes. "
        "The Fluoxetine-vs-Nortriptyline standardized mean difference is a different statistic and should "
        "not be read off by comparing the two panels.</div>")

    return widgets.VBox([
        intro,
        widgets.HBox([reset_btn, y_floor], layout=widgets.Layout(align_items="center", gap="16px")),
        widgets.HBox([p.widget for p in panels]),
    ])


app = build_app()
display(app)
