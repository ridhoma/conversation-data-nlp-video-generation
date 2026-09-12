"""
Reusable plotting helpers for the analysis notebooks.

Convention for every function here:
    - pass `ax=` to draw onto an existing Axes (for subplot grids);
    - omit `ax` and the function creates its own figure.
    Either way it returns `(fig, ax)`.

    from data_science.visualisation import plot_histogram
    # or
    from data_science import visualisation as vis

    fig, ax = plot_histogram(df["turn_count"], bins=40, title="Turns per conversation")
    # or onto a grid:
    fig, axes = plt.subplots(1, 2)
    plot_histogram(a, ax=axes[0]); plot_histogram(b, ax=axes[1])
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# a small, coherent palette so plots look consistent across the deck
_C_HIST = "#4C72B0"     # bars — seaborn "deep" blue
_C_MEAN = "#C44E52"     # mean — muted red
_C_MEDIAN = "#2F2F2F"   # median — near-black
_C_INTERVAL = "#8A8A8A" # 95% interval — grey


def _fmt(x: float) -> str:
    """Compact, readable number formatting for labels/stats boxes."""
    if not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000 or (ax != 0 and ax == int(ax)):
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def plot_histogram(
    data,
    *,
    bins: int = 30,
    clip: tuple[float, float] | None = None,
    density: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    color: str = _C_HIST,
    show_stats: bool = True,
    figsize: tuple[float, float] = (8, 5),
    hist_kws: dict | None = None,
):
    """
    Histogram with summary-statistic overlays (mean, median, central 95% interval)
    drawn as vertical lines plus a stats textbox.

    Parameters
    ----------
    data : array-like
        1-D values. Non-finite entries (NaN/inf) are dropped.
    bins : int
        Number of histogram bins.
    clip : (low, high), optional
        If given, values are `np.clip`-ed to [low, high] before plotting AND
        before stats — clipped values pile up at the bounds. Use to tame long
        tails for readability; note it shifts the mean.
    density : bool
        If True, y-axis is a density (area integrates to 1); else raw counts.
    ax : matplotlib Axes, optional
        Draw onto this Axes. If omitted, a new figure is created.
    title, xlabel : str, optional
        Axes title / x-label.
    color : str
        Bar colour.
    show_stats : bool
        Draw the summary-statistics textbox.
    figsize : (w, h)
        Figure size, only used when `ax` is None.
    hist_kws : dict, optional
        Extra keyword args forwarded to `seaborn.histplot`.

    Returns
    -------
    (fig, ax)
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("plot_histogram: no finite values to plot")
    if clip is not None:
        arr = np.clip(arr, clip[0], clip[1])

    # summary statistics (on the array as plotted)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr))
    p_lo, p_hi = (float(v) for v in np.percentile(arr, [2.5, 97.5]))

    created = ax is None
    if created:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # histogram
    kws = dict(stat="density" if density else "count", color=color,
               edgecolor="white", linewidth=0.7, alpha=0.85)
    if hist_kws:
        kws.update(hist_kws)
    sns.histplot(x=arr, bins=bins, ax=ax, **kws)

    # shaded central-95% band + summary lines
    ax.axvspan(p_lo, p_hi, color=color, alpha=0.07, zorder=0)
    ax.axvline(mean, color=_C_MEAN, ls="--", lw=2, label=f"mean={_fmt(mean)}")
    ax.axvline(median, color=_C_MEDIAN, ls="-", lw=2, label=f"median={_fmt(median)}")
    ax.axvline(p_lo, color=_C_INTERVAL, ls=":", lw=1.6, label=f"95% interval=[{_fmt(p_lo)}, {_fmt(p_hi)}]")
    ax.axvline(p_hi, color=_C_INTERVAL, ls=":", lw=1.6)

    # labels & aesthetics
    ax.set_ylabel("Density" if density else "Count", fontsize=11)
    xl = xlabel if xlabel is not None else "value"
    if clip is not None:
        xl += f"  (clipped to [{_fmt(clip[0])}, {_fmt(clip[1])}])"
    ax.set_xlabel(xl, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, fancybox=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    return fig, ax


def plot_feature_effects(labels, values, *, ax=None, title=None, xlabel="effect",
                         center=None, xlim=None, figsize=(8, 6)):
    """
    Horizontal bar chart of per-feature effects — logistic-regression coefficients /
    odds ratios or tree feature importances. Bars are sorted by value; when `center`
    is given (e.g. 0 for coefficients, 1 for odds ratios) bars above/below it are
    coloured differently to show direction.

    Returns (fig, ax).
    """
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    created = ax is None
    if created:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if center is None:
        colors = _C_HIST
    else:
        colors = [(_C_MEAN if v < center else "#55A868") for v in values]
    ax.barh(range(len(values)), values - (center or 0), left=(center or 0),
            color=colors, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    if center is not None:
        ax.axvline(center, color="#2F2F2F", lw=1)
    ax.set_xlabel(xlabel, fontsize=11)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    return fig, ax


def plot_lines(x, y, labels, *, ax=None, title=None, xlabel=None, ylabel=None,
               ylim=None, marker=None, figsize=(10, 5), palette="deep"):
    """
    Multi-line chart — one line per series in `ys`. Each legend entry is annotated
    with that series' mean value, e.g. "created_to_generated  (0.68)".

    Parameters
    ----------
    x       : shared x-axis values (e.g. a time index).
    y      : list of 1-D array-likes (one per line).
    labels  : list of names, same length/order as `y`.
    ax      : draw onto this Axes; if omitted a new figure is created.
    ylim    : optional (low, high), e.g. (0, 1) for rates.
    marker  : optional matplotlib marker (e.g. "o").

    Returns (fig, ax).
    """
    if len(y) != len(labels):
        raise ValueError("plot_lines: `y` and `labels` must be the same length")

    created = ax is None
    if created:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = sns.color_palette(palette, n_colors=len(y))
    for y, label, c in zip(y, labels, colors):
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, label=f"{label}  ({np.nanmean(y):.2f})", color=c, lw=2, marker=marker)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, fancybox=True)
    sns.despine(ax=ax)
    return fig, ax


def plot_barh(labels, values, *, ax=None, title=None, xlabel=None, ylabel=None,
              sort=True, fmt="{:.2f}", color=_C_HIST, figsize=(8, 6)):
    """
    Horizontal bar chart — category on the y-axis, value on the x-axis, with each
    value printed at the end of its bar.

    Parameters
    ----------
    labels : category names (y-axis).
    values : bar values (x-axis).
    sort   : if True (default), sort bars by value (largest at the top).
    fmt    : format for the end-of-bar label, e.g. "{:.2f}", "{:.0%}", "{:,.0f}".
    ax     : draw onto this Axes; if omitted a new figure is created.

    Returns (fig, ax).
    """
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    if sort:
        order = np.argsort(values)              # ascending -> largest bar ends on top
        labels = [labels[i] for i in order]
        values = values[order]

    created = ax is None
    if created:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.barh(range(len(values)), values, color=color,
                   edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.margins(x=0.15)                          # headroom so end-of-bar labels don't clip

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    return fig, ax
