import sys
import numpy as np


def pretty_trans_hist(ax, data, bins, **hist_kwargs):
    label = hist_kwargs.pop("label", "")
    alpha = hist_kwargs.pop("alpha", 0.4)
    lw = hist_kwargs.pop("lw", 3)
    n, bins, patches = ax.hist(data, bins, alpha=alpha, **hist_kwargs)
    if "color" in hist_kwargs:
        color = hist_kwargs.pop("color")
    else:
        color = patches[0].get_facecolor()

    return ax.hist(data, bins, lw=lw, histtype="step", color=color, label=label, **hist_kwargs)


def add_histogram_on_axis(fig, ax, data, n_bins=50, fraction=0.2, which="x", hist_kwargs={}, log=False, outside=True, invert=False):
    """
    Creates histograms at the axes of a (scatter) plot to show the distribution of the plotted data.

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        Axes (plot) to add the histograms

    data : array, list
        Data to histogram

    n_bins : int
        Number of bins for the histogram. The histogram edges will be determined by the limits of ax.

    fraction : float
        The fraction of the maximum entry of the histogram

    which : str
        "x" or "y" choose which axis.

    hist_kwargs : dict
        Optional arguments for axes.hist(..)

    """

    if which not in ["x", "y"]:
        sys.exit("Selected unknown axis")

    # get second axis and limits
    if which == "x":
        if outside:
            pos = ax.get_position()
            ax2 = fig.add_axes([pos.x0, pos.y1, 0.99 - pos.x0, 0.99 - pos.y1])
        else:
            # ax2 = ax.twinx()
            # ax2.set_position(ax.get_position())
            pos = ax.get_position()
            dy = (pos.y1 - pos.y0) * fraction

            if invert:
                ax2 = fig.add_axes([pos.x0, pos.y1 - dy, pos.x1 - pos.x0, dy])
            else:
                ax2 = fig.add_axes([pos.x0, pos.y0, pos.x1 - pos.x0, dy])

        low, up = ax.get_xlim()
        hist_kwargs.update({"orientation": "vertical"})

    else:
        if outside:
            pos = ax.get_position()
            ax2 = fig.add_axes([pos.x1, pos.y0, 0.99 - pos.x0, 0.99 - pos.y1])
        else:
            # ax2 = ax.twiny()
            pos = ax.get_position()
            dx = (pos.x1 - pos.x0) * fraction

            if invert:
                ax2 = fig.add_axes([pos.x1 - dx, pos.y0, dx, pos.y1 - pos.y0])
            else:
                ax2 = fig.add_axes([pos.x0, pos.y0, dx, pos.y1 - pos.y0])


        low, up = ax.get_ylim()
        hist_kwargs.update({"orientation": "horizontal"})

    if log:
        bins = np.logspace(np.log10(low), np.log10(up), n_bins)
    else:
        bins = np.linspace(low, up, n_bins)

    hist, edges, patches = pretty_trans_hist(ax2, data, bins, **hist_kwargs)

    # set the same limits as the original axis
    if which == "x":
        ax2.set_xlim(low, up)
    else:
        ax2.set_ylim(low, up)

    if invert:
        if which == "x":
            ax2.set_ylim(ax2.get_ylim()[::-1])
        else:
            ax2.set_xlim(ax2.get_xlim()[::-1])


    # if not outside:
    #     # define which fraction of the plot the histogram should occupy
    #     if invert:
    #         if which == "x":
    #             ax2.set_ylim(ax2.get_ylim()[0] / faction, None)
    #         else:
    #             ax2.set_xlim(ax2.get_xlim()[0] / faction, None)

    #     else:
    #         if which == "x":
    #             ax2.set_ylim(None, ax2.get_ylim()[1] / faction)
    #         else:
    #             ax2.set_xlim(None, ax2.get_xlim()[1] / faction)

    # dont draw any axis
    ax2.set_axis_off()

    return ax2, hist, edges, patches


def draw_residual(ax, ax2, ref=None, grid=True, relative=True, symetric_limits=True, label="residual"):
    """ Draws a residual for data of a given axis (ax) on a second axis (ax2).

    It uses the data in `ax.lines` and `ax.collections` to draw the residual.
    The residual is calculated relative to the first line of the original axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The original axis with the data to calculate the residual from.
    ax2 : matplotlib.axes.Axes
        The axis to draw the residual on.
    grid : bool
        Whether to show a grid on the residual axis.
    relative : bool
        Whether to draw a relative (data / data_ref) or absolute residual (data - data_ref).
    symetric_limits : bool
        Whether to make the limits of the residual axis symmetric around 0 (absolute residual) or 1 (relative residual).
    """

    has_lines = len(ax.lines)  # for "ax.plot"
    has_patches = len(ax.patches)  # for "ax.hist"
    has_collections = len(ax.collections)  # for "ax.fill_between"

    if has_lines:
        if ref is None:
            line_ref = ax.lines[0]
            x_ref = line_ref.get_xdata()
            y_ref = line_ref.get_ydata()

        for idx, line in enumerate(ax.lines):

            if ref is not None:
                y_ref = ref(line.get_xdata())
                x_ref = line.get_xdata()

            if np.allclose(x_ref, line.get_xdata()):
                y = line.get_ydata()
            else:
                y = np.interp(x_ref, line.get_xdata(), line.get_ydata(), left=np.nan, right=np.nan)

            ydata = y / y_ref if relative else y - y_ref
            ax2.plot(x_ref, ydata, lw=line.get_lw(), ls=line.get_ls(), color=line.get_color())

    if has_collections:
        if not has_lines:
            raise ValueError("Cannot calculate residual for collections without lines")

        if ref is None:
            line_ref = ax.lines[0]
            x_ref = line_ref.get_xdata()
            y_ref = line_ref.get_ydata()

        for poly in ax.collections:
            x, y = poly.get_paths()[0].vertices.T

            if ref is not None:
                ref_y_data = ref(x)
            else:
                ref_y_data = np.squeeze([y_ref[x_ref == x_val] for x_val in x])

            ydata = y / ref_y_data if relative else y - ref_y_data

            ax2.plot(x, ydata, alpha=0.5, color=poly.get_facecolor())

    if has_patches:

        if ref is None:
            patch_ref = ax.patches[0]
            x_ref, y_ref = patch_ref.get_path().vertices.T

        for idx, patch in enumerate(ax.patches):
            x, y = patch.get_path().vertices.T

            if ref is not None:
                y_ref = ref(x)
                x_ref = x

            if np.allclose(x_ref, x):
                y = line.get_ydata()
            else:
                y = np.interp(x_ref, x, y, left=np.nan, right=np.nan)

            ydata = y / y_ref if relative else y - y_ref
            ax2.plot(x_ref, ydata, lw=line.get_lw(), ls=line.get_ls(), color=patch.get_edgecolor())

    ax2.grid(grid)
    ax2.set_ylabel(label)

    if symetric_limits:
        # make symmetric limits around 0 (absolute residual) or 1 (relative residual)
        lim = np.abs(np.array(ax2.get_ylim()) - int(relative)).max()
        ax2.set_ylim(-lim + int(relative), lim + int(relative))
