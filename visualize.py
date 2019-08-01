import logging
import sys
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm, trange

import processROI
from ImagingSession import ImagingSession

logger = logging.getLogger("visualize")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)

SignalByCondition = Dict[str, np.ndarray]


def pick_layout(numPlots: int) -> Tuple[int, int]:
    """Define gridsize to fit a number of subplots.

    Arguments:
        numPlots {int} -- Subplots to be arranged.

    Returns:
        Tuple[int, int] -- Subplot grid size.
    """
    numColumns = np.round(np.sqrt(numPlots))
    numRows = np.ceil(numPlots / numColumns)
    return int(numRows), int(numColumns)


def plot_downsampled_trials_by_ROI_per_condition(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    downsampleAxis: int,
    downsampleFactor: int,
    title: str = "",
    supTitle: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_r",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot downsampled trials for each ROI for each condition.

    Average groups of consecutive trials for plotting.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries.
        session {ImagingSession} -- Session specifics.
        downsampleAxis {int} -- Axis of dF_Fs to be downsampled.
        downsampleFactor {int} -- This many trials along downsampleAxis in dF_Fs will be
            averaged together into one trial in the resultant plotted data.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        supTitle {str} -- Figure title. The condition will be suffixed to it. (default:
            {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    dF_FsDownsampled: Dict[str, np.ndarray] = {}
    for condition, dF_F in dF_Fs.items():
        oldShape = dF_F.shape
        newShape = (
            oldShape[:downsampleAxis]
            + (oldShape[downsampleAxis] // downsampleFactor,)
            + oldShape[downsampleAxis + 1 :]
        )
        dF_FsDownsampled[condition] = processROI.downsample(dF_F, newShape)
    figs = plot_trials_by_ROI_per_condition(
        dF_FsDownsampled,
        session,
        title=title,
        supTitle=supTitle,
        figDims=figDims,
        palette=palette,
        maxSubPlots=maxSubPlots,
    )
    for fig in figs:
        for leg in fig.legends:
            leg.remove()
        numLines = sum([line.get_ls() == "-" for line in fig.axes[0].get_lines()])
        fig.legend(list(range(1, numLines + 1)), loc="lower right")
    return figs


def plot_trials_by_ROI_per_condition(
    dF_Fs: Dict[str, np.ndarray],
    session: ImagingSession,
    title: str = "",
    supTitle: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_d",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot all trials for each ROI for each condition.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries
        session {ImagingSession} -- Session specifics.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        supTitle {str} -- Figure title. The condition will be suffixed to it. (default:
            {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    sns.set(rc={"figure.figsize": figDims})
    conditions = tuple(dF_Fs)
    figs = []
    for condition in conditions:
        dF_F = dF_Fs[condition]
        numTrials = dF_F.shape[1]
        sns.set_palette(palette, numTrials)
        numROI = dF_F.shape[2]
        for i_fig in range(int(np.ceil(numROI / maxSubPlots))):
            ROIOffset = maxSubPlots * i_fig
            selectedROIs = list(
                range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI))
            )
            fig = create_ROI_plot(
                {roi: dF_F[:, :, roi] for roi in selectedROIs},
                session.get_lock_offset(),
                session.odorCodesToNames,
                title,
                alpha=0.8,
            )
            fig.legend(
                session.trialGroups[condition], loc="lower right", title="Trials"
            )
            fig.suptitle(supTitle + f" for {condition}")
            figs.append(fig)
    return figs


def plot_conditions_by_ROI(
    dF_Fs: SignalByCondition,
    session: ImagingSession,
    title: str = "",
    figDims: Tuple[int, int] = (10, 9),
    palette: str = "Reds_d",
    maxSubPlots: int = 25,
) -> List[plt.Figure]:
    """Plot trial-mean for each condition on each ROI.

    Arguments:
        dF_Fs {Dict[str, np.ndarray]} -- Dict of condition: timeseries
        session {ImagingSession} -- Session specifics.

    Keyword Arguments:
        title {str} -- Subplot title (default: {""})
        figDims {Tuple[int, int]} -- Size of figures in inches. (default: {(10, 9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_d"})
        maxSubPlots {int} -- Max number of subplots per figure (default: {25})

    Returns:
        List[plt.Figure] -- List of figures produce.
    """
    conditions = tuple(dF_Fs)
    sns.set(rc={"figure.figsize": figDims})
    sns.set_palette(palette, len(conditions))
    flattenTrialsAllConditions = np.stack(
        tuple(np.nanmean(dF_F, axis=1, keepdims=1) for dF_F in dF_Fs.values()), axis=-1
    )
    numROI = flattenTrialsAllConditions.shape[2]
    figs = []
    for i_fig in range(int(np.ceil(numROI / maxSubPlots))):
        ROIOffset = maxSubPlots * i_fig
        selectedROIs = list(range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI)))
        fig = create_ROI_plot(
            {roi: flattenTrialsAllConditions[:, :, roi, :] for roi in selectedROIs},
            session.get_lock_offset(),
            session.odorCodesToNames,
            title,
        )
        fig.legend(conditions, loc="lower right")
        figs.append(fig)
    return figs


def create_ROI_plot(
    # plotDataGenerator's output's first dimension must match the length of frameAxis
    selectedData: Mapping[int, np.ndarray],
    frameAxis: Sequence[int],
    odorNames: Mapping[str, str],
    title: str,
    alpha: float = 0.8,
    **plotKwargs,
) -> plt.Figure:
    """Helper function: creates figure with subplots for each selected ROI.

    Arguments:
        selectedData {Mapping[int, np.ndarray]} -- Mapping from ROI ID to its plot data.
        frameAxis {Sequence[int]} -- Frame index (plot x-axis labels).
        odorNames {Mapping[str, str]} -- Mapping from odor code in condition names to full
            odor name.
        title {str} -- Subplot title. Appears after ROI #.

    Keyword Arguments:
        alpha {float} -- Transparency of plotted lines. (default: {0.8})
        **plotKwargs -- Any additional keyword arguments are passed on to plt.plot.

    Returns:
        plt.Figure -- Resultant figure.
    """
    numROI = len(selectedData)
    numPlots = numROI
    layout = pick_layout(numPlots)
    fig, axarr = plt.subplots(
        layout[0], layout[1], sharex=True, sharey=True, squeeze=False
    )
    for i_ROI, (roi, plotData) in enumerate(selectedData.items()):
        plotLocation = np.unravel_index(i_ROI, layout)
        axarr[plotLocation].title.set_text("ROI #" + str(roi) + ", " + title)
        plot_dF_F_timeseries(
            axarr[plotLocation],
            frameAxis,
            np.squeeze(plotData),
            alpha=alpha,
            **plotKwargs,
        )
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_ROI < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
    subtitle("Odor key: " + f"{odorNames}".replace("'", ""))
    return fig


def plot_dF_F_timeseries(
    ax: plt.Axes, frameData: Sequence[int], plotData: np.ndarray, **kwargs
) -> None:
    """Plot dF/F timeseries with consistent format.

    Arguments:
        ax {plt.Axes} -- Axes to plot onto.
        frameData {Sequence[int]} -- X-axis data.
        plotData {np.ndarray} -- Y-axis data.

    Keyword Arguments:
        **plotKwargs -- Any keyword arguments are passed on to plt.plot.
    """
    ax.set_xlabel("Frame")
    ax.set_ylabel("dF/F")
    ax.plot(frameData, np.squeeze(plotData), **kwargs)
    ax.plot([frameData[0], frameData[-1]], [0, 0], "--k", zorder=-1)


def visualize_correlation(
    correlationsByROI: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    odorNames: Mapping[str, str],
    savePath: str = None,
    title: str = "",
    figDims: Tuple[int, int] = (10, 9),
) -> List[plt.Figure]:
    sns.set(rc={"figure.figsize": figDims})
    avgRoiSize = np.median(
        [
            np.sum(mask) / np.prod(correlationsByROI[i_mask].shape)
            for i_mask, mask in enumerate(masks)
        ]
    )
    gridSizes = [x ** 2 for x in range(1, 6)]
    maxSubPlots = max(
        filter(
            lambda gridSize: (100 * np.prod(figDims) / 2 / gridSize * avgRoiSize) > 0.5,
            gridSizes,
        )
    )
    numROI = len(correlationsByROI)
    assert len(masks) == numROI
    figs = []
    for i_fig in trange(int(np.ceil(numROI / maxSubPlots))):
        ROIOffset = maxSubPlots * i_fig
        selectedROIs = list(range(0 + ROIOffset, min(maxSubPlots + ROIOffset, numROI)))
        fig = plot_correlations_by_ROI(
            correlationsByROI, masks, odorNames, title, selectedROIs
        )
        figs.append(fig)
    if savePath:
        # with PdfPages(savePath + ".pdf") as pp:
        #     for fig in figs:
        #         pp.savefig(fig)
        for i_fig, fig in enumerate(figs):
            fig.savefig(savePath + " " + str(i_fig) + ".png")
            plt.close(fig)
        logger.info(f"{str(i_fig)} figures saved to {savePath}.")
    return figs


def plot_correlations_by_ROI(
    correlationsByROI: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    odorNames: Mapping[str, str],
    suptitle: str,
    selectedROIs: Sequence[int],
    clipCorrelation: float = 1.0,  # value to clip heatmap colorbar
    colormap=sns.diverging_palette(255, 0, sep=round(0.2 * 256), as_cmap=True),
) -> plt.Figure:
    numPlots = len(selectedROIs)
    layout = pick_layout(numPlots)
    fig, axarr = plt.subplots(
        layout[0], layout[1], sharex=True, sharey=True, squeeze=False
    )
    colorbar = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    for i_ROI, roi in enumerate(selectedROIs):
        plotLocation = np.unravel_index(i_ROI, layout)
        axarr[plotLocation].title.set_text("ROI #" + str(roi))
        plotData = correlationsByROI[roi]
        sns.heatmap(
            plotData,
            ax=axarr[plotLocation],
            cmap=colormap,
            vmin=-clipCorrelation,
            vmax=clipCorrelation,
            center=0,
            xticklabels=100,
            yticklabels=200,
            cbar=i_ROI == 0,
            cbar_ax=None if i_ROI else colorbar,
        )
        # Keep y-ticklabels horizontal
        axarr[plotLocation].set_yticklabels(
            axarr[plotLocation].get_yticklabels(), rotation=0
        )
        # Draw outline of ROI for reference
        axarr[plotLocation].contour(
            masks[roi], colors="black", alpha=0.7, linestyle="dashed", linewidths=0.3
        )
    fig.suptitle(suptitle)
    subtitle("Odor key: " + f"{odorNames}".replace("'", ""))
    return fig


def visualize_conditions(
    dF_Fs: SignalByCondition,
    session: ImagingSession,
    axis: Union[int, Sequence[int]],
    title="",
    figDims=(10, 9),
    palette="Reds_r",
    sharey=True,
    showLegend=True,
    **kwargs,
) -> plt.Figure:
    """Generate plots of dF/F traces by condition.

    Each condition is plotted in its own subplot. All subplots in one figure.

    Arguments:
        dF_Fs {{str: ndarray}} -- Condition: dF/F trace. Typical trace might be frames by
            trials by ROI, but only constraint is that it play nicely with whatever is
            passed as axis argument.
        session {ImageSession} -- Imaging session metadata.
        axis {int or tuple of ints} -- The axes of each dF/F to sum over.

    Keyword Arguments:
        title {str} -- Printed above each subplot, along with that subplot's condition.
            (default: {""})
        figDims {tuple} -- Dimensions of the figure bounding all subplots (default: {(10,
            9)})
        palette {str} -- Seaborn palette for plot lines. (default: {"Reds_r"})

    Returns:
        matplotlib.figure.Figure -- The generated figure.
    """
    numConditions = len(dF_Fs)
    numPlots = numConditions
    layout = pick_layout(numPlots)
    lockOffset = session.get_lock_offset()
    sns.set(rc={"figure.figsize": figDims})
    numLines = max(np.mean(dF_Fs[list(dF_Fs)[0]], axis=axis, keepdims=True).shape[1:])
    sns.set_palette(palette, numLines)
    fig, axarr = plt.subplots(
        layout[0], layout[1], sharex=True, sharey=sharey, squeeze=False
    )
    i_condition = -1
    for condition, dF_f in dF_Fs.items():
        i_condition += 1
        plotLocation = np.unravel_index(i_condition, layout)
        with warnings.catch_warnings():
            # TODO: Catch and log numpy all-NAN warnings, instead of ignore
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            plotData = np.nanmean(dF_Fs[condition], axis=axis, keepdims=True)
        axarr[plotLocation].title.set_text(condition + ", " + title)
        plot_dF_F_timeseries(
            axarr[plotLocation], lockOffset, np.squeeze(plotData), **kwargs
        )
        # Keep subplot axis labels only for edge plots; minimize figure clutter
        if plotLocation[1] > 0:
            axarr[plotLocation].set_ylabel("")
        if i_condition < (numPlots - layout[1]):
            axarr[plotLocation].set_xlabel("")
        if ((numLines > 1) & (numLines < 10)) and showLegend:
            # This assigns the conditions to the legend, even tho this if statement only
            # indirectly confirms this is appropriate.
            axarr[plotLocation].legend(
                session.trialGroups[condition], ncol=2, fontsize="xx-small"
            )
    subtitle(
        "Legends indicate trial numbers. Odor key: "
        + f"{session.odorCodesToNames}".replace("'", "")
    )
    return fig


def subtitle(text: str) -> None:
    """Create a subtitle on the active plt.Figure.

        Position selected for a plt.subplots figure with Seaborn settings.

    Arguments:
        text {str} -- Subtitle text.
    """
    plt.figtext(
        0.5, 0.93, text, style="italic", fontsize="small", horizontalalignment="center"
    )


def get_best_ROI(dF_F: np.ndarray, numReturned: int) -> np.ndarray:
    """Return bool mask corresponding to highest signal ROIs.

    Means across trials, then sums absolute value of each frame to evaluate.

    Args:
        dF_F (np.ndarray): Frames by trials by ROI.
        numReturned (int, optional): Number of ROI to return. Defaults to 10.

    Returns:
        np.ndarray: Bool mask with shape = dF_F.shape[2:].
    """
    strengths = np.nansum(np.abs(np.nanmean(dF_F, axis=1)), axis=0)
    sorts = np.argsort(strengths, axis=None)
    numNan = np.sum(np.isnan(strengths))
    if numNan:
        sorts = sorts[:-numNan]
    bestROI = sorts[-numReturned:]
    roiShape = strengths.shape
    bestMask = np.full(roiShape, False, dtype=np.bool)
    bestMask[np.unravel_index(bestROI, roiShape)] = True
    return bestMask


def plot_allROI_best_performers(
    dF_Fs: SignalByCondition, session: ImagingSession, roiPerPlot: int = 5
) -> List[plt.Figure]:
    figs = []
    for condition, dF_F in tqdm(
        dF_Fs.items(), desc="All ROI, best performers", unit="fig"
    ):
        title = f"Top {roiPerPlot}"
        bestROI = get_best_ROI(dF_F, numReturned=roiPerPlot)
        bestOnlyDF_Fs = {
            condition: dF_F[:, :, bestROI] for condition, dF_F in dF_Fs.items()
        }
        fig = visualize_conditions(
            bestOnlyDF_Fs,
            session,
            axis=1,
            title=title,
            palette="Spectral",
            sharey=True,
            showLegend=False,
            alpha=0.9,
        )
        if bestROI.ndim == 1:
            labels = np.where(bestROI)[0]
        else:
            labels = zip(*np.where(bestROI))
        fig.legend(labels=labels, loc="lower right", title="ROI#")
        fig.suptitle(f"Most Active {roiPerPlot} ROI for {condition}")
        figs.append(fig)
    return figs


def save_figs(
    figs: Sequence[plt.Figure],
    title: str,
    saveTo: str,
    figFileType: str,
    unit: str = "figure",
    setSuptitle: bool = False,
) -> None:
    padding = 2 if (len(figs) > 9) else 1
    for i_fig, fig in enumerate(tqdm(figs, unit=unit)):
        if setSuptitle:
            fig.suptitle(title)
        figID = ("{:0" + str(padding) + "d}").format(i_fig)
        figFname = "_".join((saveTo, title.replace(" ", "_"), f"{figID}.{figFileType}"))
        fig.savefig(figFname)
        logger.debug(f"Saved {figFname}.")
        plt.close(fig)
