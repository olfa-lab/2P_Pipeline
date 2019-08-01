#!/usr/bin/env python3
"""Analyze a 2P imaging session from tiff stack, h5 metadata, and ROI masks.
"""
import argparse
import logging
import os
import sys
import time
import warnings
from glob import glob
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import bp_motioncorrection
import processROI
import visualize
from ImagingSession import H5Session, ImagingSession
from SlurmScript import SlurmScript
from TiffStack import TiffStack

logger = logging.getLogger("analyzeSession")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)

SignalByCondition = Dict[str, np.ndarray]


def process_dF_Fs(timeseries: np.ndarray, session: ImagingSession) -> SignalByCondition:
    """Produce dF/F data for given signal and session details.

    The signals are broken into conditions and timelocked to them according to session.

    Args:
        timeseries (np.ndarray): Timeframes by trial/ROI/etc.
        session (ImagingSession): Defines what time windows belong to what conditions.

    Returns:
        Dict[str, np.ndarray]: condition: dF/F. dF/F is frames by trials by ROI.
    """
    meanFs = session.get_meanFs(timeseries, frameWindow=2)
    dF_Fs = {}
    for condition in meanFs:
        dF_Fs[condition] = session.get_trial_average_data(
            timeseries, meanFs[condition], condition
        )
    return dF_Fs


def process_and_viz_correlations(
    roiDF_Fs: SignalByCondition,
    roiMasks: Sequence[np.ndarray],
    corrStack: TiffStack,
    corrSession: ImagingSession,
    savePath: str,
    window: slice = None,
):
    """Process correlation tracing imaging session.

    Arguments:
        roiDF_Fs {Dict[str, ndarray]} -- Condition: dF/F trace.
        roiMasks {Sequence[np.ndarray]} -- Boorlean masks for the dF/F traces
            corresponding to each ROI.
        corrStack {TiffStack} -- TiffStack to correlate against each ROI.
        corrSession {ImagingSession} -- Framing details for the session to correlate.
        savePath {str} -- Path at which to save figures.

    Keyword Arguments:
        window {slice} -- Correlation will only apply within given window. Default uses
            full window as defined by corrSession. (default: {None})
    """
    pixelMeanFs = corrSession.get_meanFs(corrStack.timeseries)
    assert list(roiDF_Fs) == list(pixelMeanFs), "ref and target conditions do not match"
    for condition in tqdm(pixelMeanFs, unit="condition"):
        pixeldF_Fs = corrSession.get_trial_average_data(
            corrStack.timeseries, pixelMeanFs[condition], condition
        )
        correlationsByROI = []
        numROI = roiDF_Fs[condition].shape[2]
        for i_roi in trange(numROI, desc=f"{condition}", unit="ROI"):
            # Average across trials
            roiTimeseries = np.nanmean(roiDF_Fs[condition][:, :, i_roi], axis=1)
            with warnings.catch_warnings():
                # TODO: Catch and log numpy all-NAN warnings, instead of ignore
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pixelsTimeseries = np.nanmean(pixeldF_Fs, axis=1)
            if np.isnan(pixelsTimeseries).all():
                logger.warning(
                    f"During session {corrSession.title}, ROI#{i_roi}"
                    + ", pixelsTimeseries was *all* NAN."
                )
            if window:
                assert len(roiTimeseries) == 60
                roiTimeseries = roiTimeseries[window]
                assert pixelsTimeseries.shape[0] == 60
                pixelsTimeseries = pixelsTimeseries[window]
            correlationsByROI.append(
                processROI.pixelwise_correlate(pixelsTimeseries, roiTimeseries)
            )
        title = "stack" + str(corrSession.title) + "_" + condition
        visualize.visualize_correlation(
            correlationsByROI,
            roiMasks,
            corrSession.odorCodesToNames,
            title=title,
            savePath=os.path.join(savePath, title),
        )


def launch_mc(**params):
    bp_motioncorrection.main(params)


def launch_traces(
    h5Filename: str,
    tiffFilenames: Sequence[str],
    maskFilenames: Sequence[str],
    saveDir: str,
    savePrefix: str,
    figFileType: str = "png",
    **kwargs,
) -> None:
    refStack = TiffStack(tiffFilenames, maskFilenames=maskFilenames)
    roiAverages = refStack.cut_to_averages()
    refSession = H5Session(h5Filename)
    dF_Fs = process_dF_Fs(roiAverages, refSession)
    os.makedirs(saveDir, exist_ok=True)
    saveTo = os.path.join(saveDir, savePrefix.replace(" ", "_"))

    figSettings: List[Tuple[Union[int, Sequence[int]], str]] = []
    if kwargs["allROI"] or kwargs["allFigs"]:
        numROI = roiAverages.shape[1]
        figSettings.append((1, "All ROI"))
        if numROI >= 10:
            figs = visualize.plot_allROI_best_performers(dF_Fs, refSession)
            visualize.save_figs(figs, "allROI, Best Performers", saveTo, figFileType)
    if kwargs["replicants"] or kwargs["allFigs"]:
        # TODO: Update subplot figure titles to properly reflect the number of trials
        # Currently assumes the first condition's count is correct for all
        figSettings.append((2, str(list(dF_Fs.values())[0].shape[1]) + " Replicants"))
    if kwargs["crossMean"] or kwargs["allFigs"]:
        figSettings.append(((1, 2), "Cross-trial Mean"))
    for axis, title in tqdm(figSettings, unit="meanFig"):
        fig = visualize.visualize_conditions(dF_Fs, refSession, axis=axis, title=title)
        fig.suptitle(title)
        figFname = saveTo + title.replace(" ", "_") + "." + figFileType
        fig.savefig(figFname)
        logger.debug(f"Saved {figFname}.")
        plt.close(fig)
    if figSettings:
        logger.info("Mean trace figures done.")

    # TODO: Somehow unify this all into same single figSettings loop
    if kwargs["condsROI"] or kwargs["allFigs"]:
        figs = visualize.plot_conditions_by_ROI(dF_Fs, refSession)
        title = "Conditions by ROI"
        visualize.save_figs(
            figs, title, saveTo, figFileType, unit="condROIFig", setSuptitle=True
        )
        logger.info(f"Conditions by ROI figures done. {len(figs)} figures produced.")

    if kwargs["ROIconds"] or kwargs["allFigs"]:
        title = "All ROI by Condition"
        figs = visualize.plot_trials_by_ROI_per_condition(
            dF_Fs, refSession, supTitle=title, maxSubPlots=16
        )
        visualize.save_figs(figs, title, saveTo, figFileType, unit="ROIcondsFig")
        logger.info(f"ROIs per condition figures done. {len(figs)} figures produced.")


def launch_correlation(
    h5Filename: str,
    tiffFilenames: Sequence[str],
    maskFilenames: Sequence[str],
    saveDir: str,
    savePrefix: str,
    corrPatternsFile: str,
    corrH5sFile: str,
    squashConditions: bool = False,
    **kwargs,
) -> None:
    refSession = H5Session(h5Filename, unified=squashConditions)
    refStack = TiffStack(tiffFilenames, maskFilenames=maskFilenames)
    refDF_Fs = process_dF_Fs(refStack.cut_to_averages(), refSession)
    saveTo = os.path.join(saveDir, savePrefix)
    os.makedirs(saveTo, exist_ok=True)
    corrH5s = read_h5s_file(corrH5sFile)
    corrPatterns = read_stack_patterns_file(corrPatternsFile)
    logger.debug(f"Starting correlation analysis. Ref: {refStack._tiffFilenames[0]}")
    for i_stack, corrStackPattern in enumerate(tqdm(corrPatterns, unit="correlation")):
        logger.debug(f"Loading metadata from {corrH5s[i_stack]}")
        corrSession = H5Session(
            corrH5s[i_stack], title=str(i_stack), unified=squashConditions
        )
        corrStack = TiffStack(sorted(glob(corrStackPattern)))
        process_and_viz_correlations(
            refDF_Fs, refStack.masks, corrStack, corrSession, saveTo
        )


def read_h5s_file(filename: str) -> List[str]:
    return read_stack_patterns_file(filename)


def read_stack_patterns_file(filename: str) -> List[str]:
    patterns: List[str] = []
    with open(filename, mode="r") as patternsFile:
        for line in patternsFile:
            patterns.append(line)
    return patterns


def run_on_cluster(argList: List[str], clusterSetting: Union[str, bool]):
    argList = ["python3"] + argList
    if isinstance(clusterSetting, str):
        # A str clusterSetting means an argument was given to the cluster flag. Remove
        argList.pop(argList.index("--cluster") + 1)
        if clusterSetting.lower() == "pipenv":
            argList = ["pipenv", "run"] + argList
    argList.remove("--cluster")
    clusterCommand = " ".join(argList)
    logger.debug(f"Cluster command: {clusterCommand}")
    payload = """\
        cd {workingDir}

        {clusterCommand}
        """
    script = SlurmScript(
        "submit_analyzeSession_TEMP.sh",
        payload,
        "analyzeSession",
        workingDir=os.getcwd(),
        clusterCommand=clusterCommand,
    )
    result = script.run()
    logger.debug(result)


def get_common_parser():
    commonParser = argparse.ArgumentParser(add_help=False)
    commonParser.add_argument(
        "--cluster",
        nargs="?",
        const=True,
        default=False,
        metavar="env_type",
        help="Execute in default envirnment via submission script to the cluster."
        + "  Use --cluster pipenv to execute in a pipenv environment.",
    )
    commonParser.add_argument(
        "--h5",
        dest="h5Filename",
        required=True,
        help="H5 file containing reference session metadata.",
    )
    commonParser.add_argument(
        "-T",
        "--tiffs",
        dest="tiffFilenames",
        nargs="+",
        required=True,
        metavar="tiffile",
        help="List of tiff filenames, in order, that form reference stack.",
    )
    commonParser.add_argument(
        "-M",
        "--masks",
        dest="maskFilenames",
        nargs="+",
        required=True,
        metavar="maskfile",
        help="List of ROI mask .bmp files.",
    )
    commonParser.add_argument("--saveDir", default="figures/")
    commonParser.add_argument(
        "--savePrefix", default="", help="Figure filename prefix."
    )
    commonParser.add_argument(
        "--includeEmpty",
        action="store_true",
        help="Include blank trials as their own condition.",
    )
    commonParser.add_argument(
        "--preWindow",
        type=int,
        default=500,
        metavar="length",
        help="Size of pre-timelock plotting window in milliseconds. Default: 500",
    )
    commonParser.add_argument(
        "--postWindow",
        type=int,
        default=1500,
        metavar="length",
        help="Size of post-timelock plotting window in milliseconds. Default: 1500",
    )
    return commonParser


if __name__ == "__main__":
    startTime = time.time()
    commonParser = get_common_parser()
    parser = argparse.ArgumentParser(
        description="2P Imaging Processing & Analysis Pipeline"
    )
    subparsers = parser.add_subparsers()

    # mcParser = subparsers.add_parser(
    #     "mc",
    #     parents=[bp_motioncorrection.get_mc_parser()],
    #     help="Motion correction for a tiff stack.",
    #     description="Correct for motion artifacts in a tiff stack.",
    # )
    # mcParser.set_defaults(func=launch_mc)

    tracesParser = subparsers.add_parser(
        "traces",
        parents=[commonParser],
        help="Produce trace plots.",
        description="Produces selected trace plots.",
    )
    tracesParser.set_defaults(func=launch_traces)
    vizTracersGroup = tracesParser.add_argument_group("visualization")
    vizTracersGroup.add_argument(
        "--allROI", action="store_true", help="Produce All ROI plot."
    )
    vizTracersGroup.add_argument(
        "--replicants", action="store_true", help="Produce Replicants plot."
    )
    vizTracersGroup.add_argument(
        "--crossMean", action="store_true", help="Produce Cross-trial Mean plot."
    )
    vizTracersGroup.add_argument(
        "--condsROI", action="store_true", help="Produce Conditions by ROI plots."
    )
    vizTracersGroup.add_argument(
        "--ROIconds", action="store_true", help="Produce ROI per Conditions plots."
    )
    vizTracersGroup.add_argument(
        "--all", "-A", action="store_true", dest="allFigs", help="Produce all plots."
    )

    correlationParser = subparsers.add_parser(
        "correlation",
        parents=[commonParser],
        help="Produce correlation maps between tiff stacks.",
        description="Produces correlation maps between tiff stacks.",
    )
    correlationParser.add_argument(
        "--corrPatternsFile",
        help="File cantaining filepath patterns (i.e. /path/to/Run0034Ref_00*.tif) for "
        + "each tiff stack to correlate, one per line. Order must match corrH5sFile.",
    )
    correlationParser.add_argument(
        "--corrH5sFile",
        help="File cantaining filepaths for each H5, one per line. Order must match "
        + "corrPatternsFile.",
    )
    correlationParser.set_defaults(func=launch_correlation)

    args = parser.parse_args()
    logger.debug(args)
    if hasattr(args, "cluster") and args.cluster:
        run_on_cluster(sys.argv, args.cluster)
    else:
        try:
            if args.includeEmpty:
                ImagingSession.IGNORE_ODORS.remove("empty")
        except ValueError as er:
            logger.debug(
                f"Remove failed: 'empty' not in ImagingSession.IGNORE_ODORS.\n {er}"
            )
        ImagingSession.preWindowSize = args.preWindow
        ImagingSession.postWindowSize = args.postWindow

        args.func(**vars(args))
    logger.info(f"Total run time: {time.time() - startTime:.2f} sec")
