"""Split a tiff stack by condition trials.
"""
import logging
import sys
from typing import Dict, FrozenSet, List

import numpy as np
from skimage import io

import processROI
from ImagingSession import ImagingSession
from TiffStack import TiffStack

logger = logging.getLogger("conditionSplit")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)


def find_odor_combos(session: ImagingSession) -> Dict[FrozenSet[str], List[str]]:
    """Find all maximal groupings of odors in session and their specific conditions.

    Args:
        session (ImagingSession): Reference imaging session.

    Returns:
        Dict[FrozenSet[str], List[str]]: Maps odor combo to conditions.
    """
    odorArrangements: Dict[FrozenSet[str], List[str]] = {}
    for condition in session.conditions:
        odors = frozenset(ImagingSession.odors_in_condition(condition))
        # Join any existing combos these odors are a subset of
        memberOfArrangements = [combo for combo in odorArrangements if odors <= combo]
        for combo in memberOfArrangements:
            odorArrangements[combo].append(condition)
        if not memberOfArrangements:
            odorArrangements[odors] = [condition]
            # Absorb any proper subsets already in odorArrangements
            subsetArrangements = [combo for combo in odorArrangements if combo < odors]
            for combo in subsetArrangements:
                odorArrangements[odors] += odorArrangements[combo]
                del odorArrangements[combo]
    logger.debug(f"Found odor arrangements: {odorArrangements}")
    return odorArrangements


def split_by_odor_combos(stack: TiffStack, session: ImagingSession) -> None:
    """Splits stack into multiple tiffs according to odor groupings, and saves them to disk.

    Args:
        stack (TiffStack): Source tiff stack.
        session (ImagingSession): Reference condition, trial, and frame timestamp data.
    """
    odorArrangements = find_odor_combos(session)
    for combo in odorArrangements:
        trials: List[int] = []
        for condition in odorArrangements[combo]:
            trials += session.trialGroups[condition]
        trialStartTimes = session._sessionData["fvOnTime"][trials]
        trialStartFrames = processROI.frame_from_timestamp(
            session.frameTimestamps, trialStartTimes
        )
        nextTrials = [trialNum + 1 for trialNum in trials]
        if len(session._sessionData) in nextTrials:
            # Can't use the "next" trial start time as the end time for the last trial
            lastTrialIndex = nextTrials.index(len(session._sessionData))
            nextTrials.remove(len(session._sessionData))
        trialEndTimes = session._sessionData["fvOnTime"][nextTrials]
        trialEndFrames = processROI.frame_from_timestamp(
            session.frameTimestamps, trialEndTimes
        )
        if len(trialEndFrames) < len(trials):
            # Set end frame for last trial as the last frame
            trialEndFrames = np.insert(
                trialEndFrames, lastTrialIndex, len(session.frameTimestamps)
            )
        comboFrameIndexes: List[int] = []
        for startFrame, endFrame in zip(trialStartFrames, trialEndFrames):
            comboFrameIndexes += list(range(startFrame, endFrame))
        saveFilename = f"{stack.filenamePrefix}-Odors_{'_'.join(combo)}.tif"
        logger.info(
            f"Saving combo tif sized {stack.timeseries[comboFrameIndexes].shape} "
            + f"to {saveFilename}"
        )
        io.imsave(
            saveFilename, stack.timeseries[comboFrameIndexes], check_contrast=False
        )
    logging.info(f"Odor codes: {session.odorCodesToNames}")


# Quick and dirty way to use: run from command line with first argument the H5 and the
# remaining the tiffs, in order.
if __name__ == "__main__":
    from ImagingSession import H5Session

    stack = TiffStack(sys.argv[2:])
    session = H5Session(sys.argv[1])
    logger.debug(stack.filenamePrefix)
    split_by_odor_combos(stack, session)
