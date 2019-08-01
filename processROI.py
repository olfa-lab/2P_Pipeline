"""Process ROIs
"""
import warnings
from typing import List, Sequence, Union

import h5py
import numpy as np
import pandas as pd


def get_dF_F(timeseries, width=20):
    """Calculate dF/F using local temporal window mean F.

    Arguments:
        timeseries {ndarray} -- Source timeseries. Expects time is first dim.

    Keyword Arguments:
        width {int} -- Frames before and after to include in mean. (default: {20})
    """
    numFrames = timeseries.shape[0]
    meanF = np.zeros_like(timeseries)
    # dF[0:width, :, :] =
    for i_frame, frame in enumerate(timeseries):
        meanF[i_frame] = np.mean(
            timeseries[max(0, i_frame - width) : min(numFrames, i_frame + width)],
            axis=0,
        )
    dF = (timeseries - meanF) / meanF
    dF = np.nan_to_num(dF)
    return dF


def get_trials_metadata(h5Filename: str) -> pd.DataFrame:
    """Retrieve "Trials" metadata from an h5 file.

    Arguments:
        h5Filename {str} -- Source filepath

    Returns:
        pd.DataFrame -- Each DataFrame label corresponds to a h5 dataset.
    """
    trialsMeta = pd.read_hdf(h5Filename, key="Trials")
    return trialsMeta


def get_flatten_trial_data(h5Filename, key, clean=False):
    """Extract data for key from all Trials in h5Filename.

    Arguments:
        h5Filename {str} -- Path to h5 file.

    Returns:
        ndarray -- Flat ndarray of the key data.
    """
    with h5py.File(h5Filename, "r") as h5File:
        # Assumes h5File contains Trial000# keys and one Trials key
        numTrials = len(h5File)
        keyData = np.empty(
            0, dtype=h5File["Trial{:04d}".format(numTrials - 1)][key][0].dtype
        )
        for i_trial in range(1, numTrials):
            trial = h5File["Trial{:04d}".format(i_trial)]
            for data in trial[key]:
                keyData = np.append(keyData, data)
    if clean:
        keyData = clean_data(keyData, key)
    return keyData


def clean_data(data, key):
    """Dispatch for key specific cleaning methods.

    Arguments:
        data {ndarray} -- Source data, a flattened h5 dataset.
        key {str} -- Key describing data

    Returns:
        ndarray -- Cleaned data
    """
    if key == "frame_triggers":
        data = clean_frame_trigger_data(data)
    return data


def analyze_frame_triggers(data):
    """Analyze frame_trigger data for anomalies.

    Arguments:
        data {ndarray} -- Flattened frame trigger dataset.
    """
    print("analyze_frame_triggers NOT implemented.")


def clean_frame_trigger_data(data, frameRate=100 / 3):
    """Correct common frame trigger indexing errors.

    Cleans mislabeled triggers, and imputes missing triggers.

    Arguments:
        data {ndarray} -- Numpy array of timestamps in ms.

    Keyword Arguments:
        frameRate {float} -- Expected distance between adjacent triggers. (default:
            {100/3})

    Returns:
        ndarray -- Cleaned data.
    """
    analyze_frame_triggers(data)
    # np.unique removes duplicates AND sorts
    cleanData = np.unique(data)
    interFrameIntervals = cleanData[1:] - cleanData[:-1]
    largeGapsIndexes = (interFrameIntervals > np.ceil(frameRate)).nonzero()[0]
    filledData = cleanData.copy()
    # Process gaps in reverse order so inserting fillins doesn't change indexing for
    # subsequent gap processing
    for i_gap, gapIndex in enumerate(largeGapsIndexes[::-1]):
        numFillIns = round(interFrameIntervals[gapIndex] / frameRate) - 1
        fillIns = (
            np.round((np.arange(numFillIns) + 1) * frameRate) + cleanData[gapIndex]
        )
        filledData = np.insert(filledData, gapIndex + 1, fillIns)
    assert np.all(
        np.abs((filledData[1:] - filledData[:-1]) - frameRate) < 2
    ), "Gaps 2ms or longer still exist"
    assert (
        np.sum(np.abs((filledData[1:] - filledData[:-1]) - frameRate) > 1) <= 3
    ), "More than three gaps greater than 1ms still exist"
    return filledData


def frame_from_timestamp(
    frameTriggers: Sequence[float], timestamps: Union[float, Sequence[float]]
) -> np.ndarray:
    """Map timestamps to frames

    Arguments:
        frameTriggers {[long]} -- Sequence of frame trigger timestamps.
        timestamps {[long] or long} -- Sequence or scalar of timestamps.

    Returns:
        ndarray -- Sequence of the indexes corresponding to the first timestamp greater
            than or equal to each element in timestamps.
    """
    sorters = np.argsort(frameTriggers, kind="mergesort")
    assert all(frameTriggers == frameTriggers[sorters])
    return np.searchsorted(frameTriggers, timestamps, sorter=sorters)


def upsample(signal, frameTimpstamps, targetFramerate=1.0):
    """Linearly upsample a signal.

    Note: May have difficulty with non-integer targetFramerates in some corner cases.

    Arguments:
        signal {[float]} -- Source signal.
        frameTimpstamps {[int]} -- Timestamps for each sample in signal. Required to
            be in ascending order and have no duplicate values.
        targetFramerate {float} -- Framerate to achieve.

    Returns:
        ndarray -- Upsampled signal.
    """
    targetTimestamps = np.append(
        np.round(np.arange(frameTimpstamps[0], frameTimpstamps[-1], targetFramerate)),
        frameTimpstamps[-1],
    )
    upsampledSignal = np.interp(
        targetTimestamps, frameTimpstamps, signal, right=np.nan, left=np.nan
    )
    assert not np.any(np.isnan(upsampledSignal))
    return upsampledSignal


def downsample(arr: np.ndarray, newShape: Sequence[int]) -> np.ndarray:
    assert len(arr.shape) == len(newShape)
    assert all(
        [(oldDim % newDim) == 0 for oldDim, newDim in zip(arr.shape, newShape)]
    ), "All newShape dimensions must evenly divide arr dimensions"
    shape: List[int] = []
    for i_dim, size in enumerate(newShape):
        shape += [size, arr.shape[i_dim] // newShape[i_dim]]
    return arr.reshape(shape).mean(tuple(range(1, len(newShape) * 2, 2)))


def better_correlation(
    reference: Sequence[float], columnVecs: np.ndarray
) -> np.ndarray:
    """Fast Pearson correlation coefficient between many vectors and one reference vector.

    Note that for two standard scaled vectors x,y (mean == 0, std == 1), the Pearson
    correlation coefficient formula collapses from
        1/N * sum(((x - mean(x)) * (y - mean(y))) / (std(x) * std(y))
    to
        1/N * sum(x * y).
    For a matrix X consisting of m column vectors and column vector y, this is
        1/N * dot(X.transpose, y)
    producing an m-length vector correlation coefficients.

    Arguments:
        reference {Sequence[float]} -- Reference vector.
        columnVecs {np.ndarray} -- Matrix of column vectors.

    Returns:
        np.ndarray -- Vector containing Pearson correlation coefficient for each column in
            columnVecs.
    """
    refCentered = (reference - np.nanmean(reference)) / np.nanstd(reference)
    vectorLength = len(reference)
    assert vectorLength == columnVecs.shape[0], "Column vectors must have same length."
    if columnVecs.ndim > 2:
        columnVecs = columnVecs.reshape(vectorLength, -1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        columnsCentered = (columnVecs - np.nanmean(columnVecs, axis=0)) / np.nanstd(
            columnVecs, axis=0
        )
    return (1 / vectorLength) * np.dot((columnsCentered).T, refCentered)


def pixelwise_correlate(
    pixelsTimeseries: np.ndarray, roiTimeseries: Sequence[float]
) -> np.ndarray:
    """[summary]

    Arguments:
        pixelsTimeseries {np.ndarray} -- Frames must be first dimension.
        roiTimeseries {Sequence[float]} -- Length should equal frames of pixelsTimeseries.

    Returns:
        np.ndarray -- [description]
    """
    # Only consider frames with ROI response
    nonNullFrames = ~np.isnan(roiTimeseries, dtype=np.bool)
    pixelsView = pixelsTimeseries[nonNullFrames]
    roiView = roiTimeseries[nonNullFrames]
    # Replace NAN/inf in the pixel data with -1 (no signal)
    pixelsViewNoNan = pixelsView.copy()
    pixelsViewNoNan[~np.isfinite(pixelsView, dtype=np.bool)] = -1
    correlateScores = better_correlation(roiView, pixelsViewNoNan).reshape(
        pixelsView.shape[1:]
    )
    return correlateScores
