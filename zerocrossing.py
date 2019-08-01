"""Find all zero crossings from voltage traces.
"""
import os

import h5py
import numpy as np


def findCrossings(sequence, window=20, minWidth=5):
    """Return mask of sequence where series of window positive values is followed by
        window negative values.

    Arguments:
        sequence {[numerical]} -- Sequence of values.

    Keyword Arguments:
        window {int} -- Length of positive then negative streaks to test. (default: {20})
        minWidth {int} -- Minimum streak length to test True (corner case handling).
            Only meaningful if less than window. (default: {5})

    Returns:
        [bool] -- Boolean mask of sequence.
    """
    assert window > 0
    minWidth = max(min(minWidth, window), 1)
    sequence = np.array(sequence)
    # Preload with Falses
    posWindow = np.zeros((len(sequence),), dtype=bool)
    negWindow = np.zeros((len(sequence),), dtype=bool)
    for i in range(minWidth, len(sequence) - minWidth):
        posWindow[i] = all(sequence[max(0, i - window) : i] >= 0)
        negWindow[i] = all(sequence[i : min(len(sequence), i + window)] <= 0)
    crossings = np.logical_and(posWindow, negWindow)
    return crossings


def getCrossingsFromTrial(trialDataset):
    """Get crossings from trial "sniff" dataset.

    Arguments:
        trialDataset {h5py.dataset} -- "sniff" dataset.

    Returns:
        [type] -- [description]
    """
    crossings = np.empty_like(trialDataset)
    for i_data, data in enumerate(trialDataset):
        crossings[i_data] = findCrossings(data)
    return crossings


def getSniffSignal(h5Filename):
    """Extract full raw sniff signal from an h5 file.

    Arguments:
        h5Filename {str} -- Path to h5 file.

    Returns:
        ndarray -- Flat ndarray of the signal.
    """
    with h5py.File(h5Filename, "r") as h5File:
        sniffSignal = np.empty(0, dtype=h5File["Trial0001"]["sniff"][0].dtype)
        # Assumes h5File contains Trial000# keys and one Trials key
        for i_trial in range(1, len(h5File)):
            trial = h5File["Trial{:04d}".format(i_trial)]
            for data in trial["sniff"]:
                sniffSignal = np.append(sniffSignal, data)
    return sniffSignal


def sniffIndexToTrialEvent(h5Filename, index):
    """May from sniff index to source trial and event.

    Arguments:
        h5Filename {str} -- h5 source filepath.
        index {int} -- Index to lookup.

    Returns:
        (int, int) -- Trial number, event number.
    """
    with h5py.File(h5Filename, "r") as h5File:
        # Assumes h5File contains Trial000# keys and one Trials key
        for i_trial in range(1, len(h5File)):
            trial = h5File["Trial{:04d}".format(i_trial)]
            for i_event, data in enumerate(trial["sniff"]):
                index -= len(data)
                if index <= 0:
                    return i_trial, i_event


def getSniffIndexesByTrialEvents(h5Filename):
    """Map from Trial/Event to sniff slice.

    Arguments:
        h5Filename {str} -- h5 source filepath.

    Returns:
        [[slice]] -- List (trials) of lists (events) of slices. sniffIndexes[2][10] is the
            slice object that will return the sniff samples for trial 3, event 10 from the
            fully flattened full session sniff sequence, ala getSniffSignal.
    """
    with h5py.File(h5Filename, "r") as h5File:
        sniffIndexes = []
        maxIndex = 0
        # Assumes h5File contains Trial000# keys and one Trials key
        for i_trial in range(1, len(h5File)):
            trial = h5File["Trial{:04d}".format(i_trial)]
            events = []
            for i_event, data in enumerate(trial["sniff"]):
                numSamples = len(data)
                events.append(slice(maxIndex, maxIndex + numSamples))
                maxIndex += numSamples
            sniffIndexes.append(events)
    return sniffIndexes


def getCrossingsFromFiles(filenames, rootpath=""):
    results = {}
    for filename in filenames:
        sniffSignal = getSniffSignal(os.path.join(rootpath, filename))
        allCrossings = findCrossings(sniffSignal)
        results[filename] = allCrossings
    return results


if __name__ == "__main__":
    # Unittest
    rootpath = "./data/"
    files = ["190603/1953_1_04_D2019_6_3T12_29_13_odor.h5"]
    # filenames = list(map(lambda x: os.path.join(rootpath, x), files))
    results = getCrossingsFromFiles(files, rootpath)
    assert len(results) == 1
    T12_29 = results["190603/1953_1_04_D2019_6_3T12_29_13_odor.h5"]
    assert len(T12_29) == 391406
    assert np.all(T12_29[120:125] == [False, True, False, False, False])
    print("zerocrossing.py unit test completed sucessfully.")
