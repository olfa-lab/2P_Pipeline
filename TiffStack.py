"""Manage an imaging session tiff stack and ROI masks.
"""
import glob
import logging
import os
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
from skimage import io
from tqdm import tqdm

logger = logging.getLogger("TiffStack")
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.WARNING)

BoxCoords = Tuple[List[int], List[int]]


class TiffStack:
    def __init__(self, tiffFilenames: Sequence[str], maskFilenames: Sequence[str] = []):
        """Tiffstack and its ROI masks.

        Args:
            tiffFilenames (Sequence[str]): Sequence of tiff files that constitute the
                stack, in order.
            maskFilenames (Sequence[str], optional): Sequence of bmp files for boolean
                masks. Defaults to [].

        Sets:
            .timeseries (np.ndarray): Assembled tiff stack.
            .masks (List[np.ndarray]): List of boolean masks for timeseries.
        """
        self._tiffFilenames = tiffFilenames
        self.timeseries = self.open_TIFF_stack(tiffFilenames)
        self.masks = self._get_masks(maskFilenames)

    def open_TIFF_stack(self, tiffFilenames: Sequence[str]) -> np.ndarray:
        """Open and concatenate a set of tiffs.

        Arguments:
            tiffFilenames {Sequence[str]} -- Sequence of tiff stack filenames, in order.

        Returns:
            ndarray -- Timeseries (frames by x-coord by y-coord).
        """
        startTime = time.time()
        numFiles = len(tiffFilenames)
        assert numFiles > 0, "Must provide at least one tiff file to open."
        logger.debug(f"Opening {numFiles} files.")
        stacks = []
        for fname in tqdm(
            tiffFilenames, desc=os.path.basename(tiffFilenames[0]), unit="file"
        ):
            logger.debug(f"Opening {fname}")
            stacks.append(io.imread(fname))
        logger.info(f"Concatenating {len(tiffFilenames)} tiffs.")
        stack = np.concatenate(stacks, axis=0)
        logger.debug(f"TiffStack creation time: {time.time() - startTime}")
        return stack

    def add_masks(self, maskPattern: str = "*.bmp") -> None:
        """Add masks found with a shell wildcard pattern.

        Args:
            maskPattern (str, optional): Path to the mask(s). Shell wildcard characters
                acceptable. Defaults to "*.bmp".
        """
        maskFNames = sorted(glob.glob(maskPattern))
        self.masks += self._get_masks(maskFNames)

    def _get_masks(self, maskFilenames: Sequence[str]) -> List[np.ndarray]:
        """Get ROI masks from files.

        Arguments:
            maskPattern {str} -- File glob pattern.

        Returns:
            [2d ndarray] -- List of masks.
        """
        masks: List[np.ndarray] = []
        for maskFName in maskFilenames:
            mask = io.imread(maskFName)
            invMask = mask == 0
            masks += [mask] if np.sum(mask) < np.sum(invMask) else [invMask]
        return masks

    def cut_to_averages(self, forceNew=False) -> np.ndarray:
        """Get average intensity of each ROI at each frame.

        Keyword Arguments:
            forceNew {bool} -- Force calculating averages even if .averages already
                exists. (default: {False})

        Returns:
            ndarray -- Frames by ROI.
        """
        if forceNew or (not hasattr(self, "averages")):
            ROIaverages = np.empty((self.timeseries.shape[0], len(self.masks)))
            for i_mask, mask in enumerate(self.masks):
                ROIaverages[:, i_mask] = np.mean(self.timeseries[:, mask], axis=1)
            self.averages = ROIaverages
        else:
            ROIaverages = self.averages
        return ROIaverages

    def get_bounding_boxes(self) -> List[BoxCoords]:
        """Get bounding boxes for each ROI mask.

        Returns:
            List[Tuple[List[int], List[int]]] -- List of bounding coordinates, ordered
                according to self.masks. Each entry consists of a 2-tuple, the minimum
                coordinate corner and the maximum coordinate corner.
        """
        ROIs = [self.get_bounds_index(mask) for mask in self.masks]
        return ROIs

    @staticmethod
    def get_bounds_index(mask: np.ndarray) -> BoxCoords:
        """Returns coordinates of opposing (min/max each dim) corners of bounding (hyper)box.

        Arguments:
            mask {np.ndarray} -- Boolean mask.

        Returns:
            Tuple[List[int], List[int]] -- (Coords of min corner, coords of max corner)
        """
        maxBound = mask.shape
        maskIndexes = mask.nonzero()
        maxIndex = [max(i) for i in maskIndexes]
        assert all([maski <= boundi for maski, boundi in zip(maxIndex, maxBound)])
        minIndex = [min(i) for i in maskIndexes]
        assert all([maski >= 0 for maski in minIndex])
        return minIndex, maxIndex
