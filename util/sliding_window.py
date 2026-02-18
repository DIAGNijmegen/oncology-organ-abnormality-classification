# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import numpy as np
from typing import Tuple, List, Iterator


def sliding_window_3d(
    volume: np.ndarray,
    window_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: bool = False
) -> Iterator[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """
    Generate sliding windows over a 3D volume.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        window_size: (depth, height, width) of the window
        stride: (stride_z, stride_y, stride_x)
        padding: If True, pad volume to ensure all regions are covered
    
    Yields:
        (patch, (z, y, x)) where (z, y, x) is the origin of the patch
    """
    d, h, w = volume.shape
    win_d, win_h, win_w = window_size
    stride_d, stride_h, stride_w = stride
    
    if padding:
        # Pad to ensure we can extract windows at the edges
        pad_d = max(0, win_d - stride_d)
        pad_h = max(0, win_h - stride_h)
        pad_w = max(0, win_w - stride_w)
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = volume.shape
    
    for z in range(0, d - win_d + 1, stride_d):
        for y in range(0, h - win_h + 1, stride_h):
            for x in range(0, w - win_w + 1, stride_w):
                patch = volume[z:z+win_d, y:y+win_h, x:x+win_w]
                yield patch, (z, y, x)


def sliding_window_2d_slices(
    volume: np.ndarray,
    window_size: Tuple[int, int],
    stride: int = 1,
    axis: int = 0
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generate 2D sliding windows by extracting slices along one axis.
    
    Args:
        volume: 3D numpy array
        window_size: (height, width) of the 2D window
        stride: stride along the slice axis
        axis: axis along which to extract slices (0=Z, 1=Y, 2=X)
    
    Yields:
        (patch, slice_idx) where slice_idx is the slice index
    """
    h, w = window_size
    
    if axis == 0:  # Extract XY slices along Z
        for z in range(0, volume.shape[0], stride):
            if z + 1 <= volume.shape[0]:
                # Extract 2D slice
                slice_2d = volume[z, :, :]
                # Crop or pad to window_size if needed
                if slice_2d.shape[0] >= h and slice_2d.shape[1] >= w:
                    # Center crop
                    y_start = (slice_2d.shape[0] - h) // 2
                    x_start = (slice_2d.shape[1] - w) // 2
                    patch = slice_2d[y_start:y_start+h, x_start:x_start+w]
                    yield patch, z
                elif slice_2d.shape[0] < h or slice_2d.shape[1] < w:
                    # Pad to window_size
                    pad_y = max(0, h - slice_2d.shape[0])
                    pad_x = max(0, w - slice_2d.shape[1])
                    patch = np.pad(slice_2d, ((0, pad_y), (0, pad_x)), mode='constant')
                    patch = patch[:h, :w]
                    yield patch, z
    elif axis == 1:  # Extract XZ slices along Y
        for y in range(0, volume.shape[1], stride):
            if y + 1 <= volume.shape[1]:
                slice_2d = volume[:, y, :]
                if slice_2d.shape[0] >= window_size[0] and slice_2d.shape[1] >= window_size[1]:
                    z_start = (slice_2d.shape[0] - window_size[0]) // 2
                    x_start = (slice_2d.shape[1] - window_size[1]) // 2
                    patch = slice_2d[z_start:z_start+window_size[0], x_start:x_start+window_size[1]]
                    yield patch, y
    elif axis == 2:  # Extract YZ slices along X
        for x in range(0, volume.shape[2], stride):
            if x + 1 <= volume.shape[2]:
                slice_2d = volume[:, :, x]
                if slice_2d.shape[0] >= window_size[0] and slice_2d.shape[1] >= window_size[1]:
                    z_start = (slice_2d.shape[0] - window_size[0]) // 2
                    y_start = (slice_2d.shape[1] - window_size[1]) // 2
                    patch = slice_2d[z_start:z_start+window_size[0], y_start:y_start+window_size[1]]
                    yield patch, x
