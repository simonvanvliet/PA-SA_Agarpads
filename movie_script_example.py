

import time
from typing import cast, List, Union, Dict, Optional, Any, Tuple
import warnings
import sys

import cv2
import numpy as np
import numpy.typing as npt
from scipy.io import savemat
from pathlib import Path

from . import utilities as utils

import dask.array as da
import re
import importlib
from threading import Thread

import ffmpeg
from skimage.morphology import skeletonize

import nd2

#load data
f = nd2.ND2File(path)
raw_data = f.to_dask()
f.close()

for p in range(raw_data.shape[1]):
    pos = raw_data[:,p,:,:,:]
    movie = da.map_blocks(make_movie_frame, pos)   
    #movie should be t x RGB image
    vidwrite(movie, 'str_with_save_path_for_pos')
    


def make_movie_frame(frame):
    # add normalization / clipping here (replace min/max with quantiles)
    #option 1
    frame = (frame - frame.min())/(frame.max() - frame.min())
    
    #option 2
    limits = np.quantile(frame, [0.05, 0.95])

    frame = (frame - limits[0])/(limits[1] - limits[0])
    frame[frame<0] = 0 
    frame[frame>1] = 1 

    # RGB-ify phase
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)
    
    # RGB-ify GFP
    R = np.zeros_like(frame)
    frame = np.stack([R,frame,R],axis=-1) #test!   

    # Add frame number text:
    frame = cv2.putText(
        frame,
        text=f"frame {fnb:06d}",
        org=(int(frame.shape[0] * 0.05), int(frame.shape[0] * 0.97)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(1, 1, 1, 1),
        thickness=2,
    )

    for r, roi in enumerate(self.rois):

        # Get chamber-specific variables:
        colors = utils.getrandomcolors(len(roi.lineage.cells), seed=r)
        fr = roi.label_stack[fnb]
        assert fr is not None  # FIXME: why is it not None?
        cells, contours = utils.getcellsinframe(fr, return_contours=True)
        assert isinstance(cells, list)  # needed for mypy on Python < 3.8

        if roi.box is None:
            xtl, ytl = (0, 0)
        else:
            xtl, ytl = (roi.box["xtl"], roi.box["ytl"])

        # Run through cells in labelled frame:
        for c, cell in enumerate(cells):

            # Draw contours:
            frame = cv2.drawContours(
                frame,
                contours,
                c,
                color=colors[cell],
                thickness=1,
                offset=(xtl, ytl),
            )

            # Draw poles:
            oldpole = roi.lineage.getvalue(cell, fnb, "old_pole")
            assert isinstance(oldpole, np.ndarray)  # for mypy
            frame = cv2.drawMarker(
                frame,
                (oldpole[1] + xtl, oldpole[0] + ytl),
                color=colors[cell],
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=3,
                thickness=1,
            )

            daughter = roi.lineage.getvalue(cell, fnb, "daughters")
            bornago = roi.lineage.cells[cell]["frames"].index(fnb)
            mother = roi.lineage.cells[cell]["mother"]

            if daughter is None and (bornago > 0 or mother is None):
                newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                frame = cv2.drawMarker(
                    frame,
                    (newpole[1] + xtl, newpole[0] + ytl),
                    color=[1, 1, 1],
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=3,
                    thickness=1,
                )

            # Plot division arrow:
            if daughter is not None:

                newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                daupole = roi.lineage.getvalue(daughter, fnb, "new_pole")
                # Plot arrow:
                frame = cv2.arrowedLine(
                    frame,
                    (newpole[1] + xtl, newpole[0] + ytl),
                    (daupole[1] + xtl, daupole[0] + ytl),
                    color=(1, 1, 1),
                    thickness=1,
                )
    return (frame * 255).astype(np.uint8)




def vidwrite(
    images: np.ndarray, filename: Union[str, Path], crf: int = 20, verbose: int = 1
) -> None:
    """
    Write images stack to video file with h264 compression.

    Parameters
    ----------
    images : 4D numpy array
        Stack of RGB images to write to video file.
    filename : str or Path
        File name to write video to. (Overwritten if exists)
    crf : int, optional
        Compression rate. 'Sane' values are 17-28. See
        https://trac.ffmpeg.org/wiki/Encode/H.264
        The default is 20.
    verbose : int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    None.

    """

    # Initialize ffmpeg parameters:
    height, width, _ = images[0].shape
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    quiet = [] if verbose else ["-loglevel", "error", "-hide_banner"]
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=7,
        )
        .output(
            str(filename),
            pix_fmt="yuv420p",
            vcodec="libx264",
            crf=crf,
            preset="veryslow",
        )
        .global_args(*quiet)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write frames:
    for frame in images:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()


#%% Feature extraction

