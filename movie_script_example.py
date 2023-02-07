

import time
from typing import cast, List, Union, Dict, Optional, Any, Tuple
import warnings
import sys

import cv2
import numpy as np
import numpy.typing as npt
from scipy.io import savemat
from pathlib import Path

from .data import postprocess
from . import utilities as utils
from .utilities import cfg





def results_movie(self, frames: List[int] = None) -> Any:
    """
    Generate movie illustrating segmentation and tracking

    Parameters
    ----------
    frames : list of int or None, optional
        Frames to generate the movie for. If None, all frames are run.
        The default is None.

    Returns
    -------
    movie : list of 3D numpy arrays
        List of compiled movie frames

    """

    # Re-read trans frames:
    trans_frames = self.reader.getframes(
        positions=self.position_nb,
        channels=0,
        frames=frames,
        rescale=(0, 1),
        squeeze_dimensions=False,
        rotate=self.rotate,
    )
    trans_frames = trans_frames[0, :, 0]
    if self.drift_correction:
        trans_frames, _ = utils.driftcorr(trans_frames, drift=self.drift_values)
    movie = []

    assert isinstance(frames, list)  # FIXME: what happens if frames is None?
    # Run through frames, compile movie:
    for f, fnb in enumerate(frames):

        frame = trans_frames[f]

        # RGB-ify:
        frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)

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

        # Add to movie array:
        movie += [(frame * 255).astype(np.uint8)]

    return movie




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

