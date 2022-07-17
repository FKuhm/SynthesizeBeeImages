# %% [markdown]
# @version: 14-11-2022
# @author: Florian Kuhm, Parzival Borlinghaus
# @params_to_adjust: n_bees, counter, path_to_video, path_to_output_file

from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from os.path import dirname, join, basename
from tqdm import tqdm
from os.path import join
from pathlib import Path


def apply_image_transformations(
    frame: np.ndarray,
    gray_threshold: int,
    dilation_kernel_size: int
):
    """Applies preprocessing to each frame. This helps to find the bees'
    location and to extract the bees' center points.

    Args:
        frame (np.ndarray): The input image in bgr format.
        gray_threshold (int): This threshold is applied to the grayscale image.
            Pixels darker than this value are assumed to refer to bees.
        dilation_kernel_size (int): Larger values remove more of the finer
            structures of bees (e.g. antenna and legs).

    Returns:
        np.ndarray: The transformed image data, that is a binary mask where
            zeros denote background and ones denote bees.
    """
    # If pixel has a value > thresval, its value will be set to 0.
    # Otherwise, pixel value will be left untouched.
    result = cv2.threshold(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        gray_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        np.ones(shape=(dilation_kernel_size,dilation_kernel_size))
    )
    return result


def filter_valid_bee_centroids(
    centroids: np.ndarray,
    img_width: int,
    img_height: int,
    bee_dim: int
):
    """Since only images of fully visible bees are needed, centroids that are
    too close to the edges are filtered.

    Args:
        centroids (np.ndarray): The extracted but not yet filtered centroids.
            Note: The coordinates are in (y, x) order.
        img_widht (int): The source image's width.
        img_height (int): The source image's height.
        bee_dim (int): The dimension of the cropped bee image.
    Returns:
        list: A list of valid (x, y) positions.
    """
    if len(centroids) > 1:
        raise NotImplementedError(
            "This is probably not needed in the future and therefore "
            "development is on hold."
        )

    assert img_width > img_height, "Width and height are mixed."

    valid_centroids = []
    for i, centroid in enumerate(centroids):
        if not (
            centroid[0] < bee_dim // 2 + 1 or
            centroid[1] < bee_dim // 2 + 1 or
            centroid[0] > img_height - bee_dim // 2 - 1 or
            centroid[1] > img_width - bee_dim // 2 - 1
        ):
            valid_centroids.append(centroid)
    return valid_centroids


def create_dataset(
    video_sources: List[str],
    output_dir: str,
    bee_dim: int,
    n_bees: int,
    gray_threshold: int,
    dilation_kernel_size: int,
):
    """Trigger the dataset creation.

    Args:
        video_sources (List[str]): A list of video files. This list might have
            length 1.
        output_dir (str): The directory where all output is stored.
        bee_dim (int): The dimension of a cropped bee image.
        n_bees (int): The number of bees that can/might/are/should be in the
            image. The only supported scenario at the moment is a single be
            that might visible or hidden.
        gray_threshold (int): This value is passed to the frame thresholding.
        dilation_kernel_size (int): This value is passed the dilation of the
            thresholded frame.
    """
    # create video object
    assert len(video_sources) == 1,\
        "At the moment only one video is supported."

    assert n_bees == 1,\
        "This is the only value supported at the moment."

    assert bee_dim % 2 == 0,\
        f"`bee_dim` should be an even number, not {bee_dim}."

    video_source = video_sources[0]
    vd = cv2.VideoCapture(video_source)

    success, frame = vd.read()
    img_id = 0
    progress = tqdm()  # a nice progressbar (you can skip `total` parameter.)
    while success:
        progress.update(1)  # Print progress
        assert frame.ndim == 3, (
            f"video {video_source} is broken and should be inspected. "
            "Usually we don't want to use it at all."
        )

        frame_mask = apply_image_transformations(
            frame=frame,
            gray_threshold=gray_threshold,
            dilation_kernel_size=dilation_kernel_size
        )

        # Find indices of all non-black bee pixels
        y_indices, x_indices = np.where(frame_mask > 0)
        yx_indices = np.stack([y_indices, x_indices], axis=1)

        if not len(yx_indices) > 0:
            # no bee is visible, skip frame or kmeans will crash
            success, frame = vd.read()
            continue

        # Apply kMeans only in case of at least two bees in the image
        # (legacy code)
        if n_bees > 1 and len(yx_indices) > 0:
            # Apply kmeans algorithm on all pixel coordinates computed above.
            # Because n_bees are visible in the video, k is set to n_bees:
            kmeans = KMeans(n_clusters = n_bees).fit(yx_indices)
            # For each centroid computed:
            centroids = kmeans.cluster_centers_.astype(np.int64)
        else:
            # much easier calculations for the single-bee case
            centroids = yx_indices.mean(axis=0).reshape((1, 2)).astype(np.int64)

        filtered_centroids = filter_valid_bee_centroids(
            bee_dim=bee_dim,
            centroids=centroids,
            img_width=frame.shape[1],
            img_height=frame.shape[0]
        )

        for centroid in filtered_centroids:
            # NOTE: centroids contain (y, x) coordinates
            cropped_bee_file_name = join(
                output_dir,
                "crazy_folder_structure",
                f"crazy_meta_infos_{img_id: 06d}.png"
            )
            # create folder structure if it not there yet
            Path(dirname(cropped_bee_file_name)).mkdir(
                parents=True, exist_ok=True
            )
            cv2.imwrite(
                cropped_bee_file_name,
                frame[
                    centroid[0] - bee_dim // 2:centroid[0] + bee_dim // 2,
                    centroid[1] - bee_dim // 2:centroid[1] + bee_dim // 2,
                ]
            )
            img_id += 1

        # # read more frames
        success, frame = vd.read()

    print(f"done with {basename(video_source)}")
    vd.release()

