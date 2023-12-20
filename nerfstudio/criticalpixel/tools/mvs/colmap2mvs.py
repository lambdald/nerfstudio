import numpy as np
import os
import shutil
from nerfstudio.criticalpixel.data.dataparser.colmap.model import read_model, Camera, Image, Point3D
import tyro
import cv2
import dataclasses
from pathlib import Path
import tyro

import torch
from typing import Literal, Dict
import multiprocessing as mp
import functools


def calc_score(inputs, images, points3d, extrinsic, args):
    i, j = inputs
    id_i = set(images[i + 1].point3D_ids.tolist())
    id_j = set(images[j + 1].point3D_ids.tolist())
    common_ids = list(id_i.intersection(id_j))
    cam_center_i = -np.matmul(extrinsic[i + 1][:3, :3].transpose(), extrinsic[i + 1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j + 1][:3, :3].transpose(), extrinsic[j + 1][:3, 3:4])[:, 0]
    score = 0
    angles = []

    common_points = points3d[common_ids]

    view_vec_i = cam_center_i - common_points
    view_vec_j = cam_center_j - common_points

    angles = (180 / np.pi) * np.arccos(
        np.sum(view_vec_i * view_vec_j, axis=-1)
        / np.linalg.norm(view_vec_i, axis=-1)
        / np.linalg.norm(view_vec_j, axis=-1)
    )  # triangulation angle

    score = len(common_points)

    # for pid in id_intersect:
    #     if pid == -1:
    #         continue
    #     p = points3d[pid].xyz
    #     theta = (180 / np.pi) * np.arccos(
    #         np.dot(cam_center_i - p, cam_center_j - p)
    #         / np.linalg.norm(cam_center_i - p)
    #         / np.linalg.norm(cam_center_j - p)
    #     )  # triangulation angle
    #     # score += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
    #     angles.append(theta)
    #     score += 1

    if len(angles) > 0:
        angles_sorted = sorted(angles)
        triangulationangle = angles_sorted[int(len(angles_sorted) * 0.75)]
        if triangulationangle < 1:
            score = 0.0
    return i, j, score


@dataclasses.dataclass
class Colmap2MvsConfig:
    colmap_dir: Path

    output_dir: Path

    max_depth: float = 192
    interval_scale: float = 1.0
    scale_factor: float = 1.0

    theta0: float = 5
    sigma1: float = 1
    sigma2: float = 10
    model_ext: Literal[".txt", ".bin"] = ".bin"


def processing_single_scene(config: Colmap2MvsConfig):
    image_dir = os.path.join(config.colmap_dir, "images")
    model_dir = os.path.join(config.colmap_dir, "sparse")
    cam_dir = os.path.join(config.output_dir, "cams")
    image_converted_dir = os.path.join(config.output_dir, "images")

    if os.path.exists(image_converted_dir):
        print("remove:{}".format(image_converted_dir))
        shutil.rmtree(image_converted_dir)
    os.makedirs(image_converted_dir)
    if os.path.exists(cam_dir):
        print("remove:{}".format(cam_dir))
        shutil.rmtree(cam_dir)

    cameras, images, points3d = read_model(str(model_dir), config.model_ext)
    num_images = len(list(images.items()))

    param_type = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"],
    }

    scale_factor = config.scale_factor

    # intrinsic
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        i = np.array(
            [
                [params_dict["fx"] / scale_factor, 0, params_dict["cx"] / scale_factor],
                [0, params_dict["fy"] / scale_factor, params_dict["cy"] / scale_factor],
                [0, 0, 1],
            ]
        )
        intrinsic[camera_id] = i
    print("intrinsic\n", intrinsic, end="\n\n")

    # new_images: Dict[int, Image] = {}
    # for i, image_id in enumerate(sorted(images.keys())):
    #     new_images[i + 1] = images[image_id]
    # images = new_images

    # extrinsic
    extrinsic = {}
    for image_id, image in images.items():
        extrinsic[image_id] = image.pose()
    print("extrinsic[1]\n", extrinsic[1], end="\n\n")

    # depth range and interval

    max_pt_ids = max(points3d.keys())
    points3d_np = np.zeros([max_pt_ids + 1, 3])
    for pt_id in points3d.keys():
        points3d_np[pt_id] = points3d[pt_id].xyz

    print(points3d_np.shape)

    depth_ranges = {}
    for i in range(num_images):
        zs = []

        curr_points3d_ids = np.array(images[i + 1].point3D_ids)
        valid_points3d_id = curr_points3d_ids[curr_points3d_ids != -1]

        curr_vis_points3d_w = points3d_np[valid_points3d_id]

        pose = images[i + 1].pose()

        rotmat = images[i + 1].qvec2rotmat()
        tvec = images[i + 1].tvec

        curr_vis_points3d_c = curr_vis_points3d_w @ rotmat.T + tvec

        zs = curr_vis_points3d_c[..., 2]

        depth_min = 0
        depth_max = 0

        if len(zs) != 0:
            zs_sorted = sorted(zs)
            # relaxed depth range
            depth_min = zs_sorted[int(len(zs) * 0.01)] * 0.75
            depth_max = zs_sorted[int(len(zs) * 0.99)] * 1.25

        # determine depth number by inverse depth setting, see supplementary material
        if config.max_depth == 0:
            image_int = intrinsic[images[i + 1].camera_id]
            image_ext = extrinsic[i + 1]
            image_r = image_ext[0:3, 0:3]
            image_t = image_ext[0:3, 3]
            p1 = [image_int[0, 2], image_int[1, 2], 1]
            p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
            P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
            P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
            P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
            P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
            depth_num = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
        else:
            depth_num = config.max_depth
        depth_interval = (depth_max - depth_min) / (depth_num - 1) / config.interval_scale
        depth_ranges[i + 1] = (depth_min, depth_interval, depth_num, depth_max)
    print("depth_ranges[1]\n", depth_ranges[1], end="\n\n")
    # view selection
    score = np.zeros((len(images), len(images)))
    queue = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            queue.append((i, j))

    p = mp.Pool(processes=min(10, mp.cpu_count()))
    func = functools.partial(calc_score, images=images, points3d=points3d_np, args=config, extrinsic=extrinsic)
    result = p.map(func, queue)
    for i, j, s in result:
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    num_view = min(20, len(images) - 1)
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:num_view]])
    print("view_sel[0]\n", view_sel[0], end="\n\n")

    # write
    try:
        os.makedirs(cam_dir)
    except os.error:
        print(cam_dir + " already exist.")
    for i in range(num_images):
        with open(os.path.join(cam_dir, "%08d_cam.txt" % i), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i + 1][j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i + 1].camera_id][j, k]) + " ")
                f.write("\n")
            f.write(
                "\n%f %f %f %f\n"
                % (depth_ranges[i + 1][0], depth_ranges[i + 1][1], depth_ranges[i + 1][2], depth_ranges[i + 1][3])
            )
    with open(os.path.join(config.output_dir, "pair.txt"), "w") as f:
        f.write("%d\n" % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write("%d\n%d " % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write("%d %d " % (image_id, s))
            f.write("\n")

    max_width = 0
    max_height = 0
    for i in range(num_images):
        img_path = os.path.join(image_dir, images[i + 1].name)
        img = cv2.imread(img_path)
        if max_height < img.shape[0]:
            max_height = img.shape[0]
        if max_width < img.shape[1]:
            max_width = img.shape[1]

    # convert to jpg
    for i in range(num_images):
        img_path = os.path.join(image_dir, images[i + 1].name)
        img = cv2.imread(img_path)
        pad_width = max_width - img.shape[1]
        pad_height = max_height - img.shape[0]
        img_pad = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), "constant")
        img_pad = cv2.resize(
            img_pad,
            (int(img_pad.shape[1] / scale_factor), int(img_pad.shape[0] / scale_factor)),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(os.path.join(image_converted_dir, "%08d.jpg" % i), img_pad)
        # if not img_path.endswith(".jpg"):
        #
        # else:
        #     shutil.copyfile(os.path.join(image_dir, images[i+1].name), os.path.join(image_converted_dir, '%08d.jpg' % i))


if __name__ == "__main__":
    config = tyro.cli(Colmap2MvsConfig)
    processing_single_scene(config)
