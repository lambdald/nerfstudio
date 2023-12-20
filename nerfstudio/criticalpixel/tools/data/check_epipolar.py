"""
By setting the polar coordinates, drawing the epipolar lines, and checking whether the pose is accurate.
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from nerfstudio.v3d.core import camera
sys.path.append(str(Path(__file__).parent / '../..'))

from nerfstudio.v3d.core.dataset.scene_metadata import SceneMetadata
from nerfstudio.v3d.core.camera import BaseCamera, PinholeCamera

import argparse

np.random.seed(1024)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--scene_metadata', required=False, default='scene_metadata.json')
    parser.add_argument('--image_names', nargs='+', type=str, required=False, help='The first one is the reference image ID.')
    parser.add_argument('--output_dir', required=True, type=str)

    return parser.parse_args()


def get_k(camera: PinholeCamera):
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float32)
    return k


F = None


def skew(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def get_F(pose_s2w, pose_d2w, k_s, k_d):

    pose_s2d = np.linalg.inv(pose_d2w) @ pose_s2w

    R = pose_s2d[:3, :3]
    t = pose_s2d[:3, 3]
    F = np.linalg.inv(k_d).T @ skew(t) @ R @ np.linalg.inv(k_s)
    return F



def draw_epipolar_line(img, x, y):
    global F
    h = img.shape[0]
    w = img.shape[1]//2
    # 目标图像上点击的点


    color = tuple(np.random.randint(0, 255, 3).tolist())

    cv2.circle(img, (int(x), int(y)), 1, color, -1)
    cv2.circle(img, (int(x), int(y)), 10, color, 2)
    if x < w:
        point_left = np.array([x, y]).reshape(1, 2)
        r = cv2.computeCorrespondEpilines(point_left, 1, F).reshape(-1)

        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
        cv2.line(img, (x0+w, y0), (x1+w, y1), color, 1, cv2.LINE_AA)
    else:
        point_right = np.array([x-w, y]).reshape(1,2)
        r = cv2.computeCorrespondEpilines(point_right, 2, F).reshape(-1)

        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)



def main(args):
    global F
    scene_metadata = SceneMetadata.load_scene_metadata(args.data_dir / args.scene_metadata)

    image_id_names = []
    for image_name in args.image_names:
        for image_id, image in scene_metadata.image_metadata.items():
            curr_image_name = image.unique_id
            # print(curr_image_name, image_name)
            if image_name in curr_image_name:
                image_id_names.append((image_id, image_name))
    print(image_id_names)

    print('select first image id as reference frame.')

    src_image_id, src_image_name = image_id_names[0]

    for dst_image_id, dst_image_name in image_id_names[1:]:
        print(dst_image_name)
            
        image0_matadata = scene_metadata.image_metadata[src_image_id]
        camera0 = scene_metadata.cameras[image0_matadata.camera_id]

        image1_metadata = scene_metadata.image_metadata[dst_image_id]
        camera1 = scene_metadata.cameras[image1_metadata.camera_id]

        camera1.resize(camera0.height, camera0.width)
        image1_metadata.set_image_size(camera1.height, camera1.width)


        print(camera0)
        print(camera1)

        # assert camera0.model.model_name == 'pinhole'
        # assert camera1.model.model_name == 'pinhole'
        k0 = get_k(camera0)
        k1 = get_k(camera1)

        pose0_w = image0_matadata.pose
        pose1_w = image1_metadata.pose

        F = get_F(pose0_w, pose1_w, k0, k1)

        image0 = image0_matadata.load_image()
        image1 = image1_metadata.load_image()
        
        if camera0.model.model_name in ['fisheye', 'opencv', 'full_opencv']:
            print('undistort image0')
            image0 = camera0.undistort_image(image0)
        if camera1.model.model_name in ['fisheye', 'opencv', 'full_opencv']:
            print('undistort image1')
            image1 = camera1.undistort_image(image1)

        frame = np.hstack([image0, image1])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = image0.shape[:2]

        N = 5

        for i in range(1, N):
            for j in range(1, N):
                x = int(w/N*i)
                y = int(h/N*j)
                draw_epipolar_line(frame, x, y)

        # write your epipolar point

        # for x, y in [(231, 624), (233, 611), (341, 641), (295, 777), (107, 811), (450, 749), (541, 651), (920, 192), (975, 188), (542, 456), (1848, 612), (1680, 543)]:
        #     draw_epipolar_line(frame, x, y)


        name = src_image_name.replace('/', '_') + '_' + dst_image_name.replace('/', '_')
        output_path = f'{args.output_dir}/{name}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), frame)


if __name__ == '__main__':
    main(get_args())