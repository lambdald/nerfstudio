"""Meshes

Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as onp
import trimesh

import viser
import viser.transforms as tf
import tyro
import open3d as o3d
import numpy as np

def main(
    point_path: Path,
    host: str,
    port: int,
):

    pcd = o3d.io.read_point_cloud(str(point_path))
    print(pcd)
    points = np.asarray(pcd.points)  # 将点转换为numpy数组
    colors = np.asarray(pcd.colors)
    if len(colors) != len(points):
        colors=np.zeros_like(points)

    # points[:, 1:3] *= -1
    
    points = points - points.mean(axis=0, keepdims=True)

    server = viser.ViserServer(host=host, port=port)
    gui_point_size = server.add_gui_number("Point size", initial_value=0.05)

    server.add_point_cloud(
        name="/colmap/pcd",
        points=points,
        colors=colors,
        point_size=gui_point_size.value,
    )

    while True:
        time.sleep(10.0)


if __name__ == '__main__':
    tyro.cli(main)
