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

def main(
    mesh_path: Path,
    host: str,
    port: int,
):

    # mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
    # mesh = trimesh.load_mesh('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-camera3d/lidong57/data/drone/guangzhou495/neusv1_0508/000/nerf_mesh.ply')
    mesh = trimesh.load_mesh(str(mesh_path))
    assert isinstance(mesh, trimesh.Trimesh)

    vertices = mesh.vertices * 0.5
    vertices[:, 1:3] *= -1
    faces = mesh.faces
    print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

    server = viser.ViserServer(host=host, port=port)
    server.add_mesh(
        name="/frame",
        vertices=vertices,
        faces=faces,
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2, 0.0, 0.0])).wxyz,
        position=(0.0, 0.0, 0.0),
    )

    while True:
        time.sleep(10.0)


if __name__ == '__main__':
    tyro.cli(main)
