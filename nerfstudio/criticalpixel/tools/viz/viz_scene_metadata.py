"""v3d scene metadata visualizer

Visualize scene_metdata.json.
TODO: convert cv to gl.
"""

import os.path as osp
import sys
import time
from pathlib import Path

import numpy as onp
import trimesh
import tyro
from tqdm.auto import tqdm


import viser
import viser.transforms as tf

from nerfstudio.criticalpixel.data.dataset.scene_metadata import SceneMetadata, FrameItemType


def main(
    scene_metadata_path: Path,
    host: str = "localhost",
    port: int = 8080,
    downsample_factor: int = 8,
) -> None:
    """Visualize SceneMetadata sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    # server = viser.ViserServer()

    server = viser.ViserServer(host=host, port=port)
    # server.configure_theme(canvas_background_color=(230, 230, 230))

    scene_metadata = SceneMetadata.load(scene_metadata_path)

    # Load the colmap info.
    gui_reset_up = server.add_gui_button("Reset up direction")

    @gui_reset_up.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array([0.0, -1.0, 0.0])

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(scene_metadata.points3d),
        step=1,
        initial_value=min(len(scene_metadata.points3d), 50_000),
    )
    gui_frames = server.add_gui_slider(
        "Max frames",
        min=1,
        max=len(scene_metadata),
        step=1,
        initial_value=min(len(scene_metadata), 100),
    )
    gui_point_size = server.add_gui_number("Point size", initial_value=0.05)

    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points = scene_metadata.points3d.points[:, :3]
        colors = scene_metadata.points3d.colors[:, 3:] / 255
        points_selection = onp.random.choice(points.shape[0], gui_points.value, replace=False)
        points = points[points_selection].numpy()
        colors = colors[points_selection].numpy()

        server.add_point_cloud(
            name="/point_cloud",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )

        bbox = trimesh.creation.box(bounds=scene_metadata.bbox.aabb)
        server.add_mesh_trimesh(name="scene_bbox", mesh=bbox, wireframe=True)

        # Interpret the images and cameras.
        for sensor_id in tqdm(list(scene_metadata.sensor_metadatas.keys())):
            cameras = scene_metadata.sensor_metadatas[sensor_id].camera
            frames = scene_metadata.sensor_metadatas[sensor_id].frame_metadata

            for img_id in range(len(frames)):
                data = frames[img_id]
                cam = cameras[data[FrameItemType.SensorId]]

                T_world_camera = tf.SE3.from_matrix(data[FrameItemType.Pose])
                server.add_frame(
                    f"/frame_{img_id}",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.1,
                    axes_radius=0.005,
                )

                H, W = cam.hws.squeeze().tolist()
                fy = cam.params.squeeze().tolist()[1]
                image = data[FrameItemType.Image]
                image = image[::downsample_factor, ::downsample_factor]
                server.add_camera_frustum(
                    f"/frame_{img_id}/frustum",
                    fov=2 * onp.arctan2(H / 2, fy),
                    aspect=W / H,
                    scale=0.15,
                    image=image,
                )

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False

            server.reset_scene()
            visualize_colmap()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
