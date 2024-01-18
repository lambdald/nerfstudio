"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp
import tyro
from tqdm.auto import tqdm
from pathlib import Path
import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
import tyro


def main(
    colmap_workspace: Path,
    host: str,
    port: int,
    images_reldir: str = 'images',
    model_reldir: str = 'sparse/0',
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    # server = viser.ViserServer()

    server = viser.ViserServer(host=host, port=port)
    server.configure_theme(brand_color=(230, 230, 230))

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_workspace / model_reldir / "cameras.bin")
    images = read_images_binary(colmap_workspace / model_reldir/ "images.bin")
    points3d = read_points3d_binary(colmap_workspace /model_reldir/ "points3D.bin")
    gui_reset_up = server.add_gui_button("Reset up direction")

    @gui_reset_up.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array([0.0, -1.0, 0.0])

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.add_gui_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.add_gui_number("Point size", initial_value=0.05)

    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points = onp.array([points3d[p_id].xyz for p_id in points3d])
        colors = onp.array([points3d[p_id].rgb for p_id in points3d])
        points_selection = onp.random.choice(points.shape[0], gui_points.value, replace=False)
        points = points[points_selection]
        colors = colors[points_selection]

        server.add_point_cloud(
            name="/colmap/pcd",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        onp.random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = colmap_workspace / images_reldir / img.name
            if not image_filename.exists():
                continue

            T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3(img.qvec), img.tvec).inverse()
            server.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            server.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
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
