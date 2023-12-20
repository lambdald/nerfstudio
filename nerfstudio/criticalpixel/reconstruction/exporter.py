"""
Export mesh or point cloud from v3d field.
"""

from dataclasses import dataclass
from typing import Tuple, Literal, Optional, cast
import torch
from pathlib import Path
import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.scripts.exporter import Exporter
from nerfstudio.utils.printing import CONSOLE
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.v3d.core.fields.v3d_base_field import V3dBaseField
from nerfstudio.v3d.core.fields.v3d_neuralangelo_field import V3dNeuralangeloField
from nerfstudio.v3d.core.fields.v3d_neus_field import V3dNeusField


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    precision: float = 0.5
    """Marching cube resolution."""
    mesh_relpath: Path = Path()

    # TODO: texture
    # """Marching cube resolution."""
    # simplify_mesh: bool = False
    # """Whether to simplify the mesh."""
    # px_per_uv_triangle: int = 4
    # """Number of pixels per UV triangle."""
    # unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    # """The method to use for unwrapping the mesh."""
    # num_pixels_per_side: int = 2048
    # """If using xatlas for unwrapping, the pixels per side of the texture image."""
    # target_num_faces: Optional[int] = 50000
    # """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        CONSOLE.print("Load config from ", self.load_config)
        _, pipeline, _, _ = eval_setup(self.load_config)
        scene_box = pipeline.model.scene_box.aabb

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        self.resolution = torch.round((scene_box[1] - scene_box[0]) / self.precision).to(torch.int32)
        self.resolution = ((self.resolution + 512 - 1) / 512).to(torch.int32) * 512
        CONSOLE.print("Resolution:", self.resolution)

        assert (
            self.resolution % 512
        ).sum().item() == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        geometry_fn = None
        if isinstance(pipeline.model.field, SDFField):
            geometry_fn = lambda xyz: cast(SDFField, pipeline.model.field).forward_geonetwork(xyz)[:, 0].contiguous()
        elif isinstance(pipeline.model.field, V3dBaseField):
            geometry_fn = (
                lambda xyz: cast(V3dNeusField, pipeline.model.field).get_geometry(xyz)[0].contiguous().squeeze(-1)
            )
        else:
            raise NotImplementedError(type(pipeline.model.field))

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=geometry_fn,
            resolution=self.resolution,
            bounding_box_min=tuple(scene_box[0].tolist()),
            bounding_box_max=tuple(scene_box[1].tolist()),
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / self.mesh_relpath
        CONSOLE.print("Export mesh to", filename)

        # TODO: transform from nerf to world.
        # TODO: filter mesh.
        # TODO: output textured mesh.

        multi_res_mesh.export(filename)
        # load the mesh from the marching cubes export
        # mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        # CONSOLE.print("Texturing mesh with NeRF...")
        # texture_utils.export_textured_mesh(
        #     mesh,
        #     pipeline,
        #     self.output_dir,
        #     px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
        #     unwrap_method=self.unwrap_method,
        #     num_pixels_per_side=self.num_pixels_per_side,
        # )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")

    tyro.cli(tyro.conf.FlagConversionOff[ExportMarchingCubesMesh]).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(ExportMarchingCubesMesh)  # noqa
