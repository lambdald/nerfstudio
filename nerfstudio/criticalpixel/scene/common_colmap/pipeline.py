from __future__ import annotations

import datetime
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro
from rich.console import Console

from nerfstudio.scripts.train import main as train_main
from nerfstudio.criticalpixel.scene.common_colmap.configs.method_configs import ColmapAnnotatedBaseConfigUnion

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class DataConfig:
    data_dir: Path = Path()
    """path to data dir"""
    # data_config_path: Optional[Path] = None
    # """config for data parser"""

    # exec: bool = True

    # parse_data: bool = True
    # """parse data"""
    # split_data: bool = True

    # def __post_init__(self):
    #     if self.data_config_path:
    #         self.data_config = Config.fromfile(self.data_config_path)
    #         self.data_config.data_dir = self.data_dir

    # def main(self) -> None:
    #     if not self.exec:
    #         Console().log(f"[bold][green]:tada: Skip data processing :tada:[/bold]")
    #         return
    #     if self.parse_data:
    #         # generate scene metadata
    #         parser = DroneDatasetBuilder(self.data_config)
    #         scene_metadata = parser.build_dataset()
    #         scene_metadata.save_scene_metadata(parser.get_abspath(self.data_config.scene_metadata_path))

    #     if self.split_data:
    #         """
    #         Split scene in X-Y plane.
    #         """
    #         scene_metadata = SceneMetadata.load_scene_metadata(self.data_config.data_dir / self.data_config.scene_metadata_path)
    #         spliter = DroneDatasetSpliter(self.data_config)
    #         split_info = spliter.split_scene(scene_metadata)
    #         spliter.save_split_info(split_info)
    #         spliter.save_bbox2d_image(split_info, str(self.data_config.data_dir / 'subscene_bboxes2d.png'))
    #         spliter.generate_subscene_metadata(scene_metadata)


# @dataclass
# class ReconstructionConfig(ExportMarchingCubesMesh):
#     """Export a mesh using marching cubes."""

#     load_config: Path = Path()
#     output_dir: Path = Path()
#     exec: bool = True
#     precision: float=0.2

#     def main(self) -> None:
#         """Main function."""

#         if not self.exec:
#             Console().log(f"[bold][green]:tada: Skip reconstruction :tada:[/bold]")
#             return
#         super().main()


@dataclass
class PipelineConfig:
    data_config: DataConfig
    # train_config: TrainConfig
    # recon_config: ReconstructionConfig
    train_config: ColmapAnnotatedBaseConfigUnion
    timestamp: str = ""
    exp_name: str = "default"

    def __post_init__(self):
        if not self.timestamp:
            strtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        else:
            strtime = self.timestamp
        self.timestamp = f"{strtime}-{self.exp_name}"
        Console().log(f"set timestamp to {self.timestamp}")
        self.train_config.data = self.data_config.data_dir
        self.train_config.timestamp = self.timestamp
        # self.train_config.pipeline.datamanager.dataparser.data_dir =

    def main(self):
        train_config = deepcopy(self.train_config)
        train_config.timestamp = f"{self.timestamp}"
        Console().log(f"[bold][green]:tada: Training :tada:[/bold]")
        train_main(train_config)
        Console().log(f"[bold][green]:tada: Training Done :tada:[/bold]")
        return


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    train_main(
        tyro.cli(
            ColmapAnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
