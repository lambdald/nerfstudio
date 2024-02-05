import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple, Type, Union

import torch
from torch.utils.data import DataLoader

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.criticalpixel.camera.posed_camera import PosedCamera
from nerfstudio.criticalpixel.data.datamanager.data_partation import DataPartationConfig, TrainFullDataPartationConfig
from nerfstudio.criticalpixel.data.dataset.frame_dataset import FrameDataset
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameItemType
from nerfstudio.criticalpixel.data.dataset.scene_metadata import SceneMetadata
from nerfstudio.data.datamanagers.base_datamanager import DataManager, VanillaDataManagerConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class GSplatDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: GSplatDataManager)
    spliter: DataPartationConfig = TrainFullDataPartationConfig()
    use_mask: bool = False

    def __post_init__(self):
        """Warn user of camera optimizer change."""
        if self.camera_optimizer is not None:
            import warnings

            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


def convert_enum_to_str(data: Dict[FrameItemType, torch.Tensor]):
    keys = list(data.keys())
    new_data = {}
    for key in keys:
        if isinstance(key, FrameItemType):
            new_data[key.value] = data[key]
        else:
            new_data[key] = data[key]
    return new_data


class GSplatDataManager(DataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GSplatDataManagerConfig

    def __init__(
        self,
        config: GSplatDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data

        self.dataparser = self.config.dataparser.setup()
        self.scene_metadata: SceneMetadata = self.dataparser.parse_data()
        self.spliter = self.config.spliter.setup()
        self.train_scene_metadata, self.eval_scene_metadata = self.spliter.run(self.scene_metadata)

        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataset = self.train_scene_metadata
        self.eval_dataset = self.train_scene_metadata

        print("mode:", self.test_mode, self.train_dataset, self.eval_dataset)

        super().__init__()

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_iters = {}
        scene_metadata: SceneMetadata = self.train_scene_metadata
        for name in scene_metadata.sensor_metadatas:
            self.train_image_iters[name] = iter(
                DataLoader(FrameDataset(scene_metadata.sensor_metadatas[name]), batch_size=1, shuffle=True)
            )
        self.train_count = 0

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_data_iters = {}
        self.eval_image_iters = {}
        self.eval_image_dataloaders = {}
        scene_metadata: SceneMetadata = self.eval_scene_metadata
        for name in scene_metadata.sensor_metadatas:
            self.eval_data_iters[name] = iter(
                DataLoader(FrameDataset(scene_metadata.sensor_metadatas[name]), batch_size=1, shuffle=True)
            )

            self.eval_image_iters[name] = iter(DataLoader(FrameDataset(scene_metadata.sensor_metadatas[name])))
            self.eval_image_dataloaders[name] = DataLoader(FrameDataset(scene_metadata.sensor_metadatas[name]))

        self.eval_count = 0
        self.eval_image_count = 0

    def get_image_data(self, metadata: SceneMetadata, data_iters: Dict[str, Iterable]):
        sensor_name = random.choice(
            sum(
                [[k] * len(metadata.sensor_metadatas[k].frame_metadata) for k in data_iters.keys()],
                [],
            )
        )
        try:
            image_batch = next(data_iters[sensor_name])
        except StopIteration:
            data_iters[sensor_name] = iter(DataLoader(FrameDataset(metadata.sensor_metadatas[sensor_name])))
            image_batch = next(data_iters[sensor_name])

        image_batch["sensor_name"] = sensor_name
        return image_batch

    def get_data(self, image_data: Dict, scene_metadata: SceneMetadata):
        camera = scene_metadata.sensor_metadatas[image_data["sensor_name"]].camera
        for k in image_data:
            if isinstance(image_data[k], torch.Tensor):
                image_data[k] = image_data[k].squeeze(0)
                if image_data[k].ndim == 0:
                    image_data[k] = image_data[k].view(-1)

        batch_camera = camera[image_data[FrameItemType.SensorId]]

        posed_camera = PosedCamera(
            batch_camera,
            image_data[FrameItemType.Pose],
            self.config.dataparser.coordinate_type,
            image_data[FrameItemType.UniqueId],
            batch_size=batch_camera.batch_size,
        )
        image_batch_str = convert_enum_to_str(image_data)
        return posed_camera, image_batch_str

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        image_data = self.get_image_data(self.train_scene_metadata, self.train_image_iters)
        return self.get_data(image_data, self.train_scene_metadata)

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        image_data = self.get_image_data(self.eval_scene_metadata, self.eval_image_iters)
        return self.get_data(image_data, self.eval_scene_metadata)

    def next_eval_image(self, step: int) -> Tuple[int, Dict]:
        image_data = self.get_image_data(self.eval_scene_metadata, self.eval_image_iters)
        return self.get_data(image_data, self.eval_scene_metadata)

    def iter_all_eval_image(self) -> Tuple[int, Dict]:
        for sensor_name in self.eval_image_dataloaders:
            for image_batch in self.eval_image_dataloaders[sensor_name]:
                image_batch["sensor_name"] = sensor_name
                data = self.get_data(image_batch, self.eval_scene_metadata)
                yield data

    def get_train_rays_per_batch(self) -> int:
        if self.train_sampler is not None:
            return self.train_sampler.config.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_sampler is not None:
            return self.eval_sampler.config.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
