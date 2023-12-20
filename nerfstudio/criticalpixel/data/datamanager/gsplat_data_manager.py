from nerfstudio.criticalpixel.data.dataset.scene_metadata import SceneMetadata
from dataclasses import dataclass, field
from nerfstudio.criticalpixel.data.dataparser.dataparser import DataParserConfig
from nerfstudio.criticalpixel.data.dataparser.colmap_parser import ColmapDataparserConfig
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Type, Tuple, Dict, Union
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)
from nerfstudio.criticalpixel.data.dataset.frame_dataset import FrameDataset
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameItemType
from nerfstudio.criticalpixel.data.datamanager.data_partation import DataPartationConfig, TrainFullDataPartationConfig
from nerfstudio.criticalpixel.data.dataloader.cache_dataloader import CacheDataloader
from nerfstudio.criticalpixel.data.sampler.pixel_sampler import PixelSamplerConfig, PixelSampler
from nerfstudio.criticalpixel.geometry.ray import Ray
import random
from nerfstudio.criticalpixel.camera.posed_camera import PosedCamera
from torch.utils.data import DataLoader
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
    DataManager,
    VanillaDataManagerConfig,
    VanillaDataManager,
)
from typing_extensions import TypeVar
from pathlib import Path
from nerfstudio.cameras.rays import RayBundle


@dataclass
class GsplatDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: GsplatDataManager)
    # dataparser: DataParserConfig = ColmapDataparserConfig()
    spliter: DataPartationConfig = TrainFullDataPartationConfig()
    sampler: PixelSamplerConfig = PixelSamplerConfig(num_rays_per_batch=8192)

    num_image_to_load: int = 100
    num_iter_to_resample: int = 500
    num_rays_per_batch: int = 2048
    use_mask: bool = False

    def __post_init__(self):
        """Warn user of camera optimizer change."""
        if self.camera_optimizer is not None:
            import warnings

            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


class GsplatDataManager(DataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GsplatDataManagerConfig

    train_sampler: PixelSampler
    eval_sampler: PixelSampler

    def __init__(
        self,
        config: GsplatDataManagerConfig,
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

        self.config.sampler.num_rays_per_batch = self.config.num_rays_per_batch

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

        print(self.test_mode, self.train_dataset)

        super().__init__()

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_data_iters = {}
        scene_metadata: SceneMetadata = self.train_scene_metadata
        for name in scene_metadata.sensor_metadatas:
            self.train_data_iters[name] = iter(
                CacheDataloader(
                    FrameDataset(scene_metadata.sensor_metadatas[name]),
                    num_images_to_load=self.config.num_image_to_load,
                    num_iters_to_reload_images=self.config.num_iter_to_resample,
                )
            )
        self.train_sampler = self.config.sampler.setup()
        self.train_count = 0

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_data_iters = {}
        self.eval_image_iters = {}
        scene_metadata: SceneMetadata = self.eval_scene_metadata
        for name in scene_metadata.sensor_metadatas:
            self.eval_data_iters[name] = iter(
                CacheDataloader(
                    FrameDataset(scene_metadata.sensor_metadatas[name]),
                    num_images_to_load=self.config.num_image_to_load,
                    num_iters_to_reload_images=self.config.num_iter_to_resample,
                )
            )

            self.eval_image_iters[name] = iter(DataLoader(FrameDataset(scene_metadata.sensor_metadatas[name])))

        self.eval_sampler = self.config.sampler.setup()
        self.eval_count = 0
        self.eval_image_count = 0

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        sensor_name = random.choice(
            sum(
                [
                    [k] * len(self.train_scene_metadata.sensor_metadatas[k].frame_metadata)
                    for k in self.train_data_iters.keys()
                ],
                [],
            )
        )
        image_batch = next(self.train_data_iters[sensor_name])
        assert self.train_sampler is not None
        assert isinstance(image_batch, dict)

        if self.config.use_mask and FrameItemType.Mask in image_batch:
            mask = image_batch[FrameItemType.Mask]
        else:
            mask = None
        batch = self.train_sampler.sample(image_batch, mask=mask)

        ray_indices = batch["indices"]  # [uvs]
        camera = self.train_scene_metadata.sensor_metadatas[sensor_name].camera
        posed_camera = PosedCamera(camera, self.config.dataparser.coordinate_type)
        device = torch.device(self.device)

        rays = posed_camera.get_rays(
            batch[FrameItemType.SensorId],
            ray_indices[..., 1:3].to(device),
            batch[FrameItemType.Pose].to(device).float(),
        )
        ray_bundle = RayBundle(
            origins=rays.origin,
            directions=rays.direction,
            pixel_area=None,
            camera_indices=batch[FrameItemType.SensorId],
        )
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        sensor_name = random.choice(
            sum(
                [
                    [k] * len(self.eval_scene_metadata.sensor_metadatas[k].frame_metadata)
                    for k in self.eval_data_iters.keys()
                ],
                [],
            )
        )
        image_batch = next(self.eval_data_iters[sensor_name])
        assert self.eval_sampler is not None
        assert isinstance(image_batch, dict)

        if self.config.use_mask and FrameItemType.Mask in image_batch:
            mask = image_batch[FrameItemType.Mask]
        else:
            mask = None
        batch = self.eval_sampler.sample(image_batch, mask=mask)

        ray_indices = batch["indices"]  # [uvs]
        camera = self.eval_scene_metadata.sensor_metadatas[sensor_name].camera
        posed_camera = PosedCamera(camera, self.config.dataparser.coordinate_type)
        device = torch.device(self.device)
        rays = posed_camera.get_rays(
            batch[FrameItemType.SensorId],
            ray_indices[..., 1:3].to(device),
            batch[FrameItemType.Pose].to(device).float(),
        )

        ray_bundle = RayBundle(
            origins=rays.origin,
            directions=rays.direction,
            pixel_area=None,
            camera_indices=batch[FrameItemType.SensorId],
        )

        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        sensor_name = random.choice(
            sum(
                [
                    [k] * len(self.eval_scene_metadata.sensor_metadatas[k].frame_metadata)
                    for k in self.eval_data_iters.keys()
                ],
                [],
            )
        )
        try:
            image_batch = next(self.eval_image_iters[sensor_name])
        except StopIteration as e:
            self.eval_image_iters[sensor_name] = iter(
                DataLoader(FrameDataset(self.eval_scene_metadata.sensor_metadatas[sensor_name]))
            )
            image_batch = next(self.eval_image_iters[sensor_name])

        camera = self.eval_scene_metadata.sensor_metadatas[sensor_name].camera
        posed_camera = PosedCamera(camera, self.config.dataparser.coordinate_type)

        for k in image_batch:
            image_batch[k] = image_batch[k].squeeze(0).squeeze(0)
            if image_batch[k].ndim == 0:
                image_batch[k] = image_batch[k].view(-1)
            # print(k, image_batch[k].shape)
        rays = posed_camera.get_pixelwise_rays(
            image_batch[FrameItemType.SensorId], image_batch[FrameItemType.Pose].to(self.device).float()
        )
        ray_bundle = RayBundle(
            origins=rays.origin,
            directions=rays.direction,
            pixel_area=None,
            camera_indices=image_batch[FrameItemType.SensorId],
        )
        return image_batch[FrameItemType.UniqueId], ray_bundle, image_batch

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
