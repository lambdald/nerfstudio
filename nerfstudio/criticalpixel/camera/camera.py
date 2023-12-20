from typing import Any
import torch
from enum import Enum

import torch
from tensordict import tensorclass
from enum import Enum
from dataclasses import dataclass
from typing import Union

from typing import Dict, Optional
import base64
from abc import abstractmethod
import torchvision
import cv2


@dataclass
class CameraAttribute:
    model_id: int
    model_name: str
    num_params: int

    def to_dict(self):
        data = dict(model_id=self.model_id, model_name=self.model_name, num_params=self.num_params)
        return data


class CameraModel(Enum):
    Unknown = CameraAttribute(model_id=0, model_name="unknown", num_params=0)

    SimpleRadial = CameraAttribute(model_id=1, model_name="simple_radial", num_params=4)
    Pinhole = CameraAttribute(model_id=2, model_name="pinhole", num_params=4)
    OpenCV = CameraAttribute(model_id=3, model_name="opencv", num_params=9)
    Fisheye = CameraAttribute(model_id=4, model_name="fisheye", num_params=8)
    Panoramic = CameraAttribute(model_id=5, model_name="panoramic", num_params=0)
    FoV = CameraAttribute(model_id=6, model_name="fov", num_params=2)


_camera_cls_ = {}


@tensorclass
class Camera:
    hws: torch.Tensor
    params: torch.Tensor
    model: CameraModel = CameraModel.Unknown

    # # @abstractmethod
    # # def projection_matrix(self, near: float, far: float) -> torch.Tensor:
    # #     raise NotImplementedError

    def __init_subclass__(cls) -> None:
        _camera_cls_[cls.model.value.model_id] = cls
        _camera_cls_[cls.model.value.model_name] = cls

    def to_dict(self) -> Dict:
        data = {"model": self.model.value.model_name, "hws": self.hws.tolist(), "params": self.params.tolist()}
        return data

    @abstractmethod
    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def projection_matrix(self, near: Union[torch.Tensor, float], far: Union[torch.Tensor, float]) -> torch.Tensor:
        raise NotImplementedError()

    def pixelwise_directions(self) -> torch.Tensor:
        uvs = self.pixelwise()  # batch_size + [h, w, 2]
        assert len(self.batch_size) == 1
        cam_indices = torch.arange(self.batch_size[0])
        cam_indices = cam_indices.view(-1, 1, 1).expand(uvs.shape[:-1]).long()
        cam = self[cam_indices]
        direction = cam.backproject_to_3d(uvs)
        return direction

    def is_same_size(self) -> bool:
        return torch.all(self.hws == self.hws[..., 0, :]).item()

    def pixelwise(self) -> torch.Tensor:
        if self.is_same_size():
            hw = self.hws[..., 0, :]
            h = hw[0].item()
            w = hw[1].item()
            xs = torch.linspace(0, w - 1, w, device=self.device)
            ys = torch.linspace(0, h - 1, h, device=self.device)
            uvs = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)  # [h, w, 2]

            uvs = uvs.view(len(self.batch_size) * [1] + [h, w, 2]).expand(self.batch_size + (h, w, 2))
            return uvs
        else:
            raise NotImplementedError("cameras have different image shape.")

    def to_json(
        self,
        camera_idx: int,
        pose_c2w: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        max_size: Optional[int] = None,
    ) -> Dict:
        params = self.params.flatten().tolist()

        assert self.model in [CameraModel.Fisheye, CameraModel.Pinhole, CameraModel.OpenCV]

        json_ = {
            "type": "PinholeCamera",
            "cx": params[2],
            "cy": params[3],
            "fx": params[0],
            "fy": params[1],
            "camera_to_world": pose_c2w[:3, :4].tolist(),
            "camera_index": camera_idx,
            "times": 0,
        }
        if image is not None:
            image_uint8 = (image * 255).detach().type(torch.uint8)
            if max_size is not None:
                image_uint8 = image_uint8.permute(2, 0, 1)
                image_uint8 = torchvision.transforms.functional.resize(image_uint8, max_size, antialias=None)  # type: ignore
                image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            data = cv2.imencode(".jpg", image_uint8)[1].tobytes()  # type: ignore
            json_["image"] = str("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        return json_


def create_camera(type: CameraModel, hws: torch.Tensor, params: torch.Tensor) -> Camera:
    camera_name = type.value.model_name
    batch_size = hws.shape[:-1]
    return _camera_cls_[camera_name](
        hws=hws, params=params, batch_size=batch_size, model=_camera_cls_[camera_name].model
    )


def create_camera_from_dict(data: Dict) -> Camera:
    model_name = data["model"]
    model = _camera_cls_[model_name].model
    hws = torch.tensor(data["hws"]).view(-1, 2)
    if model.value.num_params != 0:
        params = torch.tensor(data["params"]).view(-1, model.value.num_params)
        assert hws.shape[:-1] == params.shape[:-1]
    else:
        params = torch.tensor(hws.shape[:-1] + (0,))

    return create_camera(model, hws, params)


if __name__ == "__main__":
    hw = torch.tensor([2, 3])
    h = hw[0].item()
    w = hw[1].item()

    xs = torch.linspace(0, w - 1, w)
    ys = torch.linspace(0, h - 1, h)
    uvs = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)
    batch_size = [100, 5]
    uvs = uvs.view(len(batch_size) * [1] + [h, w, 2]).expand(batch_size + [h, w, 2])

    print(uvs.shape, uvs)
