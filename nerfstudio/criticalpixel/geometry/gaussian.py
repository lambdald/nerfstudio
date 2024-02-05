from tensordict import tensorclass
import torch
import pypose as pp


@tensorclass
class Gaussian2D:
    mean: torch.Tensor
    conv: torch.Tensor


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    # [N, 6] the upper triangular part of a matrix

    uncertaintyv2 = torch.empty(L.shape[:-2] + (6,), dtype=L.dtype, device=L.device)
    uncertaintyv2[..., :3] = L[..., 0, :3]
    uncertaintyv2[..., 3:5] = L[..., 1, 1:3]
    uncertaintyv2[..., 5] = L[..., 2, 2]
    return uncertaintyv2


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_scaling_rotation(s: torch.Tensor, r: pp.LieTensor) -> torch.Tensor:
    L = torch.diag_embed(s)  # [N, 3, 3]
    R = r.matrix()
    L = R @ L  # [N, 3, 3]
    return L


@tensorclass
class Gaussian3D:
    mean: torch.Tensor  # [N, 3]
    """'mean of 3d gaussian"""
    scale: torch.Tensor
    """scale of each axis [N, 3]"""
    rotation: pp.LieTensor
    """ SO3 [N, 4]"""

    @property
    def covariance(self) -> torch.Tensor:
        L = build_scaling_rotation(self.scale, self.rotation)
        covariance = L @ L.transpose(1, 2)
        return covariance

    @property
    def covariance_symmetric(self) -> torch.Tensor:
        cov = self.covariance
        symm = strip_symmetric(cov)
        return symm


if __name__ == "__main__":
    N = 1000
    mean = torch.rand((N, 3))
    s = torch.rand((N, 3))
    r = pp.randn_SO3(N, sigma=0.1)
    print(s.shape)
    print(r.shape)

    L = build_scaling_rotation(s, r)
    print(L)

    gaussian = Gaussian3D(batch_size=[N], mean=mean, scale=s, rotation=r, device="cuda")
    print(gaussian)

    print(gaussian.covariance)
    print(gaussian.covariance_symmetric)

    gaussian_slice = gaussian[:10].cpu()
    print(gaussian_slice)
