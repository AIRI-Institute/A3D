# Modified by the authors of the ICLR 2025 paper:
# "A3D: Does Diffusion Dream about 3D Alignment?"
# Based on MVDream-threestudio (https://github.com/bytedance/MVDream-threestudio)
# Licensed under the Apache License 2.0

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from threestudio.models.mesh import Mesh

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation, chunk_batch, scale_tensor
from threestudio.utils.typing import *


@threestudio.register("implicit-volume-parameterized")
class ImplicitVolumeParameterized(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        n_parameter_dims: int = 1
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.density_network = get_mlp(
            self.encoding.n_output_dims + self.cfg.n_parameter_dims,
            1,
            self.cfg.mlp_network_config
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims + self.cfg.n_parameter_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims + self.cfg.n_parameter_dims,
                3,
                self.cfg.mlp_network_config
            )

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Float[Tensor, "*N 1"] = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def _isosurface(self, bbox: Float[Tensor, "2 3"], fine_stage: bool = False, parameters=None) -> Mesh:
        def batch_func(x):
            # scale to bbox as the input vertices are in [0, 1]
            scale_x = scale_tensor(
                    x.to(bbox.device), self.isosurface_helper.points_range, bbox
                )
            batch_size = scale_x.shape[0]
            parameters_batch = parameters.repeat((batch_size, 1))
            # print(f"scale_x.shape = {scale_x.shape}, parameters.shape = {parameters_batch.shape}")

            field, deformation = self.forward_field(
                scale_x, parameters_batch
            )
            field = field.to(
                x.device
            )  # move to the same device as the input (could be CPU)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation

        assert self.isosurface_helper is not None

        field, deformation = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
        )

        threshold: float

        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            threestudio.info(
                f"Automatically determined isosurface threshold: {threshold}"
            )
        else:
            raise TypeError(
                f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}"
            )

        level = self.forward_level(field, threshold)
        mesh: Mesh = self.isosurface_helper(level, deformation=deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        mesh.add_extra("bbox", bbox)

        if self.cfg.isosurface_remove_outliers:
            # remove outliers components with small number of faces
            # only enabled when the mesh is not differentiable
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)

        return mesh

    def isosurface(self, parameters=None) -> Mesh:
        if not self.cfg.isosurface:
            raise NotImplementedError(
                "Isosurface is not enabled in the current configuration"
            )
        self._initilize_isosurface_helper()
        if self.cfg.isosurface_coarse_to_fine:
            threestudio.debug("First run isosurface to get a tight bounding box ...")
            with torch.no_grad():
                mesh_coarse = self._isosurface(self.bbox, parameters=parameters)
            vmin, vmax = mesh_coarse.v_pos.amin(dim=0), mesh_coarse.v_pos.amax(dim=0)
            vmin_ = (vmin - (vmax - vmin) * 0.1).max(self.bbox[0])
            vmax_ = (vmax + (vmax - vmin) * 0.1).min(self.bbox[1])
            threestudio.debug("Run isosurface again with the tight bounding box ...")
            mesh = self._isosurface(torch.stack([vmin_, vmax_], dim=0), fine_stage=True, parameters=parameters)
        else:
            mesh = self._isosurface(self.bbox, parameters=parameters)
        return mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, parameters=None
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        torch.set_grad_enabled(True)
        points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))

        # concatenatig parameters to the MLP input
        assert (parameters.shape[1] == self.cfg.n_parameter_dims) and (len(parameters.shape) == 2)
        enc = torch.cat([enc, parameters], dim=1)

        density = self.density_network(enc).view(*points.shape[:-1], 1)
        raw_density, density = self.get_activated_density(points_unscaled, density)

        output = {
            "density": density,
        }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(
                    density,
                    points_unscaled,
                    grad_outputs=torch.ones_like(density),
                    create_graph=True,
                )[0]
                normal = F.normalize(normal, dim=-1)
                if not grad_enabled:
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})

        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"], parameters=None) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        # concatenatig parameters to the MLP input
        assert (parameters.shape[1] == self.cfg.n_parameter_dims) and (len(parameters.shape) == 2)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        enc = torch.cat([enc, parameters], dim=1)

        density = self.density_network(
            enc
        ).reshape(*points.shape[:-1], 1)

        _, density = self.get_activated_density(points_unscaled, density)
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"], parameters=None
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points, parameters=parameters)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], parameters=None, **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))

        # concatenatig parameters to the MLP input
        assert (parameters.shape[1] == self.cfg.n_parameter_dims) and (len(parameters.shape) == 2)
        enc = torch.cat([enc, parameters], dim=1)

        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolumeParameterized":
        if isinstance(other, ImplicitVolumeParameterized):
            instance = ImplicitVolumeParameterized(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if (
                    instance.cfg.n_feature_dims > 0
                    and other.cfg.n_feature_dims == instance.cfg.n_feature_dims
                ):
                    instance.feature_network.load_state_dict(
                        other.feature_network.state_dict()
                    )
                if (
                    instance.cfg.normal_type == "pred"
                    and other.cfg.normal_type == "pred"
                ):
                    instance.normal_network.load_state_dict(
                        other.normal_network.state_dict()
                    )
            return instance
        else:
            raise TypeError(
                f"Cannot create {ImplicitVolumeParameterized.__name__} from {other.__class__.__name__}"
            )
