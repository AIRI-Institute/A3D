from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *
from cupyx.scipy.interpolate import RBFInterpolator

@threestudio.register("nerf-volume-parameterized-renderer")
class NeRFVolumeParameterizedRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False
        interpolation: Optional[dict] = None

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=32, levels=1
        )
        if not self.cfg.grid_prune:
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized

    @staticmethod
    def rbf_interpolation(points, anchors, values, function='linear', epsilon=1e-9):
        
        rbfi = RBFInterpolator(anchors, values, kernel=function, epsilon=epsilon)  # radial basis function interpolator instance
        X,Y,Z = points.unbind(dim=1)
        v = rbfi(points)
        v = v.clip(0, 1)
        
        return v

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        parameters = None,
        spatial_parameter_interpolation=None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        #Checking the dimensions of the parameters tensor
        assert len(parameters.shape) == 4 # B H W n_parameter_dims
        assert parameters.shape[0] in [1, batch_size]
        assert parameters.shape[1] in [1, height]
        assert parameters.shape[2] in [1, width] 
        assert parameters.shape[3] in [self.geometry.cfg.n_parameter_dims]
        parameters = parameters.to(rays_o.device)
        parameters_flatten = parameters.repeat(batch_size//parameters.shape[0],
                                               height//parameters.shape[1],
                                               width//parameters.shape[2],
                                               self.geometry.cfg.n_parameter_dims//parameters.shape[3])
        assert parameters_flatten.shape[:3] == rays_o.shape[:3]
        assert parameters_flatten.shape[3] == self.geometry.cfg.n_parameter_dims
        
        parameters_flatten = parameters_flatten.reshape(-1, self.geometry.cfg.n_parameter_dims)

        if spatial_parameter_interpolation is not None:
            if self.cfg.interpolation is None:
                print("No anchors points were provided, the scene will be devided in a half")
                anchors = None
                values = None
            else:
                anchors = []
                values = []
                assert len(self.cfg.interpolation.anchors.keys()) == len(self.cfg.interpolation.vals.keys())
                for point in self.cfg.interpolation.anchors.keys():
                    anchors.append(torch.tensor(self.cfg.interpolation.anchors[point]).reshape(-1,3))
                    values.append(torch.tensor(self.cfg.interpolation.vals[point]).reshape(-1,self.geometry.cfg.n_parameter_dims))
                anchors = torch.cat(anchors, axis=0)
                values = torch.cat(values, axis=0)
                
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            parameters_sampled = parameters_flatten[ray_indices]
            assert len(parameters_sampled.shape) == 2 and parameters_sampled.shape[0] == positions.shape[0] and parameters_sampled.shape[1] == self.geometry.cfg.n_parameter_dims

            # Intoducing spatial interpolation of the latent codes if the corresponding parameter is specified
            if spatial_parameter_interpolation is not None:
                if anchors is None or values is None:
                    positions_axis_of_interest = positions[:, :1]
                    positions_up_bound = positions_axis_of_interest.max()
                    positions_low_bound = positions_axis_of_interest.min()
                    parameters_sampled = parameters_sampled * (positions_axis_of_interest - positions_low_bound) / (positions_up_bound - positions_low_bound + 0.01)
                else:
                    parameters_sampled = torch.as_tensor(self.rbf_interpolation(positions, anchors, values)).to(positions.device).float()
                
                
        
            if self.training:
                sigma = self.geometry.forward_density(positions, parameters=parameters_sampled)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry.forward_density,
                    self.cfg.eval_chunk_size,
                    positions,
                    parameters=parameters_sampled
                )[..., 0]
            return sigma

        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
        else:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                )

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        parameters_sampled = parameters_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts
        assert len(parameters_sampled.shape) == 2 and parameters_sampled.shape[0] == positions.shape[0] and parameters_sampled.shape[1] == self.geometry.cfg.n_parameter_dims

        # Intoducing spatial interpolation of the latent codes if the corresponding parameter is specified
        if spatial_parameter_interpolation is not None:
            if anchors is None or values is None:
                positions_axis_of_interest = positions[:, :1]
                positions_up_bound = positions_axis_of_interest.max()
                positions_low_bound = positions_axis_of_interest.min()
                parameters_sampled = parameters_sampled * (positions_axis_of_interest - positions_low_bound) / (positions_up_bound - positions_low_bound + 0.01)
            else:
                parameters_sampled = torch.as_tensor(self.rbf_interpolation(positions, anchors, values)).to(positions.device).float()
        
        if self.training:
            geo_out = self.geometry(
                positions,
                output_normal=self.material.requires_normal,
                parameters=parameters_sampled
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
                parameters=parameters_sampled
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, _, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                        parameters=parameters_sampled
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                parameters = torch.rand([x.shape[0], self.geometry.cfg.n_parameter_dims]).to(x.device)
                density = self.geometry.forward_density(x, parameters=parameters)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size

            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
