"""
GaussianModel point-cloud initialization and PLY I/O.

The methods here convert between the in-memory Gaussian tensors and the PLY
format used for checkpoints and final outputs. Initial creation also sets up
per-camera exposure parameters.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch import nn

from utils.general_utils import RGB2SH, mkdir_p
from utils.graphics_utils import BasicPointCloud


PLY_WRITE_CHUNK_SIZE = 65_536


class GaussianModelIOMixin:
    """
        Mixin that creates Gaussian tensors from point clouds and reads/writes PLY files.
    """

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: list[Any], spatial_lr_scale: float) -> None:
        """
            Initialize Gaussian tensors from an input point cloud.

            Positions and colors come from the point cloud, scales are estimated
            from nearest-neighbor distance, rotations start as identity, and
            opacity/exposure parameters start from simple defaults.
        """
        from simple_knn._C import distCUDA2

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        if self.use_mcmc_initialization:
            scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)
        else:
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            float(self.initial_opacity) * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def construct_list_of_attributes(self) -> list[str]:
        """
            Build the ordered PLY attribute list for all saved Gaussian fields.
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # Save DC and higher-order SH coefficients as flat PLY columns.
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path: str) -> None:
        """
            Ghi Gaussian ra PLY binary theo chunk de gioi han peak RAM.
        """
        mkdir_p(os.path.dirname(path))
        point_count = int(self._xyz.shape[0])
        attribute_names = self.construct_list_of_attributes()
        element_dtype = np.dtype([(attribute, "<f4") for attribute in attribute_names])
        chunk_size = max(int(PLY_WRITE_CHUNK_SIZE), 1)
        chunk_count = max((point_count + chunk_size - 1) // chunk_size, 1)
        report_interval = max(chunk_count // 10, 1)
        temporary_path = "{}.tmp".format(path)
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex {}".format(point_count),
            *["property float {}".format(attribute) for attribute in attribute_names],
            "end_header",
            "",
        ]

        print(
            "Saving PLY {} points in {} chunks...".format(point_count, chunk_count),
            flush=True,
        )
        try:
            with open(temporary_path, "wb") as output:
                output.write("\n".join(header_lines).encode("ascii"))
                for chunk_index, start in enumerate(range(0, point_count, chunk_size), start=1):
                    end = min(start + chunk_size, point_count)
                    xyz = self._xyz[start:end].detach().cpu().numpy()
                    normals = np.zeros_like(xyz)
                    f_dc = (
                        self._features_dc[start:end]
                        .detach()
                        .transpose(1, 2)
                        .flatten(start_dim=1)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    f_rest = (
                        self._features_rest[start:end]
                        .detach()
                        .transpose(1, 2)
                        .flatten(start_dim=1)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    attributes = np.concatenate(
                        (
                            xyz,
                            normals,
                            f_dc,
                            f_rest,
                            self._opacity[start:end].detach().cpu().numpy(),
                            self._scaling[start:end].detach().cpu().numpy(),
                            self._rotation[start:end].detach().cpu().numpy(),
                        ),
                        axis=1,
                    )
                    elements = np.empty(end - start, dtype=element_dtype)
                    for attribute_index, attribute_name in enumerate(attribute_names):
                        elements[attribute_name] = attributes[:, attribute_index]
                    elements.tofile(output)
                    if chunk_index % report_interval == 0 or chunk_index == chunk_count:
                        print(
                            "  Saving PLY: {}/{} chunks".format(chunk_index, chunk_count),
                            flush=True,
                        )
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            raise
        print("Saving PLY complete: {}".format(path), flush=True)

    def load_ply(self, path: str, use_train_test_exp: bool = False) -> None:
        """
            Load raw Gaussian tensors from a saved PLY file.

            When train/test exposure is enabled, exposure matrices are loaded
            from the experiment-level `exposure.json` if it exists.
        """
        from plyfile import PlyData

        plydata = PlyData.read(path)
        vertices = plydata.elements[0]
        point_count = len(vertices.data)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        def parameter_from_numpy(values: np.ndarray) -> nn.Parameter:
            tensor = torch.from_numpy(values).to(device="cuda")
            return nn.Parameter(tensor.requires_grad_(True))

        # Moi lan chi tao mot mang float32 tren CPU, chuyen sang GPU roi giai phong ngay.
        xyz = np.empty((point_count, 3), dtype=np.float32)
        for index, attribute_name in enumerate(("x", "y", "z")):
            xyz[:, index] = np.asarray(vertices[attribute_name])
        self._xyz = parameter_from_numpy(xyz)
        del xyz

        features_dc = np.empty((point_count, 1, 3), dtype=np.float32)
        for channel in range(3):
            features_dc[:, 0, channel] = np.asarray(vertices["f_dc_{}".format(channel)])
        self._features_dc = parameter_from_numpy(features_dc)
        del features_dc

        extra_f_names = [p.name for p in vertices.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        coefficient_count = (self.max_sh_degree + 1) ** 2 - 1
        features_extra = np.empty((point_count, coefficient_count, 3), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            channel = idx // coefficient_count
            coefficient = idx % coefficient_count
            features_extra[:, coefficient, channel] = np.asarray(vertices[attr_name])
        self._features_rest = parameter_from_numpy(features_extra)
        del features_extra

        opacities = np.empty((point_count, 1), dtype=np.float32)
        opacities[:, 0] = np.asarray(vertices["opacity"])
        self._opacity = parameter_from_numpy(opacities)
        del opacities

        scale_names = [p.name for p in vertices.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.empty((point_count, len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertices[attr_name])
        self._scaling = parameter_from_numpy(scales)
        del scales

        rot_names = [p.name for p in vertices.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.empty((point_count, len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertices[attr_name])
        self._rotation = parameter_from_numpy(rots)
        del rots

        self.active_sh_degree = self.max_sh_degree
