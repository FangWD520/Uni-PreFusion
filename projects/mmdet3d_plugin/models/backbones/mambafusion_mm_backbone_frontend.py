import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from easydict import EasyDict
from mmdet.models import builder
from mmdet.models.builder import BACKBONES

from .mambafusion_stage12_frontend import MambaFusionStage12FrontEnd


def _to_easydict(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_easydict(v) for v in obj)
    return obj


def _default_mf_mm_cfg():
    return _to_easydict(dict(
        USE_WINMAMBA=True,
        USE_ALL_MAMBA=True,
        USE_MAMBA_INTER2=True,
        USE_VMAMBA_PRETRAIN=True,
        EMBED_TYPE='fixed',
        PATCH_EMBED=dict(in_channels=3, image_size=[256, 704], embed_dims=128, patch_size=8,
                         patch_norm=True, norm_cfg=dict(type='LN')),
        IMAGE_INPUT_LAYER=dict(sparse_shape=[32, 88, 1], d_model=[128], set_info=[[90, 2]],
                               window_shape=[[30, 30, 1]], hybrid_factor=[1, 1, 1],
                               shifts_list=[[[0, 0, 0], [15, 15, 0]]], input_image=True),
        LIDAR_INPUT_LAYER=dict(sparse_shape=[360, 360, 1], d_model=[128], set_info=[[90, 2]],
                               window_shape=[[30, 30, 1]], hybrid_factor=[1, 1, 1],
                               shifts_list=[[[0, 0, 0], [15, 15, 0]]]),
        set_info=[[90, 2]],
        d_model=[128],
        nhead=[8],
        dim_feedforward=[256],
        dropout=0.0,
        activation='gelu',
        checkpoint_blocks=[0, 1],
        layer_cfg=dict(use_bn=False, split_ffn=True, split_residual=True),
        FUSE_BACKBONE=dict(
            LIDAR2IMAGE=dict(
                block_start=1, block_end=2,
                point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                voxel_size=[0.3, 0.3, 8.0], sample_num=1,
                lidar2image_layer=dict(sparse_shape=[96, 264, 6], d_model=[128],
                                       set_info=[[90, 1]], window_shape=[[30, 30, 1]],
                                       hybrid_factor=[1, 1, 1],
                                       shifts_list=[[[0, 0, 0], [15, 15, 0]]],
                                       expand_max_voxels=30))),
        out_indices=[0, 1]))


class MambaFusionStateBank(nn.Module):
    """Checkpoint-compatible holder for raw MambaFusion mm_backbone tensors.

    The bank intentionally keeps the original MambaFusion key names in
    ``key_map`` and stores trainable tensors under sanitized ParameterDict names.
    This lets a UniLION model consume checkpoint keys like
    ``mm_frontend.mm_backbone.stage_0...`` without importing OpenPCDet.
    """

    def __init__(self, checkpoint_path=None, source_prefix='mm_backbone.'):
        super().__init__()
        self.source_prefix = source_prefix
        self.key_map = {}
        self.param_bank = nn.ParameterDict()
        self.buffer_bank = nn.ParameterDict()
        if checkpoint_path:
            self.init_from_checkpoint_shapes(checkpoint_path)

    @staticmethod
    def _safe_name(key):
        return key.replace('.', '__dot__')

    def init_from_checkpoint_shapes(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt.get('model_state', ckpt.get('state_dict', ckpt.get('model', ckpt)))
        for key, value in state.items():
            if not key.startswith(self.source_prefix) or not torch.is_tensor(value):
                continue
            raw_key = key[len(self.source_prefix):]
            safe_key = self._safe_name(raw_key)
            self.key_map[raw_key] = safe_key
            tensor = torch.empty_like(value, dtype=value.dtype)
            if value.dtype.is_floating_point or value.dtype.is_complex:
                self.param_bank[safe_key] = nn.Parameter(tensor)
            else:
                self.buffer_bank[safe_key] = nn.Parameter(tensor, requires_grad=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        consumed = []
        for raw_key, safe_key in self.key_map.items():
            src_key = prefix + raw_key
            if src_key not in state_dict:
                if strict:
                    missing_keys.append(src_key)
                continue
            value = state_dict[src_key]
            bank = self.param_bank if safe_key in self.param_bank else self.buffer_bank
            if safe_key not in bank:
                continue
            if tuple(bank[safe_key].shape) != tuple(value.shape):
                error_msgs.append(
                    f'size mismatch for {src_key}: checkpoint {tuple(value.shape)} vs model {tuple(bank[safe_key].shape)}')
                continue
            bank[safe_key].data.copy_(value.to(bank[safe_key].device, dtype=bank[safe_key].dtype))
            consumed.append(src_key)
        for key in consumed:
            state_dict.pop(key, None)

    def load_mambafusion_state(self, state, target_prefix=''):
        loaded, missing, mismatch = [], [], []
        for raw_key, safe_key in self.key_map.items():
            src_key = self.source_prefix + raw_key
            if src_key not in state:
                missing.append(src_key)
                continue
            value = state[src_key]
            bank = self.param_bank if safe_key in self.param_bank else self.buffer_bank
            if tuple(bank[safe_key].shape) != tuple(value.shape):
                mismatch.append([src_key, tuple(value.shape), tuple(bank[safe_key].shape)])
                continue
            bank[safe_key].data.copy_(value.to(bank[safe_key].device, dtype=bank[safe_key].dtype))
            loaded.append(target_prefix + raw_key)
        return loaded, missing, mismatch


@BACKBONES.register_module()
class MambaFusionMMBackboneFrontEnd(MambaFusionStage12FrontEnd):
    """UniLION-native MambaFusion mm_backbone migration scaffold.

    It keeps UniLION VFE/LSS/Lion3DBackbone/heads, accepts raw 256x704 images,
    and provides a checkpoint-compatible ``mm_backbone`` state bank for all
    ``checkpoint_epoch_10.pth::mm_backbone.*`` tensors. The runtime path uses a
    lightweight UniLION-compatible fusion fallback while the source-compatible
    state bank is available for full mm_backbone weight import and future exact
    operator replacement.
    """

    uses_raw_image = True

    def __init__(self,
                 img_in_channels=256,
                 pts_in_channels=128,
                 hidden_channels=128,
                 img_pool_stride=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 drop_path=0.0,
                 mf_checkpoint='ckpts/checkpoint_epoch_10.pth',
                 use_original_mm_backbone=False,
                 mf_mm_cfg=None,
                 fallback_on_error=True,
                 pillar_adapter='learned',
                 restoration_strategy='residual',
                 max_z_bins=64,
                 mf_img_backbone=None,
                 mf_img_neck=None):
        super().__init__(
            img_in_channels=img_in_channels,
            pts_in_channels=pts_in_channels,
            hidden_channels=hidden_channels,
            img_pool_stride=img_pool_stride,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path)
        self.use_original_mm_backbone = use_original_mm_backbone
        self.fallback_on_error = fallback_on_error
        self.pillar_adapter = pillar_adapter
        valid_restoration = ('residual', 'replace', 'none')
        if restoration_strategy not in valid_restoration:
            raise ValueError(f'restoration_strategy must be one of {valid_restoration}, got {restoration_strategy}')
        self.restoration_strategy = restoration_strategy
        self.max_z_bins = max_z_bins
        self.z_embed = nn.Embedding(max_z_bins, pts_in_channels)
        self.pillar_value = nn.Sequential(
            nn.LayerNorm(pts_in_channels),
            nn.Linear(pts_in_channels, pts_in_channels),
            nn.GELU(),
            nn.Linear(pts_in_channels, pts_in_channels))
        self.pillar_score = nn.Sequential(
            nn.LayerNorm(pts_in_channels),
            nn.Linear(pts_in_channels, 1))
        if self.pillar_adapter in ('mean', 'max'):
            for module in (self.z_embed, self.pillar_value, self.pillar_score):
                for param in module.parameters():
                    param.requires_grad = False
        elif self.pillar_adapter == 'learned_no_height':
            for param in self.z_embed.parameters():
                param.requires_grad = False
        self.mf_img_backbone = builder.build_backbone(mf_img_backbone) if mf_img_backbone else None
        if self.mf_img_backbone is not None and hasattr(self.mf_img_backbone, 'init_weights'):
            self.mf_img_backbone.init_weights()
        self.mf_img_neck = builder.build_backbone(mf_img_neck) if mf_img_neck else None
        self.mm_backbone_runtime = None
        if use_original_mm_backbone:
            self.mm_backbone = self._build_original_mm_backbone(mf_mm_cfg, mf_checkpoint)
        else:
            self.mm_backbone = MambaFusionStateBank(mf_checkpoint)
        self.raw_img_stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, img_in_channels, 1))
        if use_original_mm_backbone and not fallback_on_error:
            self._freeze_fallback_modules()


    def _freeze_fallback_modules(self):
        fallback_names = [
            'img_to_hidden', 'hidden_to_img',
            'stage1_img_norm', 'stage2_img_norm',
            'stage1_img_block', 'stage2_img_block',
            'stage1_img_cross', 'stage2_img_cross',
            'stage1_pts_cross', 'stage2_pts_cross',
            'raw_img_stem',
        ]
        for name in fallback_names:
            module = getattr(self, name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def _build_original_mm_backbone(self, mf_mm_cfg, mf_checkpoint):
        from pcdet.models.mm_backbone.mambafusion import MambaFusion
        cfg = _to_easydict(mf_mm_cfg) if mf_mm_cfg is not None else _default_mf_mm_cfg()
        if self.mf_img_backbone is not None:
            cfg.USE_PREBACKBONE = True
            cfg.USE_VMAMBA_PRETRAIN = False
        self._ensure_hilbert_templates(mf_checkpoint)
        cwd = os.getcwd()
        pcdet_dir = str(Path(__file__).resolve().parents[4] / 'pcdet')
        try:
            os.chdir(pcdet_dir)
            module = MambaFusion(cfg)
        finally:
            os.chdir(cwd)
        return module

    def _ensure_hilbert_templates(self, mf_checkpoint):
        root = Path(__file__).resolve().parents[4]
        dst = root / 'ckpts' / 'hilbert_template'
        dst.mkdir(parents=True, exist_ok=True)
        candidates = []
        if mf_checkpoint:
            ckpt = Path(mf_checkpoint)
            if not ckpt.is_absolute():
                ckpt = root / ckpt
            candidates.append(ckpt.parent / 'hilbert_template')
        candidates += [
            root.parent / 'MambaFusion_orin' / 'ckpts' / 'hilbert_template',
            root.parent / 'MambaFusion_our' / 'ckpts' / 'hilbert_template',
        ]
        for rank in [7, 8, 9, 10]:
            name = f'curve_template_3d_rank_{rank}.pth'
            target = dst / name
            if target.exists():
                continue
            for src_dir in candidates:
                src = src_dir / name
                if src.exists():
                    target.write_bytes(src.read_bytes())
                    break

    @staticmethod
    def _compose_lidar2image(img_inputs):
        sensor2keyegos, _, intrins, post_rots, post_trans, _ = img_inputs
        b, n = sensor2keyegos.shape[:2]
        lidar2cam = torch.inverse(sensor2keyegos.float())
        lidar2image = lidar2cam.new_zeros(b, n, 4, 4)
        lidar2image[..., 3, 3] = 1
        lidar2image[..., :3, :3] = intrins.float().matmul(lidar2cam[..., :3, :3])
        lidar2image[..., :3, 3] = intrins.float().matmul(lidar2cam[..., :3, 3:4]).squeeze(-1)
        img_aug_matrix = lidar2cam.new_zeros(b, n, 4, 4)
        img_aug_matrix[..., 3, 3] = 1
        img_aug_matrix[..., :3, :3] = post_rots.float()
        img_aug_matrix[..., :3, 3] = post_trans.float()
        return lidar2image, img_aug_matrix


    def _pillar_pool(self, voxel_features, voxel_coords):
        """Pool UniLION 3D voxels into MambaFusion pillar tokens.

        The default learned adapter is height-aware: it adds a learnable z
        embedding to every voxel, predicts per-voxel values and scores, then
        applies a softmax-weighted sum inside each (batch, y, x) pillar.
        """
        pillar_keys = torch.stack(
            [voxel_coords[:, 0], voxel_coords[:, 2], voxel_coords[:, 3]], dim=1).long()
        unique_keys, inverse = torch.unique(pillar_keys, dim=0, sorted=True, return_inverse=True)
        num_pillars = unique_keys.shape[0]

        pillar_coords = voxel_coords.new_zeros((num_pillars, 4))
        pillar_coords[:, 0] = unique_keys[:, 0]
        pillar_coords[:, 2] = unique_keys[:, 1]
        pillar_coords[:, 3] = unique_keys[:, 2]

        if self.pillar_adapter == 'mean':
            pillar_features = torch_scatter.scatter_mean(voxel_features, inverse, dim=0, dim_size=num_pillars)
            return pillar_features, pillar_coords.int(), inverse
        if self.pillar_adapter == 'max':
            pillar_features = torch_scatter.scatter_max(voxel_features, inverse, dim=0, dim_size=num_pillars)[0]
            return pillar_features, pillar_coords.int(), inverse

        if self.pillar_adapter == 'learned_no_height':
            height_aware = voxel_features
        else:
            z_ids = voxel_coords[:, 1].long().clamp(min=0, max=self.max_z_bins - 1)
            height_aware = voxel_features + self.z_embed(z_ids).to(voxel_features.dtype)
        values = self.pillar_value(height_aware)
        scores = self.pillar_score(height_aware).squeeze(-1)
        max_scores = torch_scatter.scatter_max(scores, inverse, dim=0, dim_size=num_pillars)[0]
        weights = torch.exp(scores - max_scores[inverse])
        denom = torch_scatter.scatter_add(weights, inverse, dim=0, dim_size=num_pillars).clamp_min(1e-6)
        weighted_values = values * (weights / denom[inverse]).unsqueeze(-1)
        pillar_features = torch_scatter.scatter_add(weighted_values, inverse, dim=0, dim_size=num_pillars)
        return pillar_features, pillar_coords.int(), inverse

    def _broadcast_pillar_features(self, pillar_features, inverse, base_features):
        if self.restoration_strategy == 'none' or pillar_features.shape[0] == 0:
            return base_features
        broadcast_features = pillar_features[inverse].to(base_features.dtype)
        if self.restoration_strategy == 'replace':
            return broadcast_features
        return base_features + broadcast_features

    def _run_original_mm_backbone(self, img, pts_voxel_features, pts_voxel_coords, img_inputs):
        assert img is not None and img_inputs is not None
        b = img.shape[0]
        lidar2image, img_aug_matrix = self._compose_lidar2image(img_inputs)
        pillar_features, pillar_coords, pillar_inverse = self._pillar_pool(
            pts_voxel_features, pts_voxel_coords)
        camera_imgs = F.interpolate(
            img.reshape(-1, img.shape[2], img.shape[3], img.shape[4]).float(),
            size=self.mm_backbone.image_shape, mode='bilinear', align_corners=False).reshape(
                b, img.shape[1], img.shape[2], self.mm_backbone.image_shape[0], self.mm_backbone.image_shape[1])
        batch_dict = dict(
            camera_imgs=camera_imgs,
            voxel_features=pillar_features,
            voxel_coords=pillar_coords,
            ori_voxel_coords=pts_voxel_coords,
            pillar_inverse=pillar_inverse,
            batch_size=b,
            lidar2image=lidar2image,
            img_aug_matrix=img_aug_matrix,
        )
        if self.mf_img_backbone is not None:
            batch_dict['backbone_features'] = self.mf_img_backbone(camera_imgs)
        out = self.mm_backbone(batch_dict)
        raw_image_features = list(out.get('image_features', []))
        if self.mf_img_neck is not None and raw_image_features:
            merged_feature = self.mf_img_neck(raw_image_features)
            feature_maps = [merged_feature.reshape(
                b, img.shape[1], merged_feature.shape[1], merged_feature.shape[2], merged_feature.shape[3])]
        else:
            feature_maps = [feat.reshape(b, img.shape[1], feat.shape[1], feat.shape[2], feat.shape[3])
                            for feat in raw_image_features]
            # UniLION LSSViewTransformer expects exactly one image feature map before
            # the six geometric img_inputs. MambaFusion may emit multiple out_indices;
            # keep the last/deepest fused map for LSS when no neck is used.
            if len(feature_maps) > 1:
                feature_maps = [feature_maps[-1]]
            if feature_maps and feature_maps[0].shape[2] != self.hidden_to_img.out_features:
                feature_maps[0] = feature_maps[0].repeat_interleave(
                    self.hidden_to_img.out_features // feature_maps[0].shape[2], dim=2)
        refined_pillar_features = out.get('voxel_features', out.get('pillar_features', pillar_features))
        if refined_pillar_features.shape[0] == pillar_features.shape[0]:
            refined_features = self._broadcast_pillar_features(
                refined_pillar_features, pillar_inverse, pts_voxel_features)
            refined_coords = pts_voxel_coords
        else:
            refined_features = refined_pillar_features
            refined_coords = out.get('voxel_coords', pillar_coords)
        return dict(
            feature_maps=feature_maps if feature_maps else None,
            pts_voxel_features=refined_features,
            pts_voxel_coords=refined_coords,
        )

    def image_stem(self, img):
        b, n, c, h, w = img.shape
        feat = self.raw_img_stem(img.reshape(b * n, c, h, w))
        return [feat.reshape(b, n, feat.shape[1], feat.shape[2], feat.shape[3])]

    def forward(self, feature_maps=None, pts_voxel_features=None, pts_voxel_coords=None,
                img=None, img_inputs=None, img_metas=None):
        if self.use_original_mm_backbone:
            try:
                out = self._run_original_mm_backbone(img, pts_voxel_features, pts_voxel_coords, img_inputs)
                if out['feature_maps'] is not None:
                    return out
            except Exception:
                if not self.fallback_on_error:
                    raise
        if feature_maps is None:
            assert img is not None, 'raw img is required when feature_maps is None'
            feature_maps = self.image_stem(img)
        return super().forward(
            feature_maps=feature_maps,
            pts_voxel_features=pts_voxel_features,
            pts_voxel_coords=pts_voxel_coords,
            img_inputs=img_inputs,
            img_metas=img_metas)
