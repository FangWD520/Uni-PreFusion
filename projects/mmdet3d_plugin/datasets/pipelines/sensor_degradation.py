import numpy as np
import torch
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SensorDegradation(object):
    """Test-time sensor degradation for robustness evaluation.

    Modes:
      - missing_front_camera: zero CAM_FRONT image and its projected depth map.
      - lidar_dropout: randomly drop a ratio of points before depth-map generation.
      - camera_lidar_misalignment: perturb camera extrinsic translations used by
        LSS/MambaFusion, while keeping labels/evaluation unchanged.
    """

    def __init__(self,
                 mode,
                 cam_name='CAM_FRONT',
                 cam_names=None,
                 dropout_ratio=0.5,
                 trans_shift=(0.5, 0.0, 0.0),
                 yaw_deg=0.0,
                 seed=0):
        self.mode = mode
        self.cam_name = cam_name
        self.cam_names = cam_names if cam_names is not None else [cam_name]
        self.dropout_ratio = dropout_ratio
        self.trans_shift = np.array(trans_shift, dtype=np.float32)
        self.yaw_deg = yaw_deg
        self.seed = seed

    def _cam_indices(self, results):
        if 'cam_names' not in results:
            return [0]
        return [results['cam_names'].index(name) for name in self.cam_names
                if name in results['cam_names']]

    def _dropout_points(self, results):
        points = results['points']
        tensor = points.tensor
        n = tensor.shape[0]
        if n == 0 or self.dropout_ratio <= 0:
            return results
        keep_ratio = max(0.0, min(1.0, 1.0 - self.dropout_ratio))
        g = torch.Generator(device=tensor.device)
        token = str(results.get('sample_idx', ''))
        token_hash = sum(ord(c) for c in token) % 1000003
        g.manual_seed(self.seed + token_hash)
        mask = torch.rand(n, generator=g, device=tensor.device) < keep_ratio
        if mask.sum() == 0:
            mask[torch.randint(0, n, (1,), generator=g, device=tensor.device)] = True
        results['points'] = points.new_point(tensor[mask])
        return results

    def _zero_like(self, x):
        if torch.is_tensor(x):
            return torch.zeros_like(x)
        if isinstance(x, np.ndarray):
            return np.zeros_like(x)
        return x * 0

    def _missing_front_camera(self, results):
        for cid in self._cam_indices(results):
            if 'img' in results:
                if isinstance(results['img'], list):
                    results['img'][cid] = self._zero_like(results['img'][cid])
                elif torch.is_tensor(results['img']) or isinstance(results['img'], np.ndarray):
                    results['img'][cid] = self._zero_like(results['img'][cid])
            if 'canvas' in results and isinstance(results['canvas'], list):
                results['canvas'][cid] = np.zeros_like(results['canvas'][cid])
            if 'gt_depth' in results:
                results['gt_depth'][cid] = self._zero_like(results['gt_depth'][cid])
        return results

    def _misalign(self, results):
        # img_inputs are [sensor2egos, ego2globals, intrins, post_rots,
        # post_trans, bda]. This transform is inserted after
        # PointToMultiViewDepth, so projected gt_depth and labels remain clean;
        # only the model's camera-LiDAR geometry is perturbed.
        if 'img_inputs' in results:
            inputs = list(results['img_inputs'])
            sensor2egos = inputs[0].clone()
            shift = torch.as_tensor(self.trans_shift, dtype=sensor2egos.dtype, device=sensor2egos.device)
            trans = sensor2egos[..., :3, 3]
            view_shape = [1] * trans.dim()
            view_shape[-1] = 3
            sensor2egos[..., :3, 3] = trans + shift.view(*view_shape)
            if abs(self.yaw_deg) > 1e-6:
                yaw = torch.as_tensor(self.yaw_deg * np.pi / 180.0, dtype=sensor2egos.dtype, device=sensor2egos.device)
                c, s = torch.cos(yaw), torch.sin(yaw)
                rot = sensor2egos.new_tensor([[1., 0., 0.], [0., c, -s], [0., s, c]])
                sensor2egos[..., :3, :3] = sensor2egos[..., :3, :3].matmul(rot)
            inputs[0] = sensor2egos
            results['img_inputs'] = tuple(inputs)
        return results

    def __call__(self, results):
        if self.mode == 'missing_front_camera':
            return self._missing_front_camera(results)
        if self.mode == 'lidar_dropout':
            return self._dropout_points(results)
        if self.mode == 'camera_lidar_misalignment':
            return self._misalign(results)
        raise ValueError(f'Unknown SensorDegradation mode: {self.mode}')
