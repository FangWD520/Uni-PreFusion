import os
import os.path as osp
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from pyquaternion import Quaternion
try:
    import open3d as o3d
except ImportError:
    o3d = None

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

CMD_LIST = ['Turn Left', 'Turn Right', 'Go Straight']
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
color_mapping = np.asarray([
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255

MAP_PALETTE = np.asarray([
    [166, 206, 227],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [106, 61, 154],
])

occ_color_map = np.array([
    [0, 0, 0, 255],    # others
    [255, 120, 50, 255],  # barrier              orangey
    [255, 192, 203, 255],  # bicycle              pink
    [255, 255, 0, 255],  # bus                  yellow
    [0, 150, 245, 255],  # car                  blue
    [0, 255, 255, 255],  # construction_vehicle cyan
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [135, 60, 0, 255],  # trailer              brown
    [160, 32, 240, 255],  # truck                purple
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dard purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)

class BEVRender:
    def __init__(
        self,
        plot_choices,
        out_dir,
        xlim=40,
        ylim=40,
        motion_horizon=None,
        planning_horizon=None,
        occ_view='bev',
        occ_elev=62,
        occ_azim=-90,
        occ_marker_size=2.0,
        occ_alpha=0.9,
        occ_max_points=60000,
        occ_zlim=(-1.0, 5.4),
        occ_open3d_view_json=None,
        occ_open3d_use_car=True,
        occ_open3d_wait_time=0.1,
        occ_focus_range=42.0,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.motion_horizon = motion_horizon
        self.planning_horizon = planning_horizon
        self.occ_view = occ_view
        self.occ_elev = occ_elev
        self.occ_azim = occ_azim
        self.occ_marker_size = occ_marker_size
        self.occ_alpha = occ_alpha
        self.occ_max_points = occ_max_points
        self.occ_zlim = occ_zlim
        self.occ_open3d_view_json = occ_open3d_view_json
        self.occ_open3d_use_car = occ_open3d_use_car
        self.occ_open3d_wait_time = occ_open3d_wait_time
        self.occ_focus_range = occ_focus_range
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        if self.plot_choices.get('occ', False) and self.occ_view == 'oblique':
            self.fig = plt.figure(figsize=(20, 14))
            self.axes = self.fig.add_subplot(111, projection='3d')
            self.axes.view_init(elev=self.occ_elev, azim=self.occ_azim)
            self.axes.set_xlim(-self.xlim, self.xlim)
            self.axes.set_ylim(-self.ylim, self.ylim)
            self.axes.set_zlim(*self.occ_zlim)
            try:
                self.axes.set_box_aspect((2 * self.xlim, 2 * self.ylim,
                                          self.occ_zlim[1] - self.occ_zlim[0]))
            except AttributeError:
                pass
            self.axes.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.5)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.axes.set_zticks([])
            self.axes.set_xlabel('')
            self.axes.set_ylabel('')
            self.axes.set_zlabel('')
            self.axes.set_facecolor('white')
            return
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        self.axes.set_xlim(- self.xlim, self.xlim)
        self.axes.set_ylim(- self.ylim, self.ylim)
        self.axes.set_aspect('equal')
        self.axes.axis('off')

    def render(
        self,
        data,
        result,
        index,
    ):
        self.reset_canvas()

        if self.plot_choices['det'] or self.plot_choices['planning'] or self.plot_choices['motion']:
            self.draw_point(data)
            self._render_sdc_car()
        if self.plot_choices['det']:
            self.draw_detection_gt(data)
        if self.plot_choices['motion']:
            self.draw_motion_gt(data)
        if self.plot_choices['occ'] and self.occ_view == 'open3d':
            save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
            self.render_occ_open3d(self.load_occ_gt(data), save_path_gt)
        else:
            if self.plot_choices['occ']:
                self.draw_occ_gt(data)
            if self.plot_choices['map']:
                self.draw_map_gt(data)
        if self.plot_choices['planning']:
            self.draw_planning_gt(data)
            # self._render_command(data)
            # self._render_legend()

        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        if not (self.plot_choices['occ'] and self.occ_view == 'open3d'):
            self.save_fig(save_path_gt)
            self.reset_canvas()

        if self.plot_choices['det'] or self.plot_choices['planning'] or self.plot_choices['motion']:
            self.draw_point(data)
            self._render_sdc_car()
        if self.plot_choices['det']:
            self.draw_detection_pred(result['pts_bbox'])
        # self.draw_track_pred(result)
        if self.plot_choices['motion']:
            self.draw_motion_pred(result['pts_bbox'])
        if self.plot_choices['map']:
            self.draw_map_pred(result)
        if self.plot_choices['occ'] and self.occ_view == 'open3d':
            save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
            self.render_occ_open3d(self.load_occ_pred(result), save_path_pred)
        elif self.plot_choices['occ']:
            self.draw_occ_pred(result)
        if self.plot_choices['planning']:
            self.draw_planning_pred(data, result['pts_plan_reg'])
            # self._render_command(data)
            # self._render_legend()

        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        if not (self.plot_choices['occ'] and self.occ_view == 'open3d'):
            self.save_fig(save_path_pred)

        return save_path_gt, save_path_pred

    def render_det_motion_planning_overlay(self, data, baseline_result, ours_result, index):
        """Render baseline, ours and GT in a single BEV panel."""
        self.reset_canvas()
        self.draw_point(data)
        self._render_sdc_car()
        base_color = np.asarray([0.12, 0.35, 1.0])
        ours_color = np.asarray([0.95, 0.10, 0.10])
        gt_color = np.asarray([0.0, 0.70, 0.0])
        base_bbox = baseline_result.get('pts_bbox', baseline_result)
        ours_bbox = ours_result.get('pts_bbox', ours_result)
        base_plan = baseline_result.get('pts_plan_reg', {})
        ours_plan = ours_result.get('pts_plan_reg', {})
        self.draw_detection_pred(base_bbox, color=base_color, linestyle='--', linewidth=2.0, alpha=0.85)
        self.draw_detection_pred(ours_bbox, color=ours_color, linestyle='-', linewidth=2.2, alpha=0.9)
        self.draw_detection_gt(data, color=gt_color, linestyle='-', linewidth=2.4, alpha=0.85)
        self.draw_motion_pred(base_bbox, top_k=1, color=base_color, dot_size_scale=0.28, alpha=0.55)
        self.draw_motion_pred(ours_bbox, top_k=1, color=ours_color, dot_size_scale=0.28, alpha=0.60)
        self.draw_motion_gt(data, color=gt_color, dot_size_scale=0.23, alpha=0.55)
        self.draw_planning_pred(data, base_plan, color=base_color, dot_size=38, alpha=0.9)
        self.draw_planning_pred(data, ours_plan, color=ours_color, dot_size=48, alpha=0.95)
        self.draw_planning_gt(data, color=gt_color, dot_size=42, alpha=0.9)
        self.axes.plot([], [], color=base_color, linestyle='--', linewidth=3, label='Baseline')
        self.axes.plot([], [], color=ours_color, linestyle='-', linewidth=3, label='Ours')
        self.axes.plot([], [], color=gt_color, linestyle='-', linewidth=3, label='GT')
        self.axes.legend(loc='upper right', fontsize=24, framealpha=0.85)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '_overlay.jpg')
        self.save_fig(save_path)
        return save_path

    def render_det_motion_planning_pair_overlay(self, data, pred_result, index, name='pred', pred_color=None):
        """Render one prediction source against GT in a single BEV panel."""
        self.reset_canvas()
        self.draw_point(data)
        self._render_sdc_car()
        pred_color = np.asarray(pred_color if pred_color is not None else [0.12, 0.35, 1.0])
        gt_color = np.asarray([0.0, 0.70, 0.0])
        pred_bbox = pred_result.get('pts_bbox', pred_result)
        pred_plan = pred_result.get('pts_plan_reg', {})
        self.draw_detection_pred(pred_bbox, color=pred_color, linestyle='-', linewidth=2.2, alpha=0.9)
        self.draw_detection_gt(data, color=gt_color, linestyle='--', linewidth=2.2, alpha=0.85)
        self.draw_motion_pred(pred_bbox, top_k=1, color=pred_color, dot_size_scale=0.28, alpha=0.60)
        self.draw_motion_gt(data, color=gt_color, dot_size_scale=0.23, alpha=0.55)
        self.draw_planning_pred(data, pred_plan, color=pred_color, dot_size=46, alpha=0.95)
        self.draw_planning_gt(data, color=gt_color, dot_size=42, alpha=0.9)
        self.axes.plot([], [], color=pred_color, linestyle='-', linewidth=3, label=name)
        self.axes.plot([], [], color=gt_color, linestyle='--', linewidth=3, label='GT')
        self.axes.legend(loc='upper right', fontsize=24, framealpha=0.85)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + f'_{name}_vs_gt.jpg')
        self.save_fig(save_path)
        return save_path


    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        if getattr(self.axes, 'name', '') == '3d':
            self.axes.margins(x=0, y=0, z=0)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
        else:
            plt.margins(0, 0)
            plt.savefig(filename)

    def draw_point(self, data):
        lidar_path = data['pts_filename']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(
            data['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = data['lidar2ego_translation']

        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(
            data['ego2global_rotation']).rotation_matrix
        lidarego2global[:3, 3] = data['ego2global_translation']

        camego2global = np.eye(4, dtype=np.float32)
        camego2global[:3, :3] = Quaternion(
            data['cams']['CAM_FRONT']
            ['ego2global_rotation']).rotation_matrix
        camego2global[:3, 3] = data['cams']['CAM_FRONT'][
            'ego2global_translation']
        lidar2camego = np.linalg.inv(camego2global) @ lidarego2global @ lidar2lidarego

        points_ego = lidar2camego[:3, :3].reshape(1, 3, 3) @ \
                     points[:, :3].reshape(-1, 3, 1) + \
                     lidar2camego[:3, 3].reshape(1, 3, 1)
        points[:, :3] = points_ego.squeeze(-1)

        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        norm = Normalize(vmin=min(distances), vmax=max(distances))

        self.axes.scatter(
            points[:, 0],
            points[:, 1],
            s=1,
            alpha=0.5,
            c=distances, cmap='jet', norm=norm
        )

    def draw_detection_gt(self, data, color=None, linestyle='-', linewidth=3, alpha=1.0):
        for i in range(data['ann_info']['gt_labels_3d'].shape[0]):
            label = data['ann_info']['gt_labels_3d'][i]
            if label == -1:
                continue
            draw_color = np.asarray(color if color is not None else [0, 1.0, 0])

            # draw corners
            corners = box3d_to_corners(data['ann_info']['gt_bboxes_3d'].tensor)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=draw_color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=draw_color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    def draw_occ_gt(self, data):
        occ_path = data['ann_info']['occ_path']
        gt_occ = np.load(occ_path)
        gt_occ = gt_occ['semantics']
        if self.occ_view == 'oblique':
            self.draw_occ_3d(gt_occ)
        else:
            self.draw_occ_bev(gt_occ)

    def draw_occ_bev(self, occ):
        H, W, D = occ.shape

        free_id = len(occ_class_names) - 1
        semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

        for i in range(D):
            semantics_i = occ[..., i]
            non_free_mask = (semantics_i != free_id)
            semantics_2d[non_free_mask] = semantics_i[non_free_mask]

        viz = occ_color_map[semantics_2d]
        viz = viz[..., :3]
        self.axes.imshow(
            viz.transpose(1, 0, 2)[::-1, :, :],
            extent=(-self.xlim, self.xlim, -self.ylim, self.ylim),
            origin='lower', alpha=1.0, zorder=0)

    def load_occ_gt(self, data):
        occ_path = data['ann_info']['occ_path']
        gt_occ = np.load(occ_path)
        return gt_occ['semantics']

    def load_occ_pred(self, result):
        return np.asarray(result['pts_occ']['occ'])

    def _occ_to_points_colors(self, occ, ignore_labels=(17, 255)):
        occ = np.asarray(occ)
        H, W, D = occ.shape
        mask = np.ones_like(occ, dtype=bool)
        for label in ignore_labels:
            mask &= occ != label
        xs, ys, zs = np.where(mask)
        if xs.size == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        labels = occ[xs, ys, zs].astype(np.int64)
        if self.occ_max_points and xs.size > self.occ_max_points:
            keep = np.linspace(0, xs.size - 1, self.occ_max_points).astype(np.int64)
            xs, ys, zs, labels = xs[keep], ys[keep], zs[keep], labels[keep]
        z_min, z_max = self.occ_zlim
        points = np.stack([
            xs.astype(np.float32) / H * (2 * self.xlim) - self.xlim + self.xlim / H,
            ys.astype(np.float32) / W * (2 * self.ylim) - self.ylim + self.ylim / W,
            zs.astype(np.float32) / D * (z_max - z_min) + z_min + (z_max - z_min) / (2 * D),
        ], axis=1)
        colors = occ_color_map[np.clip(labels, 0, len(occ_color_map) - 1), :3].astype(np.float32) / 255.0
        return points, colors

    def _ego_car_points_colors(self):
        # Same idea as STCOCC: explicitly add a small voxelized ego car at the origin.
        ego_range = [-1.0, -1.0, 0.0, 3.0, 1.0, 1.5]
        ego_voxel = [0.2, 0.2, 0.2]
        xs = np.arange(ego_range[0] + ego_voxel[0] / 2, ego_range[3], ego_voxel[0])
        ys = np.arange(ego_range[1] + ego_voxel[1] / 2, ego_range[4], ego_voxel[1])
        zs = np.arange(ego_range[2] + ego_voxel[2] / 2, ego_range[5], ego_voxel[2])
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
        colors = np.tile(np.asarray([[0.02, 0.02, 0.02]], dtype=np.float32), (grid.shape[0], 1))
        return grid, colors

    def _make_ego_car_box(self):
        car_points, car_colors = self._ego_car_points_colors()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(car_points)
        pcd.colors = o3d.utility.Vector3dVector(car_colors)
        return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)

    def render_occ_open3d(self, occ, save_path):
        if o3d is None:
            print('[WARN] Open3D is not available; falling back to Matplotlib voxel-block OCC rendering.')
            self.render_occ_matplotlib_blocks(occ, save_path)
            return
        points, colors = self._occ_to_points_colors(occ)
        vis = o3d.visualization.Visualizer()
        window_ok = vis.create_window(width=1920, height=1080, visible=False)
        opt = vis.get_render_option() if window_ok else None
        if opt is None:
            print('[WARN] Open3D cannot create a rendering window in this environment; '
                  'falling back to Matplotlib voxel-block OCC rendering. '
                  'For native STCOCC rendering, run with a display/xvfb or install OSMesa-enabled Open3D.')
            try:
                vis.destroy_window()
            except Exception:
                pass
            self.render_occ_matplotlib_blocks(occ, save_path)
            return
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 4.0

        if points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            voxel_size = (2 * self.xlim) / occ.shape[0]
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            vis.add_geometry(voxel_grid)
            vis.add_geometry(pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(frame)
        if self.occ_open3d_use_car:
            vis.add_geometry(self._make_ego_car_box())

        ctr = vis.get_view_control()
        if self.occ_open3d_view_json and osp.exists(self.occ_open3d_view_json):
            param = o3d.io.read_pinhole_camera_parameters(self.occ_open3d_view_json)
            ctr.convert_from_pinhole_camera_parameters(param)
        else:
            # BEV-centered, slightly tilted view: similar to STCOCC but zoomed around ego.
            ctr.set_lookat([0.0, 0.0, 1.2])
            ctr.set_front([0.0, -0.42, 0.91])
            ctr.set_up([0.0, 0.91, 0.42])
            ctr.set_zoom(0.34)

        vis.poll_events()
        vis.update_renderer()
        if self.occ_open3d_wait_time and self.occ_open3d_wait_time > 0:
            import time
            end_time = time.time() + self.occ_open3d_wait_time
            while time.time() < end_time:
                vis.poll_events()
                vis.update_renderer()
        vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()

    def render_occ_matplotlib_blocks(self, occ, save_path):
        points, colors = self._occ_to_points_colors(occ)
        # Matplotlib cube rendering is slower than Open3D; cap fallback panels to keep it usable.
        fallback_max = 18000
        if points.shape[0] > fallback_max:
            keep = np.linspace(0, points.shape[0] - 1, fallback_max).astype(np.int64)
            points, colors = points[keep], colors[keep]
        plt.close()
        self.fig = plt.figure(figsize=(16, 9))
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.view_init(elev=self.occ_elev, azim=self.occ_azim)
        r = self.occ_focus_range
        self.axes.set_xlim(-r, r)
        self.axes.set_ylim(-r, r)
        self.axes.set_zlim(*self.occ_zlim)
        self.axes.set_box_aspect((2 * r, 2 * r, self.occ_zlim[1] - self.occ_zlim[0]))
        self.axes.set_facecolor('white')
        self.axes.axis('off')
        if points.shape[0] > 0:
            dx = (2 * self.xlim) / occ.shape[0] * 0.96
            dy = (2 * self.ylim) / occ.shape[1] * 0.96
            dz = (self.occ_zlim[1] - self.occ_zlim[0]) / occ.shape[2] * 0.96
            self.axes.bar3d(points[:, 0] - dx / 2, points[:, 1] - dy / 2, points[:, 2] - dz / 2,
                            dx, dy, dz, color=colors, shade=False, alpha=self.occ_alpha, linewidth=0)
        if self.occ_open3d_use_car:
            car_points, car_colors = self._ego_car_points_colors()
            self.axes.scatter(car_points[:, 0], car_points[:, 1], car_points[:, 2],
                              c=car_colors, marker='s', s=8, alpha=1.0, depthshade=False)
        self.save_fig(save_path)

    def draw_occ_3d(self, occ):
        occ = np.asarray(occ)
        H, W, D = occ.shape
        free_id = len(occ_class_names) - 1
        mask = occ != free_id
        xs, ys, zs = np.where(mask)
        if xs.size == 0:
            return
        labels = occ[xs, ys, zs].astype(np.int64)
        if self.occ_max_points and xs.size > self.occ_max_points:
            keep = np.linspace(0, xs.size - 1, self.occ_max_points).astype(np.int64)
            xs, ys, zs, labels = xs[keep], ys[keep], zs[keep], labels[keep]

        # Match the existing BEV extent: occupancy H/W are mapped to [-xlim, xlim]
        # and [-ylim, ylim], while D is shown as height in the configured z range.
        x = (xs.astype(np.float32) + 0.5) / H * (2 * self.xlim) - self.xlim
        y = (ys.astype(np.float32) + 0.5) / W * (2 * self.ylim) - self.ylim
        z = self.occ_zlim[0] + (zs.astype(np.float32) + 0.5) / D * (self.occ_zlim[1] - self.occ_zlim[0])
        colors = occ_color_map[np.clip(labels, 0, len(occ_color_map) - 1), :3] / 255.0
        self.axes.scatter(x, y, z, c=colors, s=self.occ_marker_size,
                          alpha=self.occ_alpha, marker='o', linewidths=0, depthshade=False)

    def draw_detection_pred(self, result, color=None, linestyle='-', linewidth=3, alpha=1.0):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = torch.cat([result['boxes_3d'].gravity_center, result['boxes_3d'].tensor[..., 3:]], dim=-1)
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            draw_color = np.asarray(color if color is not None else [0, 0, 1.0])

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=draw_color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=draw_color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    def draw_track_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['track'] and "anchor_queue" in result):
            return

        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[result['labels_3d'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data, color=None, dot_size_scale=1.0, alpha=1.0):
        if not self.plot_choices['motion']:
            return

        for i in range(data['ann_info']['gt_labels_3d'].shape[0]):
            label = data['ann_info']['gt_labels_3d'][i]
            if label == -1:
                continue
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            dot_size = 150 if label in vehicle_id_list else 25
            dot_size = max(4, dot_size * dot_size_scale)

            center = data['ann_info']['gt_bboxes_3d'].tensor[i, :2]
            masks = data['ann_info']['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['ann_info']['gt_agent_fut_trajs'][i][masks]
            if self.motion_horizon is not None:
                trajs = trajs[:self.motion_horizon]
            trajs = trajs.cumsum(axis=0) + center.numpy()
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            self._render_traj(trajs, traj_score=1.0,
                              colormap='winter', dot_size=dot_size, color=color, alpha=alpha)

    def draw_motion_pred(self, result, top_k=3, color=None, dot_size_scale=1.0, alpha=1.0):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].tensor
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]

            if score < SCORE_THRESH:
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            dot_size = 150 if label in vehicle_id_list else 25
            dot_size = max(4, dot_size * dot_size_scale)

            traj = result['trajs_3d'][i].numpy()
            if self.motion_horizon is not None:
                traj = traj[:, :self.motion_horizon]
            traj_cls = result['trajs_cls_3d'][i].sigmoid()
            order = np.argsort(traj_cls.numpy() if hasattr(traj_cls, 'numpy') else np.asarray(traj_cls))[::-1]
            drawn = 0
            for j in order:
                if traj_cls[j] < SCORE_THRESH:
                    continue
                center = bboxes[i, :2][None].repeat(1, 1).numpy()
                traj_j = np.concatenate([center, traj[j]], axis=0)
                self._render_traj(traj_j, traj_score=score.item(),
                                  colormap='winter', dot_size=dot_size, color=color, alpha=alpha)
                drawn += 1
                if top_k is not None and drawn >= top_k:
                    break


    # def draw_map_gt(self, data):
    #     vectors = data['map_infos']
    #     for label, vector_list in vectors.items():
    #         color = COLOR_VECTORS[label]
    #         for vector in vector_list:
    #             pts = vector[:, :2]
    #             x = np.array([pt[0] for pt in pts])
    #             y = np.array([pt[1] for pt in pts])
    #             self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_occ_pred(self, result):
        pred_occ = result['pts_occ']['occ']
        pred_occ = np.array(pred_occ)
        if self.occ_view == 'oblique':
            self.draw_occ_3d(pred_occ)
        else:
            self.draw_occ_bev(pred_occ)

    def draw_map_gt(self, data):
        gt_bev_masks = data['ann_info']['gt_bev_masks']
        gt_bev_masks = np.array(gt_bev_masks)

        canvas = np.zeros((*gt_bev_masks.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        for i in range(len(gt_bev_masks)):
            canvas[gt_bev_masks[i].astype(np.bool), :] = MAP_PALETTE[i]
        canvas = canvas[::-1, :, :]
        self.axes.imshow(
            canvas,
            extent=(-self.xlim, self.xlim, -self.ylim, self.ylim),
            origin='lower', alpha=1.0, zorder=1)

    # def draw_map_pred(self, result):
    #     if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result):
    #         return

    #     for i in range(len(result['scores'])):
    #         score = result['scores'][i]
    #         if score < MAP_SCORE_THRESH:
    #             continue
    #         color = COLOR_VECTORS[result['labels'][i]]
    #         pts = np.array(result['vectors'][i])
    #         x = pts[:, 0]
    #         y = pts[:, 1]
    #         plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "pts_seg_map" in result):
            return

        map_pred = result['pts_seg_map']['seg']
        map_pred = np.array(map_pred)

        canvas = np.zeros((*map_pred.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        for i in range(len(map_pred)):
            canvas[(map_pred[i] > 0.2).astype(np.bool), :] = MAP_PALETTE[i]
        canvas = canvas[::-1, :, :]
        self.axes.imshow(
            canvas,
            extent=(-self.xlim, self.xlim, -self.ylim, self.ylim),
            origin='lower', alpha=1.0, zorder=1)


    def draw_planning_gt(self, data, color=None, dot_size=50, alpha=1.0):
        # if not self.plot_choices['planning']:
        #     return

        # draw planning gt
        masks = data['ann_info']['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['ann_info']['gt_ego_fut_trajs'][masks]
            if self.planning_horizon is not None:
                plan_traj = plan_traj[:self.planning_horizon]
            cmd = data['gt_ego_fut_cmd']
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0,
                              colormap='autumn', dot_size=dot_size, color=color, alpha=alpha)

    def draw_planning_pred(self, data, result, top_k=3, color=None, dot_size=50, alpha=1.0):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning']):
            return

        plan_trajs = np.array(result['plan_reg'])
        if self.planning_horizon is not None:
            plan_trajs = plan_trajs[:self.planning_horizon]
        plan_trajs = np.concatenate((np.zeros((1, 2)), plan_trajs), axis=0)
        viz_traj = plan_trajs
        self._render_traj(viz_traj, traj_score=1, colormap='autumn', dot_size=dot_size, color=color, alpha=alpha)


    def _render_traj(
        self,
        future_traj,
        traj_score=1,
        colormap='winter',
        points_per_step=20,
        dot_size=25,
        color=None,
        alpha=1.0,
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        if color is None:
            dot_colors = matplotlib.colormaps[colormap](
                np.linspace(0, 1, total_steps))[:, :3]
            dot_colors = dot_colors * traj_score + \
                (1 - traj_score) * np.ones_like(dot_colors)
        else:
            dot_colors = np.tile(np.asarray(color).reshape(1, 3), (total_steps, 1))
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size, alpha=alpha)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        if sdc_car_png is None:
            x = [2, 2, -2, -2, 2]
            y = [1, -1, -1, 1, 1]
            self.axes.plot(x, y, color='black', linewidth=3)
            return
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png.transpose(1, 0, 2), extent=(2, -2, -1, 1))
        im.set_zorder(1)
        # W = 1.85
        # H = 4.084
        # x = [H / 2 + 0.5 + 0.985793, H / 2 + 0.5 + 0.985793, -H / 2 + 0.5 + 0.985793, -H / 2 + 0.5 + 0.985793, H / 2 + 0.5 + 0.985793]
        # y = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
        # self.axes.plot(x, y, color=[0, 0, 0], linewidth=3, linestyle='-')

    def _render_legend(self):
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)