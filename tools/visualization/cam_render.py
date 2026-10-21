import os
import numpy as np
import cv2
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from tools.visualization.bev_render import (
    color_mapping,
    SCORE_THRESH,
    MAP_SCORE_THRESH,
    CMD_LIST
)


CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

class CamRender:
    def __init__(
        self,
        plot_choices,
        out_dir,
        save_separate_cams=False,
        save_grid=True,
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        self.separate_dir = os.path.join(self.pred_dir, "separate_cams")
        self.save_separate_cams = save_separate_cams
        self.save_grid = save_grid
        os.makedirs(self.pred_dir, exist_ok=True)
        if self.save_separate_cams:
            os.makedirs(self.separate_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(160 /3 , 20))
        plt.tight_layout()

    def render(
        self,
        data,
        result,
        index,
    ):
        self.reset_canvas()
        self.render_image_data(data, index)
        self.draw_detection_pred(data, result['pts_bbox'])
        # self.draw_motion_pred(data, result['pts_bbox'])
        self.draw_planning_pred(data, result['pts_plan_reg'])
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path)
        return save_path

    def render_pred_gt_overlay(self, data, result, index, name='Pred', pred_color=None):
        """Render one prediction source overlaid with GT in all camera views."""
        self.reset_canvas()
        self.render_image_data(data, index)

        pred_color = np.asarray(pred_color if pred_color is not None else [0.12, 0.35, 1.0])
        gt_color = np.asarray([0.0, 0.75, 0.0])
        pred_bbox = result.get('pts_bbox', result)
        pred_plan = result.get('pts_plan_reg', {})

        self.draw_detection_pred(data, pred_bbox, color=pred_color, linewidth=4)
        self.draw_detection_gt(data, color=gt_color, linewidth=4)
        self.draw_planning_pred(data, pred_plan, color=pred_color, s=150)
        self.draw_planning_gt(data, color=gt_color, s=120)
        self._render_overlay_legend(name, pred_color, gt_color)

        safe_name = str(name).replace(' ', '_').lower()
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + f'_{safe_name}_vs_gt.jpg')
        if self.save_grid:
            self.save_fig(save_path)
        if self.save_separate_cams:
            prefix = os.path.join(self.separate_dir, str(index).zfill(4) + f'_{safe_name}_vs_gt')
            self.save_cam_panels(prefix)
        return save_path

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index//3, index % 3]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def save_cam_panels(self, prefix):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for i, cam in enumerate(CAM_NAMES_NUSC):
            ax = self.get_axis(i)
            bbox = ax.get_window_extent(renderer).expanded(1.02, 1.02)
            out_path = f'{prefix}_{cam}.jpg'
            self.fig.savefig(out_path, bbox_inches=bbox.transformed(self.fig.dpi_scale_trans.inverted()), pad_inches=0)

    def render_image_data(self, data, index):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            img_path = data['img_filename'][idx]
            image = self.load_image(img_path, cam)
            self.update_image(image, i, cam)

    def draw_detection_pred(self, data, result, color=None, linewidth=4, alpha=1.0):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = np.concatenate([result['boxes_3d'].gravity_center.numpy(),
                                 result['boxes_3d'].tensor[..., 3:].numpy()], axis=-1)
        self._draw_boxes_on_cams(
            data,
            bboxes,
            result['labels_3d'],
            scores=result.get('scores_3d', None),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    def draw_detection_gt(self, data, color=None, linewidth=4, alpha=1.0):
        if not self.plot_choices['det']:
            return
        ann_info = data.get('ann_info', {})
        if 'gt_bboxes_3d' not in ann_info or 'gt_labels_3d' not in ann_info:
            return

        gt_boxes = ann_info['gt_bboxes_3d']
        if hasattr(gt_boxes, 'gravity_center'):
            bboxes = np.concatenate([gt_boxes.gravity_center.numpy(), gt_boxes.tensor[..., 3:].numpy()], axis=-1)
        else:
            bboxes = gt_boxes.tensor
            if hasattr(bboxes, 'numpy'):
                bboxes = bboxes.numpy()
        labels = ann_info['gt_labels_3d']
        self._draw_boxes_on_cams(
            data,
            bboxes,
            labels,
            scores=None,
            color=color if color is not None else np.asarray([0.0, 0.75, 0.0]),
            linewidth=linewidth,
            alpha=alpha,
        )

    def _draw_boxes_on_cams(self, data, bboxes, labels, scores=None, color=None, linewidth=4, alpha=1.0):
        labels = np.asarray(labels)
        if scores is not None:
            scores = np.asarray(scores)
        for j, cam in enumerate(CAM_NAMES_NUSC):
            cam_intrinsic = data['cams'][cam]['cam_intrinsic']
            extrinsic = data['cams'][cam]['lidar2cam']
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(labels.shape[0]):
                if labels[i] == -1:
                    continue
                if scores is not None and scores[i] < SCORE_THRESH:
                    continue
                draw_color = np.asarray(color if color is not None else color_mapping[labels[i] % len(color_mapping)])
                draw_color = tuple((draw_color * alpha).tolist())
                center = bboxes[i, 0:3]
                box_dims = bboxes[i, 3:6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(center, nusc_dims, quat)
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3],
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(draw_color, draw_color, draw_color),
                        linewidth=linewidth,
                    )

            self.axes[j // 3, j % 3].set_xlim(0, imsize[0])
            self.axes[j // 3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].tensor.numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cams'][cam]['cam_intrinsic']
            extrinsic = data['cams'][cam]['lidar2cam']
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH:
                    continue
                color = color_mapping[result['labels_3d'][i] % len(color_mapping)]

                traj = result['trajs_3d'][i].numpy()
                for t in range(len(traj)):
                    traj_t = traj[t]
                    origin = bboxes[i, :2][None]
                    traj_t = np.concatenate([origin, traj_t], axis=0)
                    traj_expand = np.ones((traj_t.shape[0], 1))
                    traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
                    traj_t = np.concatenate([traj_t, traj_expand], axis=1)

                    center = bboxes[i, 0 : 3]
                    box_dims = bboxes[i, 3 : 6]
                    nusc_dims = box_dims[..., [1, 0, 2]]
                    quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                    box = NuScenesBox(
                        center,
                        nusc_dims,
                        quat
                    )
                    box.rotate(rot)
                    box.translate(trans)
                    if not box_in_image(box, cam_intrinsic, imsize):
                        continue
                    traj_points = traj_t @ extrinsic[:3, :3] + trans
                    self._render_traj(traj_points, cam_intrinsic, [900, 1600], j, color=color, s=15)


    def draw_planning_pred(self, data, result, color=(1, 0.5, 0), s=150):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning']):
            return
        if 'plan_reg' not in result:
            return

        plan_traj = np.array(result['plan_reg'])
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        self._draw_planning_traj(data, plan_traj, color=color, s=s)

    def draw_planning_gt(self, data, color=(0.0, 0.75, 0.0), s=120):
        if not self.plot_choices['planning']:
            return
        ann_info = data.get('ann_info', {})
        if 'gt_ego_fut_masks' not in ann_info or 'gt_ego_fut_trajs' not in ann_info:
            return
        masks = ann_info['gt_ego_fut_masks'].astype(bool)
        if masks[0] == 0:
            return

        plan_traj = ann_info['gt_ego_fut_trajs'][masks]
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = plan_traj.cumsum(axis=0)
        plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
        self._draw_planning_traj(data, plan_traj, color=color, s=s)

    def _draw_planning_traj(self, data, plan_traj, color=(1, 0.5, 0), s=150):
        idx = 0  ## front camera
        cam = CAM_NAMES_NUSC_converter[idx]
        cam_intrinsic = data['cams'][cam]['cam_intrinsic']
        extrinsic = data['cams'][cam]['lidar2cam']
        trans = extrinsic[3, :3]
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)
        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        size = [900, 1600]
        self._render_traj(traj_points, cam_intrinsic, size, j=1, color=color, s=s)

    def _render_overlay_legend(self, pred_name, pred_color, gt_color):
        ax = self.axes[0, 1]
        ax.plot([], [], color=pred_color, linewidth=6, label=pred_name)
        ax.plot([], [], color=gt_color, linewidth=6, label='GT')
        ax.legend(loc='upper right', fontsize=26, framealpha=0.85)

    def _render_traj(self, traj_points, cam_intrinsic, size, j, color=(1, 0.5, 0), s=150, points_per_step=10):
        total_steps = (len(traj_points)-1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps-1):
            unit_vec = traj_points[k//points_per_step +
                                    1] - traj_points[k//points_per_step]
            total_xy[k] = (k/points_per_step - k//points_per_step) * \
                unit_vec + traj_points[k//points_per_step]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(
            total_xy.T, cam_intrinsic, normalize=True)[:2, :]

        traj_points = traj_points[:2, in_range_mask]

        in_img_mask = (traj_points[0, :] < size[1]) & (traj_points[1, :] < size[0])
        traj_points = traj_points[:2, in_img_mask]

        self.axes[j // 3, j % 3].scatter(traj_points[0], traj_points[1], color=color, s=s)