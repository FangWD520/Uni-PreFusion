import os
from os import path as osp
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

plot_choices = dict(
    draw_pred=True,  # True: draw gt and pred; False: only draw gt
    det=False,
    occ=True,
    track=False,  # True: draw history tracked boxes
    motion=False,
    map=False,
    planning=False,
)
START = 0
END = 300
INTERVAL = 10


class Visualizer:
    def __init__(
            self,
            args,
            plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)

        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)

        self.results = mmcv.load(args.result_path)

        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        token = data['token']

        result = self.results[index]

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)

    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width // downsample, height //
                                   downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', default=None,
                        help='prediction result to visualize')
    parser.add_argument('--out-dir', default='vis',
                        help='directory where results will be saved')
    parser.add_argument('--full-stack', action='store_true',
                        help='Draw det, occ, motion, map and planning together.')
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--occ', action='store_true')
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--map', action='store_true')
    parser.add_argument('--planning', action='store_true')
    parser.add_argument('--no-pred', action='store_true', help='Only draw GT.')
    parser.add_argument('--indices', nargs='+', type=int, default=None,
                        help='Specific sample indices to visualize.')
    parser.add_argument('--start', type=int, default=START)
    parser.add_argument('--end', type=int, default=END)
    parser.add_argument('--interval', type=int, default=INTERVAL)
    parser.add_argument('--no-video', action='store_true')
    return parser.parse_args()


def resolve_plot_choices(args):
    choices = dict(plot_choices)
    if args.full_stack:
        choices.update(dict(det=True, occ=True, motion=True, map=True, planning=True))
    else:
        choices['det'] = choices['det'] or args.det
        choices['occ'] = choices['occ'] or args.occ
        choices['motion'] = choices['motion'] or args.motion
        choices['map'] = choices['map'] or args.map
        choices['planning'] = choices['planning'] or args.planning
    if args.no_pred:
        choices['draw_pred'] = False
    return choices


def main():
    args = parse_args()
    choices = resolve_plot_choices(args)
    visualizer = Visualizer(args, choices)

    indices = args.indices if args.indices is not None else range(args.start, args.end, args.interval)
    for idx in tqdm(indices):
        if idx >= len(visualizer.results):
            break
        visualizer.add_vis(idx)

    if not args.no_video:
        visualizer.image2video()


if __name__ == '__main__':
    main()
