#!/usr/bin/env python3
import argparse
import copy
import os
import pickle


DET_KEYS = ['boxes_3d', 'scores_3d', 'labels_3d']
MOTION_KEYS = ['trajs_3d', 'trajs_cls_3d']


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser('Swap detection fields in a UniLION results.pkl.')
    parser.add_argument('--base', required=True,
                        help='Full-stack results.pkl to keep non-detection tasks from.')
    parser.add_argument('--det-source', required=True,
                        help='Detection-only or better-det results.pkl to take pts_bbox det fields from.')
    parser.add_argument('--out', required=True,
                        help='Output swapped results.pkl path.')
    parser.add_argument('--drop-motion', action='store_true',
                        help='Remove motion trajs if det-source has no motion outputs.')
    args = parser.parse_args()

    base = load(args.base)
    det_src = load(args.det_source)
    assert len(base) == len(det_src), f'Length mismatch: {len(base)} vs {len(det_src)}'

    merged = []
    for i, (base_item, det_item) in enumerate(zip(base, det_src)):
        assert 'pts_bbox' in base_item, f'base[{i}] has no pts_bbox'
        assert 'pts_bbox' in det_item, f'det_source[{i}] has no pts_bbox'
        out_item = copy.copy(base_item)
        out_bbox = dict(base_item['pts_bbox'])
        det_bbox = det_item['pts_bbox']
        for key in DET_KEYS:
            assert key in det_bbox, f'det_source[{i}][pts_bbox] missing {key}'
            out_bbox[key] = det_bbox[key]
        if args.drop_motion:
            for key in MOTION_KEYS:
                out_bbox.pop(key, None)
        out_item['pts_bbox'] = out_bbox
        merged.append(out_item)

    dump(merged, args.out)
    print(f'Saved swapped results to {args.out}')
    print(f'Num samples: {len(merged)}')
    print('Swapped keys:', DET_KEYS)
    if args.drop_motion:
        print('Dropped motion keys:', MOTION_KEYS)
    else:
        print('Kept non-det pts_bbox keys from base:', MOTION_KEYS)


if __name__ == '__main__':
    main()
