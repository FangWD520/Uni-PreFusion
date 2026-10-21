import torch

from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds
from pcdet.ops.win_coors.flattened_window_cuda import (
    expand_selected_coords,
    flattened_window_mapping,
    get_window_coors_shift_v2,
    get_window_coors_shift_v3,
    map_points,
    map_points_v2,
)
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import three_nn


def main():
    print('ingroup_inds:', ingroup_inds)
    print('get_window_coors_shift_v2:', get_window_coors_shift_v2)
    print('get_window_coors_shift_v3:', get_window_coors_shift_v3)
    print('flattened_window_mapping:', flattened_window_mapping)
    print('expand_selected_coords:', expand_selected_coords)
    print('map_points:', map_points)
    print('map_points_v2:', map_points_v2)
    print('three_nn:', three_nn)

    if torch.cuda.is_available():
        group_inds = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device='cuda')
        try:
            out = ingroup_inds(group_inds)
        except RuntimeError:
            group_inds = group_inds.int()
            out = ingroup_inds(group_inds)
        print('ingroup_inds smoke:', out.detach().cpu().tolist())
    else:
        print('CUDA is not available; import-only check finished.')

    print('MambaFusion ops import check passed.')


if __name__ == '__main__':
    main()
