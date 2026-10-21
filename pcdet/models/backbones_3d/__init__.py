from .dsvt import DSVT
from .lion_backbone_one_stride import LION3DBackboneOneStride, LION3DBackboneOneStride_Sparse

try:
    from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
    from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
    from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
    from .spconv_backbone_focal import VoxelBackBone8xFocal
    from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
    from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
    from .spconv_unet import UNetV2
except ImportError:
    PointNet2Backbone = PointNet2MSG = None
    VoxelBackBone8x = VoxelResBackBone8x = None
    PillarBackBone8x = PillarRes18BackBone8x = None
    VoxelBackBone8xFocal = None
    VoxelResBackBone8xVoxelNeXt = VoxelResBackBone8xVoxelNeXt2D = None
    UNetV2 = None

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'DSVT': DSVT,
    'LION3DBackboneOneStride': LION3DBackboneOneStride,
    'LION3DBackboneOneStride_Sparse': LION3DBackboneOneStride_Sparse,
}
