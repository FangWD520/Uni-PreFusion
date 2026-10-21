import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), src) for src in sources],
    )

setup(
    name='unilion_mf_ops',
    packages=find_packages(include=['pcdet', 'pcdet.*']),
    ext_modules=[
        make_cuda_ext(
            name='ingroup_inds_cuda',
            module='pcdet.ops.ingroup_inds',
            sources=['src/ingroup_inds.cpp', 'src/ingroup_inds_kernel.cu'],
        ),
        make_cuda_ext(
            name='flattened_window_cuda',
            module='pcdet.ops.win_coors',
            sources=['src/flattened_window.cpp', 'src/flattened_window_kernel.cu'],
        ),
        make_cuda_ext(
            name='pointnet2_stack_cuda',
            module='pcdet.ops.pointnet2.pointnet2_stack',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/voxel_query.cpp',
                'src/voxel_query_gpu.cu',
                'src/vector_pool.cpp',
                'src/vector_pool_gpu.cu',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
