import os
import pickle

# Limit BLAS/OpenMP threads before importing numpy / sklearn.
# This keeps the original script logic unchanged while avoiding
# OpenBLAS thread over-allocation crashes during KMeans.
for env_name in (
    'OPENBLAS_NUM_THREADS',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
):
    os.environ.setdefault(env_name, '1')

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv


K = 4096

fp = 'data/nuscenes/nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
navi_trajs = [[], [], []]
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
    plan_mask = info['gt_ego_fut_masks']
    cmd = info['gt_ego_fut_cmd'].astype(np.int32)
    cmd = cmd.argmax(axis=-1)
    if not plan_mask.sum() == 6:
        continue
    navi_trajs[cmd].append(plan_traj)

clusters = []
for cmd_idx, trajs in enumerate(navi_trajs):
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    if trajs.shape[0] < K:
        raise RuntimeError(
            f'Not enough planning trajectories for command {cmd_idx}: '
            f'{trajs.shape[0]} samples, but K={K}'
        )
    cluster = KMeans(n_clusters=K, n_init=10).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
for j in range(K):
    plt.scatter(cluster[j, :, 0], cluster[j, :,1])

# plt.savefig(f'/data/zliu/vis/motion_planning_{K}', bbox_inches='tight')
# plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/nuscenes/kmeans_planning_{K}.npy', clusters)