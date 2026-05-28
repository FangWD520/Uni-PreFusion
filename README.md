<div align="center">
  
# Uni-DenseFusion

### Height-Aware Dense LiDAR-Camera Fusion for Unified Autonomous Driving

</div>

Unified autonomous driving requires a shared bird's-eye-view (BEV) and three-dimensional (3D) representation for object detection, map segmentation, occupancy prediction, motion forecasting, and planning. Existing Light Detection and Ranging (LiDAR)-camera frameworks often perform multimodal interaction after camera features are lifted into 3D space, where depth-induced spatial uncertainty can weaken downstream volumetric reasoning. We propose Uni-DenseFusion, a LiDAR-camera-temporal framework that introduces height-aware dense fusion before depth-based lifting. To bridge sparse 3D LiDAR features and dense image tokens, a sparse-dense adaptation interface compresses sparse LiDAR voxels into height-aware pillar tokens for dense image-LiDAR interaction, and restores the refined tokens to voxel-compatible sparse features through residual broadcasting. This design allows LiDAR geometry to guide image feature construction before view transformation while preserving sparse voxel coordinates for temporal reasoning. The refined multimodal representation is then processed by a sparse temporal backbone and task-specific heads for full-stack driving tasks. Experiments on the nuScenes validation set show that Uni-DenseFusion achieves competitive performance at a lower image resolution and improves both BEV-level and voxel-level scene understanding over the reproduced baseline.


![framework](figures/overview.png)
