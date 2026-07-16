<div align="center">
  
# Uni-DenseFusion

### Height-Aware Dense LiDAR-Camera Fusion for Unified Autonomous Driving

</div>

---
## 🔥 Overview
<p align="center">
  <img src="figures/overview.png" width="1000">
</p>

Most autonomous driving systems remain task-specific or cover only part of the perception-prediction-planning pipeline. Existing unified LiDAR-camera frameworks often fuse modalities after camera features are lifted into 3D space, allowing depth-induced spatial uncertainty to affect subsequent voxel reasoning. We propose Uni-PreFusion, a LiDAR-camera-temporal framework with a height-aware pre-lifting interface. It compresses sparse LiDAR voxels into height-aware pillar tokens and interacts them with dense multi-view image features before depth-based lifting. Refined image features are then lifted into 3D space, while refined pillar tokens are restored to voxel-compatible sparse features through residual broadcasting. This design preserves vertical LiDAR cues and supports sparse temporal reasoning for detection, map segmentation, occupancy prediction, motion forecasting, and planning. Experiments on the nuScenes validation set demonstrate gains in map segmentation, ray-wise occupancy prediction, and planning accuracy while using a lower image resolution and fewer computational resources. Overall, Uni-PreFusion improves unified 3D scene understanding and downstream planning with higher computational efficiency.

---
## 📊 Main Results on nuScenes Validation Set

| Modality | NDS ↑ | mAP ↑ | mIoU ↑ | RayIoU ↑ | Planning L2 ↓ |
|---|---|---|---|---|---|
| LC | 74.2 | 71.9 | 70.4 | 50.3 | 0.65 |
| LCT | 74.8 | 72.4 | 72.4 | 51.7 | 0.62 |

---

## 📌 News 
- [2026/05/28] Initial release of code and models.


---
# TODO List

- [ ] Release code
- [ ] Release checkpoints
- [ ] Release visualization tools
