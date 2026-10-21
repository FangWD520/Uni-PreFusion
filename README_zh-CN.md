# Uni-PreFusion

**面向统一自动驾驶的高度感知提升前激光雷达与相机融合前端**

[English](README.md) · [环境安装](docs/INSTALL.md) · [数据准备](docs/data_preparation.md) · [模型下载](docs/checkpoints.md) · [训练说明](docs/training.md)

论文题目：*Uni-PreFusion: A Height-Aware Pre-Lifting LiDAR-Camera Fusion Frontend for Unified Autonomous Driving*

作者：Xuerui Fang、Gulimila Kezierbieke、Chunyang Chen、Congcong Bian、Zehao Xue、Yeerjiang Halimu、Hui Li。

单位：新疆农业大学、江南大学。

## 研究思路

相机特征先提升到三维空间、再进行融合时，深度估计误差已经影响了特征的空间位置。Uni-PreFusion 将激光雷达与图像的交互提前到深度提升之前，让几何信息先参与图像特征的细化。

![整体架构](figures/overview.png)

1. **高度感知的体素到柱聚合。** 将稀疏激光雷达体素压缩为柱 token，通过可学习的高度编码和柱内加权聚合保留垂直结构信息。
2. **提升前跨模态交互。** 在 MambaFusion 风格的交互骨干中融合柱 token 与多视角图像特征，在图像特征进入三维空间之前引入激光雷达几何信息。
3. **残差恢复到原体素。** 将融合后的柱上下文广播回原始稀疏体素，并通过残差连接保留原体素表示及坐标。
4. **统一时序多任务后端。** 将细化后的图像特征提升到三维，与稀疏激光雷达特征共同输入基于 UniLION 的时序后端，支持三维检测、BEV 地图分割、占据预测、运动预测和自车规划。

本工作的重点是高度感知的提升前接口及其集成。UniLION 后端、MambaFusion 交互模块与 VMamba 图像编码器均为继承组件，不作为本工作新设计的算子。

## 论文报告结果

以下为论文主表中的 nuScenes 验证集结果，图像分辨率为 **704 × 256**，不代表本次代码整理重新复现了全部任务。L、C、T 分别表示激光雷达、相机和时序信息。

| 输入 | NDS ↑ | 检测 mAP ↑ | 地图 mIoU ↑ | 占据 RayIoU ↑ |
|:--|--:|--:|--:|--:|
| LC | 74.2 | 71.9 | 70.4 | 50.2 |
| LCT | 74.8 | 72.4 | 72.4 | 51.7 |

| 输入 | 车辆 minADE ↓（米） | 行人 minADE ↓（米） | 规划 L2 ↓（米） | 碰撞率 ↓（%） |
|:--|--:|--:|--:|--:|
| LC | 0.61 | 0.38 | 0.65 | 0.23 |
| LCT | 0.57 | 0.36 | 0.62 | 0.23 |

模型约 **81.7M 参数**。规划遵循论文中的开环协议，不额外输入自车状态或导航指令。本仓库不宣称论文已被接收或发表。

## 使用方式

先按[安装说明](docs/INSTALL.md)构建环境，再执行：

```bash
python tools/download_models.py
python tools/check_release.py
```

完整 LCT 权重与四个必需的 Hilbert 查表文件通过 [GitHub Releases](https://github.com/FangWD520/Uni-PreFusion/releases/tag/v1.0.0) 提供，下载工具校验文件大小和 SHA-256。两个任务 anchor 数组随代码提供。数据集及其标注需要按各自授权自行获取，详见[数据准备](docs/data_preparation.md)。

```bash
export UNILION_MAMBA_BACKEND=external
export PYTHONPATH="$PWD:$PWD/projects:$PWD/mamba_diffv/mamba:$PYTHONPATH"
bash tools/dist_test.sh configs/uniprefusion_seq_e2e.py \
    ckpts/uniprefusion-final.pth 1 --eval mAP
```

参数顺序为“配置、权重、GPU 数量”。最终配置开启全部五个任务，实际显存需求取决于任务输出、时序状态及算子环境。

## 版本与复现说明

公开名称为 **Uni-PreFusion**，内部保留 `UniLION` 注册名、模块路径与权重键，以兼容已有训练权重；公开配置入口使用 `uniprefusion_*.py` 命名。

当前权重对应 `UNILION_MAMBA_BACKEND=external`。切换为双向 `auto` 后端会改变结构，不能直接替代。两套 Mamba 的包与扩展名保持分离。

本次发布保留作者原始代码与权重，不合入后续稀疏化实验或诊断修复。原前端曾观察到跨进程数值差异，严格重复性尚未建立，详见[复现说明](docs/reproducibility.md)。分阶段训练的初始化资源与最终权重评估所需资源分别说明。

引用格式见[英文首页](README.md)。第三方组件的版权和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
