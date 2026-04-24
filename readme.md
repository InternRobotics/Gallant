## Project Gallant

This repository contains the official code for the **CVPR 2026** paper:

**Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains**

[[ArXiv](https://arxiv.org/abs/2511.14625)] · [[Project page](https://gallantloco.github.io/)]

### Installation

1. **Install Active Adaptation**  
   Follow the instructions in the [active-adaptation](https://github.com/btx0424/active-adaptation) repository.
2. **Install Gallant (editable mode)**  
   From the root of this repository:
   ```bash
   pip install -e .
   ```
3. **Register Gallant projects with Active Adaptation**  
   Run:
   ```bash
   aa-discover-projects
   ```
   and enable `gallant` and `gallant_learning` when prompted.

### Training

The following command launches distributed PPO training for the Gallant humanoid locomotion task (8 GPUs in this example):

```bash
scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 train_ppo.py task=G1LidarNew task.num_envs=1024 algo=ppo_gallant
```

Adjust the GPU list, `task`, and `task.num_envs` according to your hardware and experiment configuration.

### Playing / Evaluation

We will release detailed commands and evaluation scripts soon. For now, please refer to the paper and project page for qualitative results.

### Citation

If you find Gallant useful in your research, please cite:

```text
@article{ben2025gallant,
  title={Gallant: Voxel grid-based humanoid locomotion and local-navigation across 3d constrained terrains},
  author={Ben, Qingwei and Xu, Botian and Li, Kailin and Jia, Feiyu and Zhang, Wentao and Wang, Jingping and Wang, Jingbo and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2511.14625},
  year={2025}
}
```

### Communication
If you have any specific questions, please drop an e-mail to elgceben@gmail.com
