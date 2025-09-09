<div align=“center” style=“font-family: charter;”>
<h1 align="center">Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue'></a> &nbsp;
</div>
<div align="center">
  Xiangwei Shen<sup>1,2*</sup>,
  <a href="https://scholar.google.com/citations?user=Lnr1FQEAAAAJ&hl=zh-CN" target="_blank"><b>Zhimin Li</b></a><sup>1*</sup>,
  <a href="https://scholar.google.com.hk/citations?user=Fz3X5FwAAAAJ" target="_blank"><b>Zhantao Yang</b></a><sup>1</sup>, 
  <a href="https://shiyi-zh0408.github.io/" target="_blank"><b>Shiyi Zhang</b></a><sup>3</sup>,
  Yingfang Zhang<sup>1</sup>,
  Donghao Li<sup>1</sup>,
  <br>
  <a href="https://scholar.google.com/citations?user=VXQV5xwAAAAJ&hl=en" target="_blank"><b>Chunyu Wang</b></a><sup>1</sup>,
  <a href="https://openreview.net/profile?id=%7EQinglin_Lu2" target="_blank"><b>Qinglin Lu</b></a><sup>1</sup>,
  <a href="https://andytang15.github.io" target="_blank"><b>Yansong Tang</b></a><sup>3,✝</sup>
</div>
<div align="center">
  <sup>1</sup>Hunyuan, Tencent 
  <br>
  <sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen 
  <br>
  <sup>3</sup>Shenzhen International Graduate School, Tsinghua University 
  <br>
  <sup>*</sup>Equal contribution 
  <sup>✝</sup>Corresponding author
</div>

![head](assets/head.jpg)

## Key Features
1. **Direct Align**: We introduce a new sampling strategy for diffusion fine-tuning that can effectively restore highly noisy images, leading to an optimization process that is more stable and less computationally demanding, especially during the initial timesteps.
2. **Faster Training**:   By rolling out only a single image and optimizing directly with analytical gradients—a key distinction from GRPO—our method achieves significant performance improvements for FLUX.1.dev in under 10 minutes of training. To further accelerate the process, our method supports replacing online rollouts entirely with a small dataset of real images; we find that fewer than 1500 images are sufficient to effectively train FLUX.1.dev.
3. **Free of Reward Hacking**: We have improved the training strategy for method that direct backpropagation on reward signal (such as ReFL and DRaFT). Moreover, we directly regularize the model using negative rewards, without the need for KL divergence or a separate reward system. In our experiments, this approach achieves comparable performance with multiple different rewards, improving the perceptual quality of FLUX.1.dev without suffering from reward hacking issues, such as overfitting to color or oversaturation preferences.
4. **Potential for Controllable Fine-tuning**: For the first time in online RL, we incorporate dynamically controllable text conditions, enabling on-the-fly adjustment of reward preference towards styles within the scope of the reward model.

## Updates

- __[2025.9.8]__:  We released the paper, checkpoint, inference code

##  TODO
- [ ] Open training code in next week

## Inference
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py \
```
## BiB
If you find SRPO useful for your research and applications, please cite using this BibTeX:
```
@misc{shen2025directlyaligningdiffusiontrajectory,
      title={Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference}, 
      author={Xiangwei Shen and Zhimin Li and Zhantao Yang and Shiyi Zhang and Yingfang Zhang and Donghao Li and Chunyu Wang and Qinglin Lu and Yansong Tang},
      year={2025},
      eprint={2509.06942},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.06942}, 
}
```
