<div align="center">
  
# 【NeurIPS'2025】Rebalancing Contrastive Alignment with Bottlenecked Semantic Increments in Text-Video Retrieval
[![Conference](https://img.shields.io/badge/NeurIPS-2025-ff69b4.svg)](https://nips.cc/Conferences/2025)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2505.12499-FF6B6B.svg)](https://arxiv.org/abs/2505.12499)
</div>

## 📚 Abstract
Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods often overlook the impact of modality gaps, which causes anchor representations to undergo in-place optimization (i.e., optimization tension), where gradients from positive and negative pairs cancel out, limiting alignment capacity. Moreover, noisy hard negatives further distort semantic learning of anchors. To address these issues, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment $\Delta_{ij}$ between text $t_i$ and video $v_j$, redistributing gradients away from the anchor to offload tension, while absorbing noise from hard negatives to mitigate semantic bias. We derive $\Delta_{ij}$ via a multivariate first-order Taylor expansion of the InfoNCE loss under a trust-region constraint, showing its role in guiding updates along locally consistent descent directions. To couple increments across batches, we implement $\Delta_{ij}$ through a lightweight neural module conditioned on the semantic gap, enabling structure-aware corrections. To further stabilize training and regularize $\Delta$ representations, we treat $\Delta_{ij}$ as a latent variable within a variational information bottleneck. We relax the compression upper bound on the text side, yielding a video-conditioned objective that over-regularizes the visual side and reduces redundancy. In addition, we apply a trust-region constraint to bound update magnitudes and a directional diversity loss to promote angular separation. Experiments on four benchmarks demonstrate that GARE consistently improves alignment accuracy and robustness, validating the effectiveness of gap-aware tension mitigation.

<div align="center">
<img src="pictures/img.png" width="700px">
</div>

## 🚀 Finetune on MSR-VTT

### Setup code environment
```shell
conda create -n GARE python=3.8
conda activate GARE
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Download CLIP Model

```shell
cd tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### Compress Video
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

###  Train on MSR-VTT
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH=/home/username/gare/data/MST-VTT
python -m torch.distributed.launch \
--master_port 29510 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/3fps_videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--temp 3 \
--alpha 2 \
--beta 0.07 \
--lambda_dir 0.01 \
--lambda_epsilon 0.01 \
--lambda_lower 0.5
```

## 🎗️ Acknowledgments
* This code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [DRL](https://github.com/foolwood/DRL), and [EMCL](https://github.com/jpthu17/EMCL).
We sincerely appreciate for their contributions.
