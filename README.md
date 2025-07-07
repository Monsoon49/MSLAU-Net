# MSLAU-Net: A Hybird CNN-Transformer Network for Medical Image Segmentation
> **Abstract:** Accurate medical image segmentation allows for the precise delineation of anatomical structures and pathological regions, which is essential for treatment planning, surgical navigation, and disease monitoring. Both CNN-based and Transformer-based methods have achieved remarkable success in medical image segmentation tasks. However, CNN-based methods struggle to effectively capture global contextual information due to the inherent limitations of convolution operations. Meanwhile, Transformer-based methods suffer from insufficient local feature modeling and face challenges related to the high computational complexity caused by the self-attention mechanism. To address these limitations, we propose a novel hybrid CNN-Transformer architecture, named MSLAU-Net, which integrates the strengths of both paradigms. The proposed MSLAU-Net incorporates two key ideas. First, it introduces Multi-Scale Linear Attention, designed to efficiently extract multi-scale features from medical images while modeling long-range dependencies with low computational complexity. Second, it adopts a top-down feature aggregation mechanism, which performs multi-level feature aggregation and restores spatial resolution using a lightweight structure. Extensive experiments conducted on benchmark datasets covering three imaging modalities demonstrate that the proposed MSLAU-Net outperforms other state-of-the-art methods on nearly all evaluation metrics, validating the superiority, effectiveness, and robustness of our approach.

## 1. Installation
* Clone this repo:
```
git clone https://github.com/Monsoon49/MSLAU-Net.git
cd MSLAU-Net
```
* Install packages:
```
pip install -r requirements.txt
```
## 2. Synapse Data Preparation
* The Synapse dataset we used is provided by TransUnet's author: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd.

## 3. Download Pre-trained  Model
* The pre-trained model weights (best.pth) can be saved to any desired location. Just make sure to set the correct path in the code accordingly.
 [Download pre-trained model in this link](https://drive.google.com/file/d/1GAASxAWTLkDPeGHcyjpLdKrU_ZipIF93/view?usp=sharing)

## 4. Synapse Train/Test
- Train
```bash
python train.py --dataset Synapse --root_path YOUR_TRAINING_DATA_DIR --max_epochs 400 --output_dir YOUR_OUTPUT_DIR --img_size 224 --base_lr 0.05 --batch_size 24
```
- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path YOUR_TEST_DATA_DIR --output_dir YOUR_OUTPUT_DIR --img_size 224 --num_classes 9
```

## 5. References
*  [TransUNet](https://github.com/Beckschen/TransUNet)
*  [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
*  [BRAU-Net++](https://github.com/Caipengzhou/BRAU-Netplusplus)
*  [UniFormer](https://github.com/Sense-X/UniFormer)

## 6. Citation
```bash
@article{lan2025mslau,
  title={MSLAU-Net: A Hybird CNN-Transformer Network for Medical Image Segmentation},
  author={Lan, Libin and Li, Yanxin and Liu, Xiaojuan and Zhou, Juan and Zhang, Jianxun and Huang, Nannan and Zhang, Yudong},
  journal={arXiv preprint arXiv:2505.18823},
  year={2025}
}
```
