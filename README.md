<p align="center">
  <h1 align="center">Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation</h1>
  <h3 align="center"><b>CVPR 2025</b></h3>
  <p align="center">
    <h3 align="center">
      <a href="https://kochanha.github.io/"><strong>Chanyoung Kim</strong></a> · 
      <a href="https://jdy77.github.io/"><strong>Dayun Ju</strong></a> · 
      <a href="https://dnwjddl.github.io/"><strong>Woojung Han</strong></a> · 
      <a href="https://faculty.ucmerced.edu/mhyang/"><strong>Ming-Hsuan Yang</strong></a> · 
      <a href="https://micv-yonsei.github.io/#professor"><strong>Seong Jae Hwang</strong></a>
    </h3>
    <h3 align="center">
      Yonsei University ·
      University of California, Merced
    </h3>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2411.17150"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2411.17150-b31b1b.svg"></a>
    <a href="https://micv-yonsei.github.io/cass/"><img alt='Project Page' src="https://img.shields.io/badge/Project-Website-orange"></a>
  </p>
  <br>
</p>
 
<be>
<img width="1775" alt="Image" src="https://github.com/user-attachments/assets/f9def7fd-537c-4ecc-897b-57edacc72efb" />

> #### **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation**<be>  
>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025  
>Chanyoung Kim, Dayun Ju, Woojung Han, Ming-Hsuan Yang, Seong Jae Hwang  
>Yonsei University & University of California, Merced
### Abstract
Open-Vocabulary Semantic Segmentation (OVSS) has advanced with recent vision-language models (VLMs), enabling segmentation beyond predefined categories through
various learning schemes. Notably, training-free methods
offer scalable, easily deployable solutions for handling unseen data, a key goal of OVSS. Yet, a critical issue persists: lack of object-level context consideration when segmenting complex objects in the challenging environment
of OVSS based on arbitrary query prompts. This oversight limits models' ability to group semantically consistent elements within object and map them precisely to userdefined arbitrary classes. In this work, we introduce a novel
approach that overcomes this limitation by incorporating
object-level contextual knowledge within images. Specifically, our model enhances intra-object consistency by distilling spectral-driven features from vision foundation models into the attention mechanism of the visual encoder, enabling semantically coherent components to form a single
object mask. Additionally, we refine the text embeddings
with zero-shot object presence likelihood to ensure accurate
alignment with the specific objects represented in the images. By leveraging object-level contextual knowledge, our
proposed approach achieves state-of-the-art performance
with strong generalizability across diverse datasets.

## :book: Contents
<!--ts-->
   * [Installation](#installation)
   * [Evaluation](#evaluation)
   * [About CASS](#about-cass)
      * [Spectral Object-Level Context Distillation](#spectral-object-level-context-distillation)
      * [Object Presence-Driven Object-Level Context](#object-presence-driven-object-level-context)
      * [Qualitative Results](#qualitative-results)
      * [Quantitative  Results](#quantitative-results)
   * [Citation](#citation)

<!--te-->


## Installation


### Requirements
```shell script
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim scikit-learn scikit-image
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex numpy==1.23.5 yapf==0.40.1
```

### Download Datasets
We include the listed dataset configurations in this repo, following [SCLIP](https://github.com/wangf3014/SCLIP): PASCAL VOC (with and without the background category), PASCAL Context (with and without the background category), Cityscapes, ADE20k, COCO-Stuff164k, and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```shell script
python ./datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO_OBJECT
```

## Evaluation
To evaluate CASS on a single benchmark, run the following command:
```shell script
python eval.py --config ./configs/cfg_{benchmark_name}.py --pamr off
```
To evaluate CASS on a all 8 benchmarks, run the following command:
```shell script
bash eval_all.sh
```

## About CASS

### Spectral Object-Level Context Distillation
Detailed illustration of our proposed training-free spectral object-level context distillation mechanism. By matching the attention graphs of VFM and CLIP head-by-head to establish complementary relationships, and distilling the fundamental object-level context of the VFM graph to CLIP, we enhance CLIP's ability to capture intra-object contextual coherence.
<div align="center">
  <img width="800" alt="spectral" src="https://micv-yonsei.github.io/cass/static/images/local.png">
</div>

### Object Presence-Driven Object-Level Context
Detailed illustration of our object presence prior-guided text embedding adjustment module. The CLIP text encoder generates text embeddings for each object class, and the object presence prior is derived from both visual and text embeddings. Within hierarchically defined class groups, text embeddings are selected based on object presence prior, then refined in an object-specific direction to align with components likely present in the image.

<div align="center">
  <img width="600" alt="Object Presence-Driven Object-Level Context" src="https://micv-yonsei.github.io/cass/static/images/OTA.png">
</div>


### Qualitative Results
Qualitative comparison across the Pascal VOC, Pascal Context, COCO, and ADE20K datasets using CLIP ViT-B/16.
<div align="center">
  <img width="900" alt="main_results" src="https://micv-yonsei.github.io/cass/static/images/qualitative.png">
</div>

### Quantitative Results
Quantitative results with state-of-the-art unsupervised open-vocabulary semantic segmentation models on eight datasets.
<div align="center">
  <img width="900" alt="quan_results" src="https://micv-yonsei.github.io/cass/static/images/miou.png">
</div>

## Citation
If you found this code useful, please cite the following paper:  
```
@InProceedings{kim2024distilling,
    author    = {Kim, Chanyoung and Ju, Dayun and Han, Woojung and Yang, Ming-Hsuan and Hwang, Seong Jae},
    title     = {Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025}
}
```

## :scroll: Acknowledgement
This repository has been developed based on the [NACLIP](https://github.com/sinahmr/NACLIP) and [SCLIP](https://github.com/wangf3014/SCLIP) repository. Thanks for the great work!