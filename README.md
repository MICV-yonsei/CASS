# Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation
#### CVPR 2025 

[[Project Page]](https://micv-yonsei.github.io/cass/) [[arXiv]](https://arxiv.org/pdf/2411.17150)  
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
of OVSS based on arbitrary query prompts. This oversight limits models’ ability to group semantically consistent elements within object and map them precisely to userdefined arbitrary classes. In this work, we introduce a novel
approach that overcomes this limitation by incorporating
object-level contextual knowledge within images. Specifically, our model enhances intra-object consistency by distilling spectral-driven features from vision foundation models into the attention mechanism of the visual encoder, enabling semantically coherent components to form a single
object mask. Additionally, we refine the text embeddings
with zero-shot object presence likelihood to ensure accurate
alignment with the specific objects represented in the images. By leveraging object-level contextual knowledge, our
proposed approach achieves state-of-the-art performance
with strong generalizability across diverse datasets.
