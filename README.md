# Text-Guided Brain Tumor Segmentation using Vision-Language Models

## Introduction
Brain tumor segmentation from Magnetic Resonance Imaging (MRI) is a critical step in clinical diagnosis, surgical planning, and post-operative monitoring. Traditional deep learning approaches rely entirely on visual feature extraction. However, in clinical practice, radiologists evaluate MRI scans in conjunction with prior knowledge and written clinical descriptions. 

This project bridges this gap by introducing **TG-SwinUNet** (Text-Guided Swin-UNet), a highly sophisticated Vision-Language Model (VLM) for the BraTS 2020 challenge. The model takes multi-slice FLAIR MRI volumes and their corresponding patient-level radiology text descriptions as joint input to explicitly condition spatial segmentation on clinical language.

## Key Features & Contributions
* **Pseudo-3D Visual Encoding:** Utilizes a Swin-Transformer (Tiny) backbone to process 5-slice stacked MRI inputs, extracting global spatial context without the heavy memory overhead of true 3D CNNs.
* **Domain-Adapted Text Encoding:** Employs the BioViL-T (BiomedVLP-CXR-BERT) model with partial unfreezing (layers 8-11) to adapt to brain MRI vocabulary without catastrophic forgetting.
* **Multi-stage Multimodal Fusion:** 
  * Early-stage Feature-wise Linear Modulation (FiLM) layers.
  * Bottleneck Cross-Attention mapping explicit textual semantics to 7x7 spatial feature grids.
  * Text-Guided Attention Gates (TGAG) acting on skip-connections to filter irrelevant spatial noise.
* **Custom Compound Loss Function:** Combines Dice, Focal, Deep Supervision, Boundary Loss, and a CLIP-style Contrastive Alignment Loss.

## Dataset and Preprocessing
The model is trained and evaluated on a paired text-image variation of the Brain Tumor Segmentation (BraTS) 2020 challenge dataset.
* **Image Data:** 2D axial FLAIR slices extracted from 344 patients (258 train, 86 validation).
* **Text Data:** Synthesized radiology reports corresponding to patient directories.
* **Preprocessing Pipeline:** 
  * Volumes resized to 128 x 128 x 128.
  * Foreground Z-Score Normalization applied.
  * Strict slice filtering: Slices containing fewer than 50 tumor pixels are removed to ensure robust gradient descent.
  * Text tokenized to exactly 128 tokens using specialized medical Byte-Pair Encoding.

## Architecture Deep-Dive
The TG-SwinUNet represents a fundamental shift from traditional unimodal segmentation, operating via four meticulously integrated subsystems:

1. **Visual Encoding via Swin Transformer:** Instead of standard CNNs, the model utilizes a hierarchical `Swin-Tiny` backbone. By employing Shifted Window Self-Attention, the encoder effectively captures global anatomical context across four spatial resolutions, projecting the 5-channel pseudo-3D MRI input into a rich 7x7x768 bottleneck feature map.
2. **Domain-Adaptive Text Encoding (BioViL-T):** Clinical reports are tokenized and processed through a Medical BERT. To achieve domain adaptation without catastrophic forgetting, layers 0-7 are frozen to retain general medical English, while layers 8-11 are fine-tuned for neuro-oncology. The resulting embeddings are passed through a Projection MLP to achieve a shared 256-D latent space.
3. **Multi-Scale Modality Fusion:** 
    * **Early Fusion (FiLM):** Feature-wise Linear Modulation applies scaling and shifting to the shallow visual features based directly on the text `CLS` token, acting as a semantic filter.
    * **Bottleneck Cross-Attention:** At the deepest encoder level, the visual patches act as *Queries* against the text tokens (*Keys/Values*), enabling explicit word-to-pixel mapping.
4. **Text-Guided Decoder:** The transposed convolution decoder is upgraded with Text-Guided Attention Gates (TGAG). These gates intercept standard skip-connections, utilizing the text embeddings to generate spatial attention masks that actively suppress irrelevant background signals before upsampling.

## Compound Loss Formulation
Optimizing a multimodal network for highly irregular tumor boundaries requires a specialized loss landscape. The model is trained using a massive compound loss function designed to penalize distinct failure modes simultaneously:

```text
L_Total = L_Seg(Final) + 0.5 * L_Seg(Dec3) + 0.25 * L_Seg(Dec4) + 0.2 * L_Boundary + 0.1 * L_Contrastive
```

* **Segmentation Loss (`L_Seg`):** A combination of standard **Dice Loss** (optimizing volume overlap) and **Focal Loss** (penalizing hard-to-predict, class-imbalanced pixels at the tumor fringe).
* **Deep Supervision (`Dec3`, `Dec4`):** Auxiliary segmentation heads extract intermediate predictions from the decoder. This provides direct gradient pathways to the deep layers, combating the vanishing gradient problem and accelerating convergence.
* **Boundary Loss (`L_Boundary`):** Standard Dice loss often fails on sharp contours. This term applies a severe 6x morphological penalty to predictions that bleed over the tumor edge, strictly optimizing the Hausdorff Distance.
* **Contrastive Alignment Loss (`L_Contrastive`):** A CLIP-style temperature-scaled cosine similarity loss. This forces the 256-D Global Image Embeddings and the 256-D Text Embeddings to mathematically converge, ensuring the Vision and Language modalities operate within an identical semantic vector space.

## Results & Ablation Study
The multimodal architecture was evaluated against a structurally identical Image-Only baseline to rigorously prove the value of textual conditioning.

| Metric | Image-Only Baseline | Multimodal (TG-SwinUNet) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Dice Score** | 0.846 | **0.856** | +0.010 |
| **Intersection over Union (IoU)** | 0.765 | **0.776** | +0.011 |
| **Hausdorff Distance** | 22.978 | **21.662** | -1.316 |
| **Precision** | 0.874 | 0.874 | 0.000 |
| **Recall (Sensitivity)** | 0.857 | **0.870** | +0.013 |

Rigorous explainability testing (Grad-CAM and t-SNE) physically confirmed that the Text and Image features converge towards each other, proving that the NLP data explicitly guides the spatial segmentation boundaries.

## Repository Files
* `Brain_Tumor.ipynb`: Core source code containing PyTorch dataloaders, the TG-SwinUNet architecture, loss functions, training loop, and evaluation/explainability metric generation.
* `report.tex`: Comprehensive academic report detailing the exact methodology.
* `*.png` / `*.svg`: High-resolution architectural diagrams and output plots.

## Authors
* Rohit Kumar (CS23B2053)
* Sarvan Kumar (ME23B1065)
