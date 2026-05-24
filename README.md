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
The TG-SwinUNet operates via four primary subsystems:

1. **Image Encoder:** A Swin-Tiny backbone projects the 5-slice input to 3 channels via a 1x1 convolution, extracting hierarchical features at four distinct spatial resolutions.
2. **Text Encoder:** The 128 text tokens yield a global CLS embedding and sequence embeddings, which are then passed through a Projection MLP to map 768-D embeddings down to a 256-D shared latent space.
3. **Fusion Mechanisms:** The text CLS token modulates visual features via FiLM at shallow encoder layers, while the sequence embeddings act as Keys/Values in the Cross-Attention bottleneck where visual patches act as Queries.
4. **Decoder:** A transposed convolution decoder utilizing Text-Guided Attention Gates (TGAG) to condition spatial skip-connections and mute irrelevant background signal.

## Loss Formulation
The network is optimized using a compound loss structure designed specifically for complex morphological segmentation:

L_Total = L_Seg(Final) + 0.5 * L_Seg(Dec3) + 0.25 * L_Seg(Dec4) + 0.2 * L_Boundary + 0.1 * L_Contrastive

Where L_Seg is the sum of Dice Loss and Focal Loss. Deep Supervision at blocks Dec3 and Dec4 mitigates vanishing gradients. Boundary Loss heavily penalizes edge misclassifications, and Contrastive Loss ensures textual and visual embeddings map to the exact same semantic vector space.

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
