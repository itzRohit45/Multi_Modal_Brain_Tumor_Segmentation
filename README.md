# Text-Guided Brain Tumor Segmentation using Vision-Language Models 🧠

**TG-SwinUNet: A Multimodal Deep Learning System for Medical Image Segmentation**

![Project Status](https://img.shields.io/badge/Status-Completed-success) 
![Architecture](https://img.shields.io/badge/Architecture-Swin%20Transformer%20%2B%20BioViL--T-blue)

Traditional brain tumor segmentation models rely entirely on visual feature extraction. However, in clinical practice, radiologists evaluate MRI scans in conjunction with written clinical descriptions. This project bridges that gap by introducing **TG-SwinUNet**, a Vision-Language Model (VLM) that feeds textual radiology reports directly into the neural network to explicitly guide the AI in drawing sharper, more accurate tumor boundaries.

---

## 🏗 Architecture Overview

The architecture features a dual-encoder design that mathematically aligns visual and semantic spaces:

1. **Visual Encoder (Swin Transformer):** Processes a 5-slice pseudo-3D FLAIR MRI window using Shifted Window Self-Attention to extract hierarchical spatial features.
2. **Text Encoder (BioViL-T):** A specialized Medical BERT. Layers 0–7 are frozen to retain general medical English, while layers 8–11 are trainable for specific Brain MRI domain adaptation.
3. **Multi-Level FiLM Fusion:** Early skip connections in the Swin Encoder are explicitly modulated (scaled and shifted) by the global text embedding.
4. **Cross-Attention Bottleneck:** At the deepest $7 \times 7$ layer, the visual image patches act as *Queries* to attend over the 128-token *Key/Value* text sequence.
5. **Text-Guided Attention Gates (TGAG):** Inside the decoder, skip-connection features are filtered by an attention mask generated directly from the text embedding, muting irrelevant background noise.

## 📊 Compound Loss Function

To stabilize training and enforce multi-modal alignment, the model uses a massive custom loss function:

$$ \mathcal{L}_\text{Total} = \mathcal{L}_\text{Seg}(\text{Final}) + 0.5\mathcal{L}_\text{Seg}(\text{Dec3}) + 0.25\mathcal{L}_\text{Seg}(\text{Dec4}) + 0.2\mathcal{L}_\text{Boundary} + 0.1\mathcal{L}_\text{Contrastive} $$

*   **Segmentation Loss (Dice + Focal):** Optimizes volume overlap and penalizes hard-to-predict pixels.
*   **Deep Supervision:** Mini-predictions at intermediate decoder blocks (`Dec3`, `Dec4`) force early layers to learn semantic tumor boundaries quickly.
*   **Boundary Loss:** A $6\times$ morphological penalty that specifically optimizes the Hausdorff Distance to guarantee sharp edges.
*   **Contrastive Alignment Loss:** A CLIP-style cosine similarity loss that forces the high-dimensional Image Embeddings and Text Embeddings to mathematically converge in a shared 256-D latent space.

---

## 💾 Dataset Details

We utilized a paired text-image variation of the **Brain Tumor Segmentation (BraTS) 2020** challenge dataset.

*   **Image Modality:** 2D axial slices extracted from 3D FLAIR MRI volumes.
*   **Text Modality:** Synthesized radiology reports describing tumor location and properties.
*   **Total Patients:** 344 text-matched patient directories (258 train, 86 validation).
*   **Pre-processing:** Volumes resized to $128 \times 128 \times 128$. Slices containing fewer than 50 tumor pixels were removed to ensure training stability.

---

## 📈 Final Results

An exhaustive ablation study was conducted against a structurally identical Image-Only baseline to mathematically prove the benefit of Natural Language Processing.

| Metric | Image-Only Baseline | Multimodal (TG-SwinUNet) | Improvement |
| :--- | :---: | :---: | :---: |
| **Dice Score** | 0.846 | **0.856** | `+0.010` |
| **IoU** | 0.765 | **0.776** | `+0.011` |
| **Hausdorff Distance** | 22.978 | **21.662** | `-1.316` |
| **Recall (Sensitivity)**| 0.857 | **0.870** | `+0.013` |

By physically examining **Grad-CAM heatmaps** and **t-SNE clusters**, the model demonstrably proves that textual guidance directly influences the spatial focus of the decoder, resulting in highly accurate predictions even on morphologically complex neuro-oncological scans.

---

## 📂 Repository Structure

*   `Brain_Tumor.ipynb`: The master Jupyter Notebook containing the data pipelines, model definitions, training loops, and visualization generation code.
*   `report.tex`: The comprehensive academic LaTeX report detailing the exact methodology, preprocessing, and mathematical proofs.
*   `*.png / *.svg`: High-resolution visual diagrams of the internal subsystems and macro architecture.
