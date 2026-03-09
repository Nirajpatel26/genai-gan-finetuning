# 🧠 GAN Fine-Tuning

  
> Module: Generative Adversarial Networks — from Dense GANs to DCGANs

---

## 📌 Project Overview

This repository contains the full implementation and experimental results for the **GAN Fine-Tuning** module of the GenAI course. The project explores the progression from basic dense-layer GANs to **Deep Convolutional GANs (DCGANs)**, investigating training stability, architectural design choices, and the impact of regularization and hyperparameter tuning on generated image quality.

The core objective is to understand **why** DCGAN architectures produce more stable and higher-quality outputs than vanilla GANs — not just how to implement them.

---

## 🗂️ Repository Structure

```
genai-gan-finetuning/
│
├── notebooks/
│   ├── 01_vanilla_gan_baseline.ipynb       # Dense-layer GAN baseline
│   ├── 02_dcgan_implementation.ipynb       # Full DCGAN with Conv layers
│   ├── 03_training_stability_analysis.ipynb # Experiments on stability techniques
│   └── 04_results_and_evaluation.ipynb     # Final results, FID scores, visualizations
│
├── src/
│   ├── models/
│   │   ├── generator.py                    # Generator architecture
│   │   ├── discriminator.py                # Discriminator architecture
│   │   └── dcgan.py                        # Full DCGAN model class
│   ├── training/
│   │   ├── train.py                        # Training loop
│   │   └── utils.py                        # Helper functions
│   └── evaluation/
│       └── evaluate.py                     # FID score + visual evaluation
│
├── results/
│   ├── images/                             # Generated image samples per epoch
│   ├── loss_curves/                        # Generator & Discriminator loss plots
│   └── metrics.json                        # Quantitative evaluation results
│
├── slides/
│   └── GAN_Finetuning_Presentation.pdf     # Course presentation slides
│
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

### Vanilla GAN (Baseline)
- **Generator**: Fully-connected (Dense) layers with ReLU activations
- **Discriminator**: Fully-connected layers with LeakyReLU activations
- **Issue**: Mode collapse, training instability, blurry outputs

### DCGAN (Final Implementation)
- **Generator**: Transposed Conv2D layers + BatchNorm + ReLU → Tanh output
- **Discriminator**: Strided Conv2D layers + BatchNorm + LeakyReLU → Sigmoid
- **Key principle**: No fully-connected layers in convolutional part; no pooling layers

```
Generator Architecture:
  Latent Vector (100,) → Dense → Reshape (4×4×512)
  → ConvTranspose2D(256) + BN + ReLU    # 8×8
  → ConvTranspose2D(128) + BN + ReLU    # 16×16
  → ConvTranspose2D(64)  + BN + ReLU    # 32×32
  → ConvTranspose2D(3)   + Tanh         # 64×64×3

Discriminator Architecture:
  Input (64×64×3)
  → Conv2D(64)  + LeakyReLU(0.2)        # 32×32
  → Conv2D(128) + BN + LeakyReLU(0.2)  # 16×16
  → Conv2D(256) + BN + LeakyReLU(0.2)  # 8×8
  → Conv2D(512) + BN + LeakyReLU(0.2)  # 4×4
  → Flatten → Dense(1) + Sigmoid
```

---

## 🔬 Experiments & Results

### Experiment 1 — Vanilla GAN Baseline
| Metric | Value |
|--------|-------|
| Dataset | MNIST |
| Epochs | 100 |
| Generator Loss (final) | ~2.1 |
| Discriminator Loss (final) | ~0.45 |
| Visual Quality | Blurry, mode collapse observed |

### Experiment 2 — DCGAN (No Stability Techniques)
| Metric | Value |
|--------|-------|
| Dataset | CIFAR-10 |
| Epochs | 100 |
| Generator Loss (final) | ~1.8 |
| Discriminator Loss (final) | ~0.38 |
| Visual Quality | Improved structure, some noise |

### Experiment 3 — DCGAN + Label Smoothing + LR Tuning ✅ Best
| Metric | Value |
|--------|-------|
| Dataset | CIFAR-10 |
| Epochs | 200 |
| Generator LR | 2e-4 |
| Discriminator LR | 1e-4 |
| Label Smoothing | Real labels → 0.9 |
| Generator Loss (final) | ~1.2 |
| Discriminator Loss (final) | ~0.6 |
| Visual Quality | **Sharp, diverse, stable training** |

### Key Observations
- **Label smoothing** (0.9 for real labels) was the single most impactful stability technique
- **Asymmetric learning rates** (G: 2e-4, D: 1e-4) prevented discriminator from dominating
- **BatchNormalization** in both G and D was critical — removing it from either caused collapse
- Mode collapse was completely resolved after switching from Dense → Convolutional architecture

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Requirements
```
tensorflow>=2.10
numpy
matplotlib
scikit-learn
jupyter
tqdm
```

### Run Training
```python
# Clone the repo
git clone https://github.com/Nirajpatel26/genai-gan-finetuning.git
cd genai-gan-finetuning

# Open in Google Colab (recommended for GPU access)
# Upload to Colab and run notebooks in order: 01 → 02 → 03 → 04
```

### Google Colab (Recommended)
This project was developed on **Google Colab with T4 GPU**. For best results:
1. Open `notebooks/02_dcgan_implementation.ipynb` in Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Mount Google Drive for model checkpointing

---

## 📊 Training Dynamics

### Loss Curves — DCGAN (Best Run)
The ideal GAN training dynamic shows:
- **Generator loss** gradually decreasing from ~3.0 → ~1.2
- **Discriminator loss** stabilizing around ~0.5–0.7 (not going to 0)
- No divergence or oscillation after epoch 50

### Generated Samples Progression
| Epoch 1 | Epoch 50 | Epoch 100 | Epoch 200 |
|---------|---------|----------|----------|
| Random noise | Rough shapes | Recognizable objects | Sharp, diverse samples |

---

## 💡 Key Learnings

1. **Architecture matters more than training duration** — switching to Conv layers had more impact than doubling epochs
2. **Balance is everything in GANs** — if either G or D dominates, training collapses
3. **Label smoothing is free performance** — a one-line change that dramatically stabilizes training
4. **Batch size affects stability** — larger batches (64–128) produced smoother loss curves
5. **Monitoring both losses simultaneously** is essential; G loss alone is misleading

---

## 🐛 Debugging Log

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Mode collapse | Discriminator too strong | Reduced D learning rate |
| Vanishing gradients in G | Binary cross-entropy saturation | Added label smoothing |
| Checkerboard artifacts | Transposed conv stride mismatch | Used resize+conv instead |
| Training divergence at epoch 30 | BatchNorm missing in D | Re-added BN to all D layers |

---

## 📁 Course Context

This is Assignment 4 of the **GenAI Spring 2026** course at Northeastern University. The module builds on prior CNN work (CIFAR-10 classification with ResNet, achieving 80.95% test accuracy) and introduces **generative modeling** as a counterpart to discriminative modeling.

**Prior modules completed:**
- Module 1: MLP Classifiers + Regularization
- Module 2: Prompt Engineering + NLP
- Module 3: CNN + ResNet (CIFAR-10)
- **Module 4: GANs + DCGANs ← This repo**
- Module 5: LLM Fine-Tuning (LoRA/QLoRA)
- Module 6: Multi-Agent Systems (AutoGen + Kafka)

---

## 👤 Author

**Niraj Patel**  
MS Student, Northeastern University  
📧 patel.niraju@northeastern.edu  
🔗 [GitHub](https://github.com/Nirajpatel26)

---

## 📄 License

This project is for academic purposes as part of the Northeastern University GenAI course curriculum.
