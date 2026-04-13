# Efficient-Skin-Disease-Classification-Using-Attention-Enhanced-CNN-Transformer-Fusion

Architecture: ResNet101 backbone → CBAM → Patch embedding → Transformer → MLP head
This notebook implements a CAD (computer-aided diagnosis) scheme that combines:
1. ResNet101 + CBAM for local feature extraction with channel and spatial attention
2. Patch embedding of the CNN feature map
3. Vision Transformer for global modeling
4. MLP head for classification

Parameter	      Value
Learning Rate	  0.0005
Optimizer	      Adam (β₁=0.9, β₂=0.999)
Weight Decay	  1e-5
Batch Size	    32
Epochs (Single)	  80
Early Stopping	12 epochs
LR Scheduler	  ReduceLROnPlateau
Loss	Weighted   CrossEntropyLoss
Activation	    ReLU
Dropout	        0.35


o	**Step-by-step instructions to reproduce results**

1. oversampling = balanced data; else use weighted sampler for imbalance
2. EDA: class distribution and sample images
3. Outputs: EDA, ROC-AUC, PR curves, confusion matrices, and heatmaps.
4. ABLATION_REGISTRY: names -> constructor kwargs for HybridResNetTransformer
5. Training loop — with early stopping, LR scheduler, best model checkpoint, AMP
6. Collect test predictions with TTA (5-crop + h-flip )
7. ROC-AUC (one-vs-rest) and Precision-Recall curves
8. Normalized confusion matrix (heatmap) and classification report
9. Standard + normalized confusion matrices ; melanoma row highlighted
10. McNemar + paired prediction stats
11. Multi-seed validation: mean ± std, paired t-test & Wilcoxon
12. k-fold cross-validation + Friedman test on fold accuracie
13. Grad-CAM: gradients of target score w.r.t. last conv feature map


o	**Environment setup and dependency requirements**

PyTorch: 2.7.0a0+7c8ec84dab.nv25.03
CUDA available: True
GPU: NVIDIA B200 MIG 1g.45gb
CUDA version: 12.8

Device: CUDA — NVIDIA B200 MIG 1g.45gb 
Model parameters: 23522375

o	**Dataset access instructions**

Dataset: 1. ISIC2026 2. ISIC2017 3. HAM10000 - (70% train / 30% test).

Class-Wise Distribution:
ISIC2016 dataset-
Benign	1031 images
Malignant	248 images

the ISIC2016 dataset link is https://www.kaggle.com/datasets/mahmudulhasantasin/isic-2016-original-dataset

ISIC2017 DATASET

Nevus (Benign): 1,843 images
Melanoma (Malignant): 521 images
Seborrheic Keratosis (Benign): 386 images

https://www.kaggle.com/datasets/johnchfr/isic-2017

HAM10000 dataset -
Melanocytic nevi (nv): 6,705 images
Melanoma (mel): 1,113 images
Benign keratosis-like lesions (bkl): 1,099 images
Basal cell carcinoma (bcc): 514 images
Actinic keratoses and intraepithelial carcinoma (akiec): 327 images
Vascular lesions (vasc): 142 images
Dermatofibroma (df): 115 images

Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

o	**Clear explanation of folder structure and code organization**

1. Clear Folder Structure of HAM10000
The raw dataset is typically downloaded as several zip files, which, when extracted, result in a structure organized around images and metadata.

HAM10000/
├── HAM10000_images_part_1/     # Part 1: Contains 5,000 images (.jpg)
│   ├── ISIC_0024306.jpg
│   └── ...
├── HAM10000_images_part_2/     # Part 2: Contains 5,015 images (.jpg)
│   ├── ISIC_0029306.jpg
│   └── ...
├── HAM10000_metadata.csv       # Metadata, labels, and metadata info
├── HAM10000_images_test/       # (Optional) Separate test images
└── HAM10000_GroundTruth.csv    # (Optional) Ground truth for test set

2. Code Organization and Data PipelineBecause the images are split into two folders (part_1 and part_2), the first step in code organization is to create a mapping that allows the code to locate any image regardless of its folder.

Ablation taxonomy (expanded code / experimental design):
Setting	Description:
o baseline_cnn	-Strong CNN-only baseline (ResNet50 classifier head).
o cnn_backbone_gap_only-	Same ResNet trunk as hybrid + GAP + head — no CBAM, no Transformer (ablation).
o cnn_cbam_gap_only	-ResNet trunk + CBAM + GAP + head — no Transformer (attention on CNN features only).
o transformer_only	-Pure Transformer on image patches (no ResNet trunk).
o cnn_transformer_no_attention	-ResNet + Transformer without CBAM (no channel/spatial attention).
o cnn_transformer_with_attention	-Proposed ResNet + CBAM + Transformer (use_cbam=True, default hybrid).
o proposed_with_fusion_refine	-Proposed + feature fusion refinement MLP before the classifier head.
o proposed_no_fusion_refine	-Proposed hybrid without the refinement MLP.

Comprehensive Code Repository:
1. DataPreprocessing Pipeline-   detailed preprocessing scripts, including image resizing, normalization, augmentation techniques
2. Training Pipeline- o	Model architecture definitions
                      o	Hyperparameter settings (learning rate, batch size, optimizer, epochs, etc.)
                      o	Training procedures for both individual models and the fusion framework
                      o	Random seed initialization to ensure reproducibility
3. Evaluation and Testing - o	Accuracy, precision, recall, and F1-score computation
                            o	Confusion matrix and normalized heatmap generation
                            o	Support for multi-dataset evaluation
4. README- o	Step-by-step instructions to reproduce results
           o	Environment setup and dependency requirements
           o	Dataset access instructions


