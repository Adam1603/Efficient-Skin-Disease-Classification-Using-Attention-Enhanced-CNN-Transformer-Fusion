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


o	Step-by-step instructions to reproduce results

 Outputs: EDA, ROC-AUC, PR curves, confusion matrices, and heatmaps.

o	Environment setup and dependency requirements


o	Dataset access instructions

Dataset: 1. ISIC2026 2. ISIC2017 3. HAM10000 - (70% train / 30% test).

o	Clear explanation of folder structure and code organization
