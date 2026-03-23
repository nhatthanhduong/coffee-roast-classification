# Coffee Roast Classification using DINOv3 + CNN

A deep learning project for classifying coffee bean roast levels (Green, Light, Medium, Dark) using a hybrid architecture combining:

- Pretrained DINOv3 ConvNeXt backbone
- Custom CNN classification head

## Project Overview

This project aims to classify coffee beans based on roast level using computer vision.

To improve performance and generalization:

- Two datasets with different backgrounds are merged
- Data is restructured and balanced
- A pretrained DINOv3 model is used as a feature extractor

## Project Structure

    .
    ├── data_preprocessing.ipynb
    ├── DinoV3_CNN.ipynb
    ├── new_dataset/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── dinov3_cnn.pth
    └── README.md
    
## Data Preprocessing

1. Dataset sources
- [Kaggle Coffee Bean Dataset]
- [Roboflow Coffee Bean Dataset]

2. Key steps
- Data inspection
- Merge datasets
- Restructured directories
    ```bash
    new_dataset/
    ├── train/
    ├── val/
    └── test/
    ```
- Split ratio: 
    - Train: 70%
    - Validation: 15%
    - Test: 15%

## Model Architecture

1. Backbone

- Pretrained DINOv3 ConvNeXt
- Frozen weights (feature extractor only)

2. Custom CNN Head

   ```bash
    Conv2D → BatchNorm → ReLU
    Conv2D → BatchNorm → ReLU
    AdaptiveAvgPool
    Flatten → Dropout → Linear
   ```

3. Training Configuration

| Parameter | Value |
|----------|----------|
| Epochs | 40 |
| Batch Size | 16 |
| Learning Rate | 1e-3 |
| Optimizer | Adamax |
| Loss Function | CrossEntropy(weighted) |

3. Techniques Used

- Data Augmentation:
    - Rotation
    - Color jitter
    - Horizontal flip
    - Random erasing
- Class imbalance handling:
    - Computed weighted classes
- Optimization: 
    - Learning rate scheduling `ReduceLROnPlateau`
    - Early Stopping (patience = 10)

## Results
The best model achieved 100% accuracy on the test set.  
Download trained model [here](https://drive.google.com/file/d/1KPQCBpnbvkIygk2_pGWkZUXPDX1y_XWt/view?usp=sharing)

## How to run
1. Clone repository

   ```bash
   git clone https://github.com/nhatthanhduong/coffee-roast-classification
   cd coffee-roast-classification
   ```

2. Install dependencies

   ```bash
   pip install torch torchvision timm scikit-learn matplotlib seaborn pandas pillow
   ```

3. Run data preprocessing

    ```bash
    jupyter notebook data_preprocessing.ipynb
    ```

4. Train model

    ```bash
    jupyter notebook DinoV3_CNN.ipynb
    ```

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)
- [Roboflow Dataset](https://universe.roboflow.com/universitas-gunadarma-1zr7d/coffebeans-5p6py)