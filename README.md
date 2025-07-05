# PRODIGY_ML_03
# Cat vs Dog SVM Classifier

## Project Overview

This project implements a Support Vector Machine (SVM) classifier to distinguish between cats and dogs using image data. The implementation includes proper data preprocessing, dimensionality reduction, hyperparameter tuning, and comprehensive evaluation metrics.

## 📋 Task Description

**Task-03**: Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

**Dataset**: [Dogs vs Cats - Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

## 🚀 Key Features

### Core Functionality
- **Image Preprocessing**: Automatic resizing, normalization, and color space conversion
- **Dimensionality Reduction**: PCA to reduce computational complexity
- **Feature Scaling**: StandardScaler for optimal SVM performance
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Data Handling
- **Real Data Support**: Load and process actual Kaggle dataset
- **Synthetic Data Fallback**: Generate synthetic data when real dataset unavailable
- **Flexible Input**: Support for various image formats (PNG, JPG, JPEG)

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visual representation of classification results
- **PCA Visualization**: 2D visualization of data in reduced dimensional space
- **Cross-Validation Analysis**: Statistical significance of results

## 🛠️ Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm
```

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import cv2
import os
from tqdm import tqdm
```

## 📊 Usage

### Basic Usage
```python
# Initialize classifier
classifier = CatDogSVMClassifier(img_size=(64, 64))

# Prepare data (synthetic data for demo)
classifier.prepare_data()

# Visualize data distribution
classifier.plot_pca_visualization()

# Train the model
classifier.train_svm(use_grid_search=True)

# Evaluate performance
classifier.evaluate_model()
```

### Using Real Kaggle Dataset
```python
# Initialize classifier
classifier = CatDogSVMClassifier()

# Load real dataset
classifier.prepare_data('/path/to/kaggle/dogs-vs-cats/data')

# Train with hyperparameter tuning
classifier.train_svm(use_grid_search=True)

# Evaluate results
classifier.evaluate_model()
```

### Predicting New Images
```python
# Predict a single image
prediction, confidence = classifier.predict_new_image('/path/to/new/image.jpg')
print(f"Prediction: {'Cat' if prediction == 0 else 'Dog'}")
print(f"Confidence: {confidence:.4f}")
```

## 🔧 Architecture

### Class Structure
```
CatDogSVMClassifier
├── __init__()                     # Initialize classifier
├── load_and_preprocess_data()     # Load real dataset
├── create_synthetic_data()        # Generate synthetic data
├── prepare_data()                 # Data preparation pipeline
├── train_svm()                    # Model training
├── evaluate_model()               # Performance evaluation
├── plot_pca_visualization()       # Data visualization
└── predict_new_image()            # Single image prediction
```

### Processing Pipeline
1. **Image Loading** → Load images from directory
2. **Preprocessing** → Resize, normalize, flatten
3. **Feature Scaling** → StandardScaler normalization
4. **Dimensionality Reduction** → PCA to 100 components
5. **Model Training** → SVM with hyperparameter tuning
6. **Evaluation** → Multiple metrics and visualizations

## 📈 Model Configuration

### Default Parameters
- **Image Size**: 64x64 pixels
- **PCA Components**: 100
- **Test Split**: 20% of data
- **Cross-Validation**: 5-fold

### Hyperparameter Grid
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
```

## 🎯 Performance Metrics

The classifier provides comprehensive evaluation:

### Accuracy Metrics
- **Test Accuracy**: Overall classification accuracy
- **Cross-Validation**: Mean accuracy with confidence intervals
- **Classification Report**: Precision, recall, F1-score per class

### Visualizations
- **Confusion Matrix**: Heatmap showing classification results
- **PCA Plot**: 2D visualization of data separation
- **Performance Statistics**: Detailed metrics breakdown

## 📁 Dataset Structure

Expected directory structure for Kaggle dataset:
```
dataset/
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│   └── ...
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

## 🔍 Key Implementation Details

### Image Preprocessing
- **Color Conversion**: BGR to RGB conversion
- **Resizing**: Consistent image dimensions
- **Normalization**: Pixel values scaled to [0, 1]
- **Flattening**: Convert 2D images to 1D feature vectors

### Label Encoding
- **Cat**: Label 0
- **Dog**: Label 1
- **Automatic**: Extract labels from filenames

### Synthetic Data Generation
When real data is unavailable:
- **Cat Features**: Normal distribution with specific parameters
- **Dog Features**: Different distribution to simulate class separation
- **Realistic Dimensionality**: Match expected image feature size

## 🚨 Usage Notes

### Performance Considerations
- **Memory Usage**: Large datasets may require batch processing
- **Training Time**: Grid search can be time-intensive
- **Image Limit**: Demo version processes 2000 images for faster execution

### Data Requirements
- **File Formats**: Supports PNG, JPG, JPEG
- **Naming Convention**: Files should start with 'cat' or 'dog'
- **Image Quality**: Higher resolution images may improve accuracy

### Model Limitations
- **Pixel-Based**: Uses raw pixel values as features
- **Resolution**: Limited to specified image dimensions
- **Synthetic Data**: Demo performance may not reflect real-world results

## 📊 Expected Results

### With Synthetic Data
- **Accuracy**: ~85-90% (synthetic data optimized for separation)
- **Cross-Validation**: Stable performance across folds
- **Training Time**: Fast due to synthetic data properties

### With Real Data
- **Accuracy**: Variable based on dataset quality and preprocessing
- **Best Performance**: Achieved with grid search hyperparameter tuning
- **Computational Cost**: Higher due to real image complexity

## 🔮 Future Enhancements

### Potential Improvements
1. **Feature Engineering**: Add texture, edge, or color histogram features
2. **Data Augmentation**: Rotation, scaling, brightness variations
3. **Advanced Kernels**: Custom kernel functions for better separation
4. **Ensemble Methods**: Combine multiple SVM models
5. **Deep Learning**: CNN feature extraction for better representations

### Performance Optimization
- **Batch Processing**: Handle larger datasets efficiently
- **Parallel Processing**: Utilize multiple CPU cores
- **Memory Management**: Optimize for large image datasets
- **Caching**: Store preprocessed features for faster training

## 📖 Tutorial

### Step-by-Step Guide

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   - Visit [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
   - Download and extract dataset

3. **Run Classification**
   ```python
   python main.py
   ```

4. **View Results**
   - Check console output for metrics
   - View generated plots for visualizations

### Custom Configuration
```python
# Custom image size
classifier = CatDogSVMClassifier(img_size=(128, 128))

# Train without grid search (faster)
classifier.train_svm(use_grid_search=False)

# Train with grid search (better accuracy)
classifier.train_svm(use_grid_search=True)
```

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is intended for educational and research purposes.

## 📞 Getting Help

### Common Issues
- **Dataset Not Found**: Ensure correct path to Kaggle dataset
- **Memory Errors**: Reduce image size or limit number of samples
- **Slow Training**: Disable grid search for faster training

### Tips for Best Results
- **Use Grid Search**: Always enable for production use
- **Increase PCA Components**: For better feature representation
- **Larger Images**: Higher resolution may improve accuracy
- **More Data**: Larger dataset generally improves performance

---

**Note**: This implementation provides a solid foundation for SVM-based image classification with clean, readable code and comprehensive evaluation metrics. Perfect for learning and educational purposes!
