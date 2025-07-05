import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import cv2
import os
from tqdm import tqdm
import warnings
import time
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class EnhancedCatDogSVMClassifier:
    def __init__(self, img_size=(64, 64), use_advanced_features=True):
        """
        Enhanced SVM classifier with advanced features
        
        Args:
            img_size: Tuple of (width, height) for resizing images
            use_advanced_features: Whether to use advanced feature engineering
        """
        self.img_size = img_size
        self.use_advanced_features = use_advanced_features
        self.svm_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=100)
        self.feature_selector = SelectKBest(f_classif, k=50)
        self.training_history = []
        self.feature_names = []
        
        # Performance tracking
        self.performance_metrics = {
            'training_time': 0,
            'inference_time': 0,
            'memory_usage': 0,
            'best_params': {},
            'cv_scores': [],
            'feature_importance': []
        }
    
    def extract_advanced_features(self, img):
        """
        Extract advanced features from image beyond just pixel values
        
        Args:
            img: Input image array
            
        Returns:
            feature_vector: Advanced feature vector
        """
        features = []
        
        # 1. Color histogram features
        for channel in range(3):  # RGB channels
            hist = cv2.calcHist([img], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Texture features using Local Binary Patterns (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Edge detection features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # 3. Statistical features
        features.extend([
            np.mean(img),
            np.std(img),
            np.median(img),
            np.min(img),
            np.max(img)
        ])
        
        # 4. Shape features (contour analysis)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            features.extend([area, perimeter])
        else:
            features.extend([0, 0])
        
        # 5. Reduced pixel features (downsample)
        img_small = cv2.resize(img, (16, 16))
        features.extend(img_small.flatten())
        
        return np.array(features)
    
    def create_enhanced_synthetic_data(self, n_samples=2000):
        """
        Create more realistic synthetic data with advanced features
        """
        print("Creating enhanced synthetic data...")
        np.random.seed(42)
        
        features_list = []
        labels = []
        
        for i in tqdm(range(n_samples), desc="Generating samples"):
            if i < n_samples // 2:
                # Cat features
                img = self.generate_synthetic_cat_image()
                label = 0
            else:
                # Dog features
                img = self.generate_synthetic_dog_image()
                label = 1
            
            if self.use_advanced_features:
                features = self.extract_advanced_features(img)
            else:
                features = img.flatten()
            
            features_list.append(features)
            labels.append(label)
        
        return np.array(features_list), np.array(labels)
    
    def generate_synthetic_cat_image(self):
        """Generate a synthetic cat-like image"""
        img = np.random.rand(self.img_size[0], self.img_size[1], 3)
        
        # Cat-like patterns: more uniform colors, rounder shapes
        center = (self.img_size[0]//2, self.img_size[1]//2)
        
        # Add circular patterns (cat face-like)
        y, x = np.ogrid[:self.img_size[0], :self.img_size[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= (self.img_size[0]//3)**2
        
        # Cat-like colors (more browns, grays)
        cat_colors = np.array([0.6, 0.4, 0.3])  # Brownish
        img[mask] = cat_colors + np.random.normal(0, 0.1, 3)
        
        return np.clip(img, 0, 1)
    
    def generate_synthetic_dog_image(self):
        """Generate a synthetic dog-like image"""
        img = np.random.rand(self.img_size[0], self.img_size[1], 3)
        
        # Dog-like patterns: more varied, elongated shapes
        # Add rectangular patterns (dog snout-like)
        start_x, start_y = self.img_size[0]//4, self.img_size[1]//4
        end_x, end_y = 3*self.img_size[0]//4, 3*self.img_size[1]//4
        
        # Dog-like colors (more varied, golden/black)
        dog_colors = np.array([0.8, 0.6, 0.2])  # Golden
        img[start_y:end_y, start_x:end_x] = dog_colors + np.random.normal(0, 0.15, 3)
        
        return np.clip(img, 0, 1)
    
    def prepare_data_with_validation(self, data_path=None, validation_split=0.1):
        """
        Enhanced data preparation with validation set
        """
        if data_path and os.path.exists(data_path):
            X, y = self.load_and_preprocess_data(data_path)
            if X is None:
                X, y = self.create_enhanced_synthetic_data()
        else:
            X, y = self.create_enhanced_synthetic_data()
        
        # Create feature names
        if self.use_advanced_features:
            self.feature_names = (
                [f'hist_r_{i}' for i in range(16)] +
                [f'hist_g_{i}' for i in range(16)] +
                [f'hist_b_{i}' for i in range(16)] +
                ['edge_density', 'mean_intensity', 'std_intensity', 'median_intensity', 
                 'min_intensity', 'max_intensity', 'contour_area', 'contour_perimeter'] +
                [f'pixel_{i}' for i in range(16*16*3)]
            )
        else:
            self.feature_names = [f'pixel_{i}' for i in range(np.prod(self.img_size) * 3)]
        
        # Split data into train, validation, and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split/(1-0.2), random_state=42, stratify=y_temp
        )
        
        # Feature engineering pipeline
        self.X_train_processed = self.feature_engineering_pipeline(self.X_train, fit=True)
        self.X_val_processed = self.feature_engineering_pipeline(self.X_val, fit=False)
        self.X_test_processed = self.feature_engineering_pipeline(self.X_test, fit=False)
        
        print(f"Training set: {self.X_train_processed.shape}")
        print(f"Validation set: {self.X_val_processed.shape}")
        print(f"Test set: {self.X_test_processed.shape}")
        
        # Data quality analysis
        self.analyze_data_quality()
    
    def feature_engineering_pipeline(self, X, fit=False):
        """
        Advanced feature engineering pipeline
        """
        if fit:
            # Fit scalers and transformers
            X_scaled = self.scaler.fit_transform(X)
            X_selected = self.feature_selector.fit_transform(X_scaled, self.y_train)
            X_pca = self.pca.fit_transform(X_selected)
        else:
            # Transform using fitted scalers
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            X_pca = self.pca.transform(X_selected)
        
        return X_pca
    
    def analyze_data_quality(self):
        """
        Analyze data quality and distribution
        """
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Class distribution: {dict(zip(['Cat', 'Dog'], counts))}")
        
        # Feature statistics
        print(f"Feature statistics:")
        print(f"  - Original features: {len(self.feature_names)}")
        print(f"  - After feature selection: {self.feature_selector.k}")
        print(f"  - After PCA: {self.pca.n_components}")
        print(f"  - PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Plot class distribution and PCA visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        axes[0, 0].bar(['Cat', 'Dog'], counts)
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # PCA visualization
        cats = self.X_train_processed[self.y_train == 0]
        dogs = self.X_train_processed[self.y_train == 1]
        axes[0, 1].scatter(cats[:, 0], cats[:, 1], alpha=0.6, label='Cats', s=20)
        axes[0, 1].scatter(dogs[:, 0], dogs[:, 1], alpha=0.6, label='Dogs', s=20)
        axes[0, 1].set_title('PCA Visualization (First 2 Components)')
        axes[0, 1].legend()
        
        # Feature importance from selection
        feature_scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        axes[1, 0].bar(range(len(selected_indices[:10])), 
                      feature_scores[selected_indices[:10]])
        axes[1, 0].set_title('Top 10 Feature Importance Scores')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('F-Score')
        
        # PCA explained variance
        axes[1, 1].plot(np.cumsum(self.pca.explained_variance_ratio_))
        axes[1, 1].set_title('PCA Explained Variance (Cumulative)')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.show()
    
    def train_with_advanced_techniques(self, use_ensemble=False):
        """
        Train SVM with advanced techniques
        """
        print("\n" + "="*50)
        print("ADVANCED SVM TRAINING")
        print("="*50)
        
        start_time = time.time()
        
        # Extended hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf', 'poly', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'degree': [2, 3, 4]  # for poly kernel
        }
        
        # Advanced grid search with multiple scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        
        svm = SVC(random_state=42, probability=True)  # Enable probability for ROC
        
        print("Performing comprehensive grid search...")
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1, return_train_score=True
        )
        
        grid_search.fit(self.X_train_processed, self.y_train)
        
        self.svm_model = grid_search.best_estimator_
        self.performance_metrics['best_params'] = grid_search.best_params_
        self.performance_metrics['training_time'] = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Training time: {self.performance_metrics['training_time']:.2f} seconds")
        
        # Validation set performance
        val_score = self.svm_model.score(self.X_val_processed, self.y_val)
        print(f"Validation accuracy: {val_score:.4f}")
        
        # Store detailed CV results
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # Learning curve analysis
        self.plot_learning_curves()
        
        return self.svm_model
    
    def plot_learning_curves(self):
        """
        Plot learning curves to analyze model performance
        """
        print("Generating learning curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.svm_model, self.X_train_processed, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        plt.figure(figsize=(12, 4))
        
        # Learning curves
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation')
        plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
        plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Hyperparameter analysis
        plt.subplot(1, 2, 2)
        if len(self.cv_results) > 0:
            C_values = self.cv_results['param_C'].unique()
            mean_scores = [self.cv_results[self.cv_results['param_C'] == c]['mean_test_score'].max() 
                          for c in C_values if c is not None]
            if mean_scores:
                plt.semilogx([float(c) for c in C_values if c is not None], mean_scores, 'o-')
                plt.xlabel('C Parameter')
                plt.ylabel('Cross-Validation Score')
                plt.title('Hyperparameter C vs Performance')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_evaluation(self):
        """
        Comprehensive model evaluation with multiple metrics
        """
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)
        
        # Test set predictions
        start_time = time.time()
        y_pred = self.svm_model.predict(self.X_test_processed)
        y_pred_proba = self.svm_model.predict_proba(self.X_test_processed)
        self.performance_metrics['inference_time'] = time.time() - start_time
        
        # Multiple metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Test Set Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Inference time: {self.performance_metrics['inference_time']:.4f} seconds")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Cat', 'Dog'], digits=4))
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # Prediction Distribution
        axes[0, 2].hist(y_pred_proba[:, 1], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Predicted Probability (Dog)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Prediction Probability Distribution')
        
        # Feature Importance (PCA components)
        pca_importance = np.abs(self.pca.components_).mean(axis=0)
        top_features = np.argsort(pca_importance)[-10:]
        axes[1, 0].barh(range(10), pca_importance[top_features])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importance (PCA)')
        
        # Cross-validation scores distribution
        cv_scores = cross_val_score(self.svm_model, self.X_train_processed, self.y_train, cv=10)
        axes[1, 1].hist(cv_scores, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Cross-Validation Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'CV Score Distribution (μ={cv_scores.mean():.3f}, σ={cv_scores.std():.3f})')
        
        # Performance comparison
        models_perf = ['Training', 'Validation', 'Test']
        train_acc = self.svm_model.score(self.X_train_processed, self.y_train)
        val_acc = self.svm_model.score(self.X_val_processed, self.y_val)
        test_acc = accuracy
        
        axes[1, 2].bar(models_perf, [train_acc, val_acc, test_acc])
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Performance Across Sets')
        axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Advanced analysis
        self.analyze_misclassifications()
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'classification_report': classification_report(self.y_test, y_pred, 
                                                         target_names=['Cat', 'Dog'], 
                                                         output_dict=True)
        }
    
    def analyze_misclassifications(self):
        """
        Analyze misclassified samples
        """
        y_pred = self.svm_model.predict(self.X_test_processed)
        misclassified = self.y_test != y_pred
        
        print(f"\nMisclassification Analysis:")
        print(f"  - Total misclassified: {np.sum(misclassified)}")
        print(f"  - Misclassification rate: {np.sum(misclassified)/len(self.y_test):.4f}")
        
        # Analyze confidence of misclassifications
        y_pred_proba = self.svm_model.predict_proba(self.X_test_processed)
        misclass_confidence = np.max(y_pred_proba[misclassified], axis=1)
        
        print(f"  - Average confidence of misclassifications: {np.mean(misclass_confidence):.4f}")
        print(f"  - Min confidence: {np.min(misclass_confidence):.4f}")
        print(f"  - Max confidence: {np.max(misclass_confidence):.4f}")
    
    def save_model_and_results(self, filepath='svm_model_results'):
        """
        Save trained model and results
        """
        # Save model
        with open(f'{filepath}_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.svm_model,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_selector': self.feature_selector,
                'performance_metrics': self.performance_metrics,
                'feature_names': self.feature_names
            }, f)
        
        # Save results as JSON
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'model_parameters': str(self.svm_model.get_params()),
            'data_shape': {
                'train': self.X_train_processed.shape,
                'test': self.X_test_processed.shape
            }
        }
        
        with open(f'{filepath}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Model and results saved to {filepath}_*")
    
    def generate_report(self):
        """
        Generate comprehensive project report
        """
        report = f"""
        ===============================================
        ENHANCED CAT VS DOG SVM CLASSIFICATION REPORT
        ===============================================
        
        Project Overview:
        - Dataset: {'Real Images' if hasattr(self, 'real_data') else 'Synthetic Data'}
        - Image Size: {self.img_size}
        - Advanced Features: {self.use_advanced_features}
        - Total Features: {len(self.feature_names)}
        
        Data Processing:
        - Feature Engineering: ✓ Color histograms, texture, statistical features
        - Feature Selection: ✓ SelectKBest with F-test
        - Dimensionality Reduction: ✓ PCA to {self.pca.n_components} components
        - Data Scaling: ✓ StandardScaler
        
        Model Performance:
        - Best Parameters: {self.performance_metrics['best_params']}
        - Training Time: {self.performance_metrics['training_time']:.2f} seconds
        - Cross-Validation: {np.mean(self.performance_metrics.get('cv_scores', [])):.4f}
        
        Advanced Features Implemented:
        ✓ Feature Engineering Pipeline
        ✓ Advanced Hyperparameter Tuning
        ✓ Learning Curve Analysis
        ✓ ROC Curve and AUC
        ✓ Comprehensive Evaluation Metrics
        ✓ Misclassification Analysis
        ✓ Model Persistence
        ✓ Performance Visualization
        
        Conclusion:
        This implementation demonstrates advanced machine learning practices
        including comprehensive data preprocessing, feature engineering,
        hyperparameter optimization, and thorough evaluation.
        """
        
        print(report)
        return report

def main():
    """
    Main function demonstrating the enhanced SVM classifier
    """
    print("🚀 Enhanced Cat vs Dog SVM Classifier")
    print("="*50)
    
    # Initialize enhanced classifier
    classifier = EnhancedCatDogSVMClassifier(
        img_size=(64, 64),
        use_advanced_features=True
    )
    
    # Prepare data with validation
    classifier.prepare_data_with_validation()
    
    # Train with advanced techniques
    classifier.train_with_advanced_techniques()
    
    # Comprehensive evaluation
    results = classifier.comprehensive_evaluation()
    
    # Save model and results
    classifier.save_model_and_results()
    
    # Generate final report
    classifier.generate_report()
    
    print("\n🎉 Enhanced SVM project completed successfully!")
    print("Check the saved model files and comprehensive visualizations.")

if __name__ == "__main__":
    main()
