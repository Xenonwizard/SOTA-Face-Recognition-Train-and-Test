import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101
from PIL import Image
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import json
from collections import defaultdict
import sys
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path for importing SOTA models
current_dir = os.path.dirname(os.path.abspath(__file__))
sota_repo_dir = os.path.join(current_dir, 'SOTA-Face-Recognition-Train-and-Test')
if os.path.exists(sota_repo_dir):
    sys.path.append(sota_repo_dir)
else:
    sys.path.append(current_dir)

DATASET_DIR = "./celeb-dataset" 

warnings.filterwarnings('ignore')

class CelebDataset(Dataset):
    """Celebrity dataset using PyTorch Dataset standard"""
    def __init__(self, data_dir, transform=None, is_training=True):
        # Use standard torchvision transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Standard ImageNet size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        self.samples = []
        self.labels = []
        self.celebrity_names = []
        
        # Use sklearn LabelEncoder for robust label encoding
        self.label_encoder = LabelEncoder()
        
        ethnicities = ['caucasian', 'chinese', 'indian', 'malay']
        celebrity_names = []
        
        for ethnicity in ethnicities:
            celeb_dirs = glob.glob(f"{data_dir}/{ethnicity}/*/")
            for celeb_dir in celeb_dirs:
                celeb_name = os.path.basename(celeb_dir.rstrip('/'))
                
                # Filter training vs testing sets
                if is_training and celeb_name.endswith('_test'):
                    continue
                elif not is_training and not celeb_name.endswith('_test'):
                    continue
                
                clean_name = celeb_name.replace('_test', '')
                celebrity_names.append(clean_name)
                
                # Support multiple image formats
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                images = []
                for ext in image_extensions:
                    images.extend(glob.glob(f"{celeb_dir}/{ext}"))
                
                for img_path in images:
                    self.samples.append(img_path)
                    self.celebrity_names.append(clean_name)
        
        # Use sklearn LabelEncoder for consistent label encoding
        if celebrity_names:
            self.label_encoder.fit(list(set(celebrity_names)))
            self.labels = self.label_encoder.transform(self.celebrity_names)
        
        logger.info(f"Dataset initialized: {len(self.samples)} samples, {len(self.label_encoder.classes_)} classes")
        logger.info(f"Celebrities: {list(self.label_encoder.classes_)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        celebrity_name = self.celebrity_names[idx]
        
        try:
            # Use PIL for reliable image loading
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, celebrity_name, img_path
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a zero tensor as fallback
            if self.transform:
                dummy_img = torch.zeros(3, 224, 224)  # Standard ImageNet size
            else:
                dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, label, celebrity_name, img_path

class PretrainedFaceModel(nn.Module):
    """Wrapper using standard torchvision models with pretrained weights"""
    def __init__(self, model_name, num_classes, pretrained_path=None):
        super().__init__()
        self.model_name = model_name
        
        # Use standard torchvision ResNet architectures
        if 'r100' in model_name.lower() or 'resnet101' in model_name.lower():
            self.backbone = resnet101(pretrained=True)  # Use ImageNet pretrained weights
            feature_dim = 2048
        else:
            self.backbone = resnet50(pretrained=True)   # Use ImageNet pretrained weights  
            feature_dim = 2048
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Load SOTA pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_sota_weights(pretrained_path)
        
        # Add classification head for celebrities
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Created {model_name} with {feature_dim} features -> {num_classes} classes")
    
    def _load_sota_weights(self, pretrained_path):
        """Load SOTA pretrained weights using PyTorch's load utilities"""
        try:
            logger.info(f"Loading SOTA weights from {pretrained_path}")
            
            # Use PyTorch's robust checkpoint loading
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Clean keys and filter backbone weights only
            backbone_state_dict = {}
            for key, value in state_dict.items():
                # Remove module prefix if present
                clean_key = key.replace('module.', '')
                # Only keep backbone weights (exclude classifier layers)
                if not any(clean_key.startswith(skip) for skip in ['fc', 'classifier', 'head', 'linear']):
                    backbone_state_dict[clean_key] = value
            
            # Load weights with error handling
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            logger.info(f"✅ SOTA weights loaded - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            
        except Exception as e:
            logger.warning(f"Failed to load SOTA weights: {e}")
            logger.info("Continuing with ImageNet pretrained weights")
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        # Classify using custom head
        logits = self.classifier(features)
        return logits, features

class SOTAFaceEvaluator:
    """Evaluation pipeline using industry-standard ML libraries"""
    def __init__(self, results_dir="results"):
        self.data_dir = DATASET_DIR
        self.results_dir = os.path.join(current_dir, results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Validate dataset
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        # Use PyTorch device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"Dataset path: {self.data_dir}")
        logger.info(f"Results path: {self.results_dir}")
        
        self.model_files = {}
        models_dir = os.path.join(current_dir, '..', '..', 'model')
        # Map to actual downloaded model files
        model_filenames= {
            'AdaFace': 'adaface-r100-ms1mv2.pth',
            'ArcFace': 'arcface-r100-ms1mv2.pth', 
            'ArcFace_Combined': 'combined-r100-ms1mv2.pth',
            'CosFace': 'cosface-r100-ms1mv2.pth',
            'CurricularFace': 'curricularface-r100-ms1mv2.pth',
            'MagFace': 'magface-r100-ms1mv2.pth',
            'SphereFace': 'sphereface-r100-ms1mv2.pth',
            'UniFace': 'uniface-r100-ms1mv2.pth'
        }

        # Build full paths and verify files exist
        for model_name, filename in model_filenames.items():
            full_path = os.path.join(models_dir, filename)
            if os.path.exists(full_path):
                self.model_files[model_name] = full_path
                logger.info(f"✅ Found {model_name}: {filename}")
            else:
                logger.warning(f"❌ Missing {model_name}: {full_path}")

            logger.info(f"Models directory: {models_dir}")
            logger.info(f"Found {len(self.model_files)} out of {len(model_filenames)} model files")
        
        # Initialize datasets using PyTorch utilities
        try:
            self.train_dataset = CelebDataset(self.data_dir, is_training=True)
            self.test_dataset = CelebDataset(self.data_dir, is_training=False)
            self.num_classes = len(self.train_dataset.label_encoder.classes_)
            
            if self.num_classes == 0:
                raise ValueError("No celebrity classes found in dataset!")
                
            logger.info(f"Initialized datasets: {self.num_classes} classes")
            
        except Exception as e:
            logger.error(f"Error initializing datasets: {e}")
            raise
        
    def train_model(self, model_name, epochs=10, lr=0.001, batch_size=32):
        """Train model using PyTorch training utilities"""
        logger.info(f"Training {model_name}")
        
        # # Get pretrained model path
        # model_file = self.model_files.get(model_name)
        # model_path = os.path.join(current_dir, 'model', model_file) if model_file else None

        model_path = self.model_files.get(model_name)  # This is already the full path
    
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found for {model_name}, using ResNet50 fallback")
            model_path = None
        
        # Create model with pretrained backbone
        model = PretrainedFaceModel(model_name, self.num_classes, model_path)
        model = model.to(self.device)
        
        # Use PyTorch optimizers and loss functions
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Use PyTorch DataLoader with proper settings
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=torch.cuda.is_available()
        )
        
        if len(train_loader) == 0:
            logger.warning(f"No training data found for {model_name}")
            return model
        
        # Training loop with progress tracking
        model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Use tqdm for progress tracking
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (images, labels, names, paths) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                logits, features = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate metrics using PyTorch utilities
                epoch_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.1f}%'
                })
            
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            logger.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
            
            # Save best model using PyTorch utilities
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy
                }, f"{self.results_dir}/{model_name}_best_checkpoint.pth")
        
        return model
    
    def evaluate_model(self, model, model_name):
        """Evaluate model using sklearn metrics"""
        logger.info(f"Evaluating {model_name}")
        
        # Use PyTorch DataLoader for evaluation
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0
        )
        
        model.eval()
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_celebrity_names = []
        all_image_paths = []
        
        with torch.no_grad():
            for images, labels, celebrity_names, image_paths in tqdm(test_loader, desc=f'Testing {model_name}'):
                try:
                    images = images.to(self.device)
                    logits, features = model(images)
                    
                    # Use PyTorch softmax for confidence calculation
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_confidences.extend(confidence.cpu().numpy())
                    all_celebrity_names.extend(celebrity_names)
                    all_image_paths.extend(image_paths)
                    
                except Exception as e:
                    logger.warning(f"Error processing image: {e}")
                    # Add dummy values for failed images
                    all_predictions.append(-1)
                    all_labels.extend(labels.numpy())
                    all_confidences.append(0.0)
                    all_celebrity_names.extend(celebrity_names)
                    all_image_paths.extend(image_paths)
        
        # Convert predictions back to celebrity names using label encoder
        predicted_names = []
        true_names = []
        
        for pred, true_label in zip(all_predictions, all_labels):
            if pred >= 0:  # Valid prediction
                predicted_names.append(self.test_dataset.label_encoder.classes_[pred])
            else:
                predicted_names.append('Error')
            true_names.append(self.test_dataset.label_encoder.classes_[true_label])
        
        # Create detailed results DataFrame using pandas
        results_df = pd.DataFrame({
            'image_path': [os.path.basename(path) for path in all_image_paths],
            'true_celebrity': true_names,
            'predicted_celebrity': predicted_names,
            'confidence': all_confidences,
            'correct': [t == p for t, p in zip(true_names, predicted_names)],
            'model': model_name
        })
        
        # Calculate metrics using sklearn
        valid_mask = np.array(all_predictions) >= 0
        valid_predictions = np.array(all_predictions)[valid_mask]
        valid_labels = np.array(all_labels)[valid_mask]
        
        if len(valid_predictions) > 0:
            # Use sklearn for comprehensive metrics
            accuracy = accuracy_score(valid_labels, valid_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_labels, valid_predictions, average='weighted', zero_division=0
            )
            
            # Generate classification report using sklearn
            class_report = classification_report(
                valid_labels, valid_predictions,
                target_names=self.test_dataset.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            
            # Generate confusion matrix using sklearn
            conf_matrix = confusion_matrix(valid_labels, valid_predictions)
            
        else:
            accuracy = precision = recall = f1 = 0.0
            class_report = {}
            conf_matrix = np.array([])
        
        return results_df, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def run_evaluation_pipeline(self):
        """Run complete evaluation pipeline"""
        logger.info("🚀 Starting SOTA Face Recognition Evaluation")
        
        all_results = []
        summary_metrics = []
        
        for model_name, model_file in self.model_files.items():
            model_path = os.path.join(current_dir, 'model', model_file)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                continue
                
            try:
                start_time = time.time()
                
                # Train the classification head
                logger.info(f"🔧 Training {model_name}...")
                trained_model = self.train_model(model_name, epochs=5)
                
                # Evaluate the model
                logger.info(f"🧪 Evaluating {model_name}...")
                results_df, metrics = self.evaluate_model(trained_model, model_name)
                
                # Save detailed results using pandas
                results_df.to_csv(f"{self.results_dir}/{model_name}_detailed_results.csv", index=False)
                
                # Calculate per-celebrity metrics using pandas groupby
                celebrity_metrics = results_df.groupby('true_celebrity').agg({
                    'correct': ['sum', 'count'],
                    'confidence': 'mean'
                }).round(3)
                
                celebrity_metrics.columns = ['correct_count', 'total_count', 'avg_confidence']
                celebrity_metrics['accuracy'] = (celebrity_metrics['correct_count'] / celebrity_metrics['total_count'] * 100).round(1)
                
                # Save per-celebrity results
                celebrity_metrics.to_csv(f"{self.results_dir}/{model_name}_celebrity_metrics.csv")
                
                # Add to summary
                summary_metrics.append({
                    'model': model_name,
                    'overall_accuracy': metrics['accuracy'] * 100,
                    'precision': metrics['precision'] * 100,
                    'recall': metrics['recall'] * 100,
                    'f1_score': metrics['f1_score'] * 100,
                    'total_test_images': len(results_df),
                    'training_time_seconds': time.time() - start_time
                })
                
                all_results.append(results_df)
                
                logger.info(f"✅ {model_name}: Accuracy = {metrics['accuracy']*100:.1f}%")
                
                # Print per-celebrity results
                logger.info(f"📊 Celebrity breakdown for {model_name}:")
                for celebrity, row in celebrity_metrics.iterrows():
                    logger.info(f"  {celebrity}: {row['accuracy']:.1f}% ({row['correct_count']}/{row['total_count']})")
                
            except Exception as e:
                logger.error(f"❌ Error with {model_name}: {e}")
                summary_metrics.append({
                    'model': model_name,
                    'overall_accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'total_test_images': 0,
                    'training_time_seconds': 0,
                    'error': str(e)[:200]
                })
        
        # Create comprehensive summary using pandas
        summary_df = pd.DataFrame(summary_metrics)
        summary_df = summary_df.sort_values('overall_accuracy', ascending=False)
        summary_df.to_csv(f"{self.results_dir}/comprehensive_summary.csv", index=False)
        
        # Print final rankings
        self._print_final_summary(summary_df)
        
        return summary_df
    
    def _print_final_summary(self, summary_df):
        """Print final summary using pandas operations"""
        logger.info(f"\n{'='*80}")
        logger.info("🎯 FINAL EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info("\n🏆 Model Rankings (by Overall Accuracy):")
        for idx, row in summary_df.iterrows():
            if pd.isna(row.get('error')):
                logger.info(f"  {idx+1}. {row['model']:15}: {row['overall_accuracy']:5.1f}% "
                          f"(Precision: {row['precision']:.1f}%, Recall: {row['recall']:.1f}%, F1: {row['f1_score']:.1f}%)")
            else:
                logger.info(f"  {idx+1}. {row['model']:15}: FAILED - {row.get('error', 'Unknown error')}")
        
        # Best model details
        if not summary_df.empty and pd.isna(summary_df.iloc[0].get('error')):
            best_model = summary_df.iloc[0]
            logger.info(f"\n🥇 Best Overall Model: {best_model['model']}")
            logger.info(f"   Accuracy: {best_model['overall_accuracy']:.1f}%")
            logger.info(f"   Precision: {best_model['precision']:.1f}%")
            logger.info(f"   Recall: {best_model['recall']:.1f}%")
            logger.info(f"   F1-Score: {best_model['f1_score']:.1f}%")
            logger.info(f"   Training Time: {best_model['training_time_seconds']:.1f} seconds")

# Main execution
if __name__ == "__main__":
    try:
        # Initialize evaluator
        evaluator = SOTAFaceEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.run_evaluation_pipeline()
        
        logger.info("\n✅ All evaluations completed!")
        logger.info(f"📁 Detailed results saved to: {evaluator.results_dir}/")
        logger.info(f"📊 Summary: {evaluator.results_dir}/comprehensive_summary.csv")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise