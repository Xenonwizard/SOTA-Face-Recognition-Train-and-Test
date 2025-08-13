import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import warnings
import cv2
import json
from collections import defaultdict
import sys

# Add current directory to path for importing SOTA models
current_dir = os.path.dirname(os.path.abspath(__file__))
sota_repo_dir = os.path.join(current_dir, 'SOTA-Face-Recognition-Train-and-Test')
if os.path.exists(sota_repo_dir):
    sys.path.append(sota_repo_dir)
else:
    # If we're already in the SOTA repo directory
    sys.path.append(current_dir)

warnings.filterwarnings('ignore')

class CelebDataset(Dataset):
    """Celebrity dataset for training/testing"""
    def __init__(self, data_dir, transform=None, is_training=True):
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.samples = []
        self.label_map = {}
        ethnicities = ['caucasian', 'chinese', 'indian', 'malay']
        
        label_idx = 0
        for ethnicity in ethnicities:
            celeb_dirs = glob.glob(f"{data_dir}/{ethnicity}/*/")
            for celeb_dir in celeb_dirs:
                celeb_name = os.path.basename(celeb_dir.rstrip('/'))
                
                # Skip test folders for training, include only test for testing
                if is_training and celeb_name.endswith('_test'):
                    continue
                elif not is_training and not celeb_name.endswith('_test'):
                    continue
                
                clean_name = celeb_name.replace('_test', '')
                if clean_name not in self.label_map:
                    self.label_map[clean_name] = label_idx
                    label_idx += 1
                
                images = glob.glob(f"{celeb_dir}/*.jpg")
                for img_path in images:
                    self.samples.append((img_path, self.label_map[clean_name], clean_name))
        
        print(f"Dataset: {len(self.samples)} samples, {len(self.label_map)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, name, img_path
        except:
            # Return dummy data if image fails to load
            dummy_img = torch.zeros(3, 112, 112)
            return dummy_img, label, name, img_path

class SOTAFaceModel(nn.Module):
    """Wrapper for SOTA face recognition models with classification head"""
    def __init__(self, backbone_name, num_classes, pretrained_path=None):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Import the specific backbone (this would need actual model implementations)
        # For demo purposes, using a simple ResNet-like structure
        self.backbone = self._get_backbone(backbone_name)
        self.classifier = nn.Linear(512, num_classes)
        
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
    
    def _get_backbone(self, name):
        """Get backbone architecture from SOTA repo models"""
        try:
            # Try to import from the SOTA repo structure
            if name == 'SphereFace':
                from model.sphere_net import sphere20a
                backbone = sphere20a()
            elif name == 'CosFace':
                from model.resnet import resnet50
                backbone = resnet50(num_classes=10575)  # Remove final FC later
            elif name == 'ArcFace':
                from model.resnet import resnet50
                backbone = resnet50(num_classes=85742)  # Remove final FC later
            elif name == 'ArcFace_Combined':
                from model.resnet import resnet50
                backbone = resnet50(num_classes=85742)
            elif name == 'CurricularFace':
                from model.resnet import resnet50
                backbone = resnet50(num_classes=85742)
            elif name == 'MagFace':
                from model.iresnet import iresnet50
                backbone = iresnet50()
            elif name == 'AdaFace':
                from model.iresnet import iresnet50
                backbone = iresnet50()
            elif name == 'UniFace':
                from model.iresnet import iresnet50
                backbone = iresnet50()
            else:
                # Fallback to torchvision
                from torchvision.models import resnet50
                backbone = resnet50(pretrained=True)
            
            # Remove final classification layer and replace with identity
            if hasattr(backbone, 'fc'):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, 'classifier'):
                backbone.classifier = nn.Identity()
            elif hasattr(backbone, 'head'):
                backbone.head = nn.Identity()
                
            return backbone
            
        except ImportError as e:
            print(f"Could not import {name} model, using ResNet50 fallback: {e}")
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            backbone.fc = nn.Identity()
            return backbone
    
    def _load_pretrained(self, path):
        """Load pretrained weights from SOTA repo checkpoints"""
        try:
            # Look for pretrained weights in the model directory
            model_dir = os.path.join(current_dir, 'model')
            if not os.path.exists(model_dir):
                model_dir = os.path.join(current_dir, 'SOTA-Face-Recognition-Train-and-Test', 'model')
            
            # Common checkpoint paths for different models
            checkpoint_paths = {
                'SphereFace': os.path.join(model_dir, 'sphere20a_20171020.pth'),
                'CosFace': os.path.join(model_dir, 'cosface_resnet50.pth'),
                'ArcFace': os.path.join(model_dir, 'arcface_resnet50.pth'),
                'ArcFace_Combined': os.path.join(model_dir, 'arcface_combined.pth'),
                'CurricularFace': os.path.join(model_dir, 'curricular_face.pth'),
                'MagFace': os.path.join(model_dir, 'magface_iresnet50.pth'),
                'AdaFace': os.path.join(model_dir, 'adaface_ir50_ms1mv2.ckpt'),
                'UniFace': os.path.join(model_dir, 'uniface_ir50.pth')
            }
            
            model_path = checkpoint_paths.get(self.backbone_name, path)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                
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
                
                # Clean up keys (remove 'module.' prefix if present)
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]  # Remove 'module.' prefix
                    if not k.startswith('fc') and not k.startswith('classifier'):
                        cleaned_state_dict[k] = v
                
                # Load weights with strict=False to ignore size mismatches
                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print(f"Loaded pretrained {self.backbone_name} from {model_path}")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                print(f"Pretrained weights not found at {model_path}, using random initialization")
                
        except Exception as e:
            print(f"Failed to load pretrained weights for {self.backbone_name}: {e}")
            print("Using random initialization")
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits, features

class SOTAFaceTrainer:
    """Training pipeline for SOTA face recognition models"""
    def __init__(self, data_dir="celeb-dataset", results_dir="results"):
        # Set paths relative to current directory
        self.data_dir = os.path.join(current_dir, data_dir)
        self.results_dir = os.path.join(current_dir, results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Check if dataset exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Dataset path: {self.data_dir}")
        print(f"Results path: {self.results_dir}")
        
        # Available models from the repo
        self.models = [
            'SphereFace', 'CosFace', 'ArcFace', 'ArcFace_Combined',
            'CurricularFace', 'MagFace', 'AdaFace', 'UniFace'
        ]
        
        # Face detectors - using OpenCV and MTCNN equivalents
        self.face_detectors = ['opencv', 'mtcnn_equivalent']
        
        # Initialize datasets
        try:
            self.train_dataset = CelebDataset(self.data_dir, is_training=True)
            self.test_dataset = CelebDataset(self.data_dir, is_training=False)
            self.num_classes = len(self.train_dataset.label_map)
            print(f"Initialized datasets: {self.num_classes} classes")
        except Exception as e:
            print(f"Error initializing datasets: {e}")
            raise
        
    def train_model(self, model_name, epochs=10, lr=0.001):
        """Train/fine-tune a specific model"""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = SOTAFaceModel(model_name, self.num_classes)
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=2)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels, names, paths) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                logits, features = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{self.results_dir}/{model_name}_best.pth")
        
        return model
    
    def test_with_detector(self, model, model_name, detector_name):
        """Test model with specific face detector"""
        print(f"\nTesting {model_name} with {detector_name} detector")
        
        # Initialize face detector
        if detector_name == 'retinaface':
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
        else:  # mtcnn
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
        
        results = []
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for images, labels, names, paths in test_loader:
                try:
                    # Load original image for face detection
                    img_path = paths[0]
                    img = cv2.imread(img_path)
                    
                    # Detect faces
                    faces = app.get(img)
                    
                    if len(faces) > 0:
                        # Use the largest face
                        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        
                        # Extract face region
                        bbox = face.bbox.astype(int)
                        face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        
                        if face_img.size > 0:
                            # Preprocess for model
                            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                            transform = transforms.Compose([
                                transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
                            face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
                            
                            # Predict
                            logits, features = model(face_tensor)
                            predicted = logits.argmax(dim=1).item()
                            confidence = torch.softmax(logits, dim=1).max().item()
                            
                            # Get predicted name
                            label_to_name = {v: k for k, v in self.train_dataset.label_map.items()}
                            predicted_name = label_to_name.get(predicted, 'Unknown')
                            
                            results.append({
                                'image': os.path.basename(img_path),
                                'true_label': names[0],
                                'predicted': predicted_name,
                                'confidence': confidence,
                                'model': model_name,
                                'detector': detector_name,
                                'status': 'success'
                            })
                        else:
                            results.append({
                                'image': os.path.basename(img_path),
                                'true_label': names[0],
                                'predicted': 'No_face_extracted',
                                'confidence': 0.0,
                                'model': model_name,
                                'detector': detector_name,
                                'status': 'no_face_extracted'
                            })
                    else:
                        results.append({
                            'image': os.path.basename(img_path),
                            'true_label': names[0],
                            'predicted': 'No_face_detected',
                            'confidence': 0.0,
                            'model': model_name,
                            'detector': detector_name,
                            'status': 'no_face_detected'
                        })
                
                except Exception as e:
                    results.append({
                        'image': os.path.basename(paths[0]) if paths else 'unknown',
                        'true_label': names[0] if names else 'unknown',
                        'predicted': 'Error',
                        'confidence': 0.0,
                        'model': model_name,
                        'detector': detector_name,
                        'status': f'error: {str(e)[:50]}'
                    })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, df):
        """Calculate classification metrics"""
        # Filter out error cases
        valid_df = df[df['predicted'].isin(self.train_dataset.label_map.keys())]
        
        if len(valid_df) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'detection_rate': 0.0,
                'total_samples': len(df)
            }
        
        # Calculate metrics
        y_true = valid_df['true_label'].values
        y_pred = valid_df['predicted'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        detection_rate = len(valid_df) / len(df)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': detection_rate,
            'total_samples': len(df)
        }
    
    def run_comprehensive_experiment(self):
        """Run complete training and testing pipeline"""
        print("🚀 Starting SOTA Face Recognition Experiments")
        
        all_results = []
        summary_results = []
        
        for model_name in self.models:
            try:
                start_time = time.time()
                
                # Train model
                trained_model = self.train_model(model_name, epochs=5)  # Reduced for demo
                
                # Test with each detector
                for detector in self.face_detectors:
                    try:
                        print(f"\n🔍 Testing {model_name} with {detector}")
                        
                        test_results = self.test_with_detector(trained_model, model_name, detector)
                        metrics = self.calculate_metrics(test_results)
                        
                        # Save individual results
                        result_file = f"{self.results_dir}/{model_name}_{detector}_results.csv"
                        test_results.to_csv(result_file, index=False)
                        
                        # Add to summary
                        summary_results.append({
                            'model': model_name,
                            'detector': detector,
                            'accuracy': metrics['accuracy'] * 100,
                            'precision': metrics['precision'] * 100,
                            'recall': metrics['recall'] * 100,
                            'f1_score': metrics['f1'] * 100,
                            'detection_rate': metrics['detection_rate'] * 100,
                            'total_samples': metrics['total_samples'],
                            'training_time': time.time() - start_time
                        })
                        
                        all_results.append(test_results)
                        
                        print(f"✅ {model_name} + {detector}: "
                              f"Acc={metrics['accuracy']*100:.1f}%, "
                              f"Det={metrics['detection_rate']*100:.1f}%")
                        
                    except Exception as e:
                        print(f"❌ Error testing {model_name} + {detector}: {e}")
                        summary_results.append({
                            'model': model_name,
                            'detector': detector,
                            'accuracy': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1_score': 0.0,
                            'detection_rate': 0.0,
                            'total_samples': 0,
                            'training_time': time.time() - start_time,
                            'error': str(e)[:100]
                        })
            
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        # Save summary results
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(f"{self.results_dir}/experiment_summary.csv", index=False)
        
        # Print final results
        self.print_summary(summary_df)
        
        return summary_df
    
    def print_summary(self, summary_df):
        """Print experiment summary"""
        print(f"\n{'='*80}")
        print("🎯 EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        # Best overall
        if not summary_df.empty:
            best_combo = summary_df.loc[summary_df['accuracy'].idxmax()]
            print(f"🥇 Best Combination: {best_combo['model']} + {best_combo['detector']}")
            print(f"   Accuracy: {best_combo['accuracy']:.1f}%")
            print(f"   F1-Score: {best_combo['f1_score']:.1f}%")
            print(f"   Detection Rate: {best_combo['detection_rate']:.1f}%")
            
            # Best per model
            print(f"\n📊 Best Results per Model:")
            for model in summary_df['model'].unique():
                model_results = summary_df[summary_df['model'] == model]
                if not model_results.empty:
                    best_model = model_results.loc[model_results['accuracy'].idxmax()]
                    print(f"  {model:15}: {best_model['accuracy']:5.1f}% "
                          f"(with {best_model['detector']})")
            
            # Best per detector
            print(f"\n🔍 Best Results per Detector:")
            for detector in summary_df['detector'].unique():
                detector_results = summary_df[summary_df['detector'] == detector]
                if not detector_results.empty:
                    best_detector = detector_results.loc[detector_results['accuracy'].idxmax()]
                    print(f"  {detector:15}: {best_detector['accuracy']:5.1f}% "
                          f"(with {best_detector['model']})")

# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = SOTAFaceTrainer(data_dir="celeb-dataset")
    
    # Run experiments
    results = trainer.run_comprehensive_experiment()
    
    print("\n✅ All experiments completed!")
    print(f"📁 Results saved to: {trainer.results_dir}/")
    print(f"📊 Summary: {trainer.results_dir}/experiment_summary.csv")