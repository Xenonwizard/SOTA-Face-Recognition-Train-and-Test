# sota_fr_eval.py
import os, sys, glob, time, logging, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# ---- logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("sota_fr_eval")
warnings.filterwarnings("ignore")

# ---- paths ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = CURRENT_DIR  # code sits inside the cloned repo folder per your screenshot
DATASET_DIR = os.path.join(REPO_DIR, "celeb-dataset")
RESULTS_DIR = os.path.join(REPO_DIR, "elifiles", time.strftime("%y%m%d"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- import repo modules (do not copy) ----
sys.path.append(REPO_DIR)
# The repo provides model_zoo and feature extraction helpers
from model_zoo.iresnet import iresnet100, iresnet50  # example backbones
# NOTE: we’ll load state_dicts directly; the repo also has feature_extractor.py if you prefer a CLI.

# ---- model weight mapping (adjust filenames if needed) ----
MODEL_FILES = {
    "ArcFace":        "arcface-r100-ms1mv2.pth",
    "CurricularFace": "curricularface-r100-ms1mv2.pth",
    "MagFace":        "magface-r100-ms1mv2.pth",
    "AdaFace":        "adaface-r100-ms1mv2.pth",
    "CosFace":        "cosface-r100-ms1mv2.pth",
    "SphereFace":     "sphereface-r100-ms1mv2.pth",
    "UniFace":        "uniface-r100-ms1mv2.pth",
}
MODEL_DIR = os.path.join(REPO_DIR, "model")

# ---- face detector (MTCNN) ----
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=112, margin=20, post_process=True, device='cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    raise RuntimeError("Please `pip install facenet-pytorch` for MTCNN cropping") from e

# ---- transforms: FR-standard, not ImageNet ----
FR_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),                 # 0..1
    transforms.Normalize([0.5, 0.5, 0.5],  # -> [-1, 1]
                         [0.5, 0.5, 0.5]),
])

def list_images(root: str, is_training: bool) -> List[Tuple[str, str]]:
    """Return list of (path, class_name) respecting *_test split rule."""
    out = []
    ethnicities = ['caucasian', 'chinese', 'indian', 'malay']
    for eth in ethnicities:
        class_dirs = glob.glob(os.path.join(root, eth, "*/"))
        for cdir in class_dirs:
            cname = os.path.basename(cdir.rstrip("/"))
            is_test = cname.endswith("_test")
            if is_training and is_test:
                continue
            if (not is_training) and (not is_test):
                continue
            cname = cname.replace("_test", "")
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                for p in glob.glob(os.path.join(cdir, ext)):
                    out.append((p, cname))
    return out

def crop_face(img_path: str) -> Image.Image:
    """Detect and return an aligned 112x112 face. If detection fails, return None."""
    img = Image.open(img_path).convert("RGB")
    tensor = mtcnn(img)  # returns 3x112x112 tensor or None
    if tensor is None:
        return None
    # convert back to PIL for transform pipeline, or just normalize tensor directly
    pil = transforms.ToPILImage()(tensor)
    return pil

@dataclass
class Split:
    paths: List[str]
    names: List[str]

def build_splits() -> Tuple[Split, Split, LabelEncoder]:
    train_pairs = list_images(DATASET_DIR, is_training=True)
    test_pairs  = list_images(DATASET_DIR, is_training=False)
    if not train_pairs:
        raise FileNotFoundError(f"No training images found under {DATASET_DIR}")
    if not test_pairs:
        raise FileNotFoundError(f"No test images found under {DATASET_DIR}")

    train_paths, train_names = zip(*train_pairs)
    test_paths,  test_names  = zip(*test_pairs)

    # shared encoder with deterministic ordering
    enc = LabelEncoder()
    enc.fit(sorted(set(train_names) | set(test_names)))  # union, sorted
    return Split(list(train_paths), list(train_names)), Split(list(test_paths), list(test_names)), enc

class FaceCropDataset(Dataset):
    def __init__(self, split: Split, encoder: LabelEncoder):
        self.paths = split.paths
        self.names = split.names
        self.encoder = encoder

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        name = self.names[idx]
        y = self.encoder.transform([name])[0]
        face = crop_face(p)
        if face is None:
            # empty tensor — will be skipped later
            return None, y, name, p
        x = FR_TRANSFORM(face)
        return x, y, name, p

def collate_skip_none(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None
    xs, ys, names, paths = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), list(names), list(paths)

# --- Load FR backbone and return a feature extractor (embeddings) ---
def load_backbone(model_name: str, weight_path: str, device) -> nn.Module:
    # iresnet100 is commonly used for r100
    if "r100" in os.path.basename(weight_path) or "100" in model_name.lower():
        net = iresnet100(num_features=512)  # 512-d embeddings typical for ArcFace
    else:
        net = iresnet50(num_features=512)
    state = torch.load(weight_path, map_location='cpu')
    # accept both {'state_dict':...} and raw dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # clean 'module.' prefixes
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    log.info(f"{model_name}: loaded weights ({len(missing)} missing, {len(unexpected)} unexpected)")
    net.fc = nn.Identity()  # ensure outputs are embeddings
    net = net.to(device).eval()
    return net

@torch.no_grad()
def compute_embeddings(net: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    all_feat, all_y, all_names, all_paths = [], [], [], []
    for batch in loader:
        if batch is None:  # all skipped
            continue
        x, y, names, paths = batch
        x = x.to(device)
        feat = net(x)                      # [B, 512]
        feat = nn.functional.normalize(feat, p=2, dim=1)  # cosine-normalize
        all_feat.append(feat.cpu().numpy())
        all_y.append(y.numpy())
        all_names.extend(names)
        all_paths.extend(paths)
    if not all_feat:
        return np.empty((0, 512)), np.array([]), [], []
    return np.concatenate(all_feat, 0), np.concatenate(all_y, 0), all_names, all_paths

def evaluate_one_model(model_name: str, weight_file: str, device, enc: LabelEncoder) -> Dict:
    weight_path = os.path.join(MODEL_DIR, weight_file)
    if not os.path.exists(weight_path):
        log.warning(f"Missing weights for {model_name}: {weight_path}")
        return {"model": model_name, "error": "missing weights"}

    net = load_backbone(model_name, weight_path, device)

    # datasets & loaders
    train_ds = FaceCropDataset(TRAIN_SPLIT, enc)
    test_ds  = FaceCropDataset(TEST_SPLIT, enc)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0,
                              collate_fn=collate_skip_none, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0,
                              collate_fn=collate_skip_none, pin_memory=torch.cuda.is_available())

    # embeddings
    log.info(f"[{model_name}] Extracting train embeddings...")
    Xtr, ytr, tr_names, tr_paths = compute_embeddings(net, train_loader, device)
    log.info(f"[{model_name}] Extracting test embeddings...")
    Xte, yte, te_names, te_paths = compute_embeddings(net, test_loader, device)

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return {"model": model_name, "error": "no embeddings (face detection may have failed)"}

    # classifier: Logistic Regression on normalized embeddings
    clf = LogisticRegression(max_iter=2000, solver="liblinear", multi_class="ovr", n_jobs=1)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    yprob = None
    try:
        yprob = clf.predict_proba(Xte).max(axis=1)
    except Exception:
        pass

    # metrics (sklearn = standardized)
    acc = accuracy_score(yte, ypred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(yte, ypred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(yte, ypred, average="macro", zero_division=0)
    cls_report = classification_report(yte, ypred, target_names=list(enc.classes_), output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, ypred)

    # detailed CSV
    pred_names = enc.inverse_transform(ypred)
    true_names = enc.inverse_transform(yte)
    df = pd.DataFrame({
        "image_path": [os.path.basename(p) for p in te_paths],
        "true_celebrity": true_names,
        "predicted_celebrity": pred_names,
        "confidence": (yprob.tolist() if yprob is not None else [np.nan]*len(ypred)),
        "correct": (ypred == yte),
        "model": model_name
    })
    out_csv = os.path.join(RESULTS_DIR, f"{model_name}_detailed_results.csv")
    df.to_csv(out_csv, index=False)

    # per-celebrity summary
    celeb = df.groupby("true_celebrity").agg(correct_count=("correct","sum"),
                                             total_count=("correct","count"),
                                             avg_confidence=("confidence","mean"))
    celeb["accuracy"] = 100 * celeb["correct_count"] / celeb["total_count"]
    celeb.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_celebrity_metrics.csv"))

    return {
        "model": model_name,
        "overall_accuracy": 100*acc,
        "precision_weighted": 100*prec_w,
        "recall_weighted": 100*rec_w,
        "f1_weighted": 100*f1_w,
        "precision_macro": 100*prec_m,
        "recall_macro": 100*rec_m,
        "f1_macro": 100*f1_m,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "total_test_images": int(len(df)),
        "results_csv": out_csv
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Dataset: {DATASET_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # shared splits + encoder
    TRAIN_SPLIT, TEST_SPLIT, ENCODER = build_splits()

    all_metrics = []
    for model_name, fname in MODEL_FILES.items():
        log.info(f"=== Evaluating {model_name} ===")
        m = evaluate_one_model(model_name, fname, device, ENCODER)
        if "error" in m:
            log.warning(f"{model_name} skipped: {m['error']}")
        else:
            log.info(f"{model_name}: Acc={m['overall_accuracy']:.1f}%  F1(macro)={m['f1_macro']:.1f}%  "
                     f"F1(weighted)={m['f1_weighted']:.1f}%  N={m['total_test_images']}")
        all_metrics.append(m)

    # summary
    summary = pd.DataFrame(all_metrics)
    summary_ok = summary[~summary["overall_accuracy"].isna()].sort_values("overall_accuracy", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "comprehensive_summary.csv"), index=False)

    # pretty print
    log.info("\n================ FINAL EVALUATION SUMMARY ================\n")
    for i, row in summary_ok.reset_index(drop=True).iterrows():
        log.info(f"{i+1:>2}. {row['model']:<16} Acc {row['overall_accuracy']:5.1f}% | "
                 f"F1w {row['f1_weighted']:5.1f}% | F1m {row['f1_macro']:5.1f}% | N={int(row['total_test_images'])}")
    log.info(f"\nCSV saved to: {RESULTS_DIR}")
