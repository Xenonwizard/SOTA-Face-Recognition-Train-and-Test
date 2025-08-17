# celeb_benchmark_cosine_iresnet.py
# Evaluate pretrained SOTA FR (.pth) with repo's iresnet backbone + cosine classifiers.
# Hardcoded absolute paths. Standardized sklearn metrics only.

import sys, glob, logging, warnings
from typing import List, Tuple, Dict
from dataclasses import dataclass
import glob
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# Make repo root importable regardless of where this script is run from
import sys
from pathlib import Path

# this file lives at <repo>/elifiles/130825/sota_face_recognition.py
ROOT = Path(__file__).resolve().parents[2]  # -> <repo>
sys.path.insert(0, str(ROOT))

# Directories derived from ROOT (no hardcoding, no case issues)
REPO_DIR    = ROOT
DATASET_DIR = ROOT / "celeb-dataset"
MODEL_DIR   = ROOT / "model"
RESULTS_DIR = ROOT / "elifiles" / "130825"

# (Optional) log what we resolved
logging.info(f"ROOT={ROOT}")
logging.info(f"DATASET_DIR exists? {DATASET_DIR.exists()}")

MODEL_FILES: Dict[str, str] = {
    # keep only the ones you actually have; others will be skipped gracefully
    "ArcFace":        "/home/ssm-user/SOTA-FR-train-and-test/model/arcface-r100-ms1mv2.pth",
    "CurricularFace": "/home/ssm-user/SOTA-FR-train-and-test/model/curricularface-r100-ms1mv2.pth",
    "MagFace":        "/home/ssm-user/SOTA-FR-train-and-test/model/magface-r100-ms1mv2.pth",
    "AdaFace":        "/home/ssm-user/SOTA-FR-train-and-test/model/adaface-r100-ms1mv2.pth",
    "CosFace":        "/home/ssm-user/SOTA-FR-train-and-test/model/cosface-r100-ms1mv2.pth",
    "SphereFace":     "/home/ssm-user/SOTA-FR-train-and-test/model/sphereface-r100-ms1mv2.pth",
    "UniFace":        "/home/ssm-user/SOTA-FR-train-and-test/model/uniface-r100-ms1mv2.pth",
}

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("cosine_iresnet")
warnings.filterwarnings("ignore")

# -------------------- IMPORT REPO BACKBONE --------------------
from model.iresnet import iresnet as iresnet_factory

# -------------------- BACKBONE LOADER --------------------
def load_iresnet_r100(weight_path: str) -> nn.Module:
    # iresnet.py exposes a generic factory, not iresnet100()
    net = iresnet_factory("100", num_features=512)  # 512-d embeddings
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    log.info(f"Loaded {weight_path}: missing={len(missing)} unexpected={len(unexpected)}")
    net.fc = nn.Identity()  # remove classifier head; keep 512-d features
    net = net.to(DEVICE).eval()
    return net



# -------------------- FACE DETECTOR (MTCNN) --------------------
try:
    from facenet_pytorch import MTCNN
except Exception as e:
    raise RuntimeError("Install facenet-pytorch: `pip install facenet-pytorch`") from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=112, margin=20, post_process=True, device=str(DEVICE))

# 112x112 RGB → [-1,1] (mean=.5,std=.5) which these FR models expect
NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# -------------------- DATA --------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(root: Path, is_training: bool) -> List[Tuple[str, str]]:
    """
    Walk celeb-dataset/<ethnicity>/<class or class_test>/** and collect images.
    For training split, skip folders whose *final* name ends with '_test';
    for test split, keep only those.
    """
    out: List[Tuple[str,str]] = []
    ethnicities = ["caucasian", "chinese", "indian", "malay"]
    for eth in ethnicities:
        eth_dir = root / eth
        if not eth_dir.is_dir():
            continue
        # class dirs are immediate children of each ethnicity
        for cls_dir in sorted([p for p in eth_dir.iterdir() if p.is_dir()]):
            cname = cls_dir.name
            is_test_dir = cname.endswith("_test")
            if is_training and is_test_dir:
                continue
            if (not is_training) and (not is_test_dir):
                continue
            cname_clean = cname.replace("_test", "")
            # grab images at any depth under this class folder
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    out.append((str(p), cname_clean))
    return out

def crop_face(img_path: str):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    t = mtcnn(img)  # 3x112x112 or None
    return t

@dataclass
class Split:
    paths: List[str]
    names: List[str]

def build_splits() -> Tuple[Split, Split, LabelEncoder]:
    tr_pairs = list_images(DATASET_DIR, is_training=True)
    te_pairs = list_images(DATASET_DIR, is_training=False)
    if len(tr_pairs) == 0:
        raise FileNotFoundError(f"No training images under {DATASET_DIR}")
    if len(te_pairs) == 0:
        raise FileNotFoundError(f"No test images under {DATASET_DIR}")
    tr_paths, tr_names = zip(*tr_pairs)
    te_paths, te_names = zip(*te_pairs)
    enc = LabelEncoder()
    enc.fit(sorted(list(set(tr_names) | set(te_names))))  # deterministic union
    return Split(list(tr_paths), list(tr_names)), Split(list(te_paths), list(te_names)), enc

class FaceCropDataset(Dataset):
    def __init__(self, split: Split, encoder: LabelEncoder):
        self.paths = split.paths
        self.names = split.names
        self.encoder = encoder

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y_name = self.names[idx]
        y = int(self.encoder.transform([y_name])[0])
        t = crop_face(p)
        if t is None:
            return None, y, y_name, p
        x = NORM(t)  # already 112x112 from MTCNN
        return x, y, y_name, p

def collate_skip_none(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None
    xs, ys, names, paths = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), list(names), list(paths)

# -------------------- BACKBONE LOADER --------------------
def load_iresnet_r100(weight_path: str) -> nn.Module:
    net = iresnet100(num_features=512)  # 512-d embeddings
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    log.info(f"Loaded {weight_path}: missing={len(missing)} unexpected={len(unexpected)}")
    net.fc = nn.Identity()
    net = net.to(DEVICE).eval()
    return net

@torch.no_grad()
def compute_embeddings(net: nn.Module, loader: DataLoader):
    all_feat, all_y, all_names, all_paths = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        x, y, names, paths = batch
        x = x.to(DEVICE)
        z = net(x)                               # [B, 512]
        z = nn.functional.normalize(z, p=2, dim=1)
        all_feat.append(z.cpu().numpy())
        all_y.append(y.numpy())
        all_names.extend(names)
        all_paths.extend(paths)
    if not all_feat:
        return np.empty((0,512)), np.array([]), [], []
    return np.concatenate(all_feat, 0), np.concatenate(all_y, 0), all_names, all_paths

# -------------------- COSINE CLASSIFIERS --------------------
def cosine_prototypes(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    protos = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        if not np.any(mask):
            continue
        m = X[mask].mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        protos[c] = m.astype(np.float32)
    return protos

def predict_by_prototypes(X: np.ndarray, protos: np.ndarray):
    scores = X @ protos.T  # cosine because rows are L2-normalized
    return scores.argmax(axis=1), scores.max(axis=1)

def predict_by_1nn(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray):
    scores = test_X @ train_X.T
    idx = scores.argmax(axis=1)
    return train_y[idx], scores.max(axis=1)

# -------------------- EVALUATION --------------------
def evaluate_one_model(model_name: str, weight_path: str, enc: LabelEncoder, train_split: Split, test_split: Split) -> Dict:
    train_ds = FaceCropDataset(train_split, enc)
    test_ds  = FaceCropDataset(test_split, enc)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0,
                              collate_fn=collate_skip_none, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0,
                              collate_fn=collate_skip_none, pin_memory=torch.cuda.is_available())

    net = load_iresnet_r100(weight_path)
    logging.info(f"[{model_name}] extracting train embeddings ...")
    Xtr, ytr, _, _ = compute_embeddings(net, train_loader)
    logging.info(f"[{model_name}] extracting test embeddings ...")
    Xte, yte, te_names, te_paths = compute_embeddings(net, test_loader)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return {"model": model_name, "error": "no embeddings (face detection may have failed)"}

    n_classes = len(enc.classes_)

    # Prototype classifier (per-class mean)
    protos = cosine_prototypes(Xtr, ytr, n_classes)
    ypred_p, conf_p = predict_by_prototypes(Xte, protos)

    # 1-NN cosine for reference
    ypred_nn, conf_nn = predict_by_1nn(Xtr, ytr, Xte)

    # --- Metrics for prototype classifier (standardized via sklearn) ---
    acc = accuracy_score(yte, ypred_p)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(yte, ypred_p, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(yte, ypred_p, average="macro", zero_division=0)
    cls_report = classification_report(yte, ypred_p, target_names=list(enc.classes_), output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, ypred_p)

    # Save detailed CSVs
    pred_names_p = enc.inverse_transform(ypred_p)
    true_names = enc.inverse_transform(yte)
    df = pd.DataFrame({
        "image_path": [p.split("/")[-1] for p in te_paths],
        "true_celebrity": true_names,
        "predicted_celebrity": pred_names_p,
        "confidence": conf_p,
        "correct": (ypred_p == yte),
        "model": model_name,
        "classifier": "prototype"
    })
    df.to_csv(f"{RESULTS_DIR}/{model_name}_detailed_results_prototype.csv", index=False)

    pred_names_nn = enc.inverse_transform(ypred_nn)
    df_nn = pd.DataFrame({
        "image_path": [p.split("/")[-1] for p in te_paths],
        "true_celebrity": true_names,
        "predicted_celebrity": pred_names_nn,
        "confidence": conf_nn,
        "correct": (ypred_nn == yte),
        "model": model_name,
        "classifier": "1NN"
    })
    df_nn.to_csv(f"{RESULTS_DIR}/{model_name}_detailed_results_1nn.csv", index=False)

    # Per-celeb breakdown (prototype)
    celeb = df.groupby("true_celebrity").agg(correct_count=("correct","sum"),
                                             total_count=("correct","count"),
                                             avg_confidence=("confidence","mean"))
    celeb["accuracy"] = (100.0 * celeb["correct_count"] / celeb["total_count"]).round(2)
    celeb.to_csv(f"{RESULTS_DIR}/{model_name}_celebrity_metrics_prototype.csv")

    return {
        "model": model_name,
        "overall_accuracy": 100.0 * acc,
        "precision_weighted": 100.0 * prec_w,
        "recall_weighted": 100.0 * rec_w,
        "f1_weighted": 100.0 * f1_w,
        "precision_macro": 100.0 * prec_m,
        "recall_macro": 100.0 * rec_m,
        "f1_macro": 100.0 * f1_m,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "total_test_images": int(len(df)),
        "results_csv_prototype": f"{RESULTS_DIR}/{model_name}_detailed_results_prototype.csv",
        "results_csv_1nn": f"{RESULTS_DIR}/{model_name}_detailed_results_1nn.csv"
    }

# -------------------- MAIN --------------------
if __name__ == "__main__":
    logging.info(f"Device:  {DEVICE}")
    logging.info(f"Dataset: {DATASET_DIR}")
    logging.info(f"Models:  {MODEL_DIR}")
    logging.info(f"Results: {RESULTS_DIR}")

    TRAIN_SPLIT, TEST_SPLIT, ENCODER = build_splits()

    all_metrics = []
    for model_name, wpath in MODEL_FILES.items():
        try:
            open(wpath, "rb").close()
        except Exception:
            logging.warning(f"Skipping {model_name}: weights not found at {wpath}")
            all_metrics.append({"model": model_name, "error": "missing weights"})
            continue

        logging.info(f"=== Evaluating {model_name} (iresnet r100, cosine) ===")
        m = evaluate_one_model(model_name, wpath, ENCODER, TRAIN_SPLIT, TEST_SPLIT)
        all_metrics.append(m)
        if "error" not in m:
            logging.info(f"{model_name}: Acc={m['overall_accuracy']:.1f}%  "
                         f"F1w={m['f1_weighted']:.1f}%  F1m={m['f1_macro']:.1f}%  "
                         f"N={m['total_test_images']}")

    summary = pd.DataFrame(all_metrics)
    # Ensure metric columns exist even if no model produced them
    metric_cols = ["overall_accuracy", "f1_weighted", "f1_macro", "total_test_images"]
    for c in metric_cols:
        if c not in summary.columns:
            summary[c] = np.nan

    summary.to_csv(f"{RESULTS_DIR}/comprehensive_summary_cosine.csv", index=False)

    logging.info("\n================ FINAL EVALUATION SUMMARY (Cosine) ================\n")

    has_results = summary["overall_accuracy"].notna().any()

    if has_results:
        summary_ok = (
            summary[summary["overall_accuracy"].notna()]
            .sort_values("overall_accuracy", ascending=False)
            .reset_index(drop=True)
        )

        for i, row in summary_ok.iterrows():
            acc  = row["overall_accuracy"]
            f1w  = row["f1_weighted"]
            f1m  = row["f1_macro"]
            nimg = row["total_test_images"]

            # print safely even if some are NaN
            acc_s  = "nan" if pd.isna(acc) else f"{acc:5.1f}%"
            f1w_s  = "nan" if pd.isna(f1w) else f"{f1w:5.1f}%"
            f1m_s  = "nan" if pd.isna(f1m) else f"{f1m:5.1f}%"
            nimg_s = "?"   if pd.isna(nimg) else f"{int(nimg)}"

            logging.info(f"{i+1:>2}. {row.get('model','<?>'):<16} Acc {acc_s} | F1w {f1w_s} | F1m {f1m_s} | N={nimg_s}")
    else:
        # Nothing succeeded—print a helpful table of errors
        logging.info("No successful evaluations. Details:")
        logging.info("\n" + summary.to_string(index=False))
        logging.info(f"\nCSV saved to: {RESULTS_DIR}")