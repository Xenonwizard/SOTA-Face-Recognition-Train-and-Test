# celeb_benchmark_official_sota.py
# Evaluate pretrained SOTA FR models using official SOTA-FR-train-and-test repository methods
# Uses official feature extraction + custom celebrity classification

import sys
import logging
import warnings
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# Hardcoded paths - running from within SOTA-Face-Recognition-Train-and-Test directory
REPO_DIR = Path(".")  # Current directory
DATASET_DIR = Path("./celeb-dataset")
MODEL_DIR = Path("./model") 
RESULTS_DIR = Path("./elifiles/130825")
TEMP_DIR = Path("./temp_sota_fr")

# Model files mapping
MODEL_FILES = {
    "ArcFace": MODEL_DIR / "arcface-r100-ms1mv2.pth",
    "CurricularFace": MODEL_DIR / "curricularface-r100-ms1mv2.pth", 
    "MagFace": MODEL_DIR / "magface-r100-ms1mv2.pth",
    "AdaFace": MODEL_DIR / "adaface-r100-ms1mv2.pth",
    "CosFace": MODEL_DIR / "cosface-r100-ms1mv2.pth",
    "SphereFace": MODEL_DIR / "sphereface-r100-ms1mv2.pth",
    "UniFace": MODEL_DIR / "uniface-r100-ms1mv2.pth",
}

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sota_official_eval")
warnings.filterwarnings("ignore")

# Image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass
class Split:
    paths: List[str]
    names: List[str]

def setup_environment():
    """Setup directories - repository already exists since we're running from within it"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Running from repository directory: {REPO_DIR.resolve()}")
    log.info(f"Dataset directory: {DATASET_DIR.resolve()}")
    log.info(f"Model directory: {MODEL_DIR.resolve()}")
    log.info(f"Results directory: {RESULTS_DIR.resolve()}")

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

def build_splits() -> Tuple[Split, Split, LabelEncoder]:
    """Build training and test splits with label encoder"""
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

def create_image_list_file(split: Split, filename: str) -> str:
    """
    Create a single-column CSV of absolute image paths with ALL fields quoted.
    This is robust to commas/spaces in paths when pandas.read_csv parses it.
    """
    import csv
    filepath = TEMP_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
            escapechar="\\",
        )
        for p in split.paths:
            writer.writerow([str(Path(p).resolve())])
    return str(filepath)


# --- replace this function ---

def extract_features_official(model_path: str,
                              image_list_file: str,
                              feature_output_base: str,
                              prefix: str) -> str:
    """
    Run the official extractor and robustly find the produced .npy, regardless of
    whether --destination is treated as a folder, a file base, or ignored.
    Returns a normalized .npy path under TEMP_DIR (e.g., TEMP_DIR/<prefix>_features.npy).
    """
    try:
        out_base = Path(feature_output_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot current npys & time
        import time
        start_time = time.time()
        before = {p.resolve() for root in [REPO_DIR, TEMP_DIR] for p in root.rglob("*.npy")}

        cmd = [
            "python3", "./feature_extractor.py",
            "--model_path", model_path,
            "--model", "iresnet",
            "--depth", "100",
            "--image_paths", image_list_file,
            "--destination", str(out_base)
        ]
        log.info(f"Running official feature extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, check=True)
        log.info("Feature extraction completed successfully")

        # Find new/updated npy candidates
        candidates = []
        for root in [REPO_DIR, TEMP_DIR]:
            for p in root.rglob("*.npy"):
                try:
                    st = p.stat()
                    if (p.resolve() not in before) or (st.st_mtime >= start_time - 1):
                        candidates.append((st.st_mtime, st.st_size, p))
                except FileNotFoundError:
                    pass

        # Expected locations as fallbacks
        exp1 = out_base.with_suffix(".npy")
        exp2 = out_base if out_base.suffix == ".npy" else None
        for p in [exp1, exp2]:
            if p and p.exists():
                st = p.stat()
                candidates.append((st.st_mtime, st.st_size, p))

        if not candidates:
            log.error("No new .npy features found after extraction.")
            log.error(f"STDOUT: {result.stdout}")
            log.error(f"STDERR: {result.stderr}")
            return None

        # Prefer files containing our prefix; else newest/largest
        def score(item):
            mtime, size, path = item
            has_prefix = 1 if prefix.lower() in path.name.lower() else 0
            return (has_prefix, mtime, size)

        best = sorted(candidates, key=score, reverse=True)[0][2]

        final_path = TEMP_DIR / f"{prefix}_features.npy"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if best.resolve() != final_path.resolve():
            final_path.unlink(missing_ok=True)
            try:
                best.replace(final_path)
            except Exception:
                import shutil as _sh
                _sh.copy2(best, final_path)

        log.info(f"Features saved to: {final_path}")
        return str(final_path)

    except subprocess.CalledProcessError as e:
        log.error(f"Feature extraction failed: {e}")
        log.error(f"STDOUT: {e.stdout}")
        log.error(f"STDERR: {e.stderr}")
        return None
    
def load_features(feature_file: str,
                  split: Split,
                  encoder: LabelEncoder) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load features from many extractor variants and return:
    X: [N, D] float32 (L2-normalized), labels, names, paths.
    """
    N_expected = len(split.paths)
    obj = np.load(feature_file, allow_pickle=True)

    # Unwrap 0-D object array (np.save(dict/list/tuple))
    if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == object:
        obj = obj.item()

    X = None
    paths_from_file = None

    def _stack(seq):
        return np.vstack([np.asarray(v) for v in seq]).astype(np.float32)

    # Dict formats
    if isinstance(obj, dict):
        feat_keys = ["features", "feats", "embeddings", "em", "outputs", "x"]
        path_keys = ["paths", "image_paths", "filenames", "files"]
        kf = next((k for k in feat_keys if k in obj), None)
        if kf is None:
            raise ValueError(f"Unknown feature dict format in {feature_file}. Keys={list(obj.keys())}")
        X_raw = obj[kf]
        if isinstance(X_raw, list):
            X = _stack(X_raw)
        elif isinstance(X_raw, np.ndarray):
            if X_raw.ndim == 2:
                X = X_raw.astype(np.float32)
            elif X_raw.ndim == 1 and X_raw.dtype == object:
                X = _stack(X_raw)
            elif N_expected and X_raw.size % N_expected == 0:
                X = X_raw.reshape(N_expected, -1).astype(np.float32)
        kp = next((k for k in path_keys if k in obj), None)
        if kp is not None:
            paths_from_file = [str(p) for p in obj[kp]]

    # List/tuple formats
    elif isinstance(obj, (list, tuple)):
        first = obj[0]
        if isinstance(first, (list, np.ndarray)):
            X = _stack(obj)
        elif isinstance(first, dict):
            feat_keys = ["features", "feats", "embeddings", "em", "outputs", "x"]
            kf = next((k for k in feat_keys if k in first), None)
            if kf is None:
                raise ValueError(f"Unrecognized dict list in {feature_file}")
            X = _stack([o[kf] for o in obj])

    # 2-D numeric
    elif isinstance(obj, np.ndarray) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
        X = obj.astype(np.float32)

    # 1-D variants
    elif isinstance(obj, np.ndarray) and obj.ndim == 1:
        if obj.dtype == object:
            X = _stack(obj)
        elif np.issubdtype(obj.dtype, np.number):
            flat = obj.astype(np.float32)
            if N_expected and flat.size % N_expected == 0:
                X = flat.reshape(N_expected, -1)
            else:
                X = flat.reshape(1, -1)

    if X is None:
        raise ValueError(f"Unsupported features format in {feature_file}: type={type(obj)}, "
                         f"shape={getattr(obj, 'shape', None)}, dtype={getattr(obj, 'dtype', None)}")

    # Use extractor-provided paths if they match N
    if paths_from_file and len(paths_from_file) == X.shape[0]:
        paths = paths_from_file
        # map names by path if possible
        by_abs = {str(Path(p).resolve()): n for p, n in zip(split.paths, split.names)}
        names = [by_abs.get(str(Path(p).resolve()), split.names[0]) for p in paths]
    else:
        paths, names = split.paths, split.names

    # Align sizes (clip to min if mismatch)
    if len(paths) != X.shape[0]:
        log.warning(f"Feature/sample mismatch for {feature_file}: got N={X.shape[0]}, expected {len(paths)}. Proceeding with min().")
        N = min(len(paths), X.shape[0])
        X, paths, names = X[:N], paths[:N], names[:N]

    # L2 normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    labels = encoder.transform(names)
    return X, labels, names, paths



def cosine_prototypes(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute class prototypes using mean embeddings"""
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
    """Predict using cosine similarity to prototypes"""
    scores = X @ protos.T  # cosine because rows are L2-normalized
    return scores.argmax(axis=1), scores.max(axis=1)

def predict_by_1nn(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray):
    """Predict using 1-nearest neighbor with cosine similarity"""
    scores = test_X @ train_X.T
    idx = scores.argmax(axis=1)
    return train_y[idx], scores.max(axis=1)

def evaluate_one_model(model_name: str, weight_path: str, enc: LabelEncoder, 
                      train_split: Split, test_split: Split) -> Dict:
    """Evaluate one model using official feature extraction"""
    
    # Create temporary files for this model
    train_list_file = create_image_list_file(train_split, f"{model_name}_train_images.csv")
    test_list_file = create_image_list_file(test_split, f"{model_name}_test_images.csv")
    
   
    # Base file paths (no extension); extractor will create .npy here
    train_features_base = str(TEMP_DIR / f"{model_name}_train_features")
    test_features_base  = str(TEMP_DIR / f"{model_name}_test_features")

    log.info(f"[{model_name}] Extracting training features using official method...")
    train_features_file = extract_features_official(
        weight_path, train_list_file, train_features_base, f"{model_name}_train"
    )
    if train_features_file is None:
        return {"model": model_name, "error": "training feature extraction failed"}

    log.info(f"[{model_name}] Extracting test features using official method...")
    test_features_file = extract_features_official(
        weight_path, test_list_file, test_features_base, f"{model_name}_test"
    )
    if test_features_file is None:
        return {"model": model_name, "error": "test feature extraction failed"}
    # Load extracted features
    try:
        Xtr, ytr, tr_names, tr_paths = load_features(train_features_file, train_split, enc)
        Xte, yte, te_names, te_paths = load_features(test_features_file, test_split, enc)
    except Exception as e:
        return {"model": model_name, "error": f"feature loading failed: {e}"}
    
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return {"model": model_name, "error": "no features extracted"}
    
    n_classes = len(enc.classes_)
    
    # Prototype classifier (per-class mean)
    protos = cosine_prototypes(Xtr, ytr, n_classes)
    ypred_p, conf_p = predict_by_prototypes(Xte, protos)
    
    # 1-NN cosine for reference
    ypred_nn, conf_nn = predict_by_1nn(Xtr, ytr, Xte)
    
    # Compute standardized metrics using sklearn
    acc = accuracy_score(yte, ypred_p)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(yte, ypred_p, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(yte, ypred_p, average="macro", zero_division=0)
    # cls_report = classification_report(yte, ypred_p, target_names=list(enc.classes_), output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, ypred_p)
    present_labels = np.unique(yte)
    present_names = enc.inverse_transform(present_labels)
    cls_report = classification_report(
        yte, ypred_p,
        labels=present_labels,
        target_names=list(present_names),
        output_dict=True,
        zero_division=0
    )

    # Save detailed results
    pred_names_p = enc.inverse_transform(ypred_p)
    true_names = enc.inverse_transform(yte)
    
    # Prototype classifier results
    df = pd.DataFrame({
        "image_path": [Path(p).name for p in te_paths],
        "true_celebrity": true_names,
        "predicted_celebrity": pred_names_p,
        "confidence": conf_p,
        "correct": (ypred_p == yte),
        "model": model_name,
        "classifier": "prototype"
    })
    df.to_csv(RESULTS_DIR / f"{model_name}_detailed_results_prototype.csv", index=False)
    
    # 1-NN results
    pred_names_nn = enc.inverse_transform(ypred_nn)
    df_nn = pd.DataFrame({
        "image_path": [Path(p).name for p in te_paths],
        "true_celebrity": true_names,
        "predicted_celebrity": pred_names_nn,
        "confidence": conf_nn,
        "correct": (ypred_nn == yte),
        "model": model_name,
        "classifier": "1NN"
    })
    df_nn.to_csv(RESULTS_DIR / f"{model_name}_detailed_results_1nn.csv", index=False)
    
    # Per-celebrity breakdown
    celeb_metrics = df.groupby("true_celebrity").agg(
        correct_count=("correct", "sum"),
        total_count=("correct", "count"),
        avg_confidence=("confidence", "mean")
    )
    celeb_metrics["accuracy"] = (100.0 * celeb_metrics["correct_count"] / celeb_metrics["total_count"]).round(2)
    celeb_metrics.to_csv(RESULTS_DIR / f"{model_name}_celebrity_metrics_prototype.csv")
    
    # Cleanup temporary files
    for temp_file in [train_list_file, test_list_file]:
        Path(temp_file).unlink(missing_ok=True)
    
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
        "results_csv_prototype": str(RESULTS_DIR / f"{model_name}_detailed_results_prototype.csv"),
        "results_csv_1nn": str(RESULTS_DIR / f"{model_name}_detailed_results_1nn.csv")
    }

def main():
    """Main evaluation pipeline"""
    log.info("=== SOTA Face Recognition Evaluation with Official Methods ===")
    log.info(f"Dataset: {DATASET_DIR}")
    log.info(f"Models: {MODEL_DIR}")
    log.info(f"Results: {RESULTS_DIR}")
    
    # Setup environment - no cloning needed since we're already in the repo
    setup_environment()
    
    # Build dataset splits
    try:
        train_split, test_split, encoder = build_splits()
        log.info(f"Training images: {len(train_split.paths)}")
        log.info(f"Test images: {len(test_split.paths)}")
        log.info(f"Number of celebrities: {len(encoder.classes_)}")
    except Exception as e:
        log.error(f"Failed to build dataset splits: {e}")
        return
    
    # Evaluate each model
    all_metrics = []
    for model_name, weight_path in MODEL_FILES.items():
        if not weight_path.exists():
            log.warning(f"Skipping {model_name}: weights not found at {weight_path}")
            all_metrics.append({"model": model_name, "error": "missing weights"})
            continue
        
        log.info(f"=== Evaluating {model_name} with official SOTA methods ===")
        metrics = evaluate_one_model(model_name, str(weight_path), encoder, train_split, test_split)
        all_metrics.append(metrics)
        
        if "error" not in metrics:
            log.info(f"{model_name}: Acc={metrics['overall_accuracy']:.1f}% "
                    f"F1w={metrics['f1_weighted']:.1f}% F1m={metrics['f1_macro']:.1f}% "
                    f"N={metrics['total_test_images']}")
        else:
            log.error(f"{model_name}: {metrics['error']}")
    
    # Save comprehensive summary
    summary = pd.DataFrame(all_metrics)
    metric_cols = ["overall_accuracy", "f1_weighted", "f1_macro", "total_test_images"]
    for col in metric_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    
    summary.to_csv(RESULTS_DIR / "comprehensive_summary_official_sota.csv", index=False)
    
    # Print final summary
    log.info("\n================ FINAL EVALUATION SUMMARY (Official SOTA Methods) ================\n")
    
    has_results = summary["overall_accuracy"].notna().any()
    if has_results:
        summary_ok = (
            summary[summary["overall_accuracy"].notna()]
            .sort_values("overall_accuracy", ascending=False)
            .reset_index(drop=True)
        )
        
        for i, row in summary_ok.iterrows():
            acc = row["overall_accuracy"]
            f1w = row["f1_weighted"] 
            f1m = row["f1_macro"]
            nimg = row["total_test_images"]
            
            acc_s = "nan" if pd.isna(acc) else f"{acc:5.1f}%"
            f1w_s = "nan" if pd.isna(f1w) else f"{f1w:5.1f}%"
            f1m_s = "nan" if pd.isna(f1m) else f"{f1m:5.1f}%"
            nimg_s = "?" if pd.isna(nimg) else f"{int(nimg)}"
            
            log.info(f"{i+1:>2}. {row.get('model','<?>'):<16} Acc {acc_s} | F1w {f1w_s} | F1m {f1m_s} | N={nimg_s}")
    else:
        log.info("No successful evaluations. Details:")
        log.info("\n" + summary.to_string(index=False))
    
    log.info(f"\nResults saved to: {RESULTS_DIR}")
    
    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()