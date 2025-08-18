# celeb_benchmark_official_sota.py
# Evaluate pretrained SOTA FR models using official SOTA-FR-train-and-test repository methods
# Uses official feature extraction + custom celebrity classification

import sys
import os
import logging
import warnings
import subprocess
import shutil
import hashlib
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# --------------------------- Paths & Config ---------------------------

# Running from within SOTA-FR-train-and-test repo
REPO_DIR = Path(".")
DATASET_DIR = Path("./celeb-dataset")
MODEL_DIR = Path("./model")
RESULTS_DIR = Path("./elifiles/130825")
TEMP_DIR = Path("./temp_sota_fr")

# Model files mapping
MODEL_FILES = {
    "ArcFace":        MODEL_DIR / "arcface-r100-ms1mv2.pth",
    "CurricularFace": MODEL_DIR / "curricularface-r100-ms1mv2.pth",
    "MagFace":        MODEL_DIR / "magface-r100-ms1mv2.pth",
    "AdaFace":        MODEL_DIR / "adaface-r100-ms1mv2.pth",
    "CosFace":        MODEL_DIR / "cosface-r100-ms1mv2.pth",
    "SphereFace":     MODEL_DIR / "sphereface-r100-ms1mv2.pth",
    "UniFace":        MODEL_DIR / "uniface-r100-ms1mv2.pth",
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sota_official_eval")
warnings.filterwarnings("ignore")

# Image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --------------------------- Data Structures ---------------------------

@dataclass
class Split:
    paths: List[str]
    names: List[str]

# --------------------------- Setup & Listing ---------------------------

def setup_environment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Running from: {REPO_DIR.resolve()}")
    log.info(f"Dataset: {DATASET_DIR.resolve()}")
    log.info(f"Model dir: {MODEL_DIR.resolve()}")
    log.info(f"Results: {RESULTS_DIR.resolve()}")

def list_images(root: Path, is_training: bool) -> List[Tuple[str, str]]:
    """
    Walk celeb-dataset/<ethnicity>/<class or class_test>/** and collect images.
    For training split, skip folders whose final name ends with '_test';
    for test split, keep only those.
    """
    out: List[Tuple[str, str]] = []
    ethnicities = ["caucasian", "chinese", "indian", "malay"]
    for eth in ethnicities:
        eth_dir = root / eth
        if not eth_dir.is_dir():
            continue
        for cls_dir in sorted([p for p in eth_dir.iterdir() if p.is_dir()]):
            cname = cls_dir.name
            is_test_dir = cname.endswith("_test")
            if is_training and is_test_dir:
                continue
            if (not is_training) and (not is_test_dir):
                continue
            cname_clean = cname.replace("_test", "")
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    out.append((str(p), cname_clean))
    return out

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
    enc.fit(sorted(list(set(tr_names) | set(te_names))))
    return Split(list(tr_paths), list(tr_names)), Split(list(te_paths), list(te_names)), enc

# --------------------------- Safe Symlink Materialization ---------------------------

def _safe_name(p: Path) -> str:
    # stable, short, comma/space-free: <basename>__<8-hex>.<ext>
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    stem = p.stem.replace(",", "_").replace(" ", "_")
    return f"{stem}__{h}{p.suffix.lower()}"

def materialize_safe_symlinks(split: Split, prefix: str) -> Tuple[Split, Dict[str, str]]:
    """
    Create a sanitized (no commas/spaces) symlinked/copy view of the split under TEMP_DIR/symlinks/<prefix>/.
    Returns:
      - new Split with symlink paths
      - mapping: symlink absolute path -> original absolute path
    """
    root = TEMP_DIR / "symlinks" / prefix
    root.mkdir(parents=True, exist_ok=True)

    symlink_paths: List[str] = []
    names: List[str] = []
    mapping: Dict[str, str] = {}

    for orig, name in zip(split.paths, split.names):
        op = Path(orig).resolve()
        sp = root / _safe_name(op)
        if not sp.exists():
            try:
                if sp.exists() or sp.is_symlink():
                    sp.unlink()
                os.symlink(op, sp)
            except OSError:
                shutil.copy2(op, sp)
        ap = str(sp.resolve())
        symlink_paths.append(ap)
        names.append(name)
        mapping[ap] = str(op)
    return Split(symlink_paths, names), mapping

def create_image_list_file_txt(split: Split, filename: str) -> str:
    """
    Plain text list: one absolute path per line (robust across extractor forks).
    """
    filepath = TEMP_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for p in split.paths:
            f.write(f"{str(Path(p).resolve())}\n")
    return str(filepath)

# --------------------------- Extractor Invocation ---------------------------

def extract_features_official(model_path: str,
                              image_list_file: str,
                              feature_output_base: str,
                              prefix: str) -> str:
    """
    Run the official extractor and robustly find the produced .npy, regardless of
    whether --destination is treated as a folder, a file base, or ignored.
    Returns a normalized .npy path under TEMP_DIR (TEMP_DIR/<prefix>_features.npy).
    """
    try:
        out_base = Path(feature_output_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)

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

        candidates = []
        for root in [REPO_DIR, TEMP_DIR]:
            for p in root.rglob("*.npy"):
                try:
                    st = p.stat()
                    if (p.resolve() not in before) or (st.st_mtime >= start_time - 1):
                        candidates.append((st.st_mtime, st.st_size, p))
                except FileNotFoundError:
                    pass

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
                shutil.copy2(best, final_path)

        log.info(f"Features saved to: {final_path}")
        return str(final_path)

    except subprocess.CalledProcessError as e:
        log.error(f"Feature extraction failed: {e}")
        log.error(f"STDOUT: {e.stdout}")
        log.error(f"STDERR: {e.stderr}")
        return None

# --------------------------- Loading & Alignment ---------------------------

def remap_paths_from_symlinks(paths_from_file: List[str], symlink_to_orig: Dict[str, str]) -> List[str]:
    remapped = []
    for p in paths_from_file:
        ap = str(Path(p).resolve())
        remapped.append(symlink_to_orig.get(ap, ap))
    return remapped

def load_features(feature_file: str,
                  split: Split,
                  encoder: LabelEncoder) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load features from many extractor variants and return:
    X: [N, D] float32 (L2-normalized), labels (y), names, paths (as provided by extractor or split).
    Strict alignment: raise if counts cannot be matched.
    """
    N_expected = len(split.paths)
    obj = np.load(feature_file, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == object:
        obj = obj.item()

    X = None
    paths_from_file = None

    def _stack(seq):
        return np.vstack([np.asarray(v) for v in seq]).astype(np.float32)

    if isinstance(obj, dict):
        feat_keys = ["features", "feats", "embeddings", "em", "outputs", "x"]
        path_keys = ["paths", "image_paths", "filenames", "files"]
        kf = next((k for k in feat_keys if k in obj), None)
        if kf is None:
            raise ValueError(f"Unknown feature dict format in {feature_file}. Keys={list(obj.keys())}")
        raw = obj[kf]
        if isinstance(raw, list):
            X = _stack(raw)
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 2:
                X = raw.astype(np.float32)
            elif raw.ndim == 1 and raw.dtype == object:
                X = _stack(raw)
            elif N_expected and raw.size % N_expected == 0:
                X = raw.reshape(N_expected, -1).astype(np.float32)
        kp = next((k for k in path_keys if k in obj), None)
        if kp is not None:
            paths_from_file = [str(p) for p in obj[kp]]

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

    elif isinstance(obj, np.ndarray) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
        X = obj.astype(np.float32)

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

    # Prefer extractor-provided paths if present (remapped in evaluate() if they are symlinks)
    if paths_from_file and len(paths_from_file) == X.shape[0]:
        paths = paths_from_file
    else:
        if X.shape[0] != len(split.paths):
            raise ValueError(f"Extractor returned N={X.shape[0]} features but split has {len(split.paths)} paths.")
        paths = split.paths

    # Align names by absolute path
    def _canon(p: str) -> str:
        try:
            return str(Path(p).resolve())
        except Exception:
            return str(Path(p))

    split_abs = [_canon(p) for p in split.paths]
    name_by_abs = {a: n for a, n in zip(split_abs, split.names)}
    names: List[str] = []
    misses = []
    for p in paths:
        n = name_by_abs.get(_canon(p))
        names.append(n)
        if n is None:
            misses.append(p)

    # If there are misses, we allow evaluate() to remap via symlink->original mapping.
    # Here we keep names as-is; evaluate() will rebuild labels after remap if needed.

    # L2 normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    labels = encoder.transform([n if n is not None else split.names[0] for n in names])  # temporary; fixed in evaluate()
    return X.astype(np.float32), labels, names, paths

# --------------------------- Simple Classifiers ---------------------------

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
    scores = X @ protos.T
    return scores.argmax(axis=1), scores.max(axis=1)

def predict_by_1nn(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray):
    scores = test_X @ train_X.T
    idx = scores.argmax(axis=1)
    return train_y[idx], scores.max(axis=1)

# --------------------------- Sanity / Leakage Checks ---------------------------

def assert_no_overlap(train_split: Split, test_split: Split):
    tr = {str(Path(p).resolve()) for p in train_split.paths}
    te = {str(Path(p).resolve()) for p in test_split.paths}
    inter = tr & te
    if inter:
        raise AssertionError(f"Train/Test leakage: {len(inter)} files overlap. Example: {next(iter(inter))}")

def sanity_checks(y_true: np.ndarray, y_pred: np.ndarray, tag: str):
    if len(np.unique(y_true)) < 2:
        raise AssertionError(f"[{tag}] Only one class present in y_true — alignment likely wrong.")
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(42)
    perm_acc = accuracy_score(y_true, rng.permutation(y_pred))
    log.info(f"[{tag}] Sanity permuted-accuracy (should be ~chance): {perm_acc:.3f}")

# --------------------------- Evaluation ---------------------------

def evaluate_one_model(model_name: str, weight_path: str, enc: LabelEncoder,
                       train_split: Split, test_split: Split) -> Dict:
    """Evaluate one model using official feature extraction (robust to extractor quirks)."""

    # 1) Build sanitized symlink views (no commas/odd chars)
    tr_safe, tr_map = materialize_safe_symlinks(train_split, f"{model_name}_train")
    te_safe, te_map = materialize_safe_symlinks(test_split,  f"{model_name}_test")

    # 2) Plain-text lists for extractor
    train_list_file = create_image_list_file_txt(tr_safe, f"{model_name}_train_images.txt")
    test_list_file  = create_image_list_file_txt(te_safe, f"{model_name}_test_images.txt")

    # 3) Output base paths (no extension)
    train_features_base = str(TEMP_DIR / f"{model_name}_train_features")
    test_features_base  = str(TEMP_DIR / f"{model_name}_test_features")

    # 4) Extract features
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

    # 5) Load features (initially aligned to symlink paths)
    try:
        Xtr, ytr_tmp, tr_names_tmp, tr_paths_tmp = load_features(train_features_file, tr_safe, enc)
        Xte, yte_tmp, te_names_tmp, te_paths_tmp = load_features(test_features_file, te_safe, enc)
    except Exception as e:
        return {"model": model_name, "error": f"feature loading failed: {e}"}

    # 6) Remap symlink paths back to original absolute paths
    tr_paths = remap_paths_from_symlinks(tr_paths_tmp, tr_map)
    te_paths = remap_paths_from_symlinks(te_paths_tmp, te_map)

    # 7) Re-derive names and labels from original splits using absolute path matching
    def _canon(p: str) -> str:
        try:
            return str(Path(p).resolve())
        except Exception:
            return str(Path(p))

    # Build lookup from original splits
    tr_lookup = { _canon(p): n for p, n in zip(train_split.paths, train_split.names) }
    te_lookup = { _canon(p): n for p, n in zip(test_split.paths,  test_split.names)  }

    tr_names = []
    te_names = []
    tr_miss = []
    te_miss = []

    for p in tr_paths:
        n = tr_lookup.get(_canon(p))
        tr_names.append(n)
        if n is None:
            tr_miss.append(p)
    for p in te_paths:
        n = te_lookup.get(_canon(p))
        te_names.append(n)
        if n is None:
            te_miss.append(p)

    if tr_miss:
        return {"model": model_name, "error": f"failed to align {len(tr_miss)} training paths back to originals (e.g., {tr_miss[0]})"}
    if te_miss:
        return {"model": model_name, "error": f"failed to align {len(te_miss)} test paths back to originals (e.g., {te_miss[0]})"}

    ytr = enc.transform(tr_names)
    yte = enc.transform(te_names)

    # 8) Sanity checks
    if Xtr.shape[0] != len(tr_names) or Xte.shape[0] != len(te_names):
        return {"model": model_name, "error": f"feature/sample count mismatch after remap: "
                                              f"train {Xtr.shape[0]}/{len(tr_names)}, test {Xte.shape[0]}/{len(te_names)}"}
    # Basic sanity (raises if degenerate)
    sanity_checks(yte, yte, f"{model_name}/groundtruth")  # should not raise (has >=2 classes)
    # 9) Train simple classifiers
    n_classes = len(enc.classes_)

    protos = cosine_prototypes(Xtr, ytr, n_classes)
    ypred_p, conf_p = predict_by_prototypes(Xte, protos)

    ypred_nn, conf_nn = predict_by_1nn(Xtr, ytr, Xte)

    # Sanity: permuted accuracy control for prototype predictions
    sanity_checks(yte, ypred_p, f"{model_name}/prototype")

    # 10) Metrics
    acc = accuracy_score(yte, ypred_p)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(yte, ypred_p, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(yte, ypred_p, average="macro", zero_division=0)
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

    # 11) Save detailed results
    pred_names_p = enc.inverse_transform(ypred_p)
    true_names = enc.inverse_transform(yte)

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

    celeb_metrics = df.groupby("true_celebrity").agg(
        correct_count=("correct", "sum"),
        total_count=("correct", "count"),
        avg_confidence=("confidence", "mean")
    )
    celeb_metrics["accuracy"] = (100.0 * celeb_metrics["correct_count"] / celeb_metrics["total_count"]).round(2)
    celeb_metrics.to_csv(RESULTS_DIR / f"{model_name}_celebrity_metrics_prototype.csv")

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

# --------------------------- Main ---------------------------

def main():
    log.info("=== SOTA Face Recognition Evaluation with Official Methods ===")
    log.info(f"Dataset: {DATASET_DIR}")
    log.info(f"Models: {MODEL_DIR}")
    log.info(f"Results: {RESULTS_DIR}")

    setup_environment()

    try:
        train_split, test_split, encoder = build_splits()
        log.info(f"Training images: {len(train_split.paths)}")
        log.info(f"Test images: {len(test_split.paths)}")
        log.info(f"Number of celebrities: {len(encoder.classes_)}")
        assert_no_overlap(train_split, test_split)
    except Exception as e:
        log.error(f"Failed to prepare dataset: {e}")
        return

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

    summary = pd.DataFrame(all_metrics)
    for col in ["overall_accuracy", "f1_weighted", "f1_macro", "total_test_images"]:
        if col not in summary.columns:
            summary[col] = np.nan

    summary.to_csv(RESULTS_DIR / "comprehensive_summary_official_sota.csv", index=False)

    log.info("\n================ FINAL EVALUATION SUMMARY (Official SOTA Methods) ================\n")
    has_results = summary["overall_accuracy"].notna().any()
    if has_results:
        summary_ok = (
            summary[summary["overall_accuracy"].notna()]
            .sort_values("overall_accuracy", ascending=False)
            .reset_index(drop=True)
        )
        for i, row in summary_ok.iterrows():
            acc = row["overall_accuracy"]; f1w = row["f1_weighted"]; f1m = row["f1_macro"]; nimg = row["total_test_images"]
            acc_s = "nan" if pd.isna(acc) else f"{acc:5.1f}%"
            f1w_s = "nan" if pd.isna(f1w) else f"{f1w:5.1f}%"
            f1m_s = "nan" if pd.isna(f1m) else f"{f1m:5.1f}%"
            nimg_s = "?" if pd.isna(nimg) else f"{int(nimg)}"
            log.info(f"{i+1:>2}. {row.get('model','<?>'):<16} Acc {acc_s} | F1w {f1w_s} | F1m {f1m_s} | N={nimg_s}")
    else:
        log.info("No successful evaluations. Details:")
        log.info("\n" + summary.to_string(index=False))

    log.info(f"\nResults saved to: {RESULTS_DIR}")
    # Optional cleanup (keep TEMP_DIR if you want artifacts)
    # shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
