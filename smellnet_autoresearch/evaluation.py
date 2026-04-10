from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_to_device(x, device, dtype=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
        return x
    return x


def _build_class_to_category(
    class_names: Sequence[str],
    ingredient_to_category: Dict[str, str],
) -> tuple[Dict[int, str], set[str]]:
    missing = set()
    class_to_cat: Dict[int, str] = {}
    for i, name in enumerate(class_names):
        cat = ingredient_to_category.get(name)
        if cat is None:
            missing.add(name)
            cat = "UNKNOWN"
        class_to_cat[i] = cat
    return class_to_cat, missing


def evaluate_contrastive(
    gcms_encoder: torch.nn.Module,
    sensor_encoder: torch.nn.Module,
    *,
    gcms_data: Union[torch.Tensor, np.ndarray],
    sensor_data: Union[torch.Tensor, np.ndarray],
    sensor_labels: Union[torch.Tensor, np.ndarray],
    lengths: Optional[torch.Tensor] = None,
    logger=None,
    l2_normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    ingredient_to_category: Optional[Dict[str, str]] = None,
    class_names: Optional[Sequence[str]] = None,
    topk: Sequence[int] = (1, 5),
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    dev = device or _device()
    gcms_encoder.to(dev).eval()
    sensor_encoder.to(dev).eval()

    if not torch.is_tensor(gcms_data):
        gcms_data = torch.tensor(gcms_data)
    if not torch.is_tensor(sensor_data):
        sensor_data = torch.tensor(sensor_data)
    if not torch.is_tensor(sensor_labels):
        sensor_labels = torch.tensor(sensor_labels, dtype=torch.long)

    gcms_data = _maybe_to_device(gcms_data, dev, dtype)
    sensor_labels = _maybe_to_device(sensor_labels, dev)
    lengths = _maybe_to_device(lengths, dev)

    with torch.no_grad():
        zg = gcms_encoder.forward_features(gcms_data)
        if l2_normalize:
            zg = F.normalize(zg, dim=1)

        zs_list = []
        if batch_size is None:
            sd = _maybe_to_device(sensor_data, dev, dtype)
            z = sensor_encoder.forward_features(sd, lengths=lengths)
            zs_list.append(z)
        else:
            n = sensor_data.size(0)
            for i in range(0, n, batch_size):
                sd = _maybe_to_device(sensor_data[i : i + batch_size], dev, dtype)
                len_batch = None if lengths is None else lengths[i : i + batch_size]
                zs_list.append(sensor_encoder.forward_features(sd, lengths=len_batch))
        zs = torch.cat(zs_list, dim=0)
        if l2_normalize:
            zs = F.normalize(zs, dim=1)
        sim = zs @ zg.T
        top1 = sim.argmax(dim=1)
        y = sensor_labels
        total = y.size(0)
        results: Dict[str, Union[float, np.ndarray, Dict]] = {}
        max_k = min(max(topk), sim.shape[1])
        topk_val_all, topk_idx_all = torch.topk(sim, k=max_k, dim=1)
        for k in topk:
            kk = min(k, sim.shape[1])
            idx_k = topk_idx_all[:, :kk]
            results[f"acc@{k}"] = (idx_k == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0

    y_np = y.cpu().numpy()
    p_np = top1.cpu().numpy()
    if total > 0:
        results["precision_macro"] = precision_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["recall_macro"] = recall_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["f1_macro"] = f1_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["confusion_matrix"] = confusion_matrix(y_np, p_np)

    results["topk_idx"] = topk_idx_all[:, :max_k].cpu().numpy()
    results["topk_sim"] = topk_val_all[:, :max_k].cpu().numpy()

    if ingredient_to_category is not None and class_names is not None and total > 0:
        class_to_cat, _missing = _build_class_to_category(class_names, ingredient_to_category)
        true_cat = np.array([class_to_cat[int(c)] for c in y_np], dtype=object)
        correct1 = p_np == y_np
        include_acc5 = 5 in topk
        hits5 = None
        if include_acc5:
            kk = min(5, sim.shape[1])
            idx5 = topk_idx_all[:, :kk].cpu().numpy()
            hits5 = (idx5 == y_np[:, None]).any(axis=1)
        per_cat = {}
        for cat in np.unique(true_cat):
            mask = true_cat == cat
            n = int(mask.sum())
            row = {"n": n, "acc@1": float(correct1[mask].mean() * 100.0) if n > 0 else 0.0}
            if include_acc5:
                row["acc@5"] = float(hits5[mask].mean() * 100.0) if n > 0 else 0.0
            per_cat[str(cat)] = row
        results["per_category"] = per_cat

    if logger is None:
        printable = {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in results.items()
            if "matrix" not in k and not isinstance(v, dict)
        }
        print(printable)

    return results
