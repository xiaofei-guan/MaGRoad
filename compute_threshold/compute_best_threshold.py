import os
import argparse
import numpy as np
from typing import Tuple


def scan_dump_root(root: str) -> Tuple[int, list]:
    """
    Return (num_batches, batch_dirs_sorted)
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dump root not found: {root}")
    batch_dirs = [d for d in os.listdir(root) if d.startswith('batch_')]
    batch_dirs.sort()
    batch_paths = [os.path.join(root, d) for d in batch_dirs]
    return len(batch_paths), batch_paths


def update_histograms(pred_u8: np.ndarray, gt_u8: np.ndarray,
                      hist_pos: np.ndarray, hist_neg: np.ndarray) -> None:
    """
    Update positive/negative histograms from a batch chunk.
    pred_u8: uint8 in [0,255]
    gt_u8: uint8 in {0,1}
    hist_pos/neg: shape [256]
    """
    # Flatten to 1D for speed/memory
    pred_flat = pred_u8.reshape(-1)
    gt_flat = gt_u8.reshape(-1)

    # Positives (gt==1)
    pos_vals = pred_flat[gt_flat == 1]
    if pos_vals.size:
        hist_pos += np.bincount(pos_vals, minlength=256)

    # Negatives (gt==0)
    neg_vals = pred_flat[gt_flat == 0]
    if neg_vals.size:
        hist_neg += np.bincount(neg_vals, minlength=256)


def compute_pr_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray,
                         thresholds_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given pos/neg histograms over scores 0..255, compute precision/recall for
    thresholds (>= t) where t is uint8 threshold in [1..254] typically.
    Return (precision, recall) arrays aligned to thresholds.
    """
    # Cumulative counts from high score to low score
    # hist index i means score==i; predictions >= t means sum_{i=t..255}
    cpos = np.cumsum(hist_pos[::-1])[::-1]
    cneg = np.cumsum(hist_neg[::-1])[::-1]

    total_pos = hist_pos.sum()

    precision = np.zeros_like(thresholds_u8, dtype=np.float64)
    recall = np.zeros_like(thresholds_u8, dtype=np.float64)

    # ensure thresholds within [0,255]
    thresholds_u8 = np.clip(thresholds_u8, 0, 255)

    # vectorized indexing: tp = cpos[t], fp = cneg[t]
    tp = cpos[thresholds_u8]
    fp = cneg[thresholds_u8]
    denom_p = tp + fp

    # precision
    mask = denom_p > 0
    precision[mask] = tp[mask] / denom_p[mask]
    precision[~mask] = 0.0

    # recall
    if total_pos > 0:
        recall = tp / float(total_pos)
    else:
        recall[:] = 0.0

    return precision, recall


def find_best_threshold(hist_pos: np.ndarray, hist_neg: np.ndarray,
                        t_min: int = 1, t_max: int = 254) -> Tuple[int, float, float, float]:
    """
    Search best uint8 threshold in [t_min..t_max] maximizing F1.
    Returns (best_t, best_p, best_r, best_f1)
    """
    thresholds = np.arange(t_min, t_max + 1, dtype=np.int32)
    precision, recall = compute_pr_from_hist(hist_pos, hist_neg, thresholds)

    denom = precision + recall
    valid = denom > 0
    f1 = np.zeros_like(precision)
    f1[valid] = 2.0 * (precision[valid] * recall[valid]) / (denom[valid] + 1e-12)

    idx = int(np.argmax(f1))
    best_t = int(thresholds[idx])
    return best_t, float(precision[idx]), float(recall[idx]), float(f1[idx])


def main():
    parser = argparse.ArgumentParser(description='Compute best threshold from dumped predictions/GT')
    parser.add_argument('--dump_root', type=str, default=os.path.join(os.path.dirname(__file__), 'dump'),
                        help='Root directory containing batch_XXXXXX folders')
    parser.add_argument('--target', type=str, choices=['kp', 'road'], required=True,
                        help='Which category to compute (kp or road)')
    parser.add_argument('--tmin', type=int, default=1, help='Min uint8 threshold (inclusive)')
    parser.add_argument('--tmax', type=int, default=254, help='Max uint8 threshold (inclusive)')
    args = parser.parse_args()

    num_batches, batch_dirs = scan_dump_root(args.dump_root)
    print(f'Found {num_batches} batch folders under {args.dump_root}')

    # Histograms for 0..255
    hist_pos = np.zeros(256, dtype=np.int64)
    hist_neg = np.zeros(256, dtype=np.int64)

    pred_name = f'pred_{"kp" if args.target=="kp" else "road"}_mask.npy'
    gt_name = f'gt_{"kp" if args.target=="kp" else "road"}_mask.npy'

    for bdir in batch_dirs:
        pred_path = os.path.join(bdir, pred_name)
        gt_path = os.path.join(bdir, gt_name)
        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f'Skip missing pair in {bdir}')
            continue

        # Memory-efficient loading: np.load with mmap_mode allows not loading file fully into RAM
        pred = np.load(pred_path, mmap_mode='r')  # uint8
        gt = np.load(gt_path, mmap_mode='r')      # uint8 {0,1}

        # If arrays are large (B,H,W), iterate in chunks along the first dim
        # Expect pred shape = [B, H, W]
        if pred.ndim == 2:
            update_histograms(pred, gt, hist_pos, hist_neg)
        else:
            B = pred.shape[0]
            # Process in small slices to reduce peak RAM
            step = max(1, 64 // max(1, pred.shape[1] // 512))  # heuristic small batch
            for i in range(0, B, step):
                update_histograms(pred[i:i+step], gt[i:i+step], hist_pos, hist_neg)

        # free mmap refs
        del pred, gt

    best_t_u8, p, r, f1 = find_best_threshold(hist_pos, hist_neg, args.tmin, args.tmax)
    best_t_float = best_t_u8 / 255.0
    print(f'Best threshold (uint8) = {best_t_u8}, float={best_t_float:.6f}, P={p:.6f} R={r:.6f} F1={f1:.6f}')

    # Also save a small report
    out_txt = os.path.join(args.dump_root, f'best_threshold_{args.target}.txt')
    with open(out_txt, 'w') as f:
        f.write(f'best_t_u8={best_t_u8}\n')
        f.write(f'best_t_float={best_t_float:.6f}\n')
        f.write(f'precision={p:.6f}\n')
        f.write(f'recall={r:.6f}\n')
        f.write(f'f1={f1:.6f}\n')
    print(f'Saved report to {out_txt}')


if __name__ == '__main__':
    main()


