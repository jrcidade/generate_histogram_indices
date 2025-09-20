import os
import sys
import math
import random
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

INPUT_DIR  = "/mnt/c/Users/Users/Downloads/indices"
OUTPUT_DIR = "/mnt/c/Users/Users/Downloads/hist_out"

CLIP_RANGE = (-1.0, 1.0)   
SAMPLE_SIZE = 1_000_000    
BINS = None                
MAX_BINS = 200             


def freedman_diaconis_bins(data, max_bins=200, min_bins=20):
    data = np.asarray(data)
    if data.size < 2:
        return min_bins
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min_bins
    n = data.size
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width <= 0:
        return min_bins
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return int(np.clip(bins, min_bins, max_bins))


def robust_nan_mask(arr, nodata):
    mask = np.isfinite(arr)
    if nodata is not None and not np.isnan(nodata):
        mask &= (arr != nodata)
    return mask


def sample_array(arr, k, rng):
    n = arr.size
    if n <= k:
        return arr
    idx = rng.sample(range(n), k)
    return arr[np.array(idx, dtype=np.int64)]


def compute_stats_from_stream(src, clip_range=None, sample_size=1_000_000, histogram_bins=None, max_bins=200):
    nodata = src.nodata

    count = 0
    mean = 0.0
    M2 = 0.0
    vmin = np.inf
    vmax = -np.inf
    nodata_count = 0

    rng = random.Random(42)
    sample = []

    def iterate_blocks():
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window).astype(np.float64)
            mask = robust_nan_mask(arr, nodata)
            valid = arr[mask]
            if clip_range is not None and valid.size:
                lo, hi = clip_range
                valid = np.clip(valid, lo, hi)
            yield valid, (~mask).sum()

    for valid, n_invalid in iterate_blocks():
        nodata_count += int(n_invalid)
        if valid.size == 0:
            continue

        for chunk in np.array_split(valid, max(1, valid.size // 1_000_000 + 1)):
            n = chunk.size
            if n == 0:
                continue

            vmin = min(vmin, float(chunk.min()))
            vmax = max(vmax, float(chunk.max()))

            # Welford
            count_old = count
            count += n
            delta = chunk.mean() - mean
            mean += (n * delta) / count
            M2 += np.sum((chunk - mean) * (chunk - (mean - (n * delta) / count)))

            # amostragem para percentis
            need = max(0, sample_size - len(sample))
            if need > 0:
                take = min(need, n)
                sample.extend(sample_array(chunk, take, rng))
            else:
                for val in chunk:
                    if rng.random() < (sample_size / count):
                        i = rng.randrange(sample_size)
                        sample[i] = val

    if count == 0:
        return {
            "count": 0, "nodata_count": nodata_count, "min": np.nan, "max": np.nan,
            "mean": np.nan, "std": np.nan, "p5": np.nan, "q1": np.nan,
            "median": np.nan, "q3": np.nan, "p95": np.nan,
            "hist_counts": None, "hist_edges": None
        }

    variance = M2 / (count - 1) if count > 1 else 0.0
    std = math.sqrt(variance)

    sample_arr = np.asarray(sample, dtype=np.float64)
    if sample_arr.size == 0:
        p5 = q1 = median = q3 = p95 = np.nan
    else:
        p5, q1, median, q3, p95 = np.percentile(sample_arr, [5, 25, 50, 75, 95])

    # bins
    if histogram_bins is None:
        bins = freedman_diaconis_bins(sample_arr, max_bins=max_bins, min_bins=20)
    else:
        bins = int(histogram_bins)

    hist_counts = np.zeros(bins, dtype=np.int64)

    lo_for_hist = vmin
    hi_for_hist = vmax
    if clip_range is not None:
        lo_for_hist = max(lo_for_hist, clip_range[0])
        hi_for_hist = min(hi_for_hist, clip_range[1])
    if lo_for_hist == hi_for_hist:
        hi_for_hist = lo_for_hist + 1e-9

    edges = np.linspace(lo_for_hist, hi_for_hist, bins + 1)

    # 2º passe: histograma
    for valid, _ in iterate_blocks():
        if valid.size == 0:
            continue
        valid = valid[(valid >= lo_for_hist) & (valid <= hi_for_hist)]
        if valid.size == 0:
            continue
        c, _ = np.histogram(valid, bins=edges)
        hist_counts += c

    return {
        "count": int(count),
        "nodata_count": int(nodata_count),
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
        "std": float(std),
        "p5": float(p5),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "p95": float(p95),
        "hist_counts": hist_counts,
        "hist_edges": edges
    }


def plot_and_save_histogram(title, counts, edges, out_png):
    if counts is None or edges is None:
        print(f"[WARN] Sem histograma para: {title}")
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=(edges[1] - edges[0]), alpha=0.85, edgecolor="black")
    plt.title(f"Histograma — {title}")
    plt.xlabel("Valor do índice")
    plt.ylabel("Frequência de pixels")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # coleta rasters .tif na pasta padrão
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERRO] Pasta de entrada não encontrada: {INPUT_DIR}")
        sys.exit(1)

    rasters = [os.path.join(INPUT_DIR, f)
               for f in os.listdir(INPUT_DIR)
               if f.lower().endswith(".tif")]

    if not rasters:
        print(f"[ERRO] Nenhum .tif encontrado em: {INPUT_DIR}")
        sys.exit(1)

    rows = []
    for path in sorted(rasters):
        base = os.path.basename(path)
        print(f"[INFO] Processando: {base}")
        try:
            with rasterio.open(path) as src:
                stats = compute_stats_from_stream(
                    src,
                    clip_range=CLIP_RANGE,
                    sample_size=SAMPLE_SIZE,
                    histogram_bins=BINS,
                    max_bins=MAX_BINS
                )

            png_name = os.path.splitext(base)[0] + "_hist.png"
            png_path = os.path.join(OUTPUT_DIR, png_name)
            plot_and_save_histogram(base, stats["hist_counts"], stats["hist_edges"], png_path)
            print(f"[OK] Histograma salvo: {png_path}")

            rows.append({
                "arquivo": base,
                "count": stats["count"],
                "nodata_count": stats["nodata_count"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "p5": stats["p5"],
                "q1": stats["q1"],
                "median": stats["median"],
                "q3": stats["q3"],
                "p95": stats["p95"],
                "clip_lo": CLIP_RANGE[0] if CLIP_RANGE else "",
                "clip_hi": CLIP_RANGE[1] if CLIP_RANGE else "",
                "bins": (len(stats["hist_counts"]) if stats["hist_counts"] is not None else "")
            })

        except Exception as e:
            print(f"[ERRO] {base}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, "estatisticas_indices.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"[OK] CSV salvo: {csv_path}")
    else:
        print("[WARN] Nada para salvar no CSV.")


if __name__ == "__main__":
    main()
