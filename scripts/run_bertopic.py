"""
Train dynamic BERTopic for DTECT (Table 4 rows) and pick K by TTC.

Per dataset:
  1. load processed/{train_texts,train_times,vocab}.txt and the raw texts of the
     same documents (docs.jsonl via processed/raw_doc_ids.txt, see align_raw_docs.py)
  2. embed raw texts (all-MiniLM-L6-v2, token chunks + mean pooling)  [once, cached]
  3. UMAP (seeded)                                                     [once, cached]
  4. for K in --ks: KMeans(K) -> c-TF-IDF -> beta (T x K x V), evaluate exactly like
     notebooks/Test_Evaluation.ipynb (top-20 words, TopicQualityAssessor, topn=10,
     c_npmi, reference corpus = processed/train_texts.txt)
  5. copy the beta of the K with the best TTC to data/<ds>/BERTopic/beta.npy

Outputs in --results-dir (default results/BERTopic):
  <ds>/K<k>/beta.npy, topics.npy, top_words.txt, metrics.json
  <ds>/run_log.json        settings, timings, empty cells, chunk stats, selection
  <ds>/cache/              embeddings.npy, umap.npy (+ meta)
  summary.csv, summary.md  all runs
  env.json                 hardware + package versions

Main runs use the trainer defaults, which mirror BERTopic's topics_over_time defaults
(global_tuning=True, evolution_tuning=True; evolution tuning is a no-op in BERTopic 0.17.4).
With --notuning, a tuning-off variant is also built from the same embeddings/UMAP/KMeans
and saved as <ds>/K<k>_notuning/; only default runs are considered for selection.

Runtime per K (`pipeline_seconds`) = embedding + UMAP + KMeans/c-TF-IDF fit + beta,
counting the shared embedding and UMAP stages in full. Each stage is also logged
separately; evaluation time is logged but not part of the pipeline time.

Example (IACS desktop):
  python scripts/run_bertopic.py --datasets TCPD_IPD_Finance ACL_Anthology UN_Debates --ks 20 50 100 --notuning
Smoke test:
  python scripts/run_bertopic.py --datasets TCPD_IPD_Finance --ks 20 --max-docs 1000 \
      --results-dir results/BERTopic_smoke --no-select --notuning
"""

import argparse
import csv
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from importlib import metadata
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.models.BERTopic_trainer import BERTopicTrainer          # noqa: E402
from backend.evaluation.eval import TopicQualityAssessor             # noqa: E402
from backend.inference.process_beta import get_top_words_at_time     # noqa: E402

PACKAGES = ["bertopic", "umap-learn", "hdbscan", "scikit-learn", "sentence-transformers",
            "transformers", "tokenizers", "torch", "numpy", "scipy", "pandas", "numba",
            "pynndescent", "gensim"]


# ---------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------- #
def environment():
    import torch
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    ram_gb = None
    try:
        with open("/proc/meminfo") as f:
            ram_gb = round(int(next(l for l in f if l.startswith("MemTotal")).split()[1]) / 2**20, 1)
    except Exception:
        pass
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1),
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "os": platform.platform(),
            "python": platform.python_version(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "ram_gb": ram_gb, "gpu": gpu, "packages": versions}


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #
def read_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def load_dataset(data_dir, ds, max_docs=None, seed=42):
    """Light-weight dataset (no dense BOW) + aligned raw texts."""
    d = os.path.join(data_dir, ds)
    proc = os.path.join(d, "processed")
    texts = read_lines(os.path.join(proc, "train_texts.txt"))
    times = np.loadtxt(os.path.join(proc, "train_times.txt")).astype(int)
    vocab = read_lines(os.path.join(proc, "vocab.txt"))
    ids_path = os.path.join(proc, "raw_doc_ids.txt")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"{ids_path} missing: run scripts/align_raw_docs.py {ds}")
    raw_ids = np.loadtxt(ids_path).astype(int)
    assert len(texts) == len(times) == len(raw_ids), "processed files / raw_doc_ids length mismatch"
    if (raw_ids < 0).any():
        raise ValueError(f"{(raw_ids < 0).sum()} processed docs have no raw match")
    num_times = int(times.max()) + 1                       # same as DynamicDataset

    raw_all = []
    with open(os.path.join(d, "docs.jsonl"), encoding="utf-8") as f:
        for line in f:
            raw_all.append(json.loads(line)["text"])
    raw = [raw_all[i] for i in raw_ids]

    idx = np.arange(len(texts))
    if max_docs and max_docs < len(texts):                 # smoke tests only
        idx = np.sort(np.random.RandomState(seed).choice(len(texts), max_docs, replace=False))
    dataset = SimpleNamespace(train_texts=[texts[i] for i in idx], train_times=times[idx],
                              vocab=vocab, num_times=num_times)
    return dataset, [raw[i] for i in idx], idx


# ---------------------------------------------------------------------- #
# Evaluation (same procedure as notebooks/Test_Evaluation.ipynb)
# ---------------------------------------------------------------------- #
def get_topic(beta, vocab, top_n):
    T, K, _ = beta.shape
    return [[get_top_words_at_time(beta=beta, vocab=vocab, topic_id=k, time=t, top_n=top_n)
             for k in range(K)] for t in range(T)]


def evaluate(beta, vocab, train_corpus):
    top_words_list = get_topic(beta, vocab, 20)
    assessor = TopicQualityAssessor(topics=top_words_list, train_texts=train_corpus,
                                    topn=10, coherence_type="c_npmi")
    return {k: float(v) for k, v in assessor.get_dtq_summary().items()}, top_words_list


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def run_dataset(ds, args, env):
    out_dir = os.path.join(args.results_dir, ds)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    set_seed(args.seed)

    dataset, raw_texts, idx = load_dataset(args.data_dir, ds, args.max_docs, args.seed)
    print(f"\n=== {ds}: {len(dataset.train_texts)} docs, T={dataset.num_times}, V={len(dataset.vocab)}",
          flush=True)

    # Main configuration = BERTopic topics_over_time defaults (trainer defaults)
    settings = dict(embedding_model=args.embedding_model, clustering="kmeans",
                    umap_n_neighbors=15, umap_n_components=5, embedding_batch_size=args.batch_size,
                    seed=args.seed)
    trainer = BERTopicTrainer(dataset, num_topics=args.ks[0], raw_texts=raw_texts,
                              verbose=args.verbose, **settings)
    variants = [("default", {"global_tuning": trainer.global_tuning,
                             "evolution_tuning": trainer.evolution_tuning,
                             "evolution_tuning_effective": trainer.evolution_tuning_effective})]
    if args.notuning:
        variants.append(("notuning", {"global_tuning": False, "evolution_tuning": False,
                                      "evolution_tuning_effective": False}))

    # Cache key: same docs + model + seed -> reuse embeddings / UMAP
    key = {"dataset": ds, "n_docs": len(idx), "max_docs": args.max_docs,
           "embedding_model": args.embedding_model, "seed": args.seed,
           "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine"}}
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    cache_ok = meta.get("key") == key and not args.no_cache

    # --- embeddings --------------------------------------------------------
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    if cache_ok and os.path.isfile(emb_path) and "embedding_seconds" in meta:
        embeddings = np.load(emb_path)
        t_emb, chunk_stats, emb_cached = meta["embedding_seconds"], meta["chunk_stats"], True
        print(f"[embed] cached ({t_emb:.1f}s when computed)", flush=True)
    else:
        embeddings, t_emb = timed(trainer.embed, device=args.device)
        chunk_stats, emb_cached = trainer.chunk_stats_, False
        np.save(emb_path, embeddings)
        meta = {"key": key, "embedding_seconds": t_emb, "chunk_stats": chunk_stats,
                "embedding_device": args.device or ("cuda" if env["gpu"] else "cpu")}
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[embed] {t_emb:.1f}s  {chunk_stats}", flush=True)
    trainer._encoder = None                              # free GPU memory

    # --- UMAP (once, reused for all K) --------------------------------------
    umap_path = os.path.join(cache_dir, "umap.npy")
    if cache_ok and os.path.isfile(umap_path) and "umap_seconds" in meta:
        reduced, t_umap, umap_cached = np.load(umap_path), meta["umap_seconds"], True
        print(f"[umap] cached ({t_umap:.1f}s when computed)", flush=True)
    else:
        reduced, t_umap = timed(trainer.reduce, embeddings)
        umap_cached = False
        np.save(umap_path, reduced)
        meta["umap_seconds"] = t_umap
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[umap] {t_umap:.1f}s", flush=True)

    # Reference corpus for coherence: full processed/train_texts.txt (as in the notebook)
    train_corpus = [t.split() for t in read_lines(os.path.join(args.data_dir, ds, "processed/train_texts.txt"))]

    runs = []
    for K in args.ks:
        set_seed(args.seed)
        trainer.num_topics = K
        k_dir = os.path.join(out_dir, f"K{K}")
        os.makedirs(k_dir, exist_ok=True)

        docs, times = trainer._docs_and_times()
        trainer.model = trainer._build_model(precomputed_reduction=True)
        (topics, _), t_fit = timed(trainer.model.fit_transform, docs, embeddings=reduced)
        trainer.topics_ = np.asarray(topics)
        n_topics_found = int(len(np.unique(trainer.topics_)))

        # Empirical check that evolution tuning is a no-op in the installed BERTopic
        tot_kw = dict(docs=docs, timestamps=list(times.astype(int)), global_tuning=True)
        evo_on, evo_off = (trainer.model.topics_over_time(evolution_tuning=e, **tot_kw) for e in (True, False))
        evolution_noop = bool(evo_on["Words"].tolist() == evo_off["Words"].tolist())

        for variant, flags in variants:
            trainer.global_tuning = flags["global_tuning"]
            trainer.evolution_tuning = flags["evolution_tuning"]
            v_dir = os.path.join(out_dir, f"K{K}" + ("" if variant == "default" else f"_{variant}"))
            os.makedirs(v_dir, exist_ok=True)

            beta, t_beta = timed(trainer._compute_beta, docs, times, trainer.topics_)
            trainer.beta = beta
            assert beta.shape == (dataset.num_times, K, len(dataset.vocab)), beta.shape
            assert np.allclose(beta.sum(-1), 1.0, atol=1e-4)
            check = trainer.check_against_topics_over_time(docs, times)

            (metrics, top_words), t_eval = timed(evaluate, beta, dataset.vocab, train_corpus)

            np.save(os.path.join(v_dir, "beta.npy"), beta)
            np.save(os.path.join(v_dir, "topics.npy"), trainer.topics_)
            with open(os.path.join(v_dir, "top_words.txt"), "w") as f:
                for t, topics_t in enumerate(top_words):
                    for k, words in enumerate(topics_t):
                        f.write(f"[Time:{t}][Topic:{k}]: {', '.join(words[:10])}\n")

            timings = {"embedding_seconds": t_emb, "umap_seconds": t_umap,
                       "kmeans_ctfidf_fit_seconds": t_fit, "beta_seconds": t_beta,
                       "pipeline_seconds": t_emb + t_umap + t_fit + t_beta,
                       "evaluation_seconds": t_eval,
                       "embedding_from_cache": emb_cached, "umap_from_cache": umap_cached}
            run = {"dataset": ds, "K": K, "variant": variant, **flags, **metrics, **timings,
                   "n_topics_found": n_topics_found, "empty_cells": trainer.empty_cells_,
                   "topics_over_time_check": check, "evolution_tuning_noop_in_bertopic": evolution_noop}
            json.dump({**run, "settings": {**settings, **flags}, "chunk_stats": chunk_stats},
                      open(os.path.join(v_dir, "metrics.json"), "w"), indent=2)
            runs.append(run)
            print(f"[K={K} {variant}] TTC={metrics['TTC']:.4f} TTS={metrics['TTS']:.4f} TTQ={metrics['TTQ']:.4f} "
                  f"| empty cells {trainer.empty_cells_['n_empty_cells']}/{trainer.empty_cells_['n_cells']} "
                  f"| top-5 vs topics_over_time: order {check['identical_order']:.3f}, set {check['identical_set']:.3f}, "
                  f"up to ties {check['identical_up_to_ties']:.3f} "
                  f"| evo no-op: {evolution_noop} "
                  f"| fit {t_fit:.1f}s beta {t_beta:.1f}s eval {t_eval:.1f}s "
                  f"| pipeline {timings['pipeline_seconds']:.1f}s", flush=True)

    best = max((r for r in runs if r["variant"] == "default"), key=lambda r: r["TTC"])
    selected_path = None
    if not args.no_select:
        dst_dir = os.path.join(args.data_dir, ds, "BERTopic")
        os.makedirs(dst_dir, exist_ok=True)
        selected_path = os.path.join(dst_dir, "beta.npy")
        shutil.copyfile(os.path.join(out_dir, f"K{best['K']}", "beta.npy"), selected_path)
        print(f"[select] K={best['K']} (TTC={best['TTC']:.4f}) -> {selected_path}", flush=True)

    log = {"dataset": ds, "n_docs": len(idx), "max_docs": args.max_docs,
           "num_times": dataset.num_times, "vocab_size": len(dataset.vocab),
           "settings": settings, "seed_used_for": ["UMAP random_state", "KMeans random_state",
                                                   "numpy/torch/random global seeds"],
           "variants": dict(variants),
           "empty_cell_fill": "topic's global c-TF-IDF row",
           "evaluation": {"top_words": 20, "topn": 10, "coherence": "c_npmi",
                          "reference_corpus": "processed/train_texts.txt (full)",
                          "assessor": "backend.evaluation.eval.TopicQualityAssessor.get_dtq_summary"},
           "chunk_stats": chunk_stats, "runs": runs,
           "selected": {"K": best["K"], "variant": "default", "criterion": "max TTC among default runs",
                        "beta_path": selected_path},
           "environment": env}
    json.dump(log, open(os.path.join(out_dir, "run_log.json"), "w"), indent=2)
    return runs, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["TCPD_IPD_Finance", "ACL_Anthology", "UN_Debates"])
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results", "BERTopic"))
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-docs", type=int, default=None, help="random subsample (smoke tests only)")
    ap.add_argument("--no-select", action="store_true", help="do not write data/<ds>/BERTopic/beta.npy")
    ap.add_argument("--no-cache", action="store_true", help="recompute embeddings and UMAP")
    ap.add_argument("--notuning", action="store_true",
                    help="also build a tuning-off variant (global/evolution off) from the same "
                         "embeddings/UMAP/KMeans; saved as <ds>/K<k>_notuning/, never selected")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    env = environment()
    json.dump(env, open(os.path.join(args.results_dir, "env.json"), "w"), indent=2)
    print(json.dumps(env, indent=2), flush=True)

    all_runs, selected = [], {}
    for ds in args.datasets:
        runs, best = run_dataset(ds, args, env)
        all_runs += runs
        selected[ds] = best["K"]

    cols = ["dataset", "K", "variant", "global_tuning", "evolution_tuning", "TTC", "TTS", "TTQ", "TC", "TD", "TQ", "n_topics_found",
            "embedding_seconds", "umap_seconds", "kmeans_ctfidf_fit_seconds", "beta_seconds",
            "pipeline_seconds", "evaluation_seconds"]
    is_sel = lambda r: r["variant"] == "default" and r["K"] == selected[r["dataset"]]
    with open(os.path.join(args.results_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["empty_cells", "n_cells", "selected"])
        for r in all_runs:
            w.writerow([r[c] for c in cols] + [r["empty_cells"]["n_empty_cells"], r["empty_cells"]["n_cells"],
                                               is_sel(r)])
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write("| Dataset | K | Variant | TTC | TTS | TTQ | empty cells | pipeline (s) | embed (s) | UMAP (s) | fit (s) | beta (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_runs:
            sel = " **(selected)**" if is_sel(r) else ""
            f.write(f"| {r['dataset']} | {r['K']}{sel} | {r['variant']} | {r['TTC']:.4f} | {r['TTS']:.4f} | {r['TTQ']:.4f} | "
                    f"{r['empty_cells']['n_empty_cells']}/{r['empty_cells']['n_cells']} | "
                    f"{r['pipeline_seconds']:.1f} | {r['embedding_seconds']:.1f} | {r['umap_seconds']:.1f} | "
                    f"{r['kmeans_ctfidf_fit_seconds']:.1f} | {r['beta_seconds']:.1f} |\n")
    print("\n" + open(os.path.join(args.results_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
