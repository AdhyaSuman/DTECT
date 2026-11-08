"""
Recover the mapping processed doc -> raw doc (line number in docs.jsonl).

The preprocessing step (backend/datasets/preprocess.py) drops short documents and
does not save the surviving indexes, and processed/train_texts.txt is not in the
same order as docs.jsonl. BERTopic needs the raw text of each processed document
(to embed it), so we recover the mapping by content:

  * candidates: raw docs with the same timestamp (time id) as the processed doc
  * stems: 5-character prefixes of lowercase alphabetic words (>= 3 chars).
    Prefixes make the comparison robust to lemmatisation; bigrams ('a_b') are split.
  * selection score: Jaccard(stems(processed), stems(raw) & vocab_stems), where
    vocab_stems are the stems of processed/vocab.txt (so stop words and pruned
    words in the raw text do not count). Symmetric, so a long raw doc does not beat
    the true source of a short processed doc.
  * reported quality: coverage = |stems(processed) & stems(raw)| / |stems(processed)|
    of the selected raw doc; docs with coverage < 0.7 are flagged.
  * search: the 10 rarest (within-year) stems of the processed doc retrieve
    candidates via an inverted index; the 20 raw docs with the most hits are scored.
    If the selected doc's coverage is < 0.7, all raw docs of that year are scored.

Outputs (new files, nothing existing is modified):
  data/<ds>/processed/raw_doc_ids.txt       one raw line number per processed doc
  data/<ds>/processed/raw_alignment_report.json

Ties (several raw candidates with the same best Jaccard and coverage) and raw docs used by several
processed docs are checked for (near-)identical raw text; any tied candidate may be
used. Processed docs with coverage < 0.7 are flagged.

Usage:  python scripts/align_raw_docs.py ACL_Anthology TCPD_IPD_Finance UN_Debates
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

LOW_COVERAGE = 0.7
NEAR_IDENTICAL = 0.95          # word-set Jaccard threshold for "near-identical" raw texts
N_RARE, N_CAND = 10, 20

_WORD = re.compile(r"[a-z]+")


def stems_raw(text):
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) >= 3}


def stems_proc(text):
    return {w[:5] for tok in text.split() for w in _WORD.findall(tok.lower()) if len(w) >= 3}


def norm_text(text):
    return " ".join(text.lower().split())


def text_relation(a, b):
    """'identical' | 'near' | 'different' for two raw texts."""
    if norm_text(a) == norm_text(b):
        return "identical"
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    jac = len(wa & wb) / max(len(wa | wb), 1)
    return "near" if jac >= NEAR_IDENTICAL else "different"


def align(dataset_dir):
    t0 = time.time()
    raw = []
    with open(os.path.join(dataset_dir, "docs.jsonl"), encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    with open(os.path.join(dataset_dir, "processed/train_texts.txt"), encoding="utf-8", errors="ignore") as f:
        proc = [line.strip() for line in f]
    with open(os.path.join(dataset_dir, "processed/train_times.txt")) as f:
        times = [int(float(x)) for x in f if x.strip()]
    with open(os.path.join(dataset_dir, "processed/time2id.txt")) as f:
        time2id = {str(k): v for k, v in json.load(f).items()}
    assert len(proc) == len(times), "train_texts / train_times length mismatch"

    with open(os.path.join(dataset_dir, "processed/vocab.txt"), encoding="utf-8") as f:
        vocab_stems = set().union(*(stems_proc(line.strip()) for line in f))

    # Index raw docs per time id
    raw_stems, raw_vstems, raw_tid = [], [], []
    inv = defaultdict(lambda: defaultdict(list))
    df = defaultdict(Counter)
    by_time = defaultdict(list)
    n_raw_unknown_time = 0
    for i, r in enumerate(raw):
        tid = time2id.get(str(r.get("timestamp")))
        st = stems_raw(r["text"])
        raw_stems.append(st)
        raw_vstems.append(st & vocab_stems)
        raw_tid.append(tid)
        if tid is None:
            n_raw_unknown_time += 1
            continue
        by_time[tid].append(i)
        for s in st:
            inv[tid][s].append(i)
            df[tid][s] += 1

    mapping, coverage, jaccard = [], [], []
    ties = []                       # (proc idx, [tied raw ids])
    n_fullscan = 0
    for j, (text, tid) in enumerate(zip(proc, times)):
        st = stems_proc(text)
        if not st or tid not in by_time:
            mapping.append(-1)
            coverage.append(0.0)
            continue
        rare = sorted((s for s in st if df[tid][s] > 0), key=lambda s: df[tid][s])[:N_RARE]
        hits = Counter()
        for s in rare:
            hits.update(inv[tid][s])
        cands = [i for i, _ in hits.most_common(N_CAND)]

        def score(cs):
            # (Jaccard, coverage): coverage only breaks Jaccard ties
            sc = [((len(st & raw_vstems[i]) / len(st | raw_vstems[i]),
                    len(st & raw_stems[i]) / len(st)), i) for i in cs]
            if not sc:
                return 0.0, 0.0, []
            best = max(c for c, _ in sc)
            tied = sorted(i for c, i in sc if c == best)
            return best[0], best[1], tied

        best, cov, tied = score(cands)
        if cov < LOW_COVERAGE:
            n_fullscan += 1
            best, cov, tied = score(by_time[tid])
        mapping.append(tied[0] if tied else -1)
        coverage.append(cov)
        jaccard.append(best)
        if len(tied) > 1:
            ties.append((j, tied))

    # Ties: are the tied raw candidates (near-)identical?
    tie_rel = Counter()
    tie_examples = []
    for j, tied in ties:
        rels = [text_relation(raw[tied[0]]["text"], raw[i]["text"]) for i in tied[1:]]
        rel = "different" if "different" in rels else ("near" if "near" in rels else "identical")
        tie_rel[rel] += 1
        if rel == "different" and len(tie_examples) < 20:
            tie_examples.append({"proc_idx": j, "raw_ids": tied})

    # Raw docs used by several processed docs
    users = defaultdict(list)
    for j, i in enumerate(mapping):
        if i >= 0:
            users[i].append(j)
    reused = {i: js for i, js in users.items() if len(js) > 1}
    n_reused_identical_proc = sum(len({proc[j] for j in js}) == 1 for js in reused.values())

    low = [(j, round(c, 4)) for j, c in enumerate(coverage) if c < LOW_COVERAGE]

    def quantiles(xs):
        xs = sorted(xs)
        q = lambda p: round(xs[min(int(p * len(xs)), len(xs) - 1)], 4)
        return {"min": q(0), "p01": q(0.01), "p05": q(0.05), "median": q(0.5), "max": q(1.0)}

    report = {
        "n_raw": len(raw),
        "n_processed": len(proc),
        "n_raw_unknown_timestamp": n_raw_unknown_time,
        "n_unmatched": mapping.count(-1),
        "n_raw_docs_used": len(users),
        "n_raw_docs_never_used": len(raw) - len(users),
        "coverage_quantiles": quantiles(coverage),
        "jaccard_quantiles": quantiles(jaccard),
        "low_coverage_threshold": LOW_COVERAGE,
        "n_low_coverage": len(low),
        "low_coverage_docs": [{"proc_idx": j, "coverage": c} for j, c in low],
        "n_full_year_scans": n_fullscan,
        "ties": {
            "n_processed_with_tied_candidates": len(ties),
            "identical": tie_rel["identical"],
            "near_identical": tie_rel["near"],
            "different": tie_rel["different"],
            "near_identical_definition": f"word-set Jaccard >= {NEAR_IDENTICAL}",
            "examples_different": tie_examples,
        },
        "raw_reused": {
            "n_raw_docs_used_by_multiple_processed": len(reused),
            "n_processed_docs_involved": sum(len(js) for js in reused.values()),
            "n_where_processed_texts_identical": n_reused_identical_proc,
        },
        "method": "same timestamp; select by Jaccard of 5-char stems (raw restricted to vocab stems); "
                  "report coverage; see scripts/align_raw_docs.py",
        "seconds": round(time.time() - t0, 1),
    }
    return mapping, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    for ds in args.datasets:
        d = os.path.join(args.data_dir, ds)
        mapping, report = align(d)
        with open(os.path.join(d, "processed/raw_doc_ids.txt"), "w") as f:
            f.write("\n".join(map(str, mapping)) + "\n")
        with open(os.path.join(d, "processed/raw_alignment_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        summary = {k: report[k] for k in ("n_raw", "n_processed", "n_unmatched", "n_low_coverage",
                                          "coverage_quantiles", "jaccard_quantiles", "ties", "raw_reused", "seconds")}
        summary["ties"] = {k: v for k, v in summary["ties"].items() if k != "examples_different"}
        print(f"== {ds}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
