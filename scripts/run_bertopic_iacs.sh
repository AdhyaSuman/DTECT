#!/usr/bin/env bash
# BERTopic runs for Table 4 on the IACS desktop (GTX 1660 Super, 6 GB).
# Usage (from the repo root, inside the `dtect` conda env):
#   bash scripts/run_bertopic_iacs.sh            # full runs: 3 datasets x K in {20, 50, 100}
#   bash scripts/run_bertopic_iacs.sh --smoke    # 1,000 TCPD docs, K=20, nothing written to data/
set -euo pipefail
cd "$(dirname "$0")/.."

# If the HuggingFace model must be downloaded through the IACS proxy, export
# HTTP(S)_PROXY in your shell before running this script (do not hard-code credentials).

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available - check the torch build / driver"
print("GPU:", torch.cuda.get_device_name(0))
EOF

DATASETS="TCPD_IPD_Finance ACL_Anthology UN_Debates"

# raw_doc_ids.txt is produced by the alignment step (fast, CPU only)
for ds in $DATASETS; do
  [ -f "data/$ds/processed/raw_doc_ids.txt" ] || python scripts/align_raw_docs.py "$ds"
done

mkdir -p results
if [ "${1:-}" = "--smoke" ]; then
  python scripts/run_bertopic.py --datasets TCPD_IPD_Finance --ks 20 --max-docs 1000 \
      --results-dir results/BERTopic_smoke --no-select --notuning 2>&1 | tee results/bertopic_smoke.log
else
  # embeddings on GPU; batch 64 with 256-token chunks fits easily in 6 GB
  python scripts/run_bertopic.py --datasets $DATASETS --ks 20 50 100 \
      --device cuda --batch-size 64 --notuning --results-dir results/BERTopic 2>&1 | tee results/bertopic_full.log
fi
