set -euo pipefail

DATASET_ID="dokiik/MIXTURE"
CACHE_DIR=".hf_mixture_cache"

SFT_DEST="sft/data"         # from data_sft -> data
GRPO_DEST="data"            # from data_grpo -> data

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[INFO] Installing huggingface_hub ..."
  python3 -m pip install --upgrade --user huggingface_hub
fi

echo "[INFO] Downloading dataset: ${DATASET_ID}"

huggingface-cli download \
  --repo-type dataset "${DATASET_ID}" \
  --include "data_sft/**" \
  --include "data_grpo/**" \
  --local-dir "${CACHE_DIR}" \
  --local-dir-use-symlinks False

mkdir -p "${SFT_DEST}"
mkdir -p "${GRPO_DEST}"

rm -rf "${SFT_DEST:?}/"* "${GRPO_DEST:?}/"*

# data_sft -> sft/data
if [ -d "${CACHE_DIR}/data_sft" ]; then
  echo "[INFO] Copy data_sft -> ${SFT_DEST} (renamed to 'data')"
  rsync -a "${CACHE_DIR}/data_sft/"/ "${SFT_DEST}/"
else
  echo "[WARN] ${CACHE_DIR}/data_sft not found."
fi

# data_grpo -> data
if [ -d "${CACHE_DIR}/data_grpo" ]; then
  echo "[INFO] Copy data_grpo -> ${GRPO_DEST} (renamed to 'data')"
  rsync -a "${CACHE_DIR}/data_grpo/"/ "${GRPO_DEST}/"
else
  echo "[WARN] ${CACHE_DIR}/data_grpo not found."
fi

echo "[DONE] MIXTURE fetched and placed:
- ${SFT_DEST}/  (from data_sft)
- ${GRPO_DEST}/ (from data_grpo)"