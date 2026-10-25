"""
bge_embedder.py - BGE-M3 embeddings: dense (1024) and sparse (lexical weights) in one pass.

The model is pinned to one Hugging Face revision, so vectors stay comparable across installs.
Download it once (about 2.3 GB, afterwards everything runs offline) and run a self-test:

    python scripts/bge_embedder.py
"""

from __future__ import annotations

import os

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
# Only what inference needs: weights, the ColBERT and sparse heads, tokenizer and configs.
# The model repo also holds ONNX exports and images (another ~2.3 GB) that are never used here.
MODEL_FILES = ["*.json", "*.model", "pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt"]
_REQUIRED = ("pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt", "tokenizer.json")
# Pad every batch to a multiple of 64 tokens. torch on Apple silicon (MPS) keeps one compiled
# graph per input shape and never frees it; measured on 2,400 chunks: 24.5 GB peak memory with
# free-length padding, 5.0 GB with this setting. Padding tokens are masked out, so the
# embeddings stay the same.
_PAD = {"pad_to_multiple_of": 64}


def model_path() -> str:
    """Local folder of the pinned model. Raises with instructions if it is not downloaded yet."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            "BGE-M3 is not downloaded yet: run `python scripts/bge_embedder.py` once (about 2.3 GB)."
        ) from exc
    missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(path, f))]
    if missing:
        # Without the sparse head FlagEmbedding would start a random one and return
        # meaningless sparse vectors, so stop instead.
        raise RuntimeError(f"BGE-M3 download incomplete ({', '.join(missing)}): run `python scripts/bge_embedder.py`")
    return path


class BGEEmbedder:
    """Loads the model once per process (a few seconds), then embeds queries and documents."""

    _model = None

    def __init__(self, use_fp16: bool = True):
        if BGEEmbedder._model is None:
            from FlagEmbedding import BGEM3FlagModel
            from transformers.utils import logging as transformers_logging

            # No "Loading weights" progress bar on every search: advisors read this output.
            transformers_logging.disable_progress_bar()
            BGEEmbedder._model = BGEM3FlagModel(model_path(), use_fp16=use_fp16)
        self.model = BGEEmbedder._model

    def embed_dense_and_sparse(self, text: str):
        """-> (dense vector, qdrant SparseVector) for one text."""
        return self.embed_batch_dense_and_sparse([text])[0]

    def embed_batch_dense_and_sparse(self, texts: list[str], batch_size: int = 32) -> list:
        """-> [(dense vector, qdrant SparseVector)] in one forward pass per batch."""
        from qdrant_client import models

        out = self.model.encode(texts, batch_size=batch_size, return_dense=True, return_sparse=True, **_PAD)
        results = []
        for dense_arr, weights in zip(out["dense_vecs"], out["lexical_weights"], strict=True):
            sparse = models.SparseVector(indices=[int(k) for k in weights], values=[float(v) for v in weights.values()])
            results.append(([float(x) for x in dense_arr], sparse))
        return results


if __name__ == "__main__":
    # The one place that is allowed to go online: fetch the pinned revision, then self-test.
    # (huggingface_hub reads HF_HUB_OFFLINE on import, so this must run before model_path().)
    os.environ["HF_HUB_OFFLINE"] = "0"
    print(f"model: {model_path()}")
    dense, sparse = BGEEmbedder().embed_dense_and_sparse("kb-build self-test: hybrid retrieval")
    if len(dense) != 1024 or not sparse.indices:
        raise SystemExit(f"self-test FAILED: dense={len(dense)} dims, sparse={len(sparse.indices)} terms")
    print(f"self-test OK: dense={len(dense)} dims, sparse={len(sparse.indices)} terms")
