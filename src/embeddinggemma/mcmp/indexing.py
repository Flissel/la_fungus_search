from __future__ import annotations
from typing import Optional
import numpy as np
import logging
_logger = logging.getLogger("MCMP.Indexing")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n


def build_faiss_index(embs: np.ndarray, dim: int) -> Optional[object]:
    _logger.info("build_faiss_index: dim=%d docs=%d faiss=%s", dim, int(embs.shape[0]), str(_FAISS_OK))
    if not _FAISS_OK:
        return None
    # For very small datasets, avoid IVF with huge cluster count
    n_docs = int(embs.shape[0])
    factory = "Flat" if n_docs < 4096 else "IVF4096,Flat"
    # Create index (cosine via inner product; caller should pass L2-normalized vectors if needed)
    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    # GPU diagnostics and move if possible
    try:
        gpu_count = int(getattr(faiss, 'get_num_gpus', lambda: 0)())
    except Exception:
        gpu_count = 0
    if gpu_count > 0:
        try:
            _ = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_all_gpus(index)
            _logger.info("build_faiss_index: faiss_gpus=%d using_gpu=True", gpu_count)
        except Exception as e:
            _logger.warning("build_faiss_index: failed to move index to GPU (faiss_gpus=%d): %s", gpu_count, e)
    else:
        _logger.info("build_faiss_index: faiss_gpus=0 using_gpu=False")
    # Normalize embeddings for cosine-like behavior with inner product
    embs = embs.astype(np.float32, copy=False)
    embs_n = _l2_normalize_rows(embs)
    # Train/add
    # Train only when index is trainable (IVF), skip for Flat
    try:
        index.train(embs_n)
    except Exception:
        pass
    index.add(embs_n)
    try:
        index.nprobe = 12
    except Exception:
        pass
    return index


def save_faiss_index(index, path: str) -> bool:
    """Save FAISS index to disk. GPU indexes are moved to CPU first."""
    if not _FAISS_OK or index is None:
        return False
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        # GPU index must be converted to CPU for serialization
        cpu_index = index
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            pass
        faiss.write_index(cpu_index, path)
        _logger.info("save_faiss_index: saved to %s", path)
        return True
    except Exception as e:
        _logger.warning("save_faiss_index: failed: %s", e)
        return False


def load_faiss_index(path: str, gpu: bool = True) -> Optional[object]:
    """Load FAISS index from disk, optionally move to GPU."""
    if not _FAISS_OK:
        return None
    import os
    if not os.path.exists(path):
        return None
    try:
        index = faiss.read_index(path)
        if gpu:
            try:
                gpu_count = int(getattr(faiss, 'get_num_gpus', lambda: 0)())
                if gpu_count > 0:
                    index = faiss.index_cpu_to_all_gpus(index)
                    _logger.info("load_faiss_index: moved to GPU (%d GPUs)", gpu_count)
            except Exception:
                pass
        try:
            index.nprobe = 12
        except Exception:
            pass
        _logger.info("load_faiss_index: loaded from %s (ntotal=%d)", path, int(index.ntotal))
        return index
    except Exception as e:
        _logger.warning("load_faiss_index: failed: %s", e)
        return None


def faiss_search(index, query_vec: np.ndarray, top_k: int):
    _logger.debug("faiss_search: top_k=%d", int(top_k))
    q = query_vec.reshape(1, -1).astype(np.float32)
    qn = _l2_normalize_rows(q)
    D, I = index.search(qn, int(top_k))
    return D[0], I[0]


def cosine_similarities_cpu(query_vec: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    _logger.debug("cosine_similarities_cpu: docs=%d", int(doc_embeddings.shape[0]))
    q = query_vec.reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    X = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)
    return (q @ X.T).ravel()


def cosine_similarities_torch(query_vec: np.ndarray, doc_emb_torch_norm, device: str = 'cuda') -> np.ndarray:
    _logger.debug("cosine_similarities_torch: device=%s", device)
    import torch
    with torch.no_grad():
        q = torch.tensor(query_vec.astype(np.float32), device=device).to(torch.float16)
        qn = torch.nn.functional.normalize(q, p=2, dim=0)
        sims_t = doc_emb_torch_norm @ qn
        return sims_t.float().cpu().numpy()


