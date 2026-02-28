"""Fusion utilities for combining search results."""
from typing import Any, Dict, List, Optional


def rrf_fuse(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple result lists.
    
    Args:
        result_lists: List of result lists, each from a different query/scale/source
        k: RRF constant (default 60)
        limit: Maximum results to return (None = return all)
    
    Returns:
        Deduplicated, scored results sorted by RRF score descending.
        Each result has '_rrf_score' field added.
    """
    rrf_scores: Dict[str, float] = {}
    id_to_record: Dict[str, Dict[str, Any]] = {}
    
    for results in result_lists:
        for rank, record in enumerate(results):
            uid = record.get("id", f"rank_{rank}")
            # RRF formula: 1 / (k + rank + 1)
            score = 1.0 / (k + rank + 1)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + score
            if uid not in id_to_record:
                id_to_record[uid] = record
    
    # Sort by RRF score descending
    sorted_uids = sorted(rrf_scores.keys(), key=lambda u: rrf_scores[u], reverse=True)
    
    # Build result list
    fused = []
    for uid in (sorted_uids[:limit] if limit else sorted_uids):
        record = dict(id_to_record[uid])
        record["_rrf_score"] = rrf_scores[uid]
        fused.append(record)
    
    return fused
