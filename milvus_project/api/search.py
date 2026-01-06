# milvus_project/api/search.py
import logging
from typing import List
from fastapi import APIRouter
from pymilvus import AnnSearchRequest, RRFRanker

from milvus_project.models import SearchRequest, SearchResult
from milvus_project.core.embedding import get_embeddings
from milvus_project.core.milvus_client import milvus_client, COLLECTION

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search/", response_model=List[SearchResult], summary="混合向量检索")
async def search(req: SearchRequest):
    q = req.query
    top_k = req.top_k
    initial_limit = top_k * 3 # 多召回，使用 RRF 融合

    try:
        # 1. 查询向量
        vec = get_embeddings([q])[0]
        
        # 2. 混合检索请求
        sparse_req = AnnSearchRequest(
            [q], "sparse_vector", {"metric_type": "BM25"}, limit=initial_limit)
        dense_req = AnnSearchRequest(
            [vec], "dense_vector", {"metric_type": "IP"}, limit=initial_limit)
            
        # 3. 执行混合检索 (RRF 融合)
        res = milvus_client.hybrid_search(
            COLLECTION, [sparse_req, dense_req], ranker=RRFRanker(),
            limit=initial_limit, output_fields=["content", "metadata"])[0]
        
        # 4. 取 RRF 后的 top_k 结果
        final = []
        for idx, r in enumerate(res[:top_k], 1):
            logger.info("Top-%d | score=%.4f | content=%.100s",
                        idx, r.score, r.entity.get("content", ""))
            final.append(SearchResult(
                score=r.score,
                content=r.entity["content"],
                metadata=r.entity["metadata"]))
        
        return final
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索服务失败: {e}")