# milvus_project/api/ask.py
import logging
from fastapi import APIRouter, HTTPException
from pymilvus import AnnSearchRequest, RRFRanker

from milvus_project.models import AskRequest, AskResponse
from milvus_project.core.embedding import get_embeddings
from milvus_project.core.milvus_client import milvus_client, COLLECTION
from milvus_project.core.llm import generate_rag_answer

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ask/", response_model=AskResponse, summary="RAG 问答")
async def ask(req: AskRequest):
    top_k = req.top_k
    initial_limit = top_k * 3

    try:
        # 1. 检索 (与 search 逻辑相同)
        vec = get_embeddings([req.query])[0]
        sparse_req = AnnSearchRequest(
            [req.query], "sparse_vector", {"metric_type": "BM25"}, limit=initial_limit)
        dense_req = AnnSearchRequest(
            [vec], "dense_vector", {"metric_type": "IP"}, limit=initial_limit)
        res = milvus_client.hybrid_search(
            COLLECTION, [sparse_req, dense_req], ranker=RRFRanker(),
            limit=initial_limit, output_fields=["content", "metadata"])[0]

        # 2. 构造上下文
        context_chunks = []
        for idx, r in enumerate(res[:top_k], 1):
            logger.info("Context Top-%d | score=%.4f | content=%.100s",
                        idx, r.score, r.entity.get("content", ""))
            context_chunks.append(r.entity["content"])
        context = "\n\n".join(context_chunks)

        # 3. 构造 Prompt
        prompt = f"""根据以下上下文回答问题：
{context}
问题：{req.query}
若上下文无相关信息，请回答"我无法根据提供的信息回答该问题"。"""

        # 4. 调用大模型生成答案
        answer = generate_rag_answer(prompt)
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"RAG 问答流程失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 问答服务失败: {e}")