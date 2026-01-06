# milvus_project/core/llm.py
from openai import OpenAI
from ..config import LLM_CHUNK_BASE_URL, LLM_CHUNK_API_KEY, RAG_LLM_MODEL, RAG_LLM_BASE_URL, RAG_LLM_API_KEY
import logging

logger = logging.getLogger(__name__)

# 语义切分专用的 OpenAI 客户端
chunk_llm_client = OpenAI(
    base_url=LLM_CHUNK_BASE_URL,
    api_key=LLM_CHUNK_API_KEY
)

# RAG 问答专用的 OpenAI 客户端 (可与切分使用同一配置)
rag_llm_client = OpenAI(
    base_url=RAG_LLM_BASE_URL,
    api_key=RAG_LLM_API_KEY
)

def generate_rag_answer(prompt: str) -> str:
    """调用大模型生成 RAG 问答结果"""
    try:
        llm_resp = rag_llm_client.chat.completions.create(
            model=RAG_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # 问答保持低温度
        )
        return llm_resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("RAG LLM 调用失败: %s", e)
        raise RuntimeError(f"LLM 服务调用失败: {e}")