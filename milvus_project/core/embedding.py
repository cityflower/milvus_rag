# milvus_project/core/embedding.py
import torch
import logging
from typing import List, Optional
# 使用 sentence-transformers 简化加载 Qwen-Embedding 模型
from sentence_transformers import SentenceTransformer
from ..config import EMBED_MODEL_PATH

logger = logging.getLogger(__name__)

# --- 模型加载和设备配置 ---
# 自动检测设备 (优先 MPS, 其次 CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    logger.info(f"Loading Embedding model from {EMBED_MODEL_PATH} on device {DEVICE}...")
    
    # SentenceTransformer 可以自动处理模型的加载和设备的移动
    # 注意： SentenceTransformer 需要模型文件夹符合其格式要求，但对于 Qwen/Qwen3-Embedding-0.6B 可能是有效的
    # 否则需要回退到 AutoModel 配合自定义池化。这里先使用您的 SentenceTransformer 方式。
    embedding_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
    embedding_model.eval()
    
    logger.info("Embedding model loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load local SentenceTransformer Embedding model: {e}")
    # 尝试回退到传统的 AutoModel 方式 (如果 SentenceTransformer 失败)
    try:
        from transformers import AutoModel, AutoTokenizer
        logger.info("Trying AutoModel fallback for Embedding...")
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
        model = AutoModel.from_pretrained(EMBED_MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
        
        # 定义备用池化函数 (参考标准 Qwen 方式)
        def fallback_average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        def fallback_get_embeddings(texts: List[str]) -> List[List[float]]:
            batch_dict = tokenizer(
                texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = fallback_average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()
        
        embedding_model = None # 标记 SentenceTransformer 失败
        logger.info("AutoModel Embedding fallback loaded successfully.")
    except Exception as fallback_e:
        logger.error(f"AutoModel Embedding fallback also failed: {fallback_e}")
        embedding_model = None


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """本地加载 Qwen3-Embedding-0.6B 模型生成文本嵌入向量。"""
    if embedding_model is not None:
        # SentenceTransformer 方式
        embeddings = embedding_model.encode(texts, prompt_name="query")
        return embeddings.tolist()
    elif 'fallback_get_embeddings' in locals():
        # AutoModel 回退方式 (不区分 query/document)
        return fallback_get_embeddings(texts)
    else:
        raise RuntimeError("Embedding model is not initialized. Check logs for loading errors.")

# 确定维度以供 Milvus 使用（如果模型已加载）
if embedding_model is not None:
    # 运行一次空文本获取维度
    try:
        test_emb = embedding_model.encode(["test"], prompt_name="query")[0]
        if len(test_emb) != 1024:
            logger.warning(f"Detected embedding dimension {len(test_emb)} does not match expected 1024.")
    except Exception:
        pass # 启动时 Milvus 客户端会检查维度，这里跳过