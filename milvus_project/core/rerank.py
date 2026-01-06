# milvus_project/core/rerank.py
import torch
import logging
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..config import RERANK_MODEL_PATH, RERANK_PREFIX, RERANK_SUFFIX, RERANK_MAX_LENGTH, RERANK_INSTRUCTION

logger = logging.getLogger(__name__)

# --- 模型加载和设备配置 ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    logger.info(f"Loading Reranker model from {RERANK_MODEL_PATH} on device {DEVICE}...")
    
    # 加载 Qwen3-Reranker-0.6B 模型
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH, padding_side='left')
    reranker_model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()

    # 获取 Reranker 专有 Token ID
    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    
    # 预编码前后缀
    prefix_tokens = reranker_tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix_tokens = reranker_tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)

    logger.info("Reranker model loaded and configured successfully.")

except Exception as e:
    logger.error(f"Failed to load local Reranker model: {e}")
    reranker_tokenizer = None
    reranker_model = None


def format_instruction(instruction: str, query: str, doc: str) -> str:
    """构造 reranker 所需的输入格式"""
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )

def process_inputs(pairs: List[str]):
    """对输入文本进行 tokenize 并加上前后缀，准备喂给 reranker"""
    if not reranker_tokenizer or not reranker_model:
         raise RuntimeError("Reranker model is not initialized.")

    max_len = RERANK_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)

    inputs = reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_len
    )
    
    # 给每条输入加上前缀和后缀
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        
    # 补齐长度并转成 PyTorch 张量
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors='pt', max_length=RERANK_MAX_LENGTH)
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
        
    return inputs


def compute_logits(inputs):
    """用 reranker 计算查询-文档对的相关性得分"""
    if not reranker_model:
         raise RuntimeError("Reranker model is not initialized.")
         
    with torch.no_grad():
        batch_scores = reranker_model(**inputs).logits[:, -1, :]  # 取最后一个 token 的 logits
        true_vector = batch_scores[:, token_true_id]   # "yes" 对应的 logit
        false_vector = batch_scores[:, token_false_id] # "no" 对应的 logit
        
        # Stack true/false logits and apply log_softmax
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        # 取 "yes" 的概率 (e^log_softmax(yes)) 作为最终得分
        scores = batch_scores[:, 1].exp().tolist()
        
    return scores


def rerank_documents(query: str, documents: List[str], task_instruction: str = RERANK_INSTRUCTION) -> List[Tuple[str, float]]:
    """根据查询对文档进行重排序，返回按相关性降序排列的 (文档, 得分) 列表"""
    if not reranker_model:
        # 如果模型加载失败，返回原始文档列表 (得分 0.0)
        logger.warning("Reranker model failed to load. Returning unranked documents.")
        return [(doc, 0.0) for doc in documents]

    # 构造每条输入
    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]

    # 模型推理拿分
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)

    # 把文档和得分打包并降序排列
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return doc_scores