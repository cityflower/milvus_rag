# milvus_project/config.py

# --- 0. 本地模型路径配置 ---
BASE_MODEL_PATH = "models"
EMBED_MODEL_PATH = f"{BASE_MODEL_PATH}/qwen3-embedding-0.6b"
RERANK_MODEL_PATH = f"{BASE_MODEL_PATH}/qwen3-reranker-0.6b"
VECTOR_DIM = 1024 # Qwen3-Embedding-0.6B 的维度


# Milvus 配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
MILVUS_DB_NAME = "default"
MILVUS_COLLECTION = "test"
VECTOR_DIM = 1024 # qwen3-embedding-0.6b 的维度

# LLM 语义切分配置 (代理服务)
LLM_CHUNK_BASE_URL = "http://localhost:1234/v1"
LLM_CHUNK_API_KEY = "1"
LLM_CHUNK_MODEL = "qwen/qwen3-vl-30b"
MAX_CHARS_PER_CHUNK = 4000 # 粗粒度切分的限制

# --- 3. Reranker 专有配置 ---
RERANK_MAX_LENGTH = 8192
RERANK_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query' # 默认任务描述
RERANK_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"

# RAG 问答 LLM 配置
RAG_LLM_MODEL = "qwen/qwen3-vl-30b"
RAG_LLM_BASE_URL = "http://localhost:1234/v1" # 假设与切分使用同一代理
RAG_LLM_API_KEY = "1"

# 文本切分 Prompt (保持原样)
SLICE_PROMPT = """你是文本语义切分助手，必须严格遵守以下规则：

1. **先判断主题数量** - 若全文只讲**一个人、一个产品、一个事件**，则**必须合并为 1 段**（除非原文超过 1500 汉字，才可拆成 2 段）。  
    - 若全文明显包含**多个不同主题**（如多人、多产品、多章节），则按主题切分。
    - 若原文出现一级标题标记 `#### `，则**以每个标题为边界单独成段**（含标题行），**禁止跨标题合并**。

2. **合并规则** - 同一人的"基本信息、教育、工作、项目、技能、联系方式"**全部合并为 1 段**。  
    - 同一产品的"简介、功能、参数、优势"**全部合并为 1 段**。  

3. **禁止行为** - 禁止按句子、段落、列表项切分。  
    - 禁止把同一主题拆成多段。

4. **输出格式** - 仅返回标准 JSON 列表，每项为一段**完整原文**，不增删字。  
    - 若文本为空，返回 []。

待切分文本：
---
{text}
"""