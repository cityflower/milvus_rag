# milvus_project/config.py

# 线上服务地址和密钥
EMBED_URL = "http://192.168.30.103:6014/v1/embeddings"
EMBED_KEY = "sk-zZVAfGSXnGjVpYT127Cf5aD420F648F1826355455eEaD881"
RERANK_URL = "http://192.168.30.103:6015/v1/rerank"
RERANK_KEY = "sk-zZVAfGSXnGjVpYT127Cf5aD420F648F1826355455eEaD881"

# Milvus 配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
MILVUS_DB_NAME = "default"
MILVUS_COLLECTION = "test"
VECTOR_DIM = 1024 # qwen3-embedding-0.6b 的维度

# LLM 语义切分配置 (代理服务)
LLM_CHUNK_BASE_URL = "http://localhost:1234/v1"
LLM_CHUNK_API_KEY = "1"
LLM_CHUNK_MODEL = "gpt-oss-20b"
MAX_CHARS_PER_CHUNK = 4000 # 粗粒度切分的限制

# RAG 问答 LLM 配置
RAG_LLM_MODEL = "gpt-oss-20b"
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