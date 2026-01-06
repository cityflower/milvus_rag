# milvus_project/models.py
from pydantic import BaseModel
from typing import List, Literal, Dict

# 请求模型
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

# 响应模型
class SearchResult(BaseModel):
    score: float
    content: str
    metadata: Dict

class AskResponse(BaseModel):
    answer: str

# 其他常量
ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
    "text/plain",
    "text/markdown",
}