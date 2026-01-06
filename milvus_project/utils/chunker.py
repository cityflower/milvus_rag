# milvus_project/utils/chunker.py
import json
import logging
from typing import List
from ..config import (
    SLICE_PROMPT, LLM_CHUNK_MODEL, 
    MAX_CHARS_PER_CHUNK
)
from ..core.llm import chunk_llm_client 

logger = logging.getLogger(__name__)

# 滑动窗口默认参数 (与原代码保持一致)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 128

# 确保 logger 级别足够低，能捕获 debug 信息
logger.setLevel(logging.DEBUG)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """滑动窗口切分"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def coarse_chunk(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """粗粒度切分，防止单个请求文本过长"""
    paragraphs = text.splitlines()
    chunks, buffer = [], []
    buffer_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if buffer_len + len(para) > max_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer, buffer_len = [], 0
        buffer.append(para)
        buffer_len += len(para) + 1
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks

def llm_chunk_single(text: str) -> List[str]:
    """调用 LLM 进行语义切分，失败则回退滑动窗口"""
    fallback_message = "LLM 语义切分失败，回退滑动窗口"
    
    try:
        logger.debug("Attempting LLM chunking...")
        resp = chunk_llm_client.chat.completions.create(
            model=LLM_CHUNK_MODEL,
            messages=[{"role": "user", "content": SLICE_PROMPT.format(text=text)}],
            temperature=0,
            # max_tokens=4096
        )
        
        # 获取 LLM 响应内容
        llm_output_content = resp.choices[0].message.content.strip()
        
        if not llm_output_content:
            logger.warning(f"{fallback_message}: LLM 返回内容为空。")
            # 即使 LLM 返回空，也应执行回退
            raise ValueError("LLM returned empty content.")

        # 1. 健壮地移除 Markdown 标记 (可能包含语言标识如 'json', 'jsonc')
        raw = llm_output_content
        if raw.startswith("```"):
            try:
                # 找到第一个换行符之后的内容，并去除末尾的 ```
                start_index = raw.find('\n') + 1
                raw = raw[start_index:].strip().removesuffix("```").strip()
            except Exception as e:
                logger.warning(f"{fallback_message}: 移除 Markdown 标记失败。原始输出: {llm_output_content[:500]}... Error: {e}")
                raise ValueError("Failed to clean Markdown wrapper.")
        
        # 记录清理后的 JSON 字符串（仅打印开头，防止日志过长）
        logger.debug(f"LLM cleaned response (start): {raw[:500]}...")
        
        # 2. 尝试解析 JSON
        try:
            chunks = json.loads(raw)
        except json.JSONDecodeError as e:
            # 捕获 JSON 解析错误，并打印出原始的、导致错误的字符串
            logger.error(f"{fallback_message}: JSON 解析失败。原始字符串: {raw[:500]}... 错误详情: {e}")
            raise e # 重新抛出错误，触发外层异常处理和回退
            
        # 3. 校验解析结果
        if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
            return [c.strip() for c in chunks if c.strip()]
        else:
            logger.warning(f"{fallback_message}: LLM 响应格式错误。期望 List[str]，得到 {type(chunks)}。")
            raise ValueError("Invalid LLM response format.")
            
    except Exception as e:
        # 最终捕获所有异常，执行回退
        logger.warning(f"{fallback_message}。错误类型: {type(e).__name__}, 错误信息: {e}")
    
    # LLM 切分失败，则回退到滑动窗口
    return chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)


def llm_chunk_safe(text: str) -> List[str]:
    """安全地进行 LLM 切分：先粗切分，再对每块进行 LLM 语义切分"""
    coarse_chunks = coarse_chunk(text)
    final_chunks = []
    for chunk in coarse_chunks:
        final_chunks.extend(llm_chunk_single(chunk))
    return final_chunks