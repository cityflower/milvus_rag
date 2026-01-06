# milvus_project/api/upload.py
import os
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from tempfile import NamedTemporaryFile

from milvus_project.models import ALLOWED_MIME
from milvus_project.utils.file_reader import read_file
from milvus_project.utils.chunker import llm_chunk_safe
from milvus_project.core.embedding import get_embeddings
from milvus_project.core.milvus_client import milvus_client, COLLECTION

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload/", summary="上传单个文件并入库")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")
    
    suffix = os.path.splitext(file.filename)[1].lower()
    tmp_path = None
    try:
        # 1. 保存临时文件
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # 2. 读取文件内容并切分
        text = read_file(tmp_path)
        chunks = llm_chunk_safe(text)
        if not chunks:
             return {"message": "文件内容为空或无法切分，未插入任何数据"}
             
        # 3. 获取向量
        dense_vecs = get_embeddings(chunks)
        
        # 4. 构建 Entity 并插入 Milvus
        entities = [{
            "content": ch,
            "dense_vector": vec,
            "metadata": {"source": file.filename, "topic": "upload"},
        } for ch, vec in zip(chunks, dense_vecs)]
        
        milvus_client.insert(COLLECTION, entities)
        logger.info("Inserted %d chunks from file: %s", len(entities), file.filename)
        
        return {"message": f"成功插入 {len(entities)} 个分块"}
    except Exception as e:
        logger.error(f"文件上传和入库失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件处理或入库失败: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)