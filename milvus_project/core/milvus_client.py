# milvus_project/core/milvus_client.py
import logging
from pymilvus import MilvusClient, DataType, Function, FunctionType
from ..config import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, MILVUS_COLLECTION, VECTOR_DIM

logger = logging.getLogger(__name__)

# 全局 Milvus 客户端实例
milvus_client = MilvusClient(
    uri=MILVUS_URI, 
    token=MILVUS_TOKEN, 
    db_name=MILVUS_DB_NAME
)
COLLECTION = MILVUS_COLLECTION

def init_collection():
    """初始化 Milvus 集合，若不存在则创建"""
    if milvus_client.has_collection(COLLECTION):
        milvus_client.load_collection(COLLECTION)
        logger.info("Collection %s loaded.", COLLECTION)
        return
    
    # 定义 Schema
    analyzer = {"tokenizer": {"type": "jieba", "mode": "search"}}
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
    schema.add_field("content", DataType.VARCHAR, max_length=65535,
                     analyzer_params=analyzer, enable_analyzer=True, enable_match=True)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM) 
    schema.add_field("metadata", DataType.JSON)
    
    # 添加 BM25 函数
    schema.add_function(Function(
        name="bm25", function_type=FunctionType.BM25,
        input_field_names=["content"], output_field_names="sparse_vector"))
        
    # 定义 Index
    idx = MilvusClient.prepare_index_params()
    idx.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    idx.add_index("dense_vector", index_type="FLAT", metric_type="IP")
    
    # 创建集合
    milvus_client.create_collection(COLLECTION, schema=schema, index_params=idx)
    logger.info("Collection %s created.", COLLECTION)

# 注意：为了让 main.py 可以控制初始化时机，我们不在这里自动调用 init_collection()
# 而是由 main.py 调用，以确保所有配置已加载。