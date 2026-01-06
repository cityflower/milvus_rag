# milvus_project/utils/logger.py
import logging

def setup_logging(level=logging.INFO):
    """配置全局日志格式和级别"""
    logging.basicConfig(
        level=level, 
        format="[%(levelname)s] %(name)s - %(message)s"
    )

# 默认调用一次配置
setup_logging()