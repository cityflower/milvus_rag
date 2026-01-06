# main.py
import logging
from fastapi import FastAPI
from milvus_project.api import upload, search, ask 
from milvus_project.utils.logger import setup_logging

# 1. 配置日志
# 尽管您在 utils/logger.py 中定义了 setup_logging，但为了兼容性，这里使用基本的配置。
# 实际项目中应调用 setup_logging()。
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 2. 初始化 FastAPI
app = FastAPI(title="Milvus 向量检索服务")

# 3. 注册路由
app.include_router(upload.router, tags=["File Operations"])
app.include_router(search.router, tags=["Retrieval"])
app.include_router(ask.router, tags=["RAG"])

@app.get("/")
def read_root():
    return {"message": "Milvus RAG Service is running. Access /docs for API details."}

# 4. 在应用启动时确保 Milvus 集合已初始化
from milvus_project.core.milvus_client import init_collection
init_collection()

# 运行应用 (通常通过 uvicorn main:app)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)