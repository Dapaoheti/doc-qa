FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# 创建数据目录
RUN mkdir -p /app/uploads

EXPOSE 5000

# 环境变量配置
# LLM_API_BASE - OpenAI 兼容 API 地址
# LLM_API_KEY  - API 密钥
# LLM_MODEL    - 模型名称
# PORT         - 端口号

CMD ["python", "app.py"]
