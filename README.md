# 📚 制度文件智能问答系统

上传公司制度文件（PDF/Excel/TXT），然后用自然语言提问，系统自动检索相关内容并回答。

## 快速开始

### 方式一：Docker（推荐）

```bash
# 构建镜像
docker build -t doc-qa .

# 运行（不配置 LLM，仅返回检索结果）
docker run -d -p 5000:5000 -v doc-data:/app/uploads --name doc-qa doc-qa

# 运行（配置 LLM，智能回答）
docker run -d -p 5000:5000 -v doc-data:/app/uploads \
  -e LLM_API_BASE=https://api.openai.com/v1 \
  -e LLM_API_KEY=你的API密钥 \
  -e LLM_MODEL=gpt-4o-mini \
  --name doc-qa doc-qa
```

### 方式二：直接运行

```bash
pip install -r requirements.txt
python app.py
```

然后打开浏览器访问 http://localhost:5000

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `LLM_API_BASE` | LLM API 地址（OpenAI 兼容） | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API 密钥 | 空（不配置则仅返回检索结果） |
| `LLM_MODEL` | 模型名称 | `gpt-4o-mini` |
| `RAG_TOP_K` | 检索返回的文本块数量 | `5` |
| `RAG_CHUNK_SIZE` | 每个文本块的字符数 | `500` |
| `PORT` | 服务端口 | `5000` |

## 支持的 LLM 服务

任何兼容 OpenAI API 格式的服务都可以，例如：

- **OpenAI**: `https://api.openai.com/v1`
- **Ollama**: `http://localhost:11434/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`
- **通义千问**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **智谱**: `https://open.bigmodel.cn/api/paas/v4`

## 使用流程

1. 上传制度文件（支持 PDF、Excel、TXT）
2. 系统自动解析并建立索引
3. 输入问题，系统检索相关内容
4. LLM 基于检索结果生成回答（未配置 LLM 则直接显示检索结果）

## 特点

- ✅ 支持 PDF / Excel / TXT 格式
- ✅ 中文友好（TF-IDF + n-gram 检索）
- ✅ 无需向量数据库，轻量级
- ✅ LLM 不可用时自动降级为纯检索
- ✅ Docker 一键部署
- ✅ 手机端适配
