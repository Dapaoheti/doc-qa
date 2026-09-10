"""
制度文件智能问答系统 v1.0
支持 PDF / Excel / Word 文件导入，基于 TF-IDF 检索 + LLM 回答
"""
import os, json, uuid, re, math, threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import pdfplumber
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# OCR 支持（扫描件PDF）
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ===================== 配置 =====================
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# LLM 配置（支持任何 OpenAI 兼容 API）
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# 检索配置
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))

app = Flask(__name__)

# ===================== 文档存储 =====================
documents = {}   # doc_id -> { id, name, type, chunks, upload_time }
tfidf_matrix = None
vectorizer = None
all_chunks = []  # flat list of { doc_id, doc_name, chunk_id, text }
lock = threading.Lock()

# ===================== 文件解析 =====================
def has_chinese(text):
    """检查文本是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def parse_pdf(filepath):
    """解析 PDF 文件，支持 OCR 识别扫描件"""
    chunks = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip() and has_chinese(text):
                    chunks.append({"page": page_num, "text": text.strip()})
    except Exception as e:
        print(f"[PDF解析错误] {filepath}: {e}")

    # 如果 pdfplumber 没提取到中文，尝试 OCR
    if not chunks and OCR_AVAILABLE:
        print(f"[PDF] 文字提取失败，尝试 OCR 识别...")
        try:
            images = convert_from_path(filepath, dpi=200)
            for page_num, img in enumerate(images, 1):
                text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                if text.strip():
                    chunks.append({"page": page_num, "text": text.strip()})
            print(f"[PDF] OCR 识别完成: {len(chunks)} 页")
        except Exception as e:
            print(f"[OCR错误] {e}")

    return chunks

def parse_excel(filepath):
    """解析 Excel 文件，返回文本段落列表"""
    chunks = []
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                chunks.append({"page": sheet_name, "text": "\n".join(rows)})
    except Exception as e:
        print(f"[Excel解析错误] {filepath}: {e}")
    return chunks

def parse_txt(filepath):
    """解析纯文本文件"""
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
            if text:
                chunks.append({"page": 1, "text": text})
    except Exception as e:
        print(f"[TXT解析错误] {filepath}: {e}")
    return chunks

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """将长文本按段落/句子切分为小块"""
    # 先按段落分
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < chunk_size:
            current += ("\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            # 如果单段太长，按句子切分
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[。！？；\n])', para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) < chunk_size:
                        sub_chunk += sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current = sub_chunk
                else:
                    current = ""
            else:
                current = para
    
    if current:
        chunks.append(current)
    
    return chunks

# ===================== 索引重建 =====================
def chinese_tokenizer(text):
    """中文分词器：2-4字滑动窗口 + 关键词提取"""
    tokens = []
    # 提取中文连续片段
    cn_segments = re.findall(r'[一-鿿]+', text)
    for seg in cn_segments:
        # 2-4字滑动窗口
        for n in range(2, min(5, len(seg) + 1)):
            for i in range(len(seg) - n + 1):
                tokens.append(seg[i:i+n])
        # 完整词
        if len(seg) >= 2:
            tokens.append(seg)
    # 提取英文/数字
    en_segments = re.findall(r'[a-zA-Z0-9]+', text)
    for seg in en_segments:
        tokens.append(seg.lower())
    return tokens


def rebuild_index():
    """重建 TF-IDF 索引"""
    global tfidf_matrix, vectorizer, all_chunks
    
    with lock:
        all_chunks = []
        for doc in documents.values():
            for i, chunk in enumerate(doc["chunks"]):
                all_chunks.append({
                    "doc_id": doc["id"],
                    "doc_name": doc["name"],
                    "chunk_id": i,
                    "text": chunk["text"],
                    "page": chunk.get("page", "")
                })
        
        if not all_chunks:
            tfidf_matrix = None
            vectorizer = None
            return
        
        texts = [c["text"] for c in all_chunks]
        vectorizer = TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            max_features=30000,
            ngram_range=(1, 1),
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        print(f"[索引] 重建完成: {len(all_chunks)} 个文本块")

def search(query, top_k=TOP_K):
    """TF-IDF + 中文分词检索最相关的文本块"""
    if not vectorizer or tfidf_matrix is None or not all_chunks:
        return []
    
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0.001:
            results.append({
                **all_chunks[idx],
                "score": float(scores[idx])
            })
    
    # 如果整句没匹配到，拆词逐个匹配
    if not results:
        keywords = chinese_tokenizer(query)
        if keywords:
            chunk_scores = [0.0] * len(all_chunks)
            for kw in keywords:
                kw_vec = vectorizer.transform([kw])
                kw_scores = cosine_similarity(kw_vec, tfidf_matrix).flatten()
                for i in range(len(all_chunks)):
                    chunk_scores[i] = max(chunk_scores[i], kw_scores[i])
            top_indices = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:top_k]
            for idx in top_indices:
                if chunk_scores[idx] > 0.001:
                    results.append({
                        **all_chunks[idx],
                        "score": float(chunk_scores[idx])
                    })
    
    return results

# ===================== LLM 调用 =====================
def call_llm(question, context_chunks):
    """调用 LLM 回答问题"""
    if not LLM_API_KEY:
        # 没有配置 API Key，直接返回检索结果
        if not context_chunks:
            return "未找到相关内容，请确认已上传制度文件。"
        answer = "⚠️ 未配置 LLM API，以下是检索到的相关内容：\n\n"
        for i, c in enumerate(context_chunks, 1):
            answer += f"**📄 来源 {i}：{c['doc_name']}（第{c['page']}页）**\n"
            answer += c["text"][:300] + ("..." if len(c["text"]) > 300 else "") + "\n\n"
        return answer
    
    # 构建上下文
    context = ""
    for i, c in enumerate(context_chunks, 1):
        context += f"\n[来源{i}: {c['doc_name']} 第{c['page']}页]\n{c['text']}\n"
    
    prompt = f"""你是一个企业制度文件问答助手。请根据以下制度文件内容，准确回答用户的问题。

要求：
1. 严格按照制度文件内容回答，不要编造
2. 如果文件中没有相关信息，明确告知用户
3. 回答时注明来源文件名和页码
4. 使用清晰的格式（分点、编号等）

=== 制度文件内容 ===
{context}
=== 文件内容结束 ===

用户问题：{question}"""

    try:
        import urllib.request
        url = f"{LLM_API_BASE}/chat/completions"
        body = json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }).encode()
        
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {LLM_API_KEY}")
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        # LLM 调用失败，降级返回检索结果
        if not context_chunks:
            return f"LLM 调用失败: {e}"
        answer = f"⚠️ LLM 调用失败（{e}），以下是检索到的相关内容：\n\n"
        for i, c in enumerate(context_chunks, 1):
            answer += f"**📄 {c['doc_name']}（第{c['page']}页）**\n{c['text'][:300]}\n\n"
        return answer

# ===================== API 路由 =====================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/upload", methods=["POST"])
def upload():
    """上传并解析文件"""
    if "file" not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400
    
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pdf", "xlsx", "xls", "txt", "csv"):
        return jsonify({"error": f"不支持的格式: {ext}"}), 400
    
    # 保存文件
    doc_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")
    file.save(save_path)
    
    # 解析
    if ext == "pdf":
        raw_chunks = parse_pdf(save_path)
    elif ext in ("xlsx", "xls", "csv"):
        raw_chunks = parse_excel(save_path)
    else:
        raw_chunks = parse_txt(save_path)
    
    if not raw_chunks:
        os.remove(save_path)
        return jsonify({"error": "文件内容为空或解析失败"}), 400
    
    # 切分文本
    all_text_chunks = []
    for rc in raw_chunks:
        text_parts = split_text(rc["text"])
        for part in text_parts:
            all_text_chunks.append({"text": part, "page": rc["page"]})
    
    if not all_text_chunks:
        os.remove(save_path)
        return jsonify({"error": "文本提取失败"}), 400
    
    # 存储
    doc = {
        "id": doc_id,
        "name": file.filename,
        "type": ext,
        "chunks": all_text_chunks,
        "chunk_count": len(all_text_chunks),
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    documents[doc_id] = doc
    
    # 重建索引
    rebuild_index()
    
    return jsonify({
        "id": doc_id,
        "name": file.filename,
        "chunks": len(all_text_chunks),
        "total_docs": len(documents),
        "total_chunks": len(all_chunks)
    })

@app.route("/api/docs", methods=["GET"])
def list_docs():
    """列出已上传的文档"""
    docs = []
    for d in documents.values():
        docs.append({
            "id": d["id"],
            "name": d["name"],
            "type": d["type"],
            "chunks": d["chunk_count"],
            "time": d["upload_time"]
        })
    return jsonify({"docs": docs, "total_chunks": len(all_chunks)})

@app.route("/api/docs/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    """删除文档"""
    if doc_id not in documents:
        return jsonify({"error": "文档不存在"}), 404
    
    doc = documents.pop(doc_id)
    # 删除文件
    for ext in ("pdf", "xlsx", "xls", "txt", "csv"):
        path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")
        if os.path.exists(path):
            os.remove(path)
    
    rebuild_index()
    return jsonify({"ok": True, "remaining": len(documents)})

@app.route("/api/ask", methods=["POST"])
def ask():
    """问答接口"""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    
    if not all_chunks:
        return jsonify({"error": "请先上传制度文件"}), 400
    
    # 检索
    results = search(question)
    if not results:
        return jsonify({
            "answer": "未在制度文件中找到与该问题相关的内容。请尝试换个说法提问。",
            "sources": []
        })
    
    # LLM 回答
    answer = call_llm(question, results)
    
    # 整理来源
    sources = []
    for r in results:
        sources.append({
            "doc": r["doc_name"],
            "page": r["page"],
            "score": round(r["score"], 3),
            "preview": r["text"][:150] + ("..." if len(r["text"]) > 150 else "")
        })
    
    return jsonify({"answer": answer, "sources": sources})

# ===================== 前端页面 =====================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>制度文件智能问答</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f5f7;color:#1d1d1f;min-height:100vh;display:flex;flex-direction:column}
.header{background:#fff;border-bottom:1px solid #e5e5e5;padding:16px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:10}
.header h1{font-size:1.1rem;font-weight:700;flex:1}
.container{max-width:900px;margin:0 auto;padding:16px;width:100%;flex:1;display:flex;flex-direction:column}
.upload-area{background:#fff;border:2px dashed #d1d1d6;border-radius:12px;padding:24px;text-align:center;margin-bottom:16px;cursor:pointer;transition:all .2s}
.upload-area:hover{border-color:#0071e3;background:#f0f5ff}
.upload-area.dragover{border-color:#0071e3;background:#e8f0fe}
.upload-area input{display:none}
.upload-area .icon{font-size:2rem;margin-bottom:8px}
.upload-area p{color:#86868b;font-size:.85rem}
.upload-area .formats{color:#aeaeb2;font-size:.75rem;margin-top:4px}

.doc-list{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.doc-chip{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:8px 12px;font-size:.8rem;display:flex;align-items:center;gap:6px}
.doc-chip .name{max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}
.doc-chip .meta{color:#86868b;font-size:.7rem}
.doc-chip .del{background:none;border:none;color:#ff3b30;cursor:pointer;font-size:1rem;padding:0 2px;line-height:1}

.chat-area{flex:1;display:flex;flex-direction:column;gap:12px;margin-bottom:16px;overflow-y:auto;padding-bottom:80px}
.msg{max-width:85%;padding:12px 16px;border-radius:16px;font-size:.9rem;line-height:1.6;word-break:break-word}
.msg.user{align-self:flex-end;background:#0071e3;color:#fff;border-bottom-right-radius:4px}
.msg.bot{align-self:flex-start;background:#fff;border:1px solid #e5e5e5;border-bottom-left-radius:4px}
.msg.bot .sources{margin-top:8px;padding-top:8px;border-top:1px solid #f0f0f0;font-size:.75rem;color:#86868b}
.msg.bot .sources b{color:#0071e3}
.msg.error{background:#fff0f0;color:#ff3b30;border-color:#ffcdd2}

.input-bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #e5e5e5;padding:12px 16px;display:flex;gap:8px;z-index:10}
.input-bar input{flex:1;border:1px solid #d1d1d6;border-radius:10px;padding:10px 14px;font-size:.9rem;font-family:inherit;outline:none}
.input-bar input:focus{border-color:#0071e3}
.input-bar button{background:#0071e3;color:#fff;border:none;border-radius:10px;padding:10px 20px;font-size:.9rem;font-weight:600;cursor:pointer;font-family:inherit}
.input-bar button:disabled{opacity:.4;cursor:not-allowed}
.input-bar button:active{transform:scale(.97)}

.status{text-align:center;padding:40px 20px;color:#86868b;font-size:.9rem}
.loading{display:inline-block;width:16px;height:16px;border:2px solid #e5e5e5;border-top-color:#0071e3;border-radius:50%;animation:spin .6s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
pre{white-space:pre-wrap;font-family:inherit}
</style>
</head>
<body>

<div class="header">
  <span style="font-size:1.4rem">📚</span>
  <h1>制度文件智能问答</h1>
  <span id="stats" style="font-size:.75rem;color:#86868b"></span>
</div>

<div class="container">
  <div class="upload-area" id="uploadArea">
    <div class="icon">📁</div>
    <p>点击或拖拽上传制度文件</p>
    <div class="formats">支持 PDF / Excel / TXT 格式</div>
    <input type="file" id="fileInput" accept=".pdf,.xlsx,.xls,.txt,.csv" multiple>
  </div>
  
  <div class="doc-list" id="docList"></div>
  <div class="chat-area" id="chatArea">
    <div class="status" id="emptyState">📁 请先上传制度文件，然后在此提问</div>
  </div>
</div>

<div class="input-bar">
  <input type="text" id="questionInput" placeholder="输入关于制度文件的问题..." disabled>
  <button id="askBtn" onclick="askQuestion()" disabled>提问</button>
</div>

<script>
var hasDocs = false;

// 上传
var uploadArea = document.getElementById("uploadArea");
var fileInput = document.getElementById("fileInput");

uploadArea.onclick = function(){ fileInput.click(); };
uploadArea.ondragover = function(e){ e.preventDefault(); uploadArea.classList.add("dragover"); };
uploadArea.ondragleave = function(){ uploadArea.classList.remove("dragover"); };
uploadArea.ondrop = function(e){
  e.preventDefault(); uploadArea.classList.remove("dragover");
  if(e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
};
fileInput.onchange = function(){ if(fileInput.files.length) uploadFiles(fileInput.files); };

function uploadFiles(files){
  for(var i=0;i<files.length;i++){
    var fd = new FormData();
    fd.append("file", files[i]);
    addMsg("system", "⏳ 正在上传: " + files[i].name + "...");
    
    fetch("/api/upload", {method:"POST", body:fd})
      .then(function(r){ return r.json(); })
      .then(function(d){
        if(d.error){ addMsg("error", "❌ " + d.error); return; }
        addMsg("system", "✅ " + d.name + " 上传成功（" + d.chunks + " 个文本块）");
        refreshDocs();
      })
      .catch(function(e){ addMsg("error", "上传失败: " + e); });
  }
  fileInput.value = "";
}

function refreshDocs(){
  fetch("/api/docs").then(function(r){ return r.json(); }).then(function(d){
    hasDocs = d.docs.length > 0;
    document.getElementById("questionInput").disabled = !hasDocs;
    document.getElementById("askBtn").disabled = !hasDocs;
    document.getElementById("stats").textContent = d.docs.length + " 个文件 · " + d.total_chunks + " 个文本块";
    
    var list = document.getElementById("docList");
    list.innerHTML = "";
    d.docs.forEach(function(doc){
      var chip = document.createElement("div");
      chip.className = "doc-chip";
      chip.innerHTML = '<span class="name">📄 ' + esc(doc.name) + '</span>' +
        '<span class="meta">' + doc.chunks + '块</span>' +
        '<button class="del" onclick="delDoc(\'' + doc.id + '\')">&times;</button>';
      list.appendChild(chip);
    });
    
    if(hasDocs){
      var empty = document.getElementById("emptyState");
      if(empty) empty.textContent = "💬 输入问题开始查询制度文件";
    }
  });
}

function delDoc(id){
  fetch("/api/docs/"+id, {method:"DELETE"}).then(function(){ refreshDocs(); });
}

// 提问
function askQuestion(){
  var input = document.getElementById("questionInput");
  var q = input.value.trim();
  if(!q) return;
  
  addMsg("user", q);
  input.value = "";
  
  var botMsg = addMsg("bot", '<span class="loading"></span> 正在检索和分析...');
  
  fetch("/api/ask", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({question:q})
  }).then(function(r){ return r.json(); }).then(function(d){
    if(d.error){ botMsg.innerHTML = '<span style="color:#ff3b30">' + esc(d.error) + '</span>'; return; }
    
    var html = "<pre>" + esc(d.answer) + "</pre>";
    if(d.sources && d.sources.length){
      html += '<div class="sources"><b>📎 参考来源：</b><br>';
      d.sources.forEach(function(s,i){
        html += (i+1) + ". " + esc(s.doc) + "（第" + s.page + "页，相关度" + (s.score*100).toFixed(0) + "%）<br>";
      });
      html += "</div>";
    }
    botMsg.innerHTML = html;
  }).catch(function(e){
    botMsg.innerHTML = '<span style="color:#ff3b30">请求失败: ' + esc(e.message) + '</span>';
  });
}

document.getElementById("questionInput").onkeydown = function(e){
  if(e.key === "Enter") askQuestion();
};

function addMsg(type, html){
  var empty = document.getElementById("emptyState");
  if(empty) empty.remove();
  
  var div = document.createElement("div");
  div.className = "msg " + type;
  div.innerHTML = html;
  document.getElementById("chatArea").appendChild(div);
  div.scrollIntoView({behavior:"smooth"});
  return div;
}

function esc(s){ var d=document.createElement("div"); d.textContent=s; return d.innerHTML; }

refreshDocs();
</script>
</body>
</html>"""

# ===================== 启动 =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n📚 制度文件智能问答系统 v1.0")
    print(f"📂 文件目录: {UPLOAD_DIR}")
    print(f"🤖 LLM: {LLM_MODEL} @ {LLM_API_BASE}")
    print(f"🔑 API Key: {'已配置' if LLM_API_KEY else '⚠️ 未配置（仅返回检索结果）'}")
    print(f"🌐 访问地址: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)