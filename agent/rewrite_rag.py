"""
RAG 版本的 Few-Shot 改写程序
与原 process_problems_concurrently 流程一致，但示例通过 RAG 从 ChromaDB 检索
"""

from __future__ import annotations
import json, re, os, time, random, statistics, itertools
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm
from openai import OpenAI
import chromadb

# ========================
# 配置
# ========================
from config.api_config import EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL

CHROMA_PERSIST_DIR = r"D:\bishe\CHIP\chroma_db"
ALIYUN_EMBEDDING_URL = f"{EMBEDDING_BASE_URL.rstrip('/compatible-mode/v1')}/api/v1/services/embeddings/text-embedding/text-embedding"

# ========================
# 基础工具函数
# ========================
def read_json_any(path: str) -> List[dict]:
    """读取 JSON/JSONL 文件"""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n{" in text: 
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    return obj if isinstance(obj, list) else [obj]


def write_json_like(path: str, rows: List[dict]):
    """写回 JSON/JSONL 文件"""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False)); f.write("\n")
    else:
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# ========================
# 文本处理
# ========================
_THINK_RE = re.compile(r'\s*<think>.*?</think>\s*', re.DOTALL|re.IGNORECASE)

def strip_think_blocks(t: str) -> str:
    return _THINK_RE.sub("", t or "")


def strip_outer_quotes(t: str) -> str:
    if not t: return ""
    s = t.strip()
    pairs = [('"', '"'), ('"', '"'), ("'", "'")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
            return s[1:-1].strip()
    return s


def normalize_llm_output(t: str) -> str:
    t = strip_think_blocks(t)
    t = strip_outer_quotes(t)
    t = re.sub(r'^\s*```[a-zA-Z0-9_-]*\s*', '', t)
    t = re.sub(r'\s*```\s*$', '', t)
    return t.strip()


def count_cjk(s: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', s or ""))


# ========================
# 阿里云 Embedding API
# ========================
import requests

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """调用阿里云 text-embedding-v4 获取向量"""
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": {"texts": [text]},
        "parameters": {"text_type": "query"}
    }
    
    max_retries = 3
    for i in range(max_retries):
        try:
            resp = requests.post(ALIYUN_EMBEDDING_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            embeddings = result.get("output", {}).get("embeddings", [])
            if embeddings:
                return embeddings[0].get("embedding", [])
            raise ValueError("No embedding in response")
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(1.5 ** i)
            else:
                raise e
    return []


def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """批量获取 embedding"""
    if not texts:
        return []
    
    # 过滤空字符串和过长的文本
    cleaned_texts = []
    for t in texts:
        if t and isinstance(t, str) and len(t) <= 2000:
            cleaned_texts.append(t)
        else:
            cleaned_texts.append(t[:2000] if t else "空文本")
    
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": {"texts": cleaned_texts},
        "parameters": {
            "text_type": "query"  # 检索用 query
        }
    }
    
    max_retries = 3
    for i in range(max_retries):
        try:
            resp = requests.post(ALIYUN_EMBEDDING_URL, json=payload, headers=headers, timeout=60)
            
            if resp.status_code != 200:
                error_msg = resp.text[:500] if resp.text else "No response"
                print(f"    [ERROR] API 返回错误 {resp.status_code}: {error_msg}")
            
            resp.raise_for_status()
            
            result = resp.json()
            embeddings_data = result.get("output", {}).get("embeddings", [])
            return [item.get("embedding", []) for item in embeddings_data]
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(1.5 ** i)
            else:
                raise e
    return []


# ========================
# ChromaDB RAG 检索
# ========================
def init_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """初始化 ChromaDB 客户端"""
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client: chromadb.PersistentClient, rule_id: str) -> Optional[chromadb.Collection]:
    """获取对应 rule_id 的 collection"""
    collection_name = f"rule_{rule_id.replace('-', '_')}"
    
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return None


def retrieve_similar_examples(
    client: chromadb.PersistentClient,
    rule_id: str,
    query_description: str,
    top_k: int = 5
) -> List[str]:
    """
    从 ChromaDB 检索与 query_description 最相似的 top_k 个 description
    """
    collection = get_collection(client, rule_id)
    
    if collection is None:
        return []
    
    query_embedding = get_embedding(query_description)
    
    if not query_embedding:
        return []
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )
        
        documents = results.get("documents", [[]])[0]
        return [doc for doc in documents if doc]
    
    except Exception as e:
        print(f"[WARN] 检索失败 {rule_id}: {e}")
        return []


# ========================
# LLM 调用
# ========================
def call_local_llm(client: OpenAI, model: str, prompt: str,
                   max_tokens: int = 512, retries: int = 3, backoff: float = 1.6) -> str:
    """调用本地 LLM"""
    last = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                extra_body={"enable_thinking": False}
            )
            out = (resp.choices[0].message.content or "").strip()
            return normalize_llm_output(out)
        except Exception as e:
            last = e
            time.sleep((backoff ** i) + random.random() * 0.05)
    raise last


def build_rewrite_prompt(rule_id: str, description: str, examples: List[str]) -> str:
    """构建改写 prompt"""
    lens = [count_cjk(x) for x in examples if x]
    base_len = count_cjk(description)
    target = statistics.median(lens) if lens else max(20, base_len)
    target_min = int(max(8, target * 0.8))
    target_max = int(target * 1.2)
    
    examples_block = "\n".join(f"- {x}" for x in examples[:6]) if examples else "（无示例，保持原风格）"
    
    return (
        f"你是中文医疗合规问题描述的改写助手。请在**不改变事实与肯定/否定极性**的前提下，"
        f"将下列描述改写为与示例一致风格、接近示例长度的一句话：\n"
        f"【规则】\n"
        f"1) 仅输出改写后的中文句子，不要任何解释、JSON、Markdown、<think>。\n"
        f"2) 语气客观、专业、简洁；优先医学术语；保留时间/部位/检查名/数值/诊断名。\n"
        f"3) 中文字符数目标：[{target_min}, {target_max}]；若原文很短，允许略短但不丢关键信息。\n"
        f"【rule_id】{rule_id}\n"
        f"【示例】\n{examples_block}\n"
        f"【待改写】\n{description}\n"
    )


# ========================
# 主处理函数（RAG 版本）
# ========================
def process_problems_concurrently_rag(
    in_path: str,
    out_path: Optional[str] = None,
    think_out: str = "./think_rag.json",
    # ChromaDB 配置
    chroma_persist_dir: str = CHROMA_PERSIST_DIR,
    rag_top_k: int = 5,
    # 并发 & 多端口
    base_urls: Optional[List[str]] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: str = "EMPTY",
    model: str = "qwen3-32b",
    max_workers: int = 16,
    per_host_limit: Optional[int] = None,
    # 规则参数
    short_len_cjk: int = 20,
    max_tokens_per_resp: int = 512,
):
    """
    RAG 版本的 process_problems_concurrently
    从 ChromaDB 检索相似示例，而非从 examples.json 读取
    """
    rows = read_json_any(in_path)
    if not rows:
        print("[Info] 空输入。"); return
    
    # 初始化 ChromaDB 客户端
    print(f"[Info] 初始化 ChromaDB (目录：{chroma_persist_dir})")
    chroma_client = init_chroma_client(chroma_persist_dir)
    
    # 初始化 LLM 客户端池
    urls = list(base_urls) if base_urls else [base_url]
    clients = [OpenAI(api_key=api_key, base_url=u) for u in urls]
    if per_host_limit is None:
        per_host_limit = max(1, max_workers // max(1, len(clients)))
    sems = [Semaphore(per_host_limit) for _ in clients]
    rr = itertools.cycle(range(len(clients)))
    
    think_recs: List[dict] = []
    n_think = n_short_skip = n_rewrite = n_no_examples = 0
    
    # ========================
    # Step 1: 预处理，构造待改写任务
    # ========================
    tasks: List[Tuple[int, int, str, str, List[str]]] = []
    
    for i, x in enumerate(rows):
        rid = str(x.get("record_id") or x.get("id") or f"idx_{i}")
        probs = x.get("problems") or []
        new_probs = []
        
        for a in probs:
            desc = str(a.get("description") or "")
            rule_id = str(a.get("rule_id") or "")
            
            # 1) 命中 think → 记录并删除
            if "think" in desc.lower():
                think_recs.append({"record_id": rid, "rule_id": rule_id, "description": desc})
                n_think += 1
                continue
            
            # 2) 短且含"为空" → 保留原样
            if ("为空" in desc) and (count_cjk(desc) < short_len_cjk):
                new_probs.append(a)
                n_short_skip += 1
                continue
            
            # 3) 从 ChromaDB 检索相似示例
            examples = retrieve_similar_examples(
                chroma_client, rule_id, desc, top_k=rag_top_k
            )
            
            if not examples:
                new_probs.append(a)
                n_no_examples += 1
            else:
                new_probs.append(dict(a))
                pos = len(new_probs) - 1
                tasks.append((i, pos, rule_id, desc, examples))
        
        rows[i]["problems"] = new_probs
    
    # ========================
    # Step 2: 并发执行改写
    # ========================
    def _worker(task: Tuple[int, int, str, str, List[str]], ci: int):
        rec_idx, pos, rule_id, desc, examples = task
        cli, sem = clients[ci], sems[ci]
        with sem:
            prompt = build_rewrite_prompt(rule_id, desc, examples)
            rewritten = call_local_llm(cli, model, prompt, max_tokens=max_tokens_per_resp)
            return rec_idx, pos, rewritten
    
    results: List[Tuple[int, int, Optional[str]]] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for t in tasks:
                ci = next(rr)
                futs.append(ex.submit(_worker, t, ci))
            for fut in tqdm(as_completed(futs), total=len(futs), desc="rewriting (RAG)"):
                try:
                    rec_idx, pos, rewritten = fut.result()
                except Exception:
                    rec_idx, pos, rewritten = None, None, None
                if rec_idx is not None:
                    results.append((rec_idx, pos, rewritten))
    
    # ========================
    # Step 3: 回填改写结果
    # ========================
    for rec_idx, pos, rewritten in results:
        try:
            if rewritten:
                rows[rec_idx]["problems"][pos]["description"] = rewritten
                n_rewrite += 1
        except Exception:
            pass
    
    # ========================
    # Step 4: 输出
    # ========================
    write_json_like(think_out, think_recs)
    
    out_path = out_path or in_path
    write_json_like(out_path, rows)
    
    print(f"[Done] think-removed: {n_think} | short-skip: {n_short_skip} | rewritten: {n_rewrite} | no-examples: {n_no_examples}")
    print(f"[Path] out={out_path} | think={think_out}")


# ========================
# 用法示例
# ========================
if __name__ == "__main__":
    from config.api_config import DASHSCOPE_BASE_URLS, DASHSCOPE_API_KEY, DEFAULT_MODEL, MAX_WORKERS, PER_HOST_LIMIT
    
    process_problems_concurrently_rag(
        in_path=r"D:\bishe\MedShard\out\results\un_rewrite.json",
        out_path=r"D:\bishe\MedShard\out\results\rag_rewrite.json",  # None=覆盖原文件
        think_out=r"D:\bishe\MedShard\think_rag.json",
        chroma_persist_dir=r"D:\bishe\MedShard\chroma_db",
        rag_top_k=5,
        base_urls=DASHSCOPE_BASE_URLS,
        api_key=DASHSCOPE_API_KEY,
        model=DEFAULT_MODEL,
        max_workers=MAX_WORKERS,
        per_host_limit=PER_HOST_LIMIT,
        short_len_cjk=20,
        max_tokens_per_resp=512,
    )
