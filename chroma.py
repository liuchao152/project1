r"""
ChromaDB 向量数据库入库程序
功能：按照 rule_id 将 description 存入不同的 ChromaDB 集合中
输入：D:\bishe\CHIP\submission\chip2025_results.json
Embedding 模型：阿里 text-embedding-v4
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import requests
import time

# ========================
# 配置
# ========================
from config.api_config import EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL

CHROMA_PERSIST_DIR = r"D:\bishe\CHIP\chroma_db"
INPUT_JSON_PATH = r"D:\bishe\CHIP\train-new\old1.json"
ALIYUN_EMBEDDING_URL = f"{EMBEDDING_BASE_URL.rstrip('/compatible-mode/v1')}/api/v1/services/embeddings/text-embedding/text-embedding"

# ========================
# 阿里云 Embedding API 调用
# ========================
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    调用阿里云 text-embedding-v4 获取向量
    """
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": {
            "texts": [text]
        },
        "parameters": {
            "text_type": "query"
        }
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
    """
    批量获取 embedding（阿里云支持批量）
    """
    if not texts:
        return []
    
    # 过滤空字符串和过长的文本（阿里云限制 2048 tokens）
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
    
    # 阿里云 text-embedding-v4 的正确格式
    payload = {
        "model": model,
        "input": {
            "texts": cleaned_texts
        },
        "parameters": {
            "text_type": "document"  # 入库用 document
        }
    }
    
    max_retries = 3
    for i in range(max_retries):
        try:
            resp = requests.post(ALIYUN_EMBEDDING_URL, json=payload, headers=headers, timeout=60)
            
            if resp.status_code != 200:
                error_msg = resp.text[:500] if resp.text else "No response"
                print(f"    [ERROR] API 返回错误 {resp.status_code}: {error_msg}")
                print(f"    [DEBUG] texts count: {len(cleaned_texts)}, first text: {cleaned_texts[0][:50] if cleaned_texts else 'N/A'}")
            
            resp.raise_for_status()
            
            result = resp.json()
            embeddings_data = result.get("output", {}).get("embeddings", [])
            embeddings = [item.get("embedding", []) for item in embeddings_data]
            return embeddings
        except Exception as e:
            if i < max_retries - 1:
                wait_time = 1.5 ** i
                print(f"    [RETRY] 第 {i+1} 次失败，{wait_time:.1f}s 后重试... {e}")
                time.sleep(wait_time)
            else:
                print(f"    [FATAL] 最终失败：{e}")
                raise e
    
    return []


def generate_id(text: str) -> str:
    """
    生成唯一 ID（基于文本的 hash）
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ========================
# 数据加载
# ========================
def load_input_data(path: str) -> List[dict]:
    """
    读取输入 JSON 文件（支持 JSON 数组或 JSONL）
    """
    p = Path(path)
    txt = p.read_text(encoding='utf-8').strip()
    
    if not txt:
        return []
    
    # JSONL 格式
    if "\n{" in txt:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    
    # JSON 数组或单对象
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else [obj]


def extract_descriptions_by_rule(data: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按 rule_id 提取 description
    返回：{rule_id: [{"description": str, "record_id": str, "id": str}, ...]}
    """
    rule_data = {}
    
    for record in data:
        record_id = record.get("record_id", record.get("id", ""))
        problems = record.get("problems", [])
        
        if not problems:
            continue
        
        for prob in problems:
            rule_id = prob.get("rule_id", "")
            description = prob.get("description", "")
            
            if not rule_id or not description:
                continue
            
            if rule_id not in rule_data:
                rule_data[rule_id] = []
            
            rule_data[rule_id].append({
                "description": description,
                "record_id": record_id,
                "source_rule_id": rule_id
            })
    
    return rule_data


# ========================
# ChromaDB 操作
# ========================
def init_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    初始化 ChromaDB 持久化客户端
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_or_create_collection(client: chromadb.PersistentClient, rule_id: str) -> chromadb.Collection:
    """
    获取或创建对应 rule_id 的 collection
    集合名规则：rule_{rule_id}
    """
    collection_name = f"rule_{rule_id.replace('-', '_')}"
    
    # 检查是否已存在
    existing = client.list_collections()
    for coll in existing:
        if coll.name == collection_name:
            return client.get_collection(name=collection_name)
    
    # 创建新集合
    return client.create_collection(name=collection_name)


def add_to_collection(collection: chromadb.Collection, items: List[Dict[str, Any]], batch_size: int = 10):
    """
    批量添加数据到 collection
    注意：添加前检查是否存在，避免重复添加导致索引损坏
    """
    """
    批量添加数据到 collection
    """
    if not items:
        print(f"  无数据可添加")
        return
    
    # 分组批量处理
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # 准备数据
        ids = [generate_id(item["description"] + item["record_id"]) for item in batch]
        documents = [item["description"] for item in batch]
        metadatas = [
            {
                "record_id": item["record_id"],
                "source_rule_id": item["source_rule_id"]
            }
            for item in batch
        ]
        
        # 获取 embeddings（批量）
        print(f"    获取 embedding 批次 {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}...")
        embeddings = get_embeddings_batch(documents)
        
        # 添加到 collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"    已添加 {len(batch)} 条")


# ========================
# 主流程
# ========================
def main():
    print("=" * 60)
    print("ChromaDB 入库程序启动")
    print("=" * 60)
    
    # 1. 加载输入数据
    print(f"\n[1] 加载输入文件：{INPUT_JSON_PATH}")
    data = load_input_data(INPUT_JSON_PATH)
    print(f"    共加载 {len(data)} 条记录")
    
    # 2. 按 rule_id 提取 description
    print(f"\n[2] 按 rule_id 提取 description...")
    rule_data = extract_descriptions_by_rule(data)
    print(f"    共 {len(rule_data)} 个不同的 rule_id")
    
    for rule_id, items in sorted(rule_data.items()):
        print(f"    - {rule_id}: {len(items)} 条 description")
    
    # 3. 初始化 ChromaDB
    print(f"\n[3] 初始化 ChromaDB (持久化目录：{CHROMA_PERSIST_DIR})")
    client = init_chroma_client(CHROMA_PERSIST_DIR)
    
    # 4. 逐个 rule_id 入库
    print(f"\n[4] 开始入库...")
    total_added = 0
    
    for rule_id, items in sorted(rule_data.items()):
        print(f"\n  处理 rule_id: {rule_id}")
        
        # 获取或创建 collection
        collection = get_or_create_collection(client, rule_id)
        
        # 去重（相同 description + record_id 视为重复）
        seen = set()
        unique_items = []
        for item in items:
            key = (item["description"], item["record_id"])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        print(f"    去重后：{len(unique_items)} 条")
        
        # 添加到 collection
        add_to_collection(collection, unique_items)
        total_added += len(unique_items)
    
    # 5. 完成
    print(f"\n" + "=" * 60)
    print(f"入库完成！共添加 {total_added} 条 description 到 {len(rule_data)} 个集合")
    print(f"ChromaDB 持久化目录：{CHROMA_PERSIST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
