"""
整体 few-shot 改写版本 - 使用大模型改写问题描述以提升 ROUGE-L 分数

功能：
- 读取预测结果 JSON 文件
- 对 problems[*].description 进行改写（基于 few-shot 示例）
- 支持多端口并发调用
- 自动清理 <think> 标签
- 跳过短且含"为空"的描述
"""

from __future__ import annotations
import json
import re

import os
import sys
# 获取当前脚本所在目录的上级目录（也就是项目根目录 D:\bishe\MedShard）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
import random
import statistics
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm
from openai import OpenAI


# ========================
# I/O 工具函数
# ========================
def read_json_any(path: str) -> List[dict]:
    """读取 .json（list 或单 obj）或 .jsonl 为 list[dict]"""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n{" in text:  # jsonl
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    return obj if isinstance(obj, list) else [obj]


def write_json_like(path: str, rows: List[dict]):
    """按原格式写回（json 或 jsonl）"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
    else:
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# ========================
# 文本处理函数
# ========================
_THINK_RE = re.compile(r'\s*<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE)


def strip_think_blocks(t: str) -> str:
    """删除 <think>...</think> 块"""
    return _THINK_RE.sub("", t or "")


def strip_outer_quotes(t: str) -> str:
    """删除外层成对引号"""
    if not t:
        return ""
    s = t.strip()
    pairs = [('"', '"'), ('"', '"'), ("'", "'")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq) and len(s) >= 2:
            return s[1:-1].strip()
    return s


def normalize_llm_output(t: str) -> str:
    """标准化 LLM 输出：删除 think 块、引号、Markdown 围栏"""
    t = strip_think_blocks(t)
    t = strip_outer_quotes(t)
    # 去掉围栏 ``` ```
    t = re.sub(r'^\s*```[a-zA-Z0-9_-]*\s*', '', t)
    t = re.sub(r'\s*```\s*$', '', t)
    return t.strip()


def count_cjk(s: str) -> int:
    """统计中文字符数"""
    return len(re.findall(r'[\u4e00-\u9fff]', s or ""))


# ========================
# 本地 LLM 调用（OpenAI 兼容，vLLM 等）
# ========================
def call_local_llm(client: OpenAI, model: str, prompt: str,
                   max_tokens: int = 512, retries: int = 3, backoff: float = 1.6) -> str:
    """调用本地 LLM，带重试机制"""
    last = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                extra_body={
                    "enable_thinking": False
                }
            )
            out = (resp.choices[0].message.content or "").strip()
            return normalize_llm_output(out)
        except Exception as e:
            last = e
            time.sleep((backoff ** i) + random.random() * 0.05)
    raise last


def build_rewrite_prompt(rule_id: str, description: str, examples: List[str]) -> str:
    """
    构建改写 prompt
    
    Args:
        rule_id: 规则 ID
        description: 待改写的描述
        examples: few-shot 示例列表
    
    Returns:
        完整的 prompt 字符串
    """
    lens = [count_cjk(x) for x in examples if x]
    base_len = count_cjk(description)
    target = statistics.median(lens) if lens else max(20, base_len)
    target_min = int(max(8, target * 0.8))
    target_max = int(target * 1.2)
    examples_block = "\n".join(f"- {x}" for x in examples[:6]) if examples else "（无示例，保持原风格）"
    
    return (
        f"你是中文医疗合规问题描述的改写助手。请在**不改变事实与肯定/否定极性**的前提下，"
        f"将下列描述改写为与示例中的一致风格、接近示例长度的一句话：\n"
        f"【规则】\n"
        f"1) 仅输出改写后的中文句子，不要任何解释、JSON、Markdown、<think>。\n"
        f"2) 语气客观、专业、简洁；优先医学术语；保留时间/部位/检查名/数值/诊断名。\n"
        f"3) 中文字符数目标：[{target_min}, {target_max}]；若原文很短，允许略短但不丢关键信息。\n"
        f"【rule_id】{rule_id}\n"
        f"【示例】\n{examples_block}\n"
        f"【待改写】\n{description}\n"
    )


def process_problems_concurrently(
    in_path: str,
    out_path: Optional[str] = None,
    think_out: str = "./think.json",
    examples_path: Optional[str] = None,              # JSON: {rule_id: [示例...]}
    examples_dict: Optional[Dict[str, List[str]]] = None,
    # 并发 & 多端口
    base_urls: Optional[List[str]] = None,             # 如 ["https://dashscope.aliyuncs.com/compatible-mode/v1", ...]
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",        # 当 base_urls 为空时使用
    api_key: str = "EMPTY",
    model: str = "qwen3-32b",
    max_workers: int = 16,
    per_host_limit: Optional[int] = None,
    # 规则参数
    short_len_cjk: int = 20,
    max_tokens_per_resp: int = 512,
) -> str:
    """
    并发处理 problems 中的 description 改写
    
    Args:
        in_path: 输入 JSON 文件路径
        out_path: 输出文件路径（None=覆盖原文件）
        think_out: think.json 输出路径
        examples_path: examples.json 文件路径
        examples_dict: 示例字典（优先级高于 examples_path）
        base_urls: 多个 API 端点列表
        base_url: 单个 API 端点
        api_key: API 密钥
        model: 模型名称
        max_workers: 最大并发数
        per_host_limit: 每端口限流
        short_len_cjk: 短文本阈值（中文字符数）
        max_tokens_per_resp: 最大输出 token 数
    
    Returns:
        输出文件路径
    """
    rows = read_json_any(in_path)
    if not rows:
        print("[Info] 空输入。")
        return ""

    # 加载示例
    if examples_dict is None and examples_path:
        examples_dict = json.loads(Path(examples_path).read_text(encoding="utf-8"))
    if examples_dict is None:
        examples_dict = {}

    # 客户端池 + 每端口限流
    urls = list(base_urls) if base_urls else [base_url]
    clients = [OpenAI(api_key=api_key, base_url=u) for u in urls]
    if per_host_limit is None:
        per_host_limit = max(1, max_workers // max(1, len(clients)))
    sems = [Semaphore(per_host_limit) for _ in clients]
    rr = itertools.cycle(range(len(clients)))

    think_recs: List[dict] = []
    n_think = n_short_skip = n_rewrite = n_no_examples = 0

    # 先进行单线程预处理，构造待改写任务；同时立即应用"删除 think / 跳过短为空 / 无示例保留"逻辑
    # 任务项：(rec_idx, prob_pos_in_new_probs, rule_id, description, examples)
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
                think_recs.append({
                    "record_id": rid,
                    "rule_id": rule_id,
                    "description": desc
                })
                n_think += 1
                continue

            # 2) 短且含"为空" → 保留原样
            if ("为空" in desc) and (count_cjk(desc) < short_len_cjk):
                new_probs.append(a)
                n_short_skip += 1
                continue

            # 3) 需要改写：若无示例则保留；有示例则创建并发任务
            examples = examples_dict.get(rule_id, [])
            if not examples:
                new_probs.append(a)
                n_no_examples += 1
            else:
                # 先占位，稍后填回改写后的 description
                new_probs.append(dict(a))
                pos = len(new_probs) - 1
                tasks.append((i, pos, rule_id, desc, examples))

        # 覆盖 problems 为预处理后的 new_probs
        rows[i]["problems"] = new_probs

    # 并发执行改写任务
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
            for fut in tqdm(as_completed(futs), total=len(futs), desc="rewriting"):
                try:
                    rec_idx, pos, rewritten = fut.result()
                except Exception:
                    rec_idx, pos, rewritten = None, None, None
                if rec_idx is not None:
                    results.append((rec_idx, pos, rewritten))

    # 回填改写结果（为空则保留原文）
    for rec_idx, pos, rewritten in results:
        try:
            if rewritten:
                rows[rec_idx]["problems"][pos]["description"] = rewritten
                n_rewrite += 1
        except Exception:
            # 索引异常忽略
            pass

    # 写出 think.json
    write_json_like(think_out, think_recs)

    # 写回最终结果
    out_path = out_path or in_path
    write_json_like(out_path, rows)

    print(f"[Done] think-removed: {n_think} | short-skip: {n_short_skip} | "
          f"rewritten: {n_rewrite} | no-examples: {n_no_examples}")
    print(f"[Path] out={out_path} | think={think_out}")
    
    return out_path


if __name__ == "__main__":
    from config.api_config import DASHSCOPE_BASE_URLS, DASHSCOPE_API_KEY, DEFAULT_MODEL, MAX_WORKERS, PER_HOST_LIMIT
    
    # 示例用法
    process_problems_concurrently(
        in_path=r"D:\bishe\MedShard\out\results\un_rewrite.json",
        out_path=r"D:\bishe\MedShard\out\results\fewshot_rewrite.json",      # None=覆盖原文件
        think_out=r"D:\bishe\MedShard\think.json",
        examples_path=r"D:\bishe\MedShard\in\examples.json",
        base_urls=DASHSCOPE_BASE_URLS,
        api_key=DASHSCOPE_API_KEY,
        model=DEFAULT_MODEL,
        max_workers=MAX_WORKERS,
        per_host_limit=PER_HOST_LIMIT,
        short_len_cjk=20,
        max_tokens_per_resp=8192,
    )
