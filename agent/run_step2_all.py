"""
Step2 质控 Agent + 反思 Agent - 批量运行脚本
一次性运行所有 19 个规则的质控检测
"""
import sys
import os
# 获取当前脚本所在目录的上级目录（也就是项目根目录 D:\bishe\MedShard）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import itertools, time, random

# ========================
# 配置
# ========================
from config.api_config import DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, DEFAULT_MODEL

DATA_PATH = r"D:\bishe\MedShard\out\train_latest_new.json"
TEMPLATES_PATH = r"D:\bishe\MedShard\in\templates.json"
GT_PATH = r"D:\bishe\MedShard\out\train_latest_new.json"
PRED_OUTDIR = r"D:\bishe\MedShard\out\pred_out"
EVAL_OUTDIR = r"D:\bishe\MedShard\out\eval_out"
DEDUCT_PATH = r"D:\bishe\MedShard\in\ruleid_to_deduct.json"

BASE_URL = DASHSCOPE_BASE_URL
API_KEY = DASHSCOPE_API_KEY
MODEL = DEFAULT_MODEL

# 19 个规则及其配置
RULES_CONFIG = [
    # rule_id, enable_recheck, enable_default_precheck
    #("EN-FZ-01-V1", True, True),
    ("EN-FZ-01-V1", False, True),
    ("IC-RZCB-01-V1", False, False),
    ("EN-XB-01-V1", False, False),
    ("EN-JW-01-V1", False, False),
    ("EN-RZCB-01-V1", False, False),
    ("EN-XB-02-V1", False, False),
    #("IC-XB-01-V1", True, False),
    ("IC-XB-01-V1", False, False),
    ("IC-XB-03-V1", False, False),
    #("CO-XB-03-V1", True, False),
    ("CO-XB-03-V1", False, False),
    ("CO-XB-02-V1", False, False),
    ("EN-ZS-01-V1", False, False),
    ("CO-XB-04-V1", False, False),
    #("CO-XB-01-V1", True, False),
    ("CO-XB-01-V1", False, False),
    ("DQ-RZ-01-V1", False, False),
    #("DQ-RZ-02-V1", True, False),
    ("DQ-RZ-02-V1", False, False),
    #("IC-JW-01-V1", True, False),
    ("IC-JW-01-V1", False, False),
    ("IC-XB-02-V1", False, False),
    ("IC-ZK-01-V1", False, False),
    ("IC-ZS-01-V1", False, False),
]

# ========================
# 基础工具函数
# ========================
def now_tag() -> str:
    return datetime.now().strftime("%m%d-%H%M")

def ensure_dir(d: str | Path):
    Path(d).mkdir(parents=True, exist_ok=True)

def read_json_any(path: str) -> List[dict]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if "\n{" in txt:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else [obj]

def read_templates(path: str) -> List[dict]:
    tpls = read_json_any(path)
    norm = []
    for t in tpls:
        norm.append({
            "rule_id": t.get("rule_id", ""),
            "issue_type": t.get("issue_type", ""),
            "field": t.get("field", ""),
            "sensitivity": t.get("sensitivity", ""),
            "inputs": list(t.get("inputs", [])),
            "template": t.get("template", ""),
            "template2": t.get("template2", ""),
        })
    return norm

def parse_field_content_if_needed(item: dict) -> dict:
    fc = item.get("field_content", None)
    parsed = None
    if isinstance(fc, dict):
        parsed = fc
    elif isinstance(fc, str):
        s = fc.strip()
        if s:
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
    return {"field_content": parsed if isinstance(parsed, dict) else {}, "root": item}

def pick_inputs(values: dict, keys: List[str]) -> Dict[str, str]:
    out = {}
    fc = values.get("field_content", {})
    root = values.get("root", {})
    for k in keys:
        v = None
        if isinstance(fc, dict) and k in fc and fc[k] is not None:
            v = fc[k]
        elif k in root and root[k] is not None:
            v = root[k]
        out[k] = str(v) if v is not None else ""
    return out

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")

def extract_placeholders(tpl: str) -> List[str]:
    if not tpl:
        return []
    return _PLACEHOLDER_RE.findall(tpl)

def safe_format(tpl: str, kv: Dict[str, Any]) -> str:
    class SafeDict(dict):
        def __missing__(self, key):
            return ""
    return tpl.format_map(SafeDict(**kv))

def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    return re.sub(r'\s*<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)

def strip_outer_quotes(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    pairs = [('"', '"'), ('"', '"'), ("'", "'")]
    for lq, rq in pairs:
        if t.startswith(lq) and t.endswith(rq) and len(t) >= 2:
            return t[1:-1].strip()
    return t

def normalize_llm_output(text: str) -> str:
    t = strip_think_blocks(text)
    t = strip_outer_quotes(t)
    return t.strip()

def call_llm(client: OpenAI, model: str, prompt: str,
             temperature: float = 0.0, top_p: float = 0.7, top_k: int = 20,
             max_tokens: int = 6144) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False}
    )
    return (resp.choices[0].message.content or "").strip()

def call_llm_with_retry(client: OpenAI, model: str, prompt: str,
                        temperature: float = 0.0, top_p: float = 0.7, top_k: int = 20,
                        max_tokens: int = 6144, retries: int = 3, backoff: float = 1.6) -> str:
    last = None
    for i in range(retries):
        try:
            txt = call_llm(client, model, prompt, temperature=temperature, top_p=top_p,
                           top_k=top_k, max_tokens=max_tokens)
            return normalize_llm_output(txt)
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep((backoff ** i) + random.random() * 0.05)
    raise last

NONE_LIKE = {"none", "无", "合格", "符合", "通过", "通过校验", "无异常", "未见异常", "empty", "na", "n/a"}

def is_none_like(text: str) -> bool:
    """
    判断 LLM 输出是否表示"没问题/合格"
    修改版：只要包含 NONE_LIKE 中的词汇就返回 True
    """
    if not text:
        return True
    t = text.strip().lower()
    t = t.strip(" .,!！?？；;：:*`")
    # 完全匹配
    if t in NONE_LIKE:
        return True
    # 包含匹配：检查是否包含 NONE_LIKE 中的关键词
    for keyword in {"none"}:
        if keyword in t:
            return True
    # 额外检查：独立的"None"（包括 **None**、\n\nNone 等）
    if re.search(r'\bnone\b', t, re.IGNORECASE):
        return True
    return False

PRECHECK_REGISTRY: Dict[str, Callable[[dict, dict, Dict[str,str]], Optional[List[dict]]]] = {}

def build_problem(rule: dict, description: str) -> dict:
    return {
        "field": rule.get("field",""),
        "issue_type": rule.get("issue_type",""),
        "rule_id": rule.get("rule_id",""),
        "description": description or ""
    }

def run_precheck(rule: dict, item: dict, kv: Dict[str,str],
                 enable_default_precheck: bool, precheck_min_len: int) -> Optional[List[dict]]:
    rid = rule.get("rule_id","")
    hook = PRECHECK_REGISTRY.get(rid)
    if hook is not None:
        try:
            result = hook(rule, item, kv)
            if result is not None:
                return result  
        except Exception as e:
            print(f"[WARN] precheck hook for {rid} failed: {e}")
    return None  

# ========================
# 19 个规则的 Precheck 函数
# ========================
def precheck_en_fz_01_v1(rule, item, kv):
    text1 = (kv.get("lab_exam", "") or "").strip()
    if is_none_like(text1) or not text1:
        return []  
    return None 

def precheck_ic_rzcb_01_v1(rule, item, kv):
    text = (kv.get("diagnosis_list","") or "").strip()
    if is_none_like(text) or not text:
        return [build_problem(rule, "入院诊断为空/初步诊断为空")] 
    return None  

def precheck_en_xb_01_v1(rule, item, kv):
    return None  

def precheck_en_jw_01_v1(rule, item, kv):
    text = (kv.get("history_past","") or "").strip()
    if is_none_like(text) or not text:
        return [] 
    return None  

def precheck_en_rzcb_01_v1(rule, item, kv):
    text = (kv.get("diagnosis_list","") or "").strip()
    if is_none_like(text) or not text:
        return []
    return None  

def precheck_en_xb_02_v1(rule, item, kv):
    text = (kv.get("history_present","") or "").strip()
    if is_none_like(text) or not text:
        return [build_problem(rule, "现病史为空")]  
    return None 

def precheck_ic_xb_01_v1(rule, item, kv):
    text = (kv.get("history_present","") or "").strip()
    if is_none_like(text) or not text:
        return [build_problem(rule, "现病史为空")] 
    return None

def precheck_ic_xb_03_v1(rule, item, kv):
    text = (kv.get("history_present","") or "").strip()
    if is_none_like(text) or not text:
        return [build_problem(rule, "现病史为空")] 
    return None  

def precheck_co_xb_03_v1(rule, item, kv):
    text1 = (kv.get("history_present","") or "").strip()
    text2 = (kv.get("chief_complaint","") or "").strip()
    if (is_none_like(text1) or not text1 ) and (is_none_like(text2) or not text2):
        return [build_problem(rule, "现病史和主诉都为空")]
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "现病史为空")]
    if is_none_like(text2) or not text2:
        return [build_problem(rule, "主诉为空")] 
    return None  

def precheck_co_xb_02_v1(rule, item, kv):
    text1 = (kv.get("history_present","") or "").strip()
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "现病史为空")] 
    return None  

def precheck_en_zs_01_v1(rule, item, kv):
    text = (kv.get("chief_complaint", "") or "").strip()
    if is_none_like(text) or not text :
        return [build_problem(rule, "主诉为空，不合格")]
    if len(text) > 20:
        return [build_problem(rule, "主诉字符超过 20，不合格")]
    return None  

def precheck_co_xb_04_v1(rule, item, kv):
    text1 = (kv.get("history_present","") or "").strip()
    text2 = (kv.get("chief_complaint","") or "").strip()
    if (is_none_like(text1) or not text1 ) and (is_none_like(text2) or not text2):
        return [build_problem(rule, "现病史和主诉都为空")]
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "现病史为空")]  
    if is_none_like(text2) or not text2:
        return [build_problem(rule, "主诉为空")]  
    return None  

def precheck_co_xb_01_v1(rule, item, kv):
    text1 = (kv.get("history_present","") or "").strip()
    text2 = (kv.get("chief_complaint","") or "").strip()
    if (is_none_like(text1) or not text1 ) and (is_none_like(text2) or not text2):
        return [build_problem(rule, "现病史和主诉都为空")]
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "现病史为空")]  
    if is_none_like(text2) or not text2:
        return [build_problem(rule, "主诉为空")]  
    return None 

def precheck_dq_rz_01_v1(rule, item, kv):
    text1 = (kv.get("admission_diagnosis","") or "").strip()
    text2 = (kv.get("chief_complaint","") or "").strip()
    if (is_none_like(text1) or not text1 ) and (is_none_like(text2) or not text2):
        return [build_problem(rule, "现病史和入院诊断都为空")]
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "入院诊断为空")]  
    if is_none_like(text2) or not text2:
        return [build_problem(rule, "主诉为空")]  
    return None  

def precheck_dq_rz_02_v1(rule, item, kv):
    text = (kv.get("admission_diagnosis", "") or "").strip()
    if is_none_like(text) or not text :
        return [build_problem(rule, "入院诊断为空")]  
    return None  

def precheck_ic_jw_01_v1(rule, item, kv):
    text1 = (kv.get("admission_diagnosis", "") or "").strip()
    text2 = (kv.get("history_past", "") or "").strip()
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "入院诊断为空")]  
    if is_none_like(text2) or not text2:
        return [build_problem(rule, "既往史为空")]  
    return None  

def precheck_ic_xb_02_v1(rule, item, kv):
    text1 = (kv.get("history_present", "") or "").strip()
    if is_none_like(text1) or not text1 :
        return [build_problem(rule, "现病史为空")]  
    return None 

def precheck_ic_zk_01_v1(rule, item, kv):
    text1 = (kv.get("special_exam", "") or "").strip()
    if is_none_like(text1) or not text1  or text1 == "未做" or text1 == "未查":
        return []  
    return None  

def precheck_ic_zs_01_v1(rule, item, kv):
    text1 = (kv.get("chief_complaint", "") or "").strip()
    if is_none_like(text1) or not text1:
        return [build_problem(rule, "主诉为空")] 
    return None  

# 注册所有 precheck 函数
PRECHECK_REGISTRY["EN-FZ-01-V1"] = precheck_en_fz_01_v1
PRECHECK_REGISTRY["IC-RZCB-01-V1"] = precheck_ic_rzcb_01_v1
PRECHECK_REGISTRY["EN-XB-01-V1"] = precheck_en_xb_01_v1
PRECHECK_REGISTRY["EN-JW-01-V1"] = precheck_en_jw_01_v1
PRECHECK_REGISTRY["EN-RZCB-01-V1"] = precheck_en_rzcb_01_v1
PRECHECK_REGISTRY["EN-XB-02-V1"] = precheck_en_xb_02_v1
PRECHECK_REGISTRY["IC-XB-01-V1"] = precheck_ic_xb_01_v1
PRECHECK_REGISTRY["IC-XB-03-V1"] = precheck_ic_xb_03_v1
PRECHECK_REGISTRY["CO-XB-03-V1"] = precheck_co_xb_03_v1
PRECHECK_REGISTRY["CO-XB-02-V1"] = precheck_co_xb_02_v1
PRECHECK_REGISTRY["EN-ZS-01-V1"] = precheck_en_zs_01_v1
PRECHECK_REGISTRY["CO-XB-04-V1"] = precheck_co_xb_04_v1
PRECHECK_REGISTRY["CO-XB-01-V1"] = precheck_co_xb_01_v1
PRECHECK_REGISTRY["DQ-RZ-01-V1"] = precheck_dq_rz_01_v1
PRECHECK_REGISTRY["DQ-RZ-02-V1"] = precheck_dq_rz_02_v1
PRECHECK_REGISTRY["IC-JW-01-V1"] = precheck_ic_jw_01_v1
PRECHECK_REGISTRY["IC-XB-02-V1"] = precheck_ic_xb_02_v1
PRECHECK_REGISTRY["IC-ZK-01-V1"] = precheck_ic_zk_01_v1
PRECHECK_REGISTRY["IC-ZS-01-V1"] = precheck_ic_zs_01_v1

# ========================
# 核心运行函数
# ========================
def run_predict(
    data_path: str,
    templates_path: str,
    outdir: str,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key: str = "EMPTY",
    model: str = "qwen3-32b",
    rule_ids: Optional[List[str]] = None,
    save_every: int = 20,
    precheck_min_len: int = 4,
    enable_recheck: bool = True,
    enable_default_precheck: bool = True,
    ts: Optional[str] = None,
    base_urls: Optional[List[str]] = None,   
    max_workers: Optional[int] = None,      
    bucket_by_len: bool = True,              
    per_host_limit: Optional[int] = None,  
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    if ts is None:
        ts = now_tag()

    outdir_ts = Path(outdir) / ts
    ensure_dir(outdir_ts)

    data = read_json_any(data_path)
    tpls = read_templates(templates_path)

    if rule_ids:
        allow = set(map(str, rule_ids))
        tpls = [t for t in tpls if str(t.get("rule_id")) in allow]

    if not tpls:
        print("[WARN] 没有可用的规则模板（rule_ids 过滤后为空？）")
        return {}, {}

    urls = list(base_urls) if base_urls else [base_url]
    clients: List[OpenAI] = [OpenAI(api_key=api_key, base_url=u) for u in urls]
    worker_num = max_workers or max(8, len(clients))  

    if per_host_limit is None:
        per_host_limit = max(1, worker_num // max(1, len(clients)))

    host_sems = [Semaphore(per_host_limit) for _ in clients]
    rr = itertools.cycle(range(len(clients)))

    def _est_len_for_rule_item(t: dict, item: dict) -> int:
        tpl = t.get("template", "")
        inputs = t.get("inputs", [])
        placeholders = set(extract_placeholders(tpl))
        view = parse_field_content_if_needed(item)
        if placeholders:
            input_keys = [k for k in inputs if k in placeholders]
        else:
            input_keys = list(inputs)
        kv = pick_inputs(view, input_keys)
        s = " ".join(str(v) for v in kv.values() if v)
        return len(s)

    pred_paths: Dict[str, str] = {}
    used_templates: Dict[str, Dict[str, str]] = {}

    for t in tpls:
        rid = t["rule_id"]
        issue_type = t["issue_type"]
        field = t["field"]
        inputs = t.get("inputs", [])
        tpl = t.get("template", "")
        tpl2 = t.get("template2", "")

        used_templates[rid] = {"template": tpl, "template2": tpl2}

        out_path = Path(outdir_ts) / f"{Path(data_path).stem}_predict_{rid}_{ts}.json"
        pred_paths[rid] = str(out_path)

        existing: Dict[str, dict] = {}
        if out_path.exists():
            try:
                old_list = read_json_any(str(out_path))
                for r in old_list:
                    rrid = r.get("record_id") or r.get("id")
                    if rrid:
                        existing[str(rrid)] = r
            except Exception:
                pass

        todo_items = []
        for item in data:
            record_id = item.get("record_id", item.get("id", ""))
            if not record_id:
                record_id = f"syn_{abs(hash(json.dumps(item, ensure_ascii=False)))%10**12}"
            if record_id in existing:
                continue
            item["_rid_"] = record_id
            todo_items.append(item)

        if not todo_items:
            print(f"[Info] 规则 {rid} 无待处理样本，直接复用已有结果。")
            final_list = list(existing.values())
            final_list.sort(key=lambda x: str(x.get("record_id")))
            Path(out_path).write_text(json.dumps(final_list, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Done] 规则 {rid} 完成，共写入 {len(final_list)} 条 -> {out_path}")
            continue

        if bucket_by_len:
            todo_items.sort(key=lambda it: _est_len_for_rule_item(t, it))

        print(f"[Info] 规则 {rid} 并行预测启动：端口={len(clients)} 并发={worker_num} 待处理={len(todo_items)}")
        rr = itertools.cycle(range(len(clients)))
        written = 0
        
        def _worker_one_item(item: dict, client: OpenAI) -> Tuple[str, dict]:
            record_id = item["_rid_"]
            record_type = item.get("record_type", "ADM_NOTE")

            view = parse_field_content_if_needed(item)
            placeholders = set(extract_placeholders(tpl))
            input_keys = [k for k in inputs if k in placeholders] if placeholders else list(inputs)
            kv_all = pick_inputs(view, input_keys)

            precheck_result = run_precheck(
                rule=t, item=item, kv=kv_all,
                enable_default_precheck=enable_default_precheck,
                precheck_min_len=precheck_min_len
            )
            if precheck_result is not None:
                return record_id, {"record_id": record_id, "record_type": record_type, "problems": precheck_result}

            first_prompt = "所有回答严格限制在 100 个字内。" + safe_format(tpl, kv_all)
            try:
                text = call_llm_with_retry(client, model, first_prompt)
            except Exception as e:
                return record_id, {"record_id": record_id, "record_type": record_type, "problems": []}

            first_ok = is_none_like(text)
            final_text, final_ok = text, first_ok

            if (not first_ok) and enable_recheck and tpl2 and tpl2.strip():
                placeholders2 = set(extract_placeholders(tpl2))
                if placeholders2:
                    input_keys2 = [k for k in inputs if k in placeholders2]
                    kv2 = pick_inputs(view, input_keys2)
                    prompt2 = "所有回答严格限制在 100 个字内。" + safe_format(tpl2, kv2)
                    prompt2 = prompt2 + text
                else:
                    prompt2 = "所有回答严格限制在 100 个字内。" + tpl2 + (item.get("emr_content", "") or "")
                try:
                    text2 = call_llm_with_retry(client, model, prompt2)
                    final_text = text2
                    final_ok = is_none_like(text2)
                except Exception:
                    pass

            problems = [] if final_ok else [build_problem(t, final_text)]
            return record_id, {"record_id": record_id, "record_type": record_type, "problems": problems}

        pbar = tqdm(total=len(todo_items), desc=f"{rid}", smoothing=0.05)
        futures = []
        with ThreadPoolExecutor(max_workers=worker_num) as ex:
            for it in todo_items:
                ci = next(rr)
                cli = clients[ci]
                sem = host_sems[ci]

                def _task_with_limit(item=it, client=cli, guard=sem):
                    with guard:
                        return _worker_one_item(item, client)

                futures.append(ex.submit(_task_with_limit))

            for fut in as_completed(futures):
                try:
                    record_id, rec = fut.result()
                except Exception:
                    record_id = f"syn_fail_{random.randint(1,10**12)}"
                    rec = {"record_id": record_id, "record_type": "ADM_NOTE", "problems": []}

                existing[record_id] = rec
                written += 1
                pbar.update(1)

                if written % save_every == 0:
                    _flush = list(existing.values())
                    _flush.sort(key=lambda x: str(x.get("record_id")))
                    Path(out_path).write_text(json.dumps(_flush, ensure_ascii=False, indent=2), encoding="utf-8")
        pbar.close()

        final_list = list(existing.values())
        final_list.sort(key=lambda x: str(x.get("record_id")))
        Path(out_path).write_text(json.dumps(final_list, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Done] 规则 {rid} 完成，共写入 {len(final_list)} 条 -> {out_path}")

    return pred_paths, used_templates


def run_all_rules():
    """
    一次性运行所有 19 个规则
    """
    print("=" * 60)
    print("Step2 质控 Agent 批量运行启动")
    print("=" * 60)
    
    ts = now_tag()
    all_pred_paths = {}
    all_used_templates = {}
    
    for rule_id, enable_recheck, enable_default_precheck in RULES_CONFIG:
        print(f"\n{'='*60}")
        print(f"运行规则：{rule_id}")
        print(f"  enable_recheck={enable_recheck}, enable_default_precheck={enable_default_precheck}")
        print(f"{'='*60}\n")
        
        pred_paths, used_templates = run_predict(
            data_path=DATA_PATH,
            templates_path=TEMPLATES_PATH,
            outdir=PRED_OUTDIR,
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            rule_ids=[rule_id],
            save_every=10,
            precheck_min_len=2,
            enable_recheck=enable_recheck,
            enable_default_precheck=enable_default_precheck,
            ts=ts,
            max_workers=48,
            bucket_by_len=True,
            per_host_limit=24,
        )
        
        all_pred_paths.update(pred_paths)
        all_used_templates.update(used_templates)
    
    print("\n" + "=" * 60)
    print("所有规则运行完成！")
    print("=" * 60)
    print(f"\n输出目录：{PRED_OUTDIR}\\{ts}")
    print(f"共完成 {len(all_pred_paths)} 个规则")
    
    return all_pred_paths, all_used_templates


if __name__ == "__main__":
    run_all_rules()
