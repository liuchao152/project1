import re, json, ast
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union

CANON_LABELS = [
    "主诉", "现病史", "既往史", "个人史", "家族史",
    "月经史", "婚育史", "体格检查", "专科检查",
    "辅助检查", "入院诊断", "初步诊断", "入院主诊断",
    "病史陈述者"
]

DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "主诉":        ["主 诉", "主诉"],
    "现病史":      ["现病史"],
    "既往史":      ["既往病史", "既往史"],
    "个人史":      ["个人史"],
    "家族史":      ["家族史"],
    "月经史":      ["月经史"],
    "婚育史":      ["婚姻史",  "结婚状况", "婚育史"],
    "体格检查":    ["体格检查"],
    "专科检查":    ["专科检查"],
    "辅助检查":    ["辅助检查"],
    "入院主诊断":  ["入院主诊断"],
    "入院诊断":    ["入院诊断"],
    "初步诊断":    ["初始诊断", "初步诊断"],
    "病史陈述者":  ["病史叙述者", "病史陈述者"],
}

def _now_tag() -> str:
    return datetime.now().strftime("%m%d-%H%M")

def _normalize(s: str) -> str:
    if s is None or len(s.strip()) <= 1:
        return ""
    t = s.strip()
    if t == "" or t.lower() == "none" or t == "无" or t=="无。"or t == "*" or t == "null" or t == "暂无"or t == "暂无。" or t == "补充及专科情况" or t == "请在此定义段落结构和内容":
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")
    return s.strip()

def _maybe_list(txt: str):
    """
    诊断字段解析为列表：
    - "None"/"无"/空 -> None
    - 字符串形如 "['a','b']" -> 尝试 ast 解析为列表
    - 其他 -> [原文字符串]
    """
    if txt is None:
        return None
    t = txt.strip()
    if len(t) <= 1:
        return None
    if t == "" or t.lower() == "none" or t == "无" or t=="无。"or t == "*" or t == "null" or t == "暂无"or t == "暂无。" or t == "请在此定义段落结构和内容":
        return None
    if t.startswith("[") and t.endswith("]"):
        try:
            val = ast.literal_eval(t)
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
        except Exception:
            pass
    return [t]

def _uniq(seq: List[str]) -> List[str]:
    if not seq:
        return []
    seen = OrderedDict()
    for x in seq:
        if x not in seen:
            seen[x] = 1
    return list(seen.keys())

def build_synonyms(override_path: Union[str, Path, None]) -> Dict[str, List[str]]:
    syn = {k: list(v) for k, v in DEFAULT_SYNONYMS.items()}
    if override_path:
        p = Path(override_path)
        if p.exists():
            try:
                user = json.loads(p.read_text(encoding="utf-8"))
                for canon, alias_list in user.items():
                    if canon not in syn:
                        syn[canon] = []
                    for a in alias_list:
                        if a not in syn[canon]:
                            syn[canon].append(a)
            except Exception as e:
                print(f"[WARN] 载入 synonyms 失败：{e}（将仅使用内置同义词）")
    for k in syn:
        syn[k].sort(key=lambda s: len(s), reverse=True)
    return syn

def normalize_labels_in_text(text: str, synonyms: Dict[str, List[str]]) -> str:
    """
    将行首的别名统一替换为规范标签，保留原冒号风格。
    仅处理"行首 + 可选空格 + 别名 + 冒号/："
    """
    t = _normalize(text)
    alias_pairs = []
    for canon, alias_list in synonyms.items():
        for alias in alias_list:
            alias_pairs.append((alias, canon))
    alias_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for alias, canon in alias_pairs:
        pat = re.compile(rf"(?m)^\s*({re.escape(alias)})\s*([：:])")
        def _repl(m):
            colon = m.group(2)
            return f"{canon}{colon}"
        t = pat.sub(_repl, t)
    return t

def _find_original_initial_label(raw_text: str, synonyms: Dict[str, List[str]]) -> str:
    """
    在未归一化的原文里，检测"初步诊断"的原始别名（可能为"初始诊断"或"初步诊断"）。
    若均未命中，则返回默认键名"初步诊断"。
    """
    raw = _normalize(raw_text or "")
    alias_list = synonyms.get("初步诊断", ["初步诊断"])
    # 使用"行首 + 冒号/："的模式匹配
    for alias in alias_list:
        if re.search(rf"(?m)^\s*{re.escape(alias)}\s*[：:]", raw):
            return alias
    return "初步诊断"

def build_section_regex() -> re.Pattern:
    labels_sorted = sorted(CANON_LABELS, key=len, reverse=True)
    pattern = (
        r"^(?P<label>" + "|".join(map(re.escape, labels_sorted)) + r")\s*[:：]\s*"
        r"(?P<text>.*?)\s*(?=^(?:" + "|".join(map(re.escape, labels_sorted)) + r")\s*[:：]|\Z)"
    )
    return re.compile(pattern, flags=re.S | re.M)

SECTION_RE = build_section_regex()

def parse_emr_to_field_content(emr_text: str, synonyms: Dict[str, List[str]]) -> Dict[str, Any]:
    initial_key_name = _find_original_initial_label(emr_text, synonyms)  # NEW

    text_norm = normalize_labels_in_text(emr_text or "", synonyms)
    text_norm = _normalize(text_norm)

    chunks: Dict[str, str] = {}
    for m in SECTION_RE.finditer(text_norm):
        label = m.group("label")
        seg = _normalize(m.group("text"))
        chunks[label] = seg

    diag_admission_raw = _maybe_list(chunks.get("入院诊断")) or []
    diag_admission = []
    for entry in diag_admission_raw:
        entry_clean = entry.replace('\r', '')
        idx_fix = entry_clean.find('\n修正')
        if idx_fix != -1:
            entry_clean = entry_clean[:idx_fix]
        entry_clean = entry_clean.replace('\n', '')
        diag_admission.append(entry_clean)

    
    diag_initial_raw1 = _maybe_list(chunks.get("初步诊断")) or []
    diag_initial_raw  = []
    for entry in diag_initial_raw1:
        entry_clean = entry.replace('\r', '')
        diag_initial_raw.append(entry_clean)
    diag_initial = _uniq(diag_initial_raw) if diag_initial_raw else []  
    diag_main = _maybe_list(chunks.get("入院主诊断")) or None       

    diagnosis_list_dict: Dict[str, List[str]] = {initial_key_name: diag_initial}
    if diag_admission: 
        diagnosis_list_dict["入院诊断"] = _uniq(diag_admission)

    admission_diag = _uniq(diag_admission) or None

    field_content: Dict[str, Any] = {
        "history_present":      chunks.get("现病史") or None,
        "lab_exam":             chunks.get("辅助检查") or None,
        "diagnosis_list":       diagnosis_list_dict,
        "chief_complaint":      chunks.get("主诉") or None,
        "history_past":         chunks.get("既往史") or None,
        "admission_diagnosis":  admission_diag,
        "special_exam":         chunks.get("专科检查") or None,
        "personal_history":     chunks.get("个人史") or None,
        "family_history":       chunks.get("家族史") or None,
        "menstrual_history":    chunks.get("月经史") or None,
        "marital_history":      chunks.get("婚育史") or None,
        "physical_exam":        chunks.get("体格检查") or None,
        "historian":            chunks.get("病史陈述者") or None,
        "diagnosis_main":       diag_main,
        "diagnosis_initial":    diag_initial,
    }
    return field_content

def read_json_any(path: Union[str, Path]) -> List[dict]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if "\n{" in txt:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else [obj]

def write_json(path: Union[str, Path], data: List[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def inject_field_content(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    emr_key: str = "emr_content",
    synonyms_path: Union[str, Path, None] = None,
) -> str:
    """
    读取 input_path 的 JSON/JSONL，
    - 先做行首同义词归一，
    - 再抽取段落，
    - 写回到每条记录的 'field_content'。
    返回输出文件路径。
    """
    data = read_json_any(input_path)
    synonyms = build_synonyms(synonyms_path)
    out_list: List[dict] = []

    for rec in data:
        emr = rec.get(emr_key, "") or ""
        fc = parse_emr_to_field_content(emr, synonyms)
        new_rec = dict(rec)
        new_rec["field_content"] = fc
        # new_rec["field_re"] = fc
        out_list.append(new_rec)

    if output_path is None:
        inp = Path(input_path)
        ts = _now_tag()
        output_path = inp.with_name(f"{inp.stem}_with_fields_{ts}{inp.suffix}")

    write_json(output_path, out_list)
    return str(output_path)

if __name__ == "__main__":
    # 示例用法
    inject_field_content(
        input_path="./test_B/test_B.json",
        output_path="./test_B/test_B_re.json",
        emr_key="emr_content",
    )
