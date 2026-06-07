#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病历质控测评工具 - 极简版
统计 TP / FP / FN + 平均 ROUGE-L
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import heapq


def read_json_any(path: str) -> List[dict]:
    """读取 json 或 jsonl"""
    txt = Path(path).read_text(encoding="utf-8").strip()
    if "\n{" in txt:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else [obj]


def get_problems(rec: dict) -> Dict[Tuple[str, str], List[str]]:
    """提取记录中问题：{(rule_id, field): [descriptions]}"""
    probs = defaultdict(list)
    for p in rec.get("problems", []):
        rule_id = p.get("rule_id", "")
        field = p.get("field", "")
        desc = p.get("description", "")
        if rule_id:
            probs[(rule_id, field)].append(desc)
    return probs


def lcs_length(a: str, b: str) -> int:
    """最长公共子序列长度"""
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def rouge_l_f1(pred: str, ref: str) -> float:
    """字符级 ROUGE-L F1"""
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    p = lcs / max(len(pred), 1)
    r = lcs / max(len(ref), 1)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def greedy_match(pred_descs: List[str], gt_descs: List[str]) -> Tuple[int, List[float]]:
    """贪心匹配预测和金标描述，返回匹配数和 ROUGE 分数列表"""
    if not pred_descs or not gt_descs:
        return 0, []
    
    sim = [[rouge_l_f1(pd, gd) for gd in gt_descs] for pd in pred_descs]
    h = [(-sim[i][j], i, j) for i in range(len(pred_descs)) for j in range(len(gt_descs))]
    heapq.heapify(h)
    
    used_p, used_g = set(), set()
    scores = []
    
    while h:
        neg_s, i, j = heapq.heappop(h)
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        scores.append(-neg_s)
    
    return len(scores), scores


def evaluate(pred_path: str, gt_path: str):
    """主评测函数"""
    pred_data = read_json_any(pred_path)
    gt_data = read_json_any(gt_path)
    
    pred_idx = {r.get("record_id"): r for r in pred_data}
    gt_idx = {r.get("record_id"): r for r in gt_data}
    all_ids = set(pred_idx.keys()) | set(gt_idx.keys())
    
    TP, FP, FN = 0, 0, 0
    rouge_scores = []
    
    for rid in all_ids:
        pred_probs = get_problems(pred_idx.get(rid, {"problems": []}))
        gt_probs = get_problems(gt_idx.get(rid, {"problems": []}))
        
        all_keys = set(pred_probs.keys()) | set(gt_probs.keys())
        
        for key in all_keys:
            pred_descs = pred_probs.get(key, [])
            gt_descs = gt_probs.get(key, [])
            
            if pred_descs and gt_descs:
                # TP: 预测有 & 实际有
                tp_count, scores = greedy_match(pred_descs, gt_descs)
                TP += tp_count
                rouge_scores.extend(scores)
                # 多余的预测 = FP
                FP += len(pred_descs) - tp_count
                # 多余的 gold = FN
                FN += len(gt_descs) - tp_count
            elif pred_descs and not gt_descs:
                # FP: 预测有 & 实际无
                FP += len(pred_descs)
            elif not pred_descs and gt_descs:
                # FN: 预测无 & 实际有
                FN += len(gt_descs)
    
    # 计算衍生指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    print("=" * 50)
    print("评测结果（问题级别）")
    print("=" * 50)
    print(f"TP (真阳性): {TP}  ← 预测正确的问题数")
    print(f"FP (假阳性): {FP}  ← 误报的问题数")
    print(f"FN (假阴性): {FN}  ← 漏报的问题数")
    print("-" * 50)
    print(f"预测总数：{TP + FP}")
    print(f"实际总数：{TP + FN}")
    print(f"TP 样本数：{len(rouge_scores)}")
    print("-" * 50)
    print(f"Precision (查准率): {precision*100:.2f}%")
    print(f"Recall    (查全率): {recall*100:.2f}%")
    print(f"F1        (综合分): {f1*100:.2f}%")
    print(f"ROUGE-L   (平均):  {avg_rouge*100:.2f}%")
    print("=" * 50)
    
    return {"TP": TP, "FP": FP, "FN": FN, "P": precision, "R": recall, "F1": f1, "ROUGE-L": avg_rouge}


if __name__ == "__main__":
    # 文件路径配置
    PRED_PATH = r"D:\bishe\MedShard\out\results\fewshot_rewrite.json"           # 预测结果文件
    GT_PATH = r"D:\bishe\MedShard\out\train_latest_new.json"  # 标准答案文件
    
    evaluate(PRED_PATH, GT_PATH)
