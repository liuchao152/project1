import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Union

def merge_all_problems_in_dir(input_dir: Union[str, Path], output_filename: str = "merge_all.json") -> str:
    """
    合并该目录下所有 json 文件中的 problems 字段（按 record_id 聚合），不关注时间和规则，只要 problems 合并。
    
    Args:
        input_dir: 包含多个规则质控结果 json 文件的目录
        output_filename: 输出文件名，默认 "merge_all.json"
    
    Returns:
        输出文件的完整路径
    """
    merged_data = {}
    record_problems = defaultdict(list)

    input_path = Path(input_dir)
    
    for fname in os.listdir(input_path):
        if fname.endswith('.json'):
            fpath = input_path / fname
            with open(fpath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"文件 {fname} 解析失败：{e}")
                    continue
                
                for record in data:
                    rid = record.get('record_id')
                    if rid is None:
                        continue
                    if rid not in merged_data:
                        merged_data[rid] = {
                            "record_id": rid,
                            "record_type": record.get("record_type", ""),
                            "problems": []
                        }
                    if "problems" in record and record["problems"]:
                        record_problems[rid].extend(record["problems"])

    for rid, problems in record_problems.items():
        merged_data[rid]["problems"] = problems

    out_path = input_path / output_filename

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(list(merged_data.values()), f, ensure_ascii=False, indent=2)
    
    print(f"合并完成，输出文件：{out_path}")
    return str(out_path)


def filter_problems_by_rule_and_desc(
    rule_id_list: list,
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> str:
    """
    对 rule_id 在 rule_id_list 里的元素：若 description 含"为空"则保留，否则删除
    其他 rule_id 的元素全部保留
    
    Args:
        rule_id_list: 需要特殊处理的 rule_id 列表
        input_path: 输入 json 文件路径
        output_path: 过滤后输出文件路径
    
    Returns:
        输出文件路径
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for x in data:
        if "problems" in x and isinstance(x["problems"], list):
            new_problems = []
            for a in x["problems"]:
                rule_id = a.get("rule_id", "")
                desc = a.get("description", "")
                if rule_id in rule_id_list:
                    # 规则在列表中：只保留 description 含"为空"的
                    if "为空" in desc:
                        new_problems.append(a)
                else:
                    # 规则不在列表中：全部保留
                    new_problems.append(a)
            x["problems"] = new_problems

    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"过滤完成，输出文件：{output_p}")
    return str(output_p)


if __name__ == "__main__":
    # 示例用法 1: 合并目录下所有规则的质控结果
    merge_all_problems_in_dir(
        input_dir=r"D:\bishe\MedShard\out\pred_out\0529-1717",
        output_filename="merge_all.json"
    )
    
    # 示例用法 2: 过滤掉某些规则（分布过少、模型出错率远大于正确率的规则）
    #rule_list = ["CO-XB-01-V1"]
    rule_list = []
    filter_problems_by_rule_and_desc(
        rule_id_list=rule_list,
        input_path=r"D:\bishe\MedShard\out\pred_out\0529-1717\merge_all.json",
        output_path=r"D:\bishe\MedShard\out\results\un_rewrite.json"
    )
