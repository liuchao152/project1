import json
import os

# 配置参数
input_file = r"D:\bishe\CHIP\train-new\train_latest_new.json"
output_dir = r"D:\bishe\CHIP\train-new"
output_file = os.path.join(output_dir, "old1.json")

# 要删除的范围（从第 x 项到第 y 项，索引从 0 开始）
# 例如：x=0, y=9 表示删除前 10 项（索引 0-9）
X = 150  # 起始索引（包含）
Y = 299  # 结束索引（包含）

# 读取原始文件
print(f"正在读取文件：{input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"共读取 {len(data)} 条记录")
print(f"准备删除第 {X} 项到第 {Y} 项（共 {Y - X + 1} 条）")

# 验证范围
if X < 0 or Y >= len(data) or X > Y:
    print(f"错误：无效的范围！文件共有 {len(data)} 条记录，索引范围应为 0 到 {len(data)-1}")
    exit(1)

# 删除指定范围的项
# 保留第 0 到 X-1 项，以及第 Y+1 到末尾的项
new_data = data[:X] + data[Y+1:]

print(f"删除后剩余 {len(new_data)} 条记录")

# 保存新文件
print(f"正在保存新文件：{output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("完成！")
print(f"原文件保持不变：{input_file}")
print(f"新文件已保存：{output_file}")
print(f"已删除记录索引：{X} 到 {Y}")
