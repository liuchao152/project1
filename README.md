# MedShard on CHIP2025 · 入院记录内涵质控（不微调）

---

## 📁 完整目录结构

```
MedShard/
│
├─ 📄 README.md                          # 项目说明文档
├─ 📄 requirements.txt                   # Python 依赖包列表
├─ 📄 .gitignore                         # Git 忽略配置
│
├─ 📂 config/                            # 【配置目录】API 密钥统一管理
│  ├─ 📄 api_config.py                   # API 密钥配置（⚠️ 敏感文件，已加入 .gitignore）
│  ├─ 📄 api_config.example.py           # API 配置模板（可安全提交）
│  └─ 📄 README.md                       # 配置使用说明
│
├─ 📂 in/                                # 【输入目录】原始数据和配置文件
│  ├─ 📄 templates.json                  # 19 条规则的提示词模板配置
│  ├─ 📄 examples.json                   # 各规则的正负样例集合（few-shot）
│  ├─ 📄 ruleid_to_deduct.json           # 规则 ID 到扣分项的映射
│  ├─ 📄 train.json                      # 原始数据
│  └─ 📄 质控项及标注逻辑说明.xlsx         # 质控规则标注逻辑说明
│
├─ 📂 out/                               # 【输出目录】运行结果
│  ├─ 📄 train_latest_new.json           # 清洗后的数据（用于 Step2 质控）
│  │
│  ├─ 📂 results/                        # 改写后的结果文件
│  │  ├─ 📄 un_rewrite.json              # 待改写的原始结果
│  │  ├─ 📄 fewshot_rewrite.json         # Few-shot 改写后的结果
│  │  └─ 📄 rag_rewrite.json             # RAG 改写后的结果
│  │
│  ├─ 📂 pred_out/                       # 预测结果输出
│  │  └─ [时间戳]/                       # 按时间戳组织，如 0525-1255/
│  │     ├─ 📄 new_predict_{rule_id}_{ts}.json
│  │     └─ 📄 merge_all.json            # 合并所有规则的结果
│  │
│  └─ 📂 eval_out/                       # 评估结果输出
│     └─ [时间戳]/                       # 按时间戳组织
│        ├─ 📄 summary_metrics_{name}_{ts}.json
│        ├─ 📄 summary_metrics_{name}_{ts}.csv
│        └─ 📄 per_record_compliance_{name}_{ts}.csv
│
├─ 📂 agent/                             # 【核心代码】Agent 实现
│  ├─ 📄 run_step2_all.py                # Step2 质控 Agent 批量运行（19 条规则）
│  ├─ 📄 rewrite_fewshot.py              # Few-shot 改写脚本（提升 ROUGE-L）
│  ├─ 📄 rewrite_rag.py                  # RAG 版本改写脚本（从 ChromaDB 检索示例）
│  └─ 📄 cleaning_agent_regex.py         # 清洗 Agent 正则实现（字段结构化）
│
├─ 📂 chroma_db/                         # 【向量数据库】ChromaDB 持久化目录
│  └─ (自动创建，存储向量索引)
│
└─ 🔧 根目录工具脚本
   ├─ 📄 chroma.py                       # ChromaDB 入库程序（向量存储）
   ├─ 📄 merge_rules.py                  # 规则结果合并脚本
   ├─ 📄 evaluation.py                   # 评估脚本
   ├─ 📄 wash.py                         # 脚本
   ├─ 📄 delete.py                       # 脚本
   ├─ 📄 test_api.py                     # API 连通性测试
   └─ 📄 think.json                      # LLM think 标签清理记录

```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n medshard python=3.12 -y
conda activate medshard

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 复制配置模板
cp config/api_config.example.py config/api_config.py

# 编辑配置文件，填入真实的 API 密钥
notepad config/api_config.py
```

**配置项说明**：
```python
# 阿里云 DashScope（通义千问）
DASHSCOPE_API_KEY = "sk-your-key-here"
DEFAULT_MODEL = "qwen3-32b"

# Embedding 配置（ChromaDB 用）
EMBEDDING_API_KEY = "sk-your-embedding-key-here"
EMBEDDING_MODEL = "text-embedding-v4"
```

### 3. 测试 API 连接

```bash
python test_api.py
```

### 4. 运行完整流程

```bash
# Step 1: 数据预处理（字段结构化）
python agent/cleaning_agent_regex.py

# Step 2: 运行 19 条规则的质控检测
python agent/run_step2_all.py

# Step 3: 合并所有规则的结果
python merge_rules.py

# Step 4: 改写 description 提升 ROUGE-L 分数
python agent/rewrite_fewshot.py

# Step 5: （可选）使用 RAG 版本改写
python agent/rewrite_rag.py
```

---

## 🧠 方法与流程简介

### 方法简介

* **逐规则查验**：针对每个规则独立设计 agent，避免长上下文导致的"注意力分散"。
* **多 Agent 级联**：
  - **清洗 Agent** (`cleaning_agent_regex.py`)：将原始输入拆分为结构化字段（主诉、现病史、诊断等）
  - **质控 Agent** (`run_step2_all.py`)：逐规则判断是否合格
  - **反思 Agent** (内建于质控流程)：复核初步结果，提高精确率
  - **输出 Agent** (`rewrite_fewshot.py`)：规范化 problem description，提升 ROUGE-L 分数
* **Few-shot Prompt**：添加合适的正负样例，效果远优于纯规则陈述。
* **RAG 检索** (`rewrite_rag.py`)：从 ChromaDB 动态检索相似示例，替代静态 examples.json。

### 流程简介

```
原始 EMR → 清洗 Agent → 结构化字段 → 质控 Agent → 初步结果
                                      ↓
                              反思 Agent（可选）
                                      ↓
                              输出 Agent → 最终提交
```

1. **模型选择**：使用阿里云 DashScope API 的 Qwen3-32B 模型
2. **数据预处理**：基于正则表达式拆分原始 JSON，得到 `field_content` 字段
3. **结果优化**：
   - **问题定位**：通过正例提高召回率，rethink 提高精确率
   - **Description 优化**：使用 ROUGE-L 最优短语替换

---

## ⚙️ 配置说明

### API 配置（`config/api_config.py`）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_BASE_URL` | 阿里云 DashScope API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_API_KEY` | 阿里云 API 密钥 | 需自行填写 |
| `DEFAULT_MODEL` | 默认推理模型 | `qwen3-32b` |
| `EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v4` |
| `EMBEDDING_API_KEY` | Embedding API 密钥 | 需自行填写 |
| `MAX_WORKERS` | 最大并发数 | `48` |
| `PER_HOST_LIMIT` | 每端口限流 | `24` |

---

## 📊 结果文件格式

`out/pred_out/[时间戳]/merge_all.json`：

```json
{
  "record_id": "MED_QC_ADM_0001_testB",
  "record_type": "ADM_NOTE",
  "problems": [
    {
      "field": "diagnosis_list",
      "issue_type": "INFO_COMPLETENESS",
      "rule_id": "IC-RZCB-01-V1",
      "description": "检验提示贫血未列诊断"
    },
    {
      "field": "history_present",
      "issue_type": "EXAM_RECORD_NORM",
      "rule_id": "EN-XB-02-V1",
      "description": "缺诱因"
    }
  ]
}
```

---

## 📦 依赖与版本
**完整依赖**：见 `requirements.txt`

---

## 🔗 额外资源

* **Qwen3-32B**：https://modelscope.cn/models/Qwen/Qwen3-32B
* **DashScope API**：https://dashscope.aliyun.com/
* **ChromaDB 文档**：https://docs.trychroma.com/

---

