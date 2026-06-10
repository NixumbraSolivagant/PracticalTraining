# 简历-岗位智能匹配系统

基于 TF-IDF 语义相似度 + 多维度规则评分 + PySpark MLlib 分布式引擎的人岗智能推荐平台。

## 技术架构

```
数据输入 → PySpark ETL → MLlib匹配 → 结果输出 → Streamlit UI
```

## 目录结构

```
local_resume_job_matching_project/
  src/
    config.py          # 配置中心
    clean.py           # 脏数据清洗
    structure.py       # 字段结构化
    preprocess.py      # 文本预处理（jieba分词+停用词）
    similarity.py      # TF-IDF + Jaccard 相似度 + 矩阵缓存
    scoring.py         # 6维规则评分 + 薪资平滑函数
    matcher.py         # 综合分 + Top-N + 可解释结果
    theme.py           # 配色常量 + CSS生成器
    html_templates.py  # HTML模板
    charts.py          # Plotly图表构建器（纯函数）
    ui_components.py   # Streamlit渲染组件
    hdfs_utils.py      # HDFS操作封装
  app.py              # Streamlit主页面
  run_match.py        # 本地Pipeline入口
  run_spark.py        # HDFS+Spark Pipeline入口
  config.yaml         # YAML配置
  requirements.txt    # Python依赖
  data/               # 自备测试数据（30条简历+20条岗位）
  output/             # 输出结果
  hdfs_project/      # HDFS脚本和Spark模块
```

## 算法说明

### 语义相似度（本地模式）

- **TF-IDF 余弦相似度**：使用 `scikit-learn TfidfVectorizer`，ngram_range=(1,2)，max_features=5000
- **技能 Jaccard 系数**：`Jaccard = |resume_skills ∩ job_skills| / |resume_skills ∪ job_skills|`
- **融合**：`semantic_sim = tfidf_sim × 0.6 + jaccard_sim × 0.4`
- **Jieba 自定义技能词典**：从 `data/skill_alias.json` 自动加载所有标准化技能词（359个），提升专业术语分词精度

> 注：本地模式不训练 Word2Vec（50条数据词表太小，词向量无区分度）。
> Spark 模式使用 MLlib Word2Vec（分布式训练不受样本量限制）。

### 薪资评分（平滑函数）

```
区间内 → 100分
低于期望（岗位好）：gap = (job_min - expect) / expect，每超10%扣50分
高于期望（岗位差）：gap = (expect - job_max) / job_max，每超10%扣30分
```

### 综合分权重

| 维度 | 权重 | 说明 |
|------|------|------|
| 技能匹配 | 30% | 最核心匹配依据 |
| TF-IDF 语义 | 20% | 文本整体相似度 |
| Jaccard | 15% | 技能集合重叠度 |
| 学历 | 15% | 硬性门槛 |
| 经验 | 10% | 重要但不如技能核心（平滑函数：每差1年扣100/要求年数的分） |
| 城市 | 5% | 偏好性条件 |
| 证书 | 5% | 加分项 |

## 数据流水线与字段契约

Pipeline 共 3 个阶段，每阶段输出文件及核心字段如下：

### 阶段 1：`clean.py` → `output/cleaned/cleaned_resumes.csv` / `cleaned_jobs.csv`

原始数据清洗：去重、空值填充、格式标准化、城市别名映射。

核心输出字段（简历）：`resume_id, name, education, skills, certifications, experience_years, city, expected_salary`
核心输出字段（岗位）：`job_id, job_title, required_skills, preferred_skills, required_education, min_experience_years, city, salary_min, salary_max`

### 阶段 2：`structure.py` → `output/cleaned/structured_*.csv`

技能标准化（通过 `skill_alias.json` 别名映射）、学历等级数值化。

新增字段（简历）：`skills_std, skills_list, cert_list, cert_list_str, edu_level`
新增字段（岗位）：`required_skills_std, required_skills_list, preferred_skills_std, pref_certs_str, required_edu_level, min_exp_num`

### 阶段 3：`preprocess.py` → `output/cleaned/preprocessed_*.csv`

jieba 分词 + 停用词过滤 + 文本拼接。

新增字段（简历/岗位通用）：`text_combined, text_tokenized, text_joined`

> **UI 数据加载**：`app.py` 的 `load_resumes()` 和 `load_jobs()` 读取 `preprocessed_*.csv`，因为这些文件包含完整的结构化字段（skills_list, edu_level 等）供展示使用。

### 输出文件

- `output/top_matches.csv`：Top-N 匹配结果，含所有 7 个维度分 + 综合分
- `output/explainable_match_result.csv`：推荐理由详情，含 matched_skills / missing_skills / recommendation_reason

## 快速开始

### 环境要求

- Python 3.10+
- pandas, numpy, scikit-learn, jieba
- streamlit, plotly
- pyyaml, python-dotenv

```bash
# 安装依赖
pip install -r requirements.txt

# 本地模式（推荐）
python run_match.py

# 启动 Web UI
streamlit run app.py --server.port 8501

# HDFS + Spark 模式（需要 Hadoop 集群）
bash hdfs_project/upload_to_hdfs.sh
python run_spark.py
```

## Streamlit UI 功能

- **综合概览**：评分维度说明 + 算法架构 + 综合分排名柱状图
- **学生视角**：按简历查看推荐岗位，含 Gauge 仪表盘 + 雷达图 + 推荐理由
- **HR视角**：按岗位查看匹配候选人，含雷达图 + 技能匹配详情
- **全量统计**：分数分布直方图 + Top岗位柱图 + 城市饼图 + 技能热力

## 大数据架构说明

当前使用 `crossJoin` 生成所有配对，在演示规模（30×20=600条）下无性能问题。
真实大数据场景（10万×1万）演进路径：

1. **粗排阶段**：Elasticsearch 倒排索引或 LSH（MinHash）将召回从 O(N×M) 降到 O(N×K)，K=100~500
2. **精排阶段**：对 Top-K 候选使用完整特征工程 + LightGBM 排序模型
3. **向量检索**：用 FAISS/Annoy 对简历/岗位做 embedding，近似最近邻检索
