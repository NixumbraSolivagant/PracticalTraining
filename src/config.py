"""src/config.py — 配置中心（所有路径/权重/超参数集中管理）"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env (silently skip if not present)
load_dotenv()

# ── Spark / Hadoop paths from environment ──────────────────────────────────────
SPARK_HOME = os.getenv("SPARK_HOME", "/usr/local/spark")
JAVA_HOME  = os.getenv("JAVA_HOME", "/usr/lib/jvm/jdk-11")
HADOOP_HOME = os.getenv("HADOOP_HOME", "/usr/local/hadoop")
HADOOP_CONF_DIR = os.getenv("HADOOP_CONF_DIR", "/usr/local/hadoop/etc/hadoop")
HDFS_USER = os.getenv("HDFS_USER", os.getenv("USER", "hadoop"))
HDFS_PASS = os.getenv("HDFS_PASS", "hadoop")

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
SIM_CACHE_DIR = OUTPUT_DIR / "similarity_cache"
CLEANED_DIR = OUTPUT_DIR / "cleaned"

# ──综合分权重 ─────────────────────────────────────────────────────────────────
WEIGHTS = {
    "skill":      0.30,   # 技能匹配（核心）
    "tfidf":      0.20,   # TF-IDF 语义相似度
    "jaccard":    0.15,   # 技能 Jaccard 系数
    "education":   0.15,   # 学历
    "experience":  0.10,   # 经验
    "city":       0.05,   # 城市
    "certificate": 0.05,   # 证书
}

# ── 语义分内部权重 ─────────────────────────────────────────────────────────────
SEMANTIC_WEIGHTS = {"tfidf": 0.6, "jaccard": 0.4}

# ── 学历等级 ──────────────────────────────────────────────────────────────────
EDUCATION_LEVELS = {
    "初中及以下": 0, "高中": 1, "中专": 2, "技校": 2,
    "大专": 3, "本科": 4, "硕士": 5, "博士": 6,
}

# ── 城市标准化映射 ────────────────────────────────────────────────────────────
CITY_ALIAS = {
    "北京": "北京", "北京市": "北京",
    "上海": "上海", "上海市": "上海",
    "深圳": "深圳", "深圳市": "深圳",
    "广州": "广州", "广州市": "广州",
    "杭州": "杭州", "杭州市": "杭州",
    "南昌": "南昌", "南昌市": "南昌",
    "成都": "成都", "成都市": "成都",
    "武汉": "武汉", "武汉市": "武汉",
    "西安": "西安", "西安市": "西安",
    "南京": "南京", "南京市": "南京",
    "苏州": "苏州", "苏州市": "苏州",
    "重庆": "重庆", "重庆市": "重庆",
    "天津": "天津", "天津市": "天津",
    "长沙": "长沙", "长沙市": "长沙",
    "郑州": "郑州", "郑州市": "郑州",
    "一线城市": None, "新一线": None,
    "二线城市": None, "三线城市": None,
    "未知": "未知", "其他": "未知",
}

# ── TF-IDF 参数 ────────────────────────────────────────────────────────────────
TFIDF_CONFIG = {"max_features": 5000, "ngram_range": (1, 2)}

# ── HDFS 路径 ────────────────────────────────────────────────────────────────
HDFS_BASE    = "/resume_matching"
HDFS_RAW     = f"{HDFS_BASE}/raw_data"
HDFS_CLEANED = f"{HDFS_BASE}/cleaned_data"
HDFS_RESULTS = f"{HDFS_BASE}/results"

# ── Spark 配置 ────────────────────────────────────────────────────────────────
SPARK_CONFIG = {
    "master": "local[*]",
    "app_name": "ResumeMatchingSystem",
    "driver_memory": "2g",
    "executor_memory": "2g",
}

# ── 薪资评分参数 ─────────────────────────────────────────────────────────────
SALARY_MATCH = {
    "underpay_penalty_per_10pct": 50,   # 低于期望，每10%扣50分
    "overpay_penalty_per_10pct":  30,   # 高于期望，每10%扣30分
}

# ── Top-N ────────────────────────────────────────────────────────────────────
TOP_N = 10
