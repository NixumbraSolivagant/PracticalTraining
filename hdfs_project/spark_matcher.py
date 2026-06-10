"""hdfs_project/spark_matcher.py — PySpark MLlib 分布式匹配

与本地 scoring.py 保持算法一致：
  - 语义分内部权重：TF-IDF 0.6 + Jaccard 0.4（来自 config.SEMANTIC_WEIGHTS）
  - 综合分权重：技能30% + TF-IDF 20% + Jaccard 15% + 学历15% + 经验10% + 城市5% + 证书5%

⚠️ 规模说明：当前使用 crossJoin 生成所有配对（30×20=600条），在演示规模下无性能问题。
如扩展到真实大数据场景（10万×1万=10亿条），需改用 LSH 粗排 + 精排二级架构。
"""

import os
import json
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number, udf, split, size, when, lit, concat_ws
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import FloatType, StringType


# ── 综合分权重（与 config.py 保持同步）──────────────────────────────────────────
WEIGHTS = {
    "skill": 0.30,
    "tfidf": 0.20,
    "jaccard": 0.15,
    "education": 0.15,
    "experience": 0.10,
    "city": 0.05,
    "certificate": 0.05,
}
SEMANTIC_WEIGHTS = {"tfidf": 0.6, "jaccard": 0.4}

EDUCATION_LEVELS = {
    "初中及以下": 0, "高中": 1, "中专": 2, "技校": 2,
    "大专": 3, "本科": 4, "硕士": 5, "博士": 6,
}

CITY_ALIAS = {
    "北京": "北京", "北京市": "北京",
    "上海": "上海", "上海市": "上海",
    "深圳": "深圳", "深圳市": "深圳",
    "广州": "广州", "广州市": "广州",
    "杭州": "杭州", "杭州市": "杭州",
    "南昌": "南昌", "南昌市": "南昌",
    "成都": "成都", "成都市": "成都",
    "武汉": "武汉", "武汉市": "武汉",
}


# ── Spark UDF ──────────────────────────────────────────────────────────────────

def cosine_sim_vec(v1: DenseVector, v2: DenseVector) -> float:
    dot = float(v1.dot(v2))
    n1 = float(v1.norm(2))
    n2 = float(v2.norm(2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


cosine_sim_vec_udf = udf(cosine_sim_vec, FloatType())


def jaccard_score(arr1, arr2) -> float:
    if not arr1 or not arr2:
        return 0.0
    inter = len(set(arr1) & set(arr2))
    union = len(set(arr1) | set(arr2))
    return inter / union if union > 0 else 0.0


jaccard_udf = udf(jaccard_score, FloatType())


def score_skill_udf(resume_skills_str: str, req_skills_str: str, pref_skills_str: str) -> str:
    """返回: score,matched,missing,extra（用 || 分隔）"""
    r_set = set(s.strip() for s in str(resume_skills_str).split(";") if s.strip())
    req_set = set(s.strip() for s in str(req_skills_str).split(";") if s.strip())
    pref_set = set(s.strip() for s in str(pref_skills_str).split(";") if s.strip())

    matched = sorted(r_set & req_set)
    missing_req = sorted(req_set - r_set)
    extra = sorted(r_set - req_set - pref_set)

    if len(req_set) == 0:
        base = 100.0
    else:
        base = len(matched) / len(req_set) * 100.0
    bonus = min(len(extra) * 5.0, 25.0)
    score = min(base + bonus, 100.0)
    return (f"{score:.2f}||{'|'.join(matched)}||{'|'.join(missing_req)}||{'|'.join(extra)}")


skill_udf = udf(score_skill_udf, StringType())


def score_edu_udf(resume_edu: str, job_edu: str) -> float:
    r = EDUCATION_LEVELS.get(str(resume_edu).strip(), 3)
    j = EDUCATION_LEVELS.get(str(job_edu).strip(), 3)
    if r >= j:
        return 100.0
    return max(0.0, 100.0 - (j - r) * 25.0)


edu_udf = udf(score_edu_udf, FloatType())


def score_exp_udf(resume_exp: float, job_min_exp: float) -> float:
    import math
    if math.isnan(float(resume_exp)) or math.isnan(float(job_min_exp)):
        return 60.0
    r, j = float(resume_exp), float(job_min_exp)
    if j <= 0:
        return 100.0
    if r >= j:
        return 100.0
    gap = (j - r) / j
    return max(0.0, min(100.0, 100.0 - gap * 100))


exp_udf = udf(score_exp_udf, FloatType())


def score_city_udf(resume_city: str, job_city: str) -> float:
    r = CITY_ALIAS.get(str(resume_city).strip(), resume_city)
    j = CITY_ALIAS.get(str(job_city).strip(), job_city)
    if str(r) == str(j):
        return 100.0
    if r in ("未知", "其他", "") or j in ("未知", "其他", ""):
        return 60.0
    return 20.0


city_udf = udf(score_city_udf, FloatType())


def score_cert_udf(resume_certs: str, pref_certs: str) -> float:
    r_set = set(s.strip() for s in str(resume_certs).split(";") if s.strip())
    p_set = set(s.strip() for s in str(pref_certs).split(";") if s.strip())
    if not p_set:
        return 100.0
    matched = len(r_set & p_set)
    base = matched / len(p_set) * 100.0
    extra = min((len(r_set) - matched) * 12.5, 25.0)
    return min(base + extra, 100.0)


cert_udf = udf(score_cert_udf, FloatType())


def get_spark(app_name: str = "ResumeMatching-MLlib") -> SparkSession:
    spark_home = os.getenv("SPARK_HOME", "/usr/local/spark")
    java_home = os.getenv("JAVA_HOME", "/usr/lib/jvm/jdk-11")

    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]")
               .config("spark.driver.memory", "2g")
               .config("spark.executor.memory", "2g")
               .config("spark.sql.shuffle.partitions", "8"))

    return builder.getOrCreate()


def run():
    print("=== PySpark MLlib 匹配 ===")
    spark = get_spark()

    resumes = spark.read.parquet("hdfs:///resume_matching/cleaned_data/cleaned_resumes.parquet")
    jobs = spark.read.parquet("hdfs:///resume_matching/cleaned_data/cleaned_jobs.parquet")

    resumes = resumes.withColumnRenamed("resume_id", "resume_id_r") \
                    .withColumnRenamed("name", "resume_name")
    jobs = jobs.withColumnRenamed("job_id", "job_id_j") \
               .withColumnRenamed("name", "job_title_alias")

    # ── Tokenize（技能字段）────────────────────────────────────────────────────
    tokenizer_resume = Tokenizer(inputCol="skills", outputCol="resume_words")
    resumes = tokenizer_resume.transform(resumes)
    tokenizer_job = Tokenizer(inputCol="required_skills", outputCol="job_words")
    jobs = tokenizer_job.transform(jobs)

    # ── HashingTF + IDF ────────────────────────────────────────────────────────
    hashingTF = HashingTF(inputCol="resume_words", outputCol="resume_raw_features", numFeatures=5000)
    resumes = hashingTF.transform(resumes)
    idf = IDF(inputCol="resume_raw_features", outputCol="resume_tfidf")
    resume_idf_model = idf.fit(resumes)
    resumes = resume_idf_model.transform(resumes)

    hashingTF2 = HashingTF(inputCol="job_words", outputCol="job_raw_features", numFeatures=5000)
    jobs = hashingTF2.transform(jobs)
    idf2 = IDF(inputCol="job_raw_features", outputCol="job_tfidf")
    job_idf_model = idf2.fit(jobs)
    jobs = job_idf_model.transform(jobs)

    # ── crossJoin 配对 ────────────────────────────────────────────────────────
    job_select = jobs.select(
        "job_id_j", "job_title_alias", "job_title", "company", "city",
        "job_tfidf", "required_skills", "preferred_skills", "required_education",
        "min_experience_years", "preferred_certificates",
        "salary_min", "salary_max",
    )
    pairs = resumes.crossJoin(job_select)

    # ── 相似度计算 ────────────────────────────────────────────────────────────
    pairs = pairs.withColumn("tfidf_sim",
        cosine_sim_vec_udf(col("resume_tfidf"), col("job_tfidf")))
    pairs = pairs.withColumn("jaccard_sim",
        jaccard_udf(split(col("skills"), ";"), split(col("required_skills"), ";")))

    # ── 语义融合（与本地 similarity.py 一致）───────────────────────────────────
    pairs = pairs.withColumn("semantic_sim",
        col("tfidf_sim") * lit(SEMANTIC_WEIGHTS["tfidf"])
        + col("jaccard_sim") * lit(SEMANTIC_WEIGHTS["jaccard"]))

    # ── 多维评分（与本地 scoring.py 一致）──────────────────────────────────────
    skill_result = skill_udf(col("skills"), col("required_skills"), col("preferred_skills"))
    pairs = pairs.withColumn("skill_score", split(col("skill_result"), "\\|\\|").getItem(0).cast("double"))
    pairs = pairs.withColumn("matched_skills", concat_ws(";", split(col("skill_result"), "\\|\\|").getItem(1)))
    pairs = pairs.withColumn("missing_skills", concat_ws(";", split(col("skill_result"), "\\|\\|").getItem(2)))
    pairs = pairs.withColumn("extra_skills", concat_ws(";", split(col("skill_result"), "\\|\\|").getItem(3)))

    pairs = pairs.withColumn("education_score", edu_udf(col("education"), col("required_education")))
    pairs = pairs.withColumn("experience_score", exp_udf(col("experience_years"), col("min_experience_years")))
    pairs = pairs.withColumn("city_score", city_udf(col("city"), col("city")))  # resumes.city vs jobs.city
    pairs = pairs.withColumn("cert_score", cert_udf(col("certifications"), col("preferred_certificates")))

    # ── 综合分（与本地 scoring.py 一致）───────────────────────────────────────
    pairs = pairs.withColumn("final_score",
        col("skill_score") * lit(WEIGHTS["skill"])
        + col("semantic_sim") * lit(WEIGHTS["tfidf"])
        + col("jaccard_sim") * lit(WEIGHTS["jaccard"])
        + col("education_score") * lit(WEIGHTS["education"])
        + col("experience_score") * lit(WEIGHTS["experience"])
        + col("city_score") * lit(WEIGHTS["city"])
        + col("cert_score") * lit(WEIGHTS["certificate"])
    )

    # ── 排名 Top-10 ──────────────────────────────────────────────────────────
    window = Window.partitionBy("resume_id_r").orderBy(col("final_score").desc())
    ranked = pairs.withColumn("rank", row_number().over(window))
    top_matches = ranked.filter(col("rank") <= 10)

    # ── 推荐理由生成 ──────────────────────────────────────────────────────────
    def build_reason(score_s: float, edu_s: float) -> str:
        parts = []
        if score_s >= 80:
            parts.append("核心技能与岗位高度匹配，适合推荐。")
        elif score_s >= 60:
            parts.append("技能基本匹配，部分要求略有差距。")
        else:
            parts.append("技能匹配度较低，建议优先提升相关技能。")
        if edu_s >= 100:
            parts.append("学历满足岗位要求。")
        elif edu_s >= 50:
            parts.append("学历略低于要求，但可考虑。")
        return " ".join(parts)

    build_reason_udf = udf(build_reason, StringType())
    top_matches = top_matches.withColumn("recommendation_reason",
        build_reason_udf(col("skill_score"), col("education_score")))

    # ── 输出到 HDFS（Parquet + CSV）──────────────────────────────────────────
    output_cols = [
        col("rank"),
        col("resume_id_r").alias("resume_id"),
        col("resume_name"),
        col("job_id_j").alias("job_id"),
        col("job_title_alias").alias("job_title"),
        col("company"),
        col("city"),
        col("skills").alias("resume_skills"),
        col("required_skills"),
        col("preferred_skills"),
        col("skill_score"),
        col("education_score"),
        col("experience_score"),
        col("city_score"),
        col("cert_score"),
        col("tfidf_sim"),
        col("jaccard_sim"),
        col("semantic_sim"),
        col("final_score"),
        col("matched_skills"),
        col("missing_skills"),
        col("extra_skills"),
        col("recommendation_reason"),
    ]

    result_df = top_matches.select(*output_cols)
    result_df.write.mode("overwrite").parquet(
        "hdfs:///resume_matching/results/spark_matches.parquet")
    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        "hdfs:///resume_matching/results/spark_matches_csv")

    count = result_df.count()
    print(f"PySpark 匹配完成：{count} 条结果")
    print(f"  语义权重：TF-IDF {SEMANTIC_WEIGHTS['tfidf']} + Jaccard {SEMANTIC_WEIGHTS['jaccard']}")
    print(f"  综合分权重：技能{WEIGHTS['skill']} + TF-IDF{WEIGHTS['tfidf']} + Jaccard{WEIGHTS['jaccard']} "
          f"+ 学历{WEIGHTS['education']} + 经验{WEIGHTS['experience']} + 城市{WEIGHTS['city']} + 证书{WEIGHTS['certificate']}")
    spark.stop()


if __name__ == "__main__":
    run()
