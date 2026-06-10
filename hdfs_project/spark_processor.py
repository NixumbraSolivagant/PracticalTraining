"""hdfs_project/spark_processor.py — PySpark ETL（清洗 + 写回 HDFS）"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, trim, lower, regexp_replace, when, split, size
from pyspark.sql.types import StringType, DoubleType, IntegerType


def get_spark(app_name: str = "ResumeMatching-ETL") -> SparkSession:
    spark_home = os.getenv("SPARK_HOME", "/usr/local/spark")
    java_home = os.getenv("JAVA_HOME", "/usr/lib/jvm/jdk-11")
    hadoop_conf = os.getenv("HADOOP_CONF_DIR", "/usr/local/hadoop/etc/hadoop")

    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]")
               .config("spark.driver.memory", "2g")
               .config("spark.executor.memory", "2g")
               .config("spark.sql.shuffle.partitions", "8"))

    if os.path.exists(os.path.join(spark_home, "python/pyspark")):
        builder = builder.config("spark.pyspark.python", "python3")

    return builder.getOrCreate()


def clean_resume_df(df):
    """Spark版简历清洗"""
    from pyspark.sql.functions import when
    return (df
        .dropDuplicates(["resume_id"])
        .filter(col("resume_id").isNotNull())
        .withColumn("expected_salary",
            when(col("expected_salary").isNull(), 0.0)
            .otherwise(col("expected_salary").cast("double")))
        .withColumn("experience_years",
            when(col("experience_years").isNull(), 0.0)
            .otherwise(col("experience_years").cast("double")))
        .withColumn("skills", trim(regexp_replace(col("skills"), r"\s+", " ")))
        # 统一列名：resume location → city
        .withColumn("city", when(col("location").isNotNull(), col("location")).otherwise(lit("未知")))
    )


def clean_job_df(df):
    """Spark版岗位清洗"""
    return (df
        .dropDuplicates(["job_id"])
        .filter(col("job_id").isNotNull())
        .withColumn("salary_min",
            when(col("salary_min").isNull(), 0.0)
            .otherwise(col("salary_min").cast("double")))
        .withColumn("salary_max",
            when(col("salary_max").isNull(), 0.0)
            .otherwise(col("salary_max").cast("double")))
        .withColumn("min_experience_years",
            when(col("min_experience_years").isNull(), 0.0)
            .otherwise(col("min_experience_years").cast("double")))
        .withColumn("required_skills", trim(regexp_replace(col("required_skills"), r"\s+", " ")))
    )


def run():
    print("=== PySpark ETL ===")
    spark = get_spark()

    # 读取 HDFS 数据
    resumes = spark.read.csv("hdfs:///resume_matching/raw_data/resumes.csv", header=True, inferSchema=True)
    jobs = spark.read.csv("hdfs:///resume_matching/raw_data/jobs.csv", header=True, inferSchema=True)

    resumes_clean = clean_resume_df(resumes)
    jobs_clean = clean_job_df(jobs)

    # 写回 HDFS
    resumes_clean.write.mode("overwrite").parquet("hdfs:///resume_matching/cleaned_data/cleaned_resumes.parquet")
    jobs_clean.write.mode("overwrite").parquet("hdfs:///resume_matching/cleaned_data/cleaned_jobs.parquet")

    print(f"简历 ETL 完成：{resumes_clean.count()} 条")
    print(f"岗位 ETL 完成：{jobs_clean.count()} 条")
    spark.stop()


if __name__ == "__main__":
    run()
