"""run_spark.py — HDFS + PySpark pipeline 一键入口"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    print("=" * 60)
    print("简历-岗位匹配系统 · HDFS + PySpark Pipeline")
    print("=" * 60)

    # Step 1: 上传数据到 HDFS
    print("\n[1/3] 上传数据到 HDFS...")
    import subprocess
    script_dir = os.path.join(ROOT, "hdfs_project")
    result = subprocess.run(
        ["bash", os.path.join(script_dir, "upload_to_hdfs.sh")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"⚠️  HDFS 上传失败（可能 Hadoop 未启动）：\n{result.stderr}")
    else:
        print("✅ HDFS 上传完成")

    # Step 2: PySpark ETL 清洗
    print("\n[2/3] PySpark ETL 清洗...")
    try:
        from hdfs_project import spark_processor
        spark_processor.run()
    except ImportError as e:
        print(f"⚠️  无法导入 PySpark 模块：{e}")
        print("   请确保已安装 pyspark：pip install pyspark")
        return

    # Step 3: PySpark MLlib 匹配
    print("\n[3/4] PySpark MLlib 分布式匹配...")
    try:
        from hdfs_project import spark_matcher
        spark_matcher.run()
    except ImportError as e:
        print(f"⚠️  无法导入 PySpark 模块：{e}")
        return

    # Step 4: 下载 Spark CSV 结果到本地 output 目录（供 Streamlit UI 使用）
    print("\n[4/4] 下载 Spark 结果到本地...")
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_hdfs_dir = "/resume_matching/results/spark_matches_csv"
    local_path = os.path.join(output_dir, "spark_top_matches.csv")
    sys.path.insert(0, ROOT)
    try:
        from src import hdfs_utils
        success = hdfs_utils.download_file(csv_hdfs_dir, local_path, merge=True)
        if success:
            print(f"✅ Spark 结果已下载到 {local_path}")
        else:
            print(f"⚠️  下载失败（Spark结果可能不存在），跳过")
    except ImportError as e:
        print(f"⚠️  无法导入 hdfs_utils（{e}），尝试直接用 hdfs 命令")
        try:
            result = subprocess.run(
                ["hdfs", "dfs", "-getmerge", csv_hdfs_dir, local_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                print(f"✅ Spark 结果已下载到 {local_path}")
        except FileNotFoundError:
            print("⚠️  hdfs 命令未找到，跳过下载")
        except Exception as e:
            print(f"⚠️  下载异常：{e}")

    print("\n" + "=" * 60)
    print("✅ Spark Pipeline 完成！")
    print(f"   结果已写入 HDFS：hdfs:///resume_matching/results/")
    print(f"   本地结果：output/spark_top_matches.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
