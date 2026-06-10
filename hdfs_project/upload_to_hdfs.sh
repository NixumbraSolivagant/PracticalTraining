#!/bin/bash
# upload_to_hdfs.sh — 上传本地数据到 HDFS（环境变量由 .env 注入或系统环境提供）
set -e

# 从 .env 加载环境变量（如果存在）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

# 设置 JAVA_HOME 和 HADOOP_HOME（由环境变量或默认值提供）
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/jdk-11}"
export HADOOP_HOME="${HADOOP_HOME:-/usr/local/hadoop}"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH"

HDFS_USER="${HDFS_USER:-$(whoami)}"

echo "=== HDFS 上传脚本 ==="
echo "JAVA_HOME: $JAVA_HOME"
echo "HADOOP_HOME: $HADOOP_HOME"
echo "HDFS_USER: $HDFS_USER"
echo ""

# 创建项目目录
hdfs dfs -mkdir -p /resume_matching/raw_data
hdfs dfs -mkdir -p /resume_matching/cleaned_data
hdfs dfs -mkdir -p /resume_matching/results

# 上传原始数据
DATA_DIR="$(cd "$SCRIPT_DIR/../data" && pwd)"
hdfs dfs -put "$DATA_DIR/resumes.csv" /resume_matching/raw_data/
hdfs dfs -put "$DATA_DIR/jobs.csv" /resume_matching/raw_data/

echo "=== 验证上传 ==="
hdfs dfs -ls /resume_matching/raw_data
hdfs dfs -ls /resume_matching/results
echo "=== 上传完成 ==="
