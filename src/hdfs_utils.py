"""src/hdfs_utils.py — HDFS 操作工具封装

提供 HDFS 连接检查、文件上传下载、目录列表等功能。
当 Hadoop 未启动时所有操作优雅降级，不阻塞主流程。
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


def _hdfs_cmd(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """执行 hdfs dfs 命令，返回 (returncode, stdout, stderr)"""
    env = os.environ.copy()
    env.setdefault("JAVA_HOME", os.getenv("JAVA_HOME", "/usr/lib/jvm/jdk-11"))
    env.setdefault("HADOOP_HOME", os.getenv("HADOOP_HOME", "/usr/local/hadoop"))
    env.setdefault("PATH",
        f"{env['HADOOP_HOME']}/bin:{env.get('PATH', '')}")

    try:
        result = subprocess.run(
            ["hdfs", "dfs"] + args,
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", "hdfs command not found"
    except subprocess.TimeoutExpired:
        return -2, "", "hdfs command timed out"
    except Exception as e:
        return -3, "", str(e)


def check_connection(timeout: int = 10) -> bool:
    """检查 HDFS 是否可连接"""
    rc, _, _ = _hdfs_cmd(["-ls", "/"], timeout=timeout)
    return rc == 0


def list_dir(hdfs_path: str, timeout: int = 15) -> list[str]:
    """列出 HDFS 目录内容，返回文件名列表"""
    rc, stdout, _ = _hdfs_cmd(["-ls", hdfs_path], timeout=timeout)
    if rc != 0:
        return []
    lines = stdout.strip().splitlines()
    # 跳过第一行（total ...）和标题行
    files = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 8:
            files.append(parts[-1])
    return files


def upload_file(local_path: str | Path, hdfs_path: str, overwrite: bool = True) -> bool:
    """
    上传本地文件到 HDFS。

    Args:
        local_path: 本地文件路径
        hdfs_path: HDFS 目标路径（如 /resume_matching/raw_data/file.csv）
        overwrite: 是否覆盖（默认 True）

    Returns:
        成功返回 True，失败返回 False
    """
    local_path = str(local_path)
    if not os.path.exists(local_path):
        print(f"[HDFS] 本地文件不存在：{local_path}")
        return False

    args = ["-put"]
    if overwrite:
        args += ["-f"]
    args += [local_path, hdfs_path]

    rc, stdout, stderr = _hdfs_cmd(args)
    if rc == 0:
        print(f"[HDFS] 上传成功：{local_path} -> {hdfs_path}")
        return True
    else:
        print(f"[HDFS] 上传失败（{rc}）：{stderr.strip()}")
        return False


def download_file(hdfs_path: str, local_path: str | Path, merge: bool = False) -> bool:
    """
    从 HDFS 下载文件到本地。

    Args:
        hdfs_path: HDFS 文件路径
        local_path: 本地目标路径
        merge: 是否合并多个分片（hdfs -getmerge 模式，用于 CSV）

    Returns:
        成功返回 True，失败返回 False
    """
    local_path = str(local_path)

    if merge:
        # 使用 -getmerge 合并下载（适合 CSV 结果文件）
        rc, _, stderr = _hdfs_cmd(["-getmerge", hdfs_path, local_path])
    else:
        rc, _, stderr = _hdfs_cmd(["-get", hdfs_path, local_path])

    if rc == 0:
        print(f"[HDFS] 下载成功：{hdfs_path} -> {local_path}")
        return True
    else:
        print(f"[HDFS] 下载失败（{rc}）：{stderr.strip()}")
        return False


def mkdir(hdfs_path: str, parents: bool = True) -> bool:
    """在 HDFS 上创建目录"""
    args = ["-mkdir"]
    if parents:
        args += ["-p"]
    args.append(hdfs_path)
    rc, _, _ = _hdfs_cmd(args)
    return rc == 0


def exists(hdfs_path: str, timeout: int = 10) -> bool:
    """检查 HDFS 文件或目录是否存在"""
    rc, _, _ = _hdfs_cmd(["-test", "-e", hdfs_path], timeout=timeout)
    return rc == 0
