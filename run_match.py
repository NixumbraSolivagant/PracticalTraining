"""run_match.py — 本地 pipeline 一键入口"""
import sys
import os

# 将项目根目录加入路径
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import clean, structure, preprocess, similarity, scoring, matcher


def main():
    print("=" * 60)
    print("简历-岗位匹配系统 · 本地 Pipeline")
    print("=" * 60)

    # Step 1: 数据清洗
    print("\n[1/5] 脏数据清洗...")
    resumes, jobs = clean.run()

    # Step 2: 字段结构化
    print("\n[2/5] 字段结构化...")
    resumes, jobs = structure.run()

    # Step 3: 文本预处理
    print("\n[3/5] 文本预处理...")
    resumes, jobs = preprocess.run()

    # Step 4: 相似度计算 + 综合评分与匹配（matcher 内部会调用 similarity）
    print("\n[4/5] 相似度计算 + 综合评分与匹配...")
    matches_df, explains_df = matcher.run()

    print("\n" + "=" * 60)
    print(f"✅ Pipeline 完成！")
    print(f"   简历 {len(resumes)} 条 × 岗位 {len(jobs)} 条")
    print(f"   匹配结果 {len(matches_df)} 条")
    print(f"   综合分范围：{matches_df['final_score'].min():.1f} ~ {matches_df['final_score'].max():.1f}")
    print(f"\n   输出文件：")
    print(f"   - output/top_matches.csv")
    print(f"   - output/explainable_match_result.csv")
    print(f"\n   启动 Web UI：")
    print(f"   streamlit run app.py --server.port 8501")
    print("=" * 60)


if __name__ == "__main__":
    main()
