"""src/matcher.py — 综合分 + Top-N + 可解释结果（双视角）"""
import pandas as pd
from pathlib import Path
from . import config, similarity, scoring


def build_matches(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    tfidf_sim: pd.DataFrame,
    jaccard_sim: pd.DataFrame,
    semantic_sim: pd.DataFrame,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对所有 resume-job 对计算评分，返回 matches + explains DataFrame
    """
    top_n = top_n or config.TOP_N
    records = []
    explain_records = []

    for i, resume in resumes.iterrows():
        resume_id = resume.get("resume_id", f"R{i:03d}")
        row_tfidf   = tfidf_sim.iloc[i].values if i < len(tfidf_sim) else tfidf_sim.iloc[0].values * 0
        row_jaccard = jaccard_sim.iloc[i].values if i < len(jaccard_sim) else jaccard_sim.iloc[0].values * 0
        row_semantic = semantic_sim.iloc[i].values if i < len(semantic_sim) else semantic_sim.iloc[0].values * 0

        pair_scores = []
        for j, job in jobs.iterrows():
            job_id = job.get("job_id", f"J{j:03d}")
            score_dict = scoring.score_pair(
                resume, job,
                row_tfidf[j], row_jaccard[j], row_semantic[j],
            )
            pair_scores.append((job_id, score_dict["final_score"], score_dict))

        # Top-N
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (job_id, final_score, sd) in enumerate(pair_scores[:top_n], start=1):
            job_row = jobs[jobs.get("job_id", pd.Series()) == job_id]
            if job_row.empty:
                continue
            job = job_row.iloc[0]
            rec = {
                "rank":            rank,
                "resume_id":       resume_id,
                "resume_name":     resume.get("name", ""),
                "job_id":          job_id,
                "job_title":      job.get("job_title", ""),
                "company":         job.get("company", ""),
                "city":            job.get("city", ""),
                "skill_score":     sd["skill_score"],
                "education_score": sd["education_score"],
                "experience_score":sd["experience_score"],
                "city_score":      sd["city_score"],
                "salary_score":    sd["salary_score"],
                "cert_score":      sd["cert_score"],
                "tfidf_sim":       sd["tfidf_sim"],
                "jaccard_sim":     sd["jaccard_sim"],
                "semantic_sim":    sd["semantic_sim"],
                "final_score":     final_score,
            }
            records.append(rec)
            explain_records.append({
                "resume_id":             resume_id,
                "resume_name":           resume.get("name", ""),
                "job_id":                job_id,
                "job_title":             job.get("job_title", ""),
                "company":               job.get("company", ""),
                "matched_skills":        sd["matched_skills"],
                "missing_skills":        sd["missing_skills"],
                "extra_skills":          sd["extra_skills"],
                "recommendation_reason":  sd["recommendation_reason"],
                "skill_score":           sd["skill_score"],
                "education_score":       sd["education_score"],
                "experience_score":      sd["experience_score"],
                "city_score":            sd["city_score"],
                "salary_score":          sd["salary_score"],
                "cert_score":            sd["cert_score"],
                "semantic_sim":          sd["semantic_sim"],
                "final_score":           final_score,
            })

    matches_df = pd.DataFrame(records)
    explains_df = pd.DataFrame(explain_records)
    return matches_df, explains_df


def run():
    print("=== 匹配计算 ===")
    # 先跑相似度
    tfidf_sim, jaccard_sim, semantic_sim, resumes, jobs = similarity.run()

    # 转 DataFrame（便于按行索引取值）
    tfidf_df   = pd.DataFrame(tfidf_sim)
    jaccard_df = pd.DataFrame(jaccard_sim)
    sem_df     = pd.DataFrame(semantic_sim)

    # 评分
    matches_df, explains_df = build_matches(resumes, jobs, tfidf_df, jaccard_df, sem_df)

    # 保存
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(out_dir / "top_matches.csv", index=False)
    explains_df.to_csv(out_dir / "explainable_match_result.csv", index=False)

    print(f"匹配结果：{len(matches_df)} 条 → {out_dir / 'top_matches.csv'}")
    print(f"推荐理由：{len(explains_df)} 条 → {out_dir / 'explainable_match_result.csv'}")
    print(f"综合分范围：{matches_df['final_score'].min():.1f} ~ {matches_df['final_score'].max():.1f}")
    return matches_df, explains_df


if __name__ == "__main__":
    run()
