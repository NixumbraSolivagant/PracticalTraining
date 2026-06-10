"""src/scoring.py — 6维规则评分 + 推荐理由生成"""
import numpy as np
import pandas as pd
from . import config


# ── 薪资评分（平滑函数）────────────────────────────────────────────────────────

def score_salary(resume_expect: float, job_min: float, job_max: float) -> float:
    """
    平滑薪资评分：区间内100分，超出按比例扣分
    - 低于期望（岗位好）：每超10%加50分（封底0）
    - 高于期望（岗位差）：每超10%扣30分（比低估更宽容，体现可谈空间）
    """
    if job_min <= 0 or job_max <= 0 or resume_expect <= 0:
        return 60.0  # 信息不全时给中性分
    if job_min <= resume_expect <= job_max:
        return 100.0
    if resume_expect < job_min:
        gap = (job_min - resume_expect) / resume_expect
        score = 100.0 - gap * 500
        return max(0.0, min(100.0, score))
    # resume_expect > job_max
    gap = (resume_expect - job_max) / job_max
    score = 100.0 - gap * 300
    return max(0.0, min(100.0, score))


# ── 学历评分 ─────────────────────────────────────────────────────────────────

def score_education(resume_edu_level: float, job_required_level: float) -> float:
    if resume_edu_level >= job_required_level:
        return 100.0
    diff = job_required_level - resume_edu_level
    return max(0.0, 100.0 - diff * 25.0)


# ── 经验评分 ────────────────────────────────────────────────────────────────

def score_experience(resume_years: float, job_min_years: float) -> float:
    if pd.isna(resume_years) or pd.isna(job_min_years):
        return 60.0
    if job_min_years <= 0:
        return 100.0
    if resume_years >= job_min_years:
        return 100.0
    gap = (job_min_years - resume_years) / job_min_years
    score = 100.0 - gap * 100
    return max(0.0, min(100.0, score))


# ── 城市评分 ────────────────────────────────────────────────────────────────

def score_city(resume_city: str, job_city: str) -> float:
    if pd.isna(resume_city) or pd.isna(job_city):
        return 60.0
    r = str(resume_city).strip()
    j = str(job_city).strip()
    if r == j:
        return 100.0
    if r in ("未知", "其他", "") or j in ("未知", "其他", ""):
        return 60.0
    return 20.0


# ── 技能评分 ────────────────────────────────────────────────────────────────

def score_skill(
    resume_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
) -> tuple[float, list[str], list[str], list[str]]:
    """
    返回：(score, matched, missing, extra)
    """
    r_set = set(resume_skills)
    req_set = set(required_skills)
    pref_set = set(preferred_skills)

    matched = sorted(r_set & req_set)
    missing_req = req_set - r_set
    extra = sorted(r_set - req_set - pref_set)

    if len(req_set) == 0:
        base = 100.0
    else:
        base = len(matched) / len(req_set) * 100.0

    bonus = min(len(extra) * 5.0, 25.0)
    score = min(base + bonus, 100.0)
    return score, matched, list(missing_req), extra


# ── 证书评分 ────────────────────────────────────────────────────────────────

def score_certificate(resume_certs: list[str], preferred_certs: list[str]) -> float:
    if not preferred_certs:
        return 100.0
    r_certs = set(c.strip() for c in resume_certs if c.strip())
    p_certs = set(c.strip() for c in preferred_certs if c.strip())
    if not p_certs:
        return 100.0
    matched = r_certs & p_certs
    base = len(matched) / len(p_certs) * 100.0
    extra = min((len(r_certs) - len(matched)) * 12.5, 25.0)
    return min(base + extra, 100.0)


# ── 综合评分 ────────────────────────────────────────────────────────────────

def score_pair(
    resume: pd.Series,
    job: pd.Series,
    tfidf_sim: float,
    jaccard_sim: float,
    semantic_sim: float,
) -> dict:
    """对一对 resume-job 计算所有维度分和综合分"""
    r_skills = str(resume.get("skills_list", "")).split(";")
    r_skills = [s.strip() for s in r_skills if s.strip()]
    j_req    = str(job.get("required_skills_list", "")).split(";")
    j_req    = [s.strip() for s in j_req if s.strip()]
    j_pref   = str(job.get("preferred_skills_list", "")).split(";")
    j_pref   = [s.strip() for s in j_pref if s.strip()]

    r_certs = str(resume.get("cert_list_str", "")).split(";")
    r_certs = [s.strip() for s in r_certs if s.strip()]
    j_certs = str(job.get("pref_certs_str", "")).split(";")
    j_certs = [s.strip() for s in j_certs if s.strip()]

    edu_r = float(resume.get("edu_level", 3))
    edu_j = float(job.get("required_edu_level", 3))
    exp_r = float(resume.get("experience_years", 0) if pd.notna(resume.get("experience_years")) else 0)
    exp_j = float(job.get("min_exp_num", 0))

    skill_score, matched, missing, extra = score_skill(r_skills, j_req, j_pref)
    edu_score    = score_education(edu_r, edu_j)
    exp_score    = score_experience(exp_r, exp_j)
    city_score   = score_city(resume.get("city"), job.get("city"))
    cert_score   = score_certificate(r_certs, j_certs)
    sal_score    = score_salary(
        float(resume.get("expected_salary_num", 0)),
        float(job.get("salary_min_num", 0)),
        float(job.get("salary_max_num", 0)),
    )

    # 综合分
    final = (
        skill_score * config.WEIGHTS["skill"] +
        semantic_sim * config.WEIGHTS["tfidf"] +
        jaccard_sim * config.WEIGHTS["jaccard"] +
        edu_score   * config.WEIGHTS["education"] +
        exp_score   * config.WEIGHTS["experience"] +
        city_score  * config.WEIGHTS["city"] +
        cert_score  * config.WEIGHTS["certificate"]
    )

    # 推荐理由
    reason_lines = []
    if matched:
        reason_lines.append(f"技能匹配较高，共同技能包括 {', '.join(matched[:5])}。")
    if missing:
        reason_lines.append(f"建议补充技能：{', '.join(missing[:5])}，以提高岗位竞争力。")
    if skill_score >= 80:
        reason_lines.append("核心技能与岗位高度匹配，适合推荐。")
    elif skill_score >= 60:
        reason_lines.append("技能基本匹配，部分要求略有差距。")
    else:
        reason_lines.append("技能匹配度较低，建议优先提升相关技能。")
    if edu_score >= 100:
        reason_lines.append("学历满足岗位要求。")
    elif edu_score >= 50:
        reason_lines.append("学历略低于要求，但可考虑。")
    if sal_score >= 90:
        reason_lines.append("薪资期望与岗位范围匹配良好。")
    reason = "\n".join(reason_lines) if reason_lines else "综合评估一般，建议观望。"

    return {
        "skill_score":        round(skill_score, 2),
        "education_score":    round(edu_score, 2),
        "experience_score":   round(exp_score, 2),
        "city_score":        round(city_score, 2),
        "salary_score":       round(sal_score, 2),
        "cert_score":        round(cert_score, 2),
        "tfidf_sim":         round(tfidf_sim * 100, 2),
        "jaccard_sim":       round(jaccard_sim * 100, 2),
        "semantic_sim":      round(semantic_sim, 2),
        "final_score":        round(final, 2),
        "matched_skills":    ";".join(matched),
        "missing_skills":    ";".join(missing),
        "extra_skills":      ";".join(extra),
        "recommendation_reason": reason,
    }


def run():
    print("=== 评分测试 ===")
    from . import similarity
    tfidf_sim, jaccard_sim, semantic_sim, resumes, jobs = similarity.run()

    sample_r = resumes.iloc[0]
    sample_j = jobs.iloc[0]
    result = score_pair(sample_r, sample_j,
                        tfidf_sim[0, 0], jaccard_sim[0, 0], semantic_sim[0, 0])
    print(f"示例评分：{result}")
    return result


if __name__ == "__main__":
    run()
