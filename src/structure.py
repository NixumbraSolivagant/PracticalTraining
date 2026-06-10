"""src/structure.py — 字段结构化（从清洗后数据中提取结构化字段）"""
import pandas as pd
import json
from pathlib import Path
from . import config


def standardize_skill(skill_str: str, alias_map: dict) -> list[str]:
    """技能标准化：将原始技能字符串映射为标准技能列表"""
    if not skill_str or pd.isna(skill_str):
        return []
    raw = [s.strip() for s in str(skill_str).replace("；", ";").replace(",", ";").split(";") if s.strip()]
    standardized = []
    for s in raw:
        mapped = alias_map.get(s, alias_map.get(s.lower(), s))
        if mapped and mapped not in standardized:
            standardized.append(mapped)
    return standardized


def structure_resumes(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    df = df.copy()
    # 技能标准化
    df["skills_std"] = df["skills"].apply(lambda x: standardize_skill(x, alias_map))
    df["skills_list"] = df["skills_std"].apply(lambda x: ";".join(x))
    # 证书列表
    df["cert_list"] = df["certifications"].apply(
        lambda x: [s.strip() for s in str(x).replace("；", ";").replace(",", ";").split(";") if s.strip()]
        if pd.notna(x) else []
    )
    df["cert_list_str"] = df["cert_list"].apply(lambda x: ";".join(x))
    # 学历等级
    df["edu_level"] = df["education"].map(config.EDUCATION_LEVELS).fillna(3)
    # 薪资（已有，标准化为数字）
    df["expected_salary_num"] = pd.to_numeric(df["expected_salary"], errors="coerce").fillna(0)
    # 城市标准化（已有）
    return df


def structure_jobs(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    df = df.copy()
    # 必备技能标准化
    df["required_skills_std"] = df["required_skills"].apply(lambda x: standardize_skill(x, alias_map))
    df["required_skills_list"] = df["required_skills_std"].apply(lambda x: ";".join(x))
    # 加分技能标准化
    df["preferred_skills_std"] = df["preferred_skills"].apply(lambda x: standardize_skill(x, alias_map))
    df["preferred_skills_list"] = df["preferred_skills_std"].apply(lambda x: ";".join(x))
    # 证书列表
    cert_col = "preferred_certificates" if "preferred_certificates" in df.columns else "preferred_certifications"
    df["pref_certs_list"] = df[cert_col].apply(
        lambda x: [s.strip() for s in str(x).replace("；", ";").replace(",", ";").split(";") if s.strip()]
        if pd.notna(x) else []
    )
    df["pref_certs_str"] = df["pref_certs_list"].apply(lambda x: ";".join(x))
    # 学历等级
    df["required_edu_level"] = df["required_education"].map(config.EDUCATION_LEVELS).fillna(3)
    # 薪资数字
    df["salary_min_num"] = pd.to_numeric(df["salary_min"], errors="coerce").fillna(0)
    df["salary_max_num"] = pd.to_numeric(df["salary_max"], errors="coerce").fillna(0)
    # 经验要求数字
    df["min_exp_num"] = pd.to_numeric(df["min_experience_years"], errors="coerce").fillna(0)
    return df


def run():
    print("=== 字段结构化 ===")
    alias_path = config.DATA_DIR / "skill_alias.json"
    alias_map = json.loads(alias_path.read_text(encoding="utf-8"))

    resumes = pd.read_csv(config.CLEANED_DIR / "cleaned_resumes.csv")
    jobs = pd.read_csv(config.CLEANED_DIR / "cleaned_jobs.csv")

    resumes_std = structure_resumes(resumes, alias_map)
    jobs_std = structure_jobs(jobs, alias_map)

    resumes_std.to_csv(config.CLEANED_DIR / "structured_resumes.csv", index=False)
    jobs_std.to_csv(config.CLEANED_DIR / "structured_jobs.csv", index=False)

    print(f"简历结构化完成：{len(resumes_std)} 条")
    print(f"岗位结构化完成：{len(jobs_std)} 条")
    return resumes_std, jobs_std


if __name__ == "__main__":
    run()
