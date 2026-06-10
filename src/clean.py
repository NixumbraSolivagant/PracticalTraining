"""src/clean.py — 脏数据清洗"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from . import config


def clean_resumes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. 去掉重复 ID
    df.drop_duplicates(subset=["resume_id"], keep="first", inplace=True)
    df.dropna(subset=["resume_id"], inplace=True)

    # 2. 清洗 salary
    def parse_salary(v):
        if pd.isna(v) or str(v).strip() in ("", "面议", "未知"):
            return np.nan
        v = str(v).replace("元/月", "").replace("元", "").replace(",", "").strip()
        nums = re.findall(r"\d+", v)
        if not nums:
            return np.nan
        if "以上" in str(v):
            return float(nums[0])
        if "-" in v:
            return (float(nums[0]) + float(nums[1])) / 2
        return float(nums[0])

    df["expected_salary"] = df["expected_salary"].apply(parse_salary)

    # 3. 清洗 experience_years
    def parse_years(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v in ("", "不限", "不限", "无要求"):
            return np.nan
        nums = re.findall(r"\d+", v)
        return float(nums[0]) if nums else np.nan

    df["experience_years"] = df["experience_years"].apply(parse_years)

    # 4. 清洗 age
    def parse_age(v):
        if pd.isna(v):
            return np.nan
        nums = re.findall(r"\d+", str(v))
        if not nums:
            return np.nan
        age = float(nums[0])
        return age if 18 <= age <= 65 else np.nan

    df["age"] = df["age"].apply(parse_age)

    # 5. 清洗 education
    valid_edu = list(config.EDUCATION_LEVELS.keys())
    df["education"] = df["education"].apply(
        lambda x: x if str(x).strip() in valid_edu else "本科"
        if pd.isna(x) else x
    )

    # 6. 清洗 skills（去除空白、分号统一）
    df["skills"] = df["skills"].apply(
        lambda x: ";".join(
            s.strip() for s in str(x).replace("；", ";").replace(",", ";").split(";")
            if s.strip()
        ) if pd.notna(x) else ""
    )

    # 7. 城市标准化（简历用 location 列，岗位用 city 列）
    if "location" in df.columns:
        df["city"] = df["location"].apply(lambda x: config.CITY_ALIAS.get(str(x).strip(), "未知"))
    elif "city" in df.columns:
        df["city"] = df["city"].apply(lambda x: config.CITY_ALIAS.get(str(x).strip(), "未知"))

    return df.reset_index(drop=True)


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. 去掉重复 ID
    df.drop_duplicates(subset=["job_id"], keep="first", inplace=True)
    df.dropna(subset=["job_id"], inplace=True)

    # 2. 清洗 salary
    def parse_job_salary(v, field="min"):
        if pd.isna(v) or str(v).strip() in ("", "面议", "不限"):
            return np.nan
        v = str(v).replace("元/月", "").replace("元", "").replace(",", "").strip()
        nums = re.findall(r"\d+", v)
        if not nums:
            return np.nan
        if "-" in v and len(nums) >= 2:
            return float(nums[0]) if field == "min" else float(nums[1])
        return float(nums[0])

    df["salary_min"] = df["salary_min"].apply(lambda x: parse_job_salary(x, "min"))
    df["salary_max"] = df["salary_max"].apply(lambda x: parse_job_salary(x, "max"))

    # 3. 清洗 min_experience_years
    def parse_exp(v):
        if pd.isna(v) or str(v).strip() in ("", "不限", "无要求", "经验不限"):
            return 0.0
        nums = re.findall(r"\d+", str(v))
        return float(nums[0]) if nums else 0.0

    df["min_experience_years"] = df["min_experience_years"].apply(parse_exp)

    # 4. 清洗 required_skills / preferred_skills / preferred_certificates
    for col in ["required_skills", "preferred_skills"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ";".join(
                    s.strip() for s in str(x).replace("；", ";").replace(",", ";").split(";")
                    if s.strip()
                ) if pd.notna(x) else ""
            )
    # 清洗证书偏好列
    if "preferred_certificates" in df.columns:
        df["preferred_certificates"] = df["preferred_certificates"].apply(
            lambda x: ";".join(
                s.strip() for s in str(x).replace("；", ";").replace(",", ";").split(";")
                if s.strip()
            ) if pd.notna(x) else ""
        )

    # 5. 城市标准化
    df["city"] = df["city"].apply(lambda x: config.CITY_ALIAS.get(str(x).strip(), "未知"))

    # 6. 清洗 required_education
    valid_edu = list(config.EDUCATION_LEVELS.keys())
    df["required_education"] = df["required_education"].apply(
        lambda x: x if str(x).strip() in valid_edu else "本科"
        if pd.isna(x) else x
    )

    # 7. 清洗 job_type
    df["job_type"] = df["job_type"].fillna("全职")

    return df.reset_index(drop=True)


REQUIRED_RESUME_COLS = ["resume_id", "name", "education", "skills", "experience_years", "location"]
REQUIRED_JOB_COLS = ["job_id", "job_title", "required_skills", "min_experience_years", "city"]


def validate_schema(df: pd.DataFrame, required_cols: list[str], name: str) -> list[str]:
    """检查必要列是否存在，返回缺失列列表"""
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def run():
    print("=== 数据清洗 ===")
    resumes = pd.read_csv(config.DATA_DIR / "resumes.csv")
    jobs = pd.read_csv(config.DATA_DIR / "jobs.csv")

    # Schema 验证
    missing_resume = validate_schema(resumes, REQUIRED_RESUME_COLS, "简历")
    missing_job = validate_schema(jobs, REQUIRED_JOB_COLS, "岗位")
    if missing_resume:
        print(f"警告：简历数据缺失列 {missing_resume}，这些列将被跳过")
    if missing_job:
        print(f"警告：岗位数据缺失列 {missing_job}，这些列将被跳过")

    resumes_clean = clean_resumes(resumes)
    jobs_clean = clean_jobs(jobs)

    out_dir = config.CLEANED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    resumes_clean.to_csv(out_dir / "cleaned_resumes.csv", index=False)
    jobs_clean.to_csv(out_dir / "cleaned_jobs.csv", index=False)

    print(f"简历清洗完成：{len(resumes_clean)} 条 → {out_dir / 'cleaned_resumes.csv'}")
    print(f"岗位清洗完成：{len(jobs_clean)} 条 → {out_dir / 'cleaned_jobs.csv'}")
    return resumes_clean, jobs_clean


if __name__ == "__main__":
    run()
