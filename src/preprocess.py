"""src/preprocess.py — 文本预处理：分词 + 停用词过滤"""
import jieba
import pandas as pd
import json
from pathlib import Path
from . import config


_stopwords: set[str] | None = None
_skill_dict_loaded: bool = False


def load_skill_dict() -> None:
    """从 skill_alias.json 加载所有标准化技能词到 jieba 自定义词典，提升专业术语分词精度"""
    global _skill_dict_loaded
    if _skill_dict_loaded:
        return
    alias_path = config.DATA_DIR / "skill_alias.json"
    if alias_path.exists():
        try:
            alias_map = json.loads(alias_path.read_text(encoding="utf-8"))
            unique_skills = set(alias_map.values())
            for skill in unique_skills:
                if skill and len(skill) >= 2:
                    jieba.add_word(skill, freq=100, tag="nz")
            print(f"技能词典加载完成：{len(unique_skills)} 个技能词")
        except Exception as e:
            print(f"警告：技能词典加载失败 ({e})，分词可能不准确")
    _skill_dict_loaded = True


def load_stopwords(path: Path | None = None) -> set[str]:
    global _stopwords
    if _stopwords is not None:
        return _stopwords
    path = path or (config.DATA_DIR / "stopwords.txt")
    _stopwords = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w:
                _stopwords.add(w)
    return _stopwords


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """结巴分词 + 停用词过滤"""
    if not text or pd.isna(text):
        return []
    load_skill_dict()
    sw = stopwords or load_stopwords()
    return [w.strip() for w in jieba.cut(str(text)) if w.strip() and w.strip() not in sw and len(w.strip()) > 1]


def preprocess_resumes(df: pd.DataFrame, stopwords: set[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    sw = stopwords or load_stopwords()

    def combine_text(row) -> str:
        parts = [
            str(row.get("name", "")),
            str(row.get("major", "")),
            str(row.get("skills", "")),
            str(row.get("certifications", "")),
            str(row.get("self_description", "")),
            str(row.get("project_experience", "")),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    df["text_combined"] = df.apply(combine_text, axis=1)
    df["text_tokenized"] = df["text_combined"].apply(lambda x: tokenize(x, sw))
    df["text_joined"] = df["text_tokenized"].apply(lambda x: " ".join(x))
    return df


def preprocess_jobs(df: pd.DataFrame, stopwords: set[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    sw = stopwords or load_stopwords()

    def combine_text(row) -> str:
        parts = [
            str(row.get("job_title", "")),
            str(row.get("job_description", "")),
            str(row.get("required_skills", "")),
            str(row.get("responsibilities", "")),
            str(row.get("benefits", "")),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    df["text_combined"] = df.apply(combine_text, axis=1)
    df["text_tokenized"] = df["text_combined"].apply(lambda x: tokenize(x, sw))
    df["text_joined"] = df["text_tokenized"].apply(lambda x: " ".join(x))
    return df


def run():
    print("=== 文本预处理 ===")
    load_skill_dict()
    sw = load_stopwords()
    print(f"停用词加载完成：{len(sw)} 条")

    resumes = pd.read_csv(config.CLEANED_DIR / "structured_resumes.csv")
    jobs = pd.read_csv(config.CLEANED_DIR / "structured_jobs.csv")

    resumes_pp = preprocess_resumes(resumes, sw)
    jobs_pp = preprocess_jobs(jobs, sw)

    resumes_pp.to_csv(config.CLEANED_DIR / "preprocessed_resumes.csv", index=False)
    jobs_pp.to_csv(config.CLEANED_DIR / "preprocessed_jobs.csv", index=False)

    print(f"简历预处理完成：{len(resumes_pp)} 条")
    print(f"岗位预处理完成：{len(jobs_pp)} 条")
    return resumes_pp, jobs_pp


if __name__ == "__main__":
    run()
