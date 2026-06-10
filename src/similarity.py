"""src/similarity.py — TF-IDF + Jaccard 相似度 + 矩阵缓存（缓存失效机制）"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config


# ── 缓存失效机制 ───────────────────────────────────────────────────────────────

def _cache_meta_path() -> Path:
    p = config.SIM_CACHE_DIR / ".cache_meta"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache_meta() -> dict:
    p = _cache_meta_path()
    if not p.exists():
        return {"sources": {}, "version": "1.0"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": {}, "version": "1.0"}


def _save_cache_meta(meta: dict):
    p = _cache_meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _cache_valid(resume_path: Path, job_path: Path) -> bool:
    """检查缓存是否有效：所有源文件修改时间早于缓存文件"""
    meta = _load_cache_meta()
    cached_version = meta.get("version")
    if cached_version != "1.0":
        return False
    sources = meta.get("sources", {})
    for path_key, mtime_str in sources.items():
        p = Path(path_key)
        if not p.exists():
            return False
        if abs(p.stat().st_mtime - float(mtime_str)) > 0.5:
            return False
    # 确保所有期望的缓存文件都存在
    for fname in ["tfidf_sim_matrix.npy", "jaccard_sim_matrix.npy",
                  "semantic_sim_matrix.npy", "resume_tfidf_matrix.npy",
                  "job_tfidf_matrix.npy", "vectorizer.pkl"]:
        if not (config.SIM_CACHE_DIR / fname).exists():
            return False
    return True


def _invalidate_cache(resume_path: Path, job_path: Path):
    """清除旧缓存并保存新的源文件 mtime"""
    for fname in ["tfidf_sim_matrix.npy", "jaccard_sim_matrix.npy",
                  "semantic_sim_matrix.npy", "resume_tfidf_matrix.npy",
                  "job_tfidf_matrix.npy", "vectorizer.pkl"]:
        fpath = config.SIM_CACHE_DIR / fname
        if fpath.exists():
            fpath.unlink()
    _save_cache_meta({
        "version": "1.0",
        "sources": {
            str(resume_path): str(resume_path.stat().st_mtime),
            str(job_path): str(job_path.stat().st_mtime),
        }
    })


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def compute_tfidf_matrix(resume_texts: list[str], job_texts: list[str]):
    """构建 TF-IDF 矩阵，返回 resume×job 相似度矩阵"""
    all_texts = resume_texts + job_texts
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_CONFIG["max_features"],
        ngram_range=config.TFIDF_CONFIG["ngram_range"],
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    n_res = len(resume_texts)
    resume_vectors = tfidf_matrix[:n_res]
    job_vectors = tfidf_matrix[n_res:]
    sim_matrix = cosine_similarity(resume_vectors, job_vectors)
    return sim_matrix, resume_vectors, job_vectors, vectorizer


def compute_jaccard_matrix(
    resume_skills: list[list[str]],
    job_skills: list[list[str]],
) -> np.ndarray:
    """计算技能 Jaccard 系数矩阵"""
    n_res = len(resume_skills)
    n_job = len(job_skills)
    matrix = np.zeros((n_res, n_job), dtype=np.float32)
    for i, r_skills in enumerate(resume_skills):
        r_set = set(r_skills)
        if not r_set:
            continue
        for j, j_skills in enumerate(job_skills):
            j_set = set(j_skills)
            if not j_set:
                continue
            inter = len(r_set & j_set)
            union = len(r_set | j_set)
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


def compute_semantic_matrix(
    tfidf_sim: np.ndarray,
    jaccard_sim: np.ndarray,
) -> np.ndarray:
    """融合 TF-IDF 和 Jaccard 为综合语义分（0-100）"""
    tfidf_w = config.SEMANTIC_WEIGHTS["tfidf"]
    jaccard_w = config.SEMANTIC_WEIGHTS["jaccard"]
    combined = tfidf_sim * tfidf_w + jaccard_sim * jaccard_w
    return combined * 100


def run():
    print("=== 相似度计算 ===")
    resume_path = config.CLEANED_DIR / "preprocessed_resumes.csv"
    job_path   = config.CLEANED_DIR / "preprocessed_jobs.csv"

    resumes = pd.read_csv(resume_path)
    jobs    = pd.read_csv(job_path)

    # 检查缓存
    if _cache_valid(resume_path, job_path):
        print("缓存命中，直接加载...")
        tfidf_sim   = np.load(config.SIM_CACHE_DIR / "tfidf_sim_matrix.npy")
        jaccard_sim = np.load(config.SIM_CACHE_DIR / "jaccard_sim_matrix.npy")
        semantic_sim = np.load(config.SIM_CACHE_DIR / "semantic_sim_matrix.npy")
        print("加载完成")
        return tfidf_sim, jaccard_sim, semantic_sim, resumes, jobs

    print("缓存失效，重新计算...")
    # TF-IDF
    resume_texts = resumes["text_joined"].tolist()
    job_texts    = jobs["text_joined"].tolist()
    tfidf_sim, _, _, vectorizer = compute_tfidf_matrix(resume_texts, job_texts)

    # Jaccard
    resume_skills = [str(s).split(";") for s in resumes["skills_list"]]
    job_skills    = [str(s).split(";") for s in jobs["required_skills_list"]]
    jaccard_sim = compute_jaccard_matrix(resume_skills, job_skills)

    # 综合语义分
    semantic_sim = compute_semantic_matrix(tfidf_sim, jaccard_sim)

    # 保存缓存
    _invalidate_cache(resume_path, job_path)
    np.save(config.SIM_CACHE_DIR / "tfidf_sim_matrix.npy", tfidf_sim)
    np.save(config.SIM_CACHE_DIR / "jaccard_sim_matrix.npy", jaccard_sim)
    np.save(config.SIM_CACHE_DIR / "semantic_sim_matrix.npy", semantic_sim)

    import joblib
    joblib.dump(vectorizer, config.SIM_CACHE_DIR / "vectorizer.pkl")

    print(f"TF-IDF 相似度矩阵: {tfidf_sim.shape}")
    print(f"Jaccard 相似度矩阵: {jaccard_sim.shape}")
    print(f"语义相似度矩阵: {semantic_sim.shape}")
    print("缓存已保存")
    return tfidf_sim, jaccard_sim, semantic_sim, resumes, jobs


if __name__ == "__main__":
    run()
