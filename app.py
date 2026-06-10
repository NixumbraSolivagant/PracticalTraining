"""
app.py — 人岗智能匹配系统
运行命令：streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui_components import (
    render_kpi_cards,
    render_match_table,
    render_score_chart,
    render_overall_score_chart,
    render_reason,
    render_radar_chart,
    render_stats_overview,
    render_hdfs_status,
    render_gauge_chart,
    C,
)


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="人岗智能匹配系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
* {{ font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; }}

/* 页面背景 */
.stApp {{ background: {C['bg_page']}; }}

/* 标题区 */
.page-header {{
    padding: 24px 0 16px;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 28px;
}}
.page-header h1 {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {C['text']};
    margin: 0;
}}
.page-header p {{
    font-size: 0.88rem;
    color: {C['text_muted']};
    margin: 4px 0 0;
}}

/* Tab */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 2px solid {C['border']};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {C['text_muted']};
    padding: 8px 20px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {C['text']}; background: none; }}
.stTabs [aria-selected="true"] {{
    color: {C['primary']} !important;
    border-bottom-color: {C['primary']} !important;
}}

/* 侧边栏 */
[data-testid="stSidebar"] {{
    background: white !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {{ color: {C['text']} !important; }}

/* 标签 */
.tag {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}}
.tag-purple {{ background: {C['purple_light']}; color: {C['purple']}; border: 1px solid #DDD6FE; }}
.tag-blue   {{ background: #E0F2FE; color: #0284C7; border: 1px solid #BAE6FD; }}

/* 分隔线 */
.sep {{
    border: none;
    border-top: 1px solid {C['border']};
    margin: 20px 0;
}}

/* 简历/岗位信息卡 */
.info-strip {{
    background: {C['bg']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.info-strip .field {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}}
.info-strip .field-label {{
    font-size: 11px;
    font-weight: 600;
    color: {C['text_light']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    min-width: 44px;
}}
.info-strip .field-value {{
    font-size: 14px;
    color: {C['text']};
    font-weight: 500;
}}
.skills-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_matches(use_hdfs=False):
    if use_hdfs:
        import subprocess, tempfile, glob
        with tempfile.TemporaryDirectory() as tmpdir:
            r = subprocess.run(
                ["/bin/bash", "-c",
                 f"export JAVA_HOME=/usr/lib/jvm/jdk-11 && "
                 f"export PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$PATH && "
                 f"hdfs dfs -get /resume_matching/results/spark_top_matches {tmpdir}/"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                files = glob.glob(f"{tmpdir}/**/part-*.csv", recursive=True)
                if files:
                    return pd.read_csv(files[0])
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "output", "top_matches.csv"))


@st.cache_data(ttl=3600)
def load_explains(use_hdfs=False):
    if use_hdfs:
        import subprocess, tempfile, glob
        with tempfile.TemporaryDirectory() as tmpdir:
            r = subprocess.run(
                ["/bin/bash", "-c",
                 f"export JAVA_HOME=/usr/lib/jvm/jdk-11 && "
                 f"export PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$PATH && "
                 f"hdfs dfs -get /resume_matching/results/spark_explainable {tmpdir}/"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                files = glob.glob(f"{tmpdir}/**/part-*.csv", recursive=True)
                if files:
                    return pd.read_csv(files[0])
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "output", "explainable_match_result.csv"))


@st.cache_data(ttl=3600)
def load_resumes():
    path = os.path.join(os.path.dirname(__file__), "output", "cleaned", "preprocessed_resumes.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "resumes.csv")
    )


@st.cache_data(ttl=3600)
def load_jobs():
    path = os.path.join(os.path.dirname(__file__), "output", "cleaned", "preprocessed_jobs.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "jobs.csv")
    )


@st.cache_data(ttl=3600)
def load_spark_matches():
    path = os.path.join(os.path.dirname(__file__), "output", "spark_top_matches.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "resume_skills" in df.columns and "skills" not in df.columns:
        df["skills"] = df["resume_skills"]
    return df


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("人岗智能匹配")
    st.markdown("---")

    st.markdown("**数据来源**")
    use_spark = st.radio(
        "数据来源",
        ["本地 CSV", "Spark/HDFS"],
        captions=["使用已有结果文件", "从 HDFS 重新计算"],
        label_visibility="collapsed",
        index=0,
    )
    use_spark = (use_spark == "Spark/HDFS")

    if use_spark:
        with st.spinner("检查 HDFS..."):
            import subprocess
            r = subprocess.run(
                ["/bin/bash", "-c",
                 "export JAVA_HOME=/usr/lib/jvm/jdk-11 && "
                 "export PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$PATH && "
                 "hdfs dfs -ls /resume_matching/results/spark_top_matches"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                st.success("HDFS 结果已就绪")
            else:
                spark_path = os.path.join(os.path.dirname(__file__), "output", "spark_top_matches.csv")
                if not os.path.exists(spark_path):
                    st.info("运行 `python3 run_spark.py` 生成结果")
                    if st.button("重新运行 Spark Pipeline", type="primary"):
                        with st.spinner("运行中..."):
                            result = subprocess.run(
                                [sys.executable, os.path.join(os.path.dirname(__file__), "run_spark.py")],
                                capture_output=True, text=True,
                            )
                            if result.returncode == 0:
                                st.success("Pipeline 完成！请刷新页面。")
                                st.rerun()
                            else:
                                st.error(f"失败：\n{result.stderr[-500:]}")

    st.markdown("---")
    st.markdown("**筛选条件**")
    all_cities = ["南昌", "北京", "上海", "深圳", "杭州", "广州", "成都", "武汉"]
    selected_cities = st.multiselect("期望城市（多选）", all_cities, default=all_cities)
    min_score = st.slider("最低综合分", 0, 100, 0)
    top_n = st.slider("显示排名上限", 3, 15, 10)

    st.markdown("---")
    with st.expander("关于本系统"):
        st.markdown("""
        **简历-岗位人才匹配系统**

        基于 **PySpark MLlib TF-IDF** 语义相似度
        + 分布式多维评分引擎。

        评分维度：技能、TF-IDF、Jaccard、学历、经验、城市、证书
        """)
        render_hdfs_status()


# ---------------------------------------------------------------------------
# 加载数据
# ---------------------------------------------------------------------------
resumes_df = load_resumes()
jobs_df = load_jobs()

if use_spark:
    spark_df = load_spark_matches()
    if not spark_df.empty:
        matches_df = spark_df.copy()
        explains_df = spark_df.copy()
    else:
        matches_df = load_matches(use_hdfs=False)
        explains_df = load_explains(use_hdfs=False)
        if matches_df.empty:
            st.warning("Spark 结果不存在，使用本地 CSV")
else:
    matches_df = load_matches(use_hdfs=False)
    explains_df = load_explains(use_hdfs=False)

if matches_df.empty:
    st.error("未找到匹配结果，请先运行 `python3 run_match.py` 生成数据。")
    st.stop()


# ---------------------------------------------------------------------------
# 页面头部
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="page-header">
    <h1>简历-岗位人才匹配系统</h1>
    <p>基于多维度评分的人岗智能推荐平台 &nbsp;&middot;&nbsp; Powered by PySpark MLlib</p>
</div>
""", unsafe_allow_html=True)

src_tag = (
    f'<span class="tag tag-purple">Spark/HDFS</span>'
    if use_spark else
    f'<span class="tag tag-blue">本地 CSV</span>'
)
st.markdown(
    f"<div style='margin-bottom:20px;'>{src_tag} &nbsp; "
    f"简历 {len(resumes_df)} 条 &nbsp;&middot;&nbsp; "
    f"岗位 {len(jobs_df)} 条 &nbsp;&middot;&nbsp; "
    f"匹配对 {len(matches_df)} 条</div>",
    unsafe_allow_html=True,
)

render_kpi_cards(matches_df, explains_df)

# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
tabs = st.tabs(["综合概览", "学生视角", "HR视角", "全量统计"])

# ========== Tab 0: 综合概览 ==========
with tabs[0]:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"""
        <div class="info-strip">
            <div style="font-size:15px;font-weight:700;color:{C['text']};margin-bottom:12px;">
                评分维度说明
            </div>
        </div>
        """, unsafe_allow_html=True)
        dims = [
            ("技能分", "简历与岗位技能交集比例", C["primary"]),
            ("TF-IDF", "文本整体语义相似度", C["purple"]),
            ("Jaccard", "技能集合重叠度", C["cyan"]),
            ("学历分", "是否满足岗位最低学历要求", C["green"]),
            ("经验分", "经验年限是否满足岗位要求", C["yellow"]),
            ("城市分", "候选人期望城市与岗位城市匹配", "#F97316"),
            ("证书分", "证书是否满足岗位偏好", C["cyan"]),
        ]
        for name, desc, color in dims:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px 0;"
                f"border-bottom:1px solid {C['chart_grid']};'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;'></div>"
                f"<span style='font-size:14px;font-weight:600;color:{C['text']};min-width:60px;'>{name}</span>"
                f"<span style='font-size:13px;color:{C['text_muted']};'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown(f"""
        <div class="info-strip">
            <div style="font-size:15px;font-weight:700;color:{C['text']};margin-bottom:12px;">
                算法架构
            </div>
        </div>
        """, unsafe_allow_html=True)
        arch = [
            ("数据输入", C["primary"]),
            ("脏数据清洗", C["purple"]),
            ("字段结构化", "#8B5CF6"),
            ("文本预处理", C["cyan"]),
            ("TF-IDF + Jaccard", C["green"]),
            ("多维规则评分", C["yellow"]),
            ("Streamlit UI", "#F97316"),
        ]
        for i, (name, color) in enumerate(arch):
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;padding:6px 0;'>"
                f"<div style='width:28px;height:28px;border-radius:8px;background:{color}15;"
                f"border:1px solid {color}30;display:flex;align-items:center;justify-content:center;'>"
                f"<span style='font-size:11px;font-weight:700;color:{color};'>{i+1}</span>"
                f"</div>"
                f"<span style='font-size:14px;color:#334155;'>{name}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if i < len(arch) - 1:
                st.markdown(
                    f"<div style='margin-left:14px;padding:2px 0;'>"
                    f"<svg width='16' height='16' viewBox='0 0 16 16' fill='none'>"
                    f"<path d='M8 3v10M4 9l4 4 4-4' stroke='{arch[i+1][1]}' stroke-width='1.5' "
                    f"stroke-linecap='round' stroke-linejoin='round' opacity='0.5'/>"
                    f"</svg></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown("**综合匹配分排名**")
    render_overall_score_chart(matches_df, key="overview_score")

# ========== Tab 1: 学生视角 ==========
with tabs[1]:
    resume_options = [
        f"{r.get('resume_id','')} | {r.get('name','')} | {r.get('education','')} | "
        f"{r.get('experience_years',0)}年 | {r.get('city','')}"
        for _, r in resumes_df.iterrows()
    ]
    resume_map = {opt: r.get("resume_id", "") for opt, (_, r) in zip(resume_options, resumes_df.iterrows())}
    sel_label = st.selectbox("选择简历", options=resume_options, index=0)
    sel_id = resume_map.get(sel_label, resumes_df.iloc[0].get("resume_id", ""))

    rinfo_series = resumes_df[resumes_df["resume_id"] == sel_id]
    rinfo = rinfo_series.iloc[0] if not rinfo_series.empty else resumes_df.iloc[0]

    # 简历信息条
    skills_list = [s.strip() for s in str(rinfo.get("skills", "")).split(";") if s.strip()]
    st.markdown(f"""
    <div class="info-strip">
        <div style="display:flex;gap:32px;flex-wrap:wrap;">
            <div class="field">
                <span class="field-label">姓名</span>
                <span class="field-value">{rinfo.get("name", "")}</span>
            </div>
            <div class="field">
                <span class="field-label">学历</span>
                <span class="field-value">{rinfo.get("education", "")}</span>
            </div>
            <div class="field">
                <span class="field-label">经验</span>
                <span class="field-value">{rinfo.get("experience_years", "")}年</span>
            </div>
            <div class="field">
                <span class="field-label">城市</span>
                <span class="field-value">{rinfo.get("city", "")}</span>
            </div>
        </div>
        <div class="field" style="margin-top:8px;">
            <span class="field-label">技能</span>
            <div class="skills-row">
                {"".join(f'<span class="tag tag-blue">{s}</span>' for s in skills_list)}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 筛选
    filtered = matches_df[
        (matches_df["resume_id"] == sel_id) &
        (matches_df["final_score"] >= min_score)
    ].copy()
    if selected_cities:
        filtered = filtered[filtered["city"].isin(selected_cities)]
    filtered = filtered.sort_values("final_score", ascending=False).head(top_n)

    if filtered.empty:
        st.warning("当前筛选条件下没有匹配的岗位")
    else:
        sel_idx = render_match_table(filtered, mode="resume", selectable=True)

        sel_row = filtered.iloc[sel_idx:sel_idx + 1].copy()
        sel_score = round(float(sel_row.iloc[0]["final_score"]), 1)

        # 仪表盘 + 雷达图
        col_g, col_r = st.columns([1, 1])
        with col_g:
            st.markdown("**综合分**")
            render_gauge_chart(sel_score, label="综合分", key="gauge_s")
        with col_r:
            st.markdown("**能力雷达图**")
            render_radar_chart(sel_row, mode="resume", jobs_df=jobs_df, key="radar_s")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("**各维度分数对比**")
        render_score_chart(sel_row, job_title_col="job_title", key="dim_s")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("**推荐详情**")
        render_reason(sel_row, explains_df, mode="resume")

        st.download_button(
            "下载推荐结果",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            f"resume_{sel_id}_matches.csv",
            mime="text/csv",
        )

# ========== Tab 2: HR 视角 ==========
with tabs[2]:
    job_options = [
        f"{r.get('job_id','')} | {r.get('job_title','')} | {r.get('company','')} | "
        f"{r.get('city','')}"
        for _, r in jobs_df.iterrows()
    ]
    job_map = {opt: r.get("job_id", "") for opt, (_, r) in zip(job_options, jobs_df.iterrows())}
    sel_label = st.selectbox("选择岗位", options=job_options, index=0)
    sel_id = job_map.get(sel_label, jobs_df.iloc[0].get("job_id", ""))

    jinfo_series = jobs_df[jobs_df["job_id"] == sel_id]
    jinfo = jinfo_series.iloc[0] if not jinfo_series.empty else jobs_df.iloc[0]

    req_skills = [s.strip() for s in str(jinfo.get("required_skills", "")).split(";") if s.strip()]
    pref_skills = [s.strip() for s in str(jinfo.get("preferred_skills", "")).split(";") if s.strip()]
    req_html = "".join(f'<span class="tag tag-blue">{s}</span>' for s in req_skills)
    pref_html = "".join(f'<span class="tag tag-purple">{s}</span>' for s in pref_skills)

    st.markdown(f"""
    <div class="info-strip">
        <div style="display:flex;gap:32px;flex-wrap:wrap;">
            <div class="field">
                <span class="field-label">岗位</span>
                <span class="field-value">{jinfo.get("job_title", "")}</span>
            </div>
            <div class="field">
                <span class="field-label">公司</span>
                <span class="field-value">{jinfo.get("company", "")}</span>
            </div>
            <div class="field">
                <span class="field-label">城市</span>
                <span class="field-value">{jinfo.get("city", "")}</span>
            </div>
            <div class="field">
                <span class="field-label">类型</span>
                <span class="field-value">{jinfo.get("job_type", "")}</span>
            </div>
        </div>
        <div class="field" style="margin-top:8px;">
            <span class="field-label">必备</span>
            <div class="skills-row">{req_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if pref_skills:
        st.markdown(f"""
        <div class="info-strip" style="margin-top:-8px;border-top:none;padding-top:0;">
            <div class="field">
                <span class="field-label">加分</span>
                <div class="skills-row">{pref_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    jm = matches_df[matches_df["job_id"] == sel_id].copy()
    jm = jm[jm["final_score"] >= min_score]
    if selected_cities:
        jm = jm[jm["city"].isin(selected_cities)]
    jm = jm.sort_values("final_score", ascending=False).head(top_n)

    if jm.empty:
        st.warning("当前筛选条件下没有匹配的候选人")
    else:
        sel_idx = render_match_table(jm, mode="job", selectable=True)
        sel_row = jm.iloc[sel_idx:sel_idx + 1].copy()
        sel_score = round(float(sel_row.iloc[0]["final_score"]), 1)

        col_g, col_r = st.columns([1, 1])
        with col_g:
            st.markdown("**综合分**")
            render_gauge_chart(sel_score, label="综合分", key="gauge_h")
        with col_r:
            st.markdown("**候选人雷达图**")
            render_radar_chart(sel_row, mode="job", key="radar_h")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("**各维度分数对比**")
        render_score_chart(sel_row, resume_name_col="resume_name", key="dim_h")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown("**推荐详情**")
        render_reason(sel_row, explains_df, mode="job")

        st.download_button(
            "下载匹配结果",
            jm.to_csv(index=False).encode("utf-8-sig"),
            f"job_{sel_id}_matches.csv",
            mime="text/csv",
        )

# ========== Tab 3: 全量统计 ==========
with tabs[3]:
    render_stats_overview(matches_df, explains_df)

# ---------------------------------------------------------------------------
# 底部
# ---------------------------------------------------------------------------
st.markdown("<hr class='sep'>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "下载完整匹配结果",
        matches_df.to_csv(index=False).encode("utf-8-sig"),
        "top_matches.csv",
        mime="text/csv",
    )
with col_b:
    st.download_button(
        "下载详细推荐理由",
        explains_df.to_csv(index=False).encode("utf-8-sig"),
        "explainable_match_result.csv",
        mime="text/csv",
    )
