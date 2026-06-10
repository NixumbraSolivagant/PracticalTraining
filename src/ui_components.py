"""
src/ui_components.py — Streamlit UI 组件（亮色简洁版）

设计原则：
- 配色集中在 C 字典中，单一数据源
- 所有图表使用 Plotly 原生交互
- HTML 仅用于布局卡片、标签、表格等结构元素
- 组件函数不读写文件，数据全部通过参数传入（stats_overview 除外，它直接从文件读取以保证完整性）
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


# ---------------------------------------------------------------------------
# 配色
# ---------------------------------------------------------------------------
C = {
    "primary": "#4F46E5",
    "primary_light": "#EEF2FF",
    "purple": "#7C3AED",
    "purple_light": "#F3E8FF",
    "cyan": "#06B6D4",
    "green": "#10B981",
    "green_light": "#DCFCE7",
    "green_text": "#166534",
    "yellow": "#F59E0B",
    "orange": "#F97316",
    "red": "#EF4444",
    "red_light": "#FEE2E2",
    "red_text": "#991B1B",
    "bg": "#FFFFFF",
    "bg_page": "#F8FAFC",
    "bg_hover": "#F1F5F9",
    "border": "#E2E8F0",
    "chart_grid": "#F1F5F9",
    "text": "#1E293B",
    "text_muted": "#64748B",
    "text_light": "#94A3B8",
}


def _score_color(s):
    if s >= 80: return C["green"]
    if s >= 60: return C["yellow"]
    if s >= 40: return C["orange"]
    return C["red"]


def _score_label(s):
    if s >= 80: return "优秀"
    if s >= 60: return "良好"
    if s >= 40: return "一般"
    return "较低"


def _font():
    return "Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"


# ---------------------------------------------------------------------------
# KPI 卡片
# ---------------------------------------------------------------------------
def render_kpi_cards(matches_df, explains_df):
    n_res = explains_df["resume_id"].nunique()
    n_job = explains_df["job_id"].nunique()
    mx = round(float(matches_df["final_score"].max()), 1)
    av = round(float(matches_df["final_score"].mean()), 1)

    c1, c2, c3, c4 = st.columns(4)
    items = [
        (c1, "简历总数", n_res, C["primary"]),
        (c2, "岗位总数", n_job, C["purple"]),
        (c3, "最高综合分", f"{mx}", C["green"]),
        (c4, "平均综合分", f"{av}", C["cyan"]),
    ]
    for col, label, val, color in items:
        with col:
            st.markdown(f"""
            <div style="
                background: {C['bg']};
                border: 1px solid {C['border']};
                border-radius: 16px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            ">
                <div style="
                    width: 44px; height: 44px; border-radius: 12px;
                    background: {color}18;
                    border: 1.5px solid {color}30;
                    margin: 0 auto 10px;
                    display: flex; align-items: center; justify-content: center;
                ">
                    <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
                </div>
                <div style="font-size:12px;color:{C['text_muted']};font-weight:500;margin-bottom:6px;">
                    {label}
                </div>
                <div style="font-size:2rem;font-weight:800;color:{color};line-height:1;">
                    {val}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 环形仪表盘
# ---------------------------------------------------------------------------
def render_gauge_chart(score, label="综合分", key=None):
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font.size": 32,
            "font.color": color,
            "suffix": "",
            "valueformat": ".0f",
        },
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color, "thickness": 0.15},
            "bgcolor": "white",
            "shape": "angular",
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.08)"},
                {"range": [40, 60], "color": "rgba(249,115,22,0.08)"},
                {"range": [60, 80], "color": "rgba(245,158,11,0.08)"},
                {"range": [80, 100],"color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "value": score,
            },
        },
        title={"text": label, "font.size": 13, "font.color": C["text_muted"]},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=C["text"], family=_font()),
    )
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# 综合分排名柱状图
# ---------------------------------------------------------------------------
def render_overall_score_chart(df, job_title_col="job_title", resume_name_col="resume_name", key=None):
    if df.empty:
        return
    chart_df = df.sort_values("final_score", ascending=False).copy()
    chart_df["label"] = (
        chart_df[job_title_col]
        if job_title_col in chart_df.columns
        else chart_df[resume_name_col]
    )
    colors = [_score_color(s) for s in chart_df["final_score"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(chart_df))),
        y=chart_df["final_score"],
        marker_color=colors,
        marker_line_width=0,
        text=[f"{s:.1f}" for s in chart_df["final_score"]],
        textposition="outside",
        textfont_size=11,
        hovertemplate="<b>%{customdata}</b><br>综合分: %{y:.1f}<extra></extra>",
        customdata=chart_df["label"],
    ))
    fig.update_layout(
        title={"text": "综合匹配分排名", "font.size": 15, "font.color": C["text"]},
        xaxis=dict(
            title="", showticklabels=True, tickmode="array",
            tickvals=list(range(len(chart_df))),
            ticktext=[f"#{i+1}" for i in range(len(chart_df))],
            tickfont_color=C["text_light"], gridcolor="white",
        ),
        yaxis=dict(
            title="", range=[0, 108], gridcolor=C["chart_grid"],
            tickfont_color=C["text_light"],
        ),
        height=240,
        margin=dict(b=40, t=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=_font(), color=C["text"]),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# 多维度柱状图
# ---------------------------------------------------------------------------
def render_score_chart(df, job_title_col="job_title", resume_name_col="resume_name", key=None):
    if df.empty:
        return
    chart_df = df.copy()
    chart_df["label"] = (
        chart_df[job_title_col]
        if job_title_col in chart_df.columns
        else chart_df[resume_name_col]
    )

    score_map = {
        "技能分": "skill_score",
        "学历分": "education_score",
        "经验分": "experience_score",
        "城市分": "city_score",
        "证书分": "cert_score",
    }

    melted = chart_df.melt(
        id_vars=["label", "final_score"],
        value_vars=list(score_map.values()),
        var_name="score_type",
        value_name="score",
    )
    melted["score_type"] = melted["score_type"].map(score_map)
    color_seq = [C["primary"], C["purple"], C["cyan"], C["green"], C["yellow"]]

    fig = px.bar(
        melted,
        x="label",
        y="score",
        color="score_type",
        barmode="group",
        labels={"label": "", "score": "分数", "score_type": ""},
        color_discrete_sequence=color_seq,
        text_auto=True,
    )
    fig.update_layout(
        title={"text": "各维度评分对比", "font.size": 15, "font.color": C["text"]},
        height=300,
        margin=dict(b=80, t=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="center", x=0.5, font_color=C["text_muted"],
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=_font(), color=C["text"]),
        xaxis_tickangle=-20,
    )
    fig.update_yaxes(range=[0, 105], gridcolor=C["chart_grid"], tickfont_color=C["text_light"])
    fig.update_xaxes(showgrid=False, tickfont_color=C["text_muted"])
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# 雷达图
# ---------------------------------------------------------------------------
def render_radar_chart(df, mode="resume", jobs_df=None, key=None):
    if df.empty:
        return
    top = df.iloc[0]

    student_vals = {
        "技能分": float(top.get("skill_score", 0)),
        "学历分": float(top.get("education_score", 0)),
        "经验分": float(top.get("experience_score", 0)),
        "城市分": float(top.get("city_score", 0)),
        "证书分": float(top.get("cert_score", 0)),
    }
    labels = list(student_vals.keys())
    s_vals = list(student_vals.values())
    s_vals += s_vals[:1]
    labels_closed = labels + labels[:1]

    fig = go.Figure()

    # 候选人（蓝实线）
    fig.add_trace(go.Scatterpolar(
        r=s_vals,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(79,70,229,0.15)",
        line_color=C["primary"],
        line_width=2.5,
        marker=dict(size=6, color=C["primary"]),
        name="候选人得分",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))

    # 仅学生视角：叠加岗位要求（橙虚线）
    if mode == "resume" and jobs_df is not None:
        rid = top.get("resume_id")
        jid = top.get("job_id")
        explains_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "output", "explainable_match_result.csv"
        )
        matched_str, missing_str = "", ""
        try:
            exp = pd.read_csv(explains_path)
            row = exp[(exp["resume_id"] == rid) & (exp["job_id"] == jid)]
            if not row.empty:
                matched_str = str(row.iloc[0].get("matched_skills", ""))
                missing_str = str(row.iloc[0].get("missing_skills", ""))
        except Exception:
            pass

        matched = set(m.strip().lower() for m in matched_str.split(";") if m.strip())
        missing = set(m.strip().lower() for m in missing_str.split(";") if m.strip())
        total = max(len(matched) + len(missing), 1)
        skill_req = round(len(matched) / total * 100)

        job_req_vals = [
            skill_req,
            float(top.get("education_score", 0)),
            float(top.get("experience_score", 0)),
            float(top.get("city_score", 0)),
            float(top.get("cert_score", 0)),
        ]
        job_req_vals += job_req_vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=job_req_vals,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(249,115,22,0.06)",
            line_color=C["yellow"],
            line_width=2,
            line_dash="dot",
            marker=dict(size=5, color=C["yellow"]),
            name="岗位要求",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 105], tickfont_size=11, tickcolor=C["text_light"],
                gridcolor=C["chart_grid"],
            ),
            angularaxis=dict(tickfont_size=12, tickcolor=C["text_muted"]),
            bgcolor="white",
        ),
        height=280,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06,
            xanchor="center", x=0.5,
            font_color=C["text_muted"], font_size=12,
        ),
        margin=dict(l=30, r=30, t=20, b=30),
        paper_bgcolor="white",
        font=dict(family=_font(), color=C["text"]),
    )
    st.plotly_chart(fig, width="stretch", key=key)


# ---------------------------------------------------------------------------
# 匹配结果表格
# ---------------------------------------------------------------------------
def render_match_table(df, mode="resume", selectable=False):
    if df.empty:
        st.warning("暂无匹配结果")
        return None

    if mode == "resume":
        headers = ["#", "综合分", "技能分", "岗位", "公司", "城市"]
        cols = ["rank", "final_score", "skill_score", "job_title", "company", "city"]
    else:
        name_col = next((c for c in ["resume_name", "name", "resume_id"] if c in df.columns), "resume_id")
        city_col = next((c for c in ["resume_city", "city"] if c in df.columns), "city")
        edu_col = next((c for c in ["resume_education", "education"] if c in df.columns), "")
        headers = ["#", "综合分", "技能分", "姓名", edu_col, city_col] if edu_col else ["#", "综合分", "技能分", "姓名", city_col]
        cols = ["rank", "final_score", "skill_score", name_col, edu_col, city_col] if edu_col else ["rank", "final_score", "skill_score", name_col, city_col]
        cols = [c for c in cols if c]
        headers = headers[:len(cols)]

    display_df = df[cols].copy()
    display_df.columns = headers

    sel_key = f"sel_{mode}"
    sel_idx = 0
    if selectable:
        if sel_key not in st.session_state:
            st.session_state[sel_key] = 0
        labels = [f"#{i+1} {display_df.iloc[i][headers[3]]}" for i in range(len(display_df))]
        chosen = st.selectbox(
            "选择查看",
            list(range(len(labels))),
            index=st.session_state[sel_key],
            format_func=lambda i: labels[i],
            key=sel_key,
            label_visibility="collapsed",
        )
        sel_idx = st.session_state[sel_key]

    # Build HTML rows — highlight selected row
    html_rows = ""
    for idx, (_, row) in enumerate(display_df.iterrows()):
        fs = round(float(row["综合分"]), 1)
        ss = round(float(row["技能分"]), 1)
        color = _score_color(fs)
        label = _score_label(fs)
        bg = C["bg_hover"] if idx == sel_idx else "white"

        html_rows += f"""<tr>
            <td style="text-align:center;font-weight:700;color:{color};font-size:13px;">{row['#']}</td>
            <td>
                <span style="
                    display:inline-flex;align-items:center;gap:6px;
                    background:{color}15;color:{color};
                    border:1px solid {color}30;
                    padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">
                    {fs} <span style="font-weight:400;font-size:11px;opacity:0.7;">{label}</span>
                </span>
            </td>
            <td>
                <div style="display:flex;align-items:center;gap:8px;min-width:80px;">
                    <div style="flex:1;height:5px;background:{C['border']};border-radius:3px;overflow:hidden;">
                        <div style="width:{ss}%;height:100%;background:{_score_color(ss)};border-radius:3px;"></div>
                    </div>
                    <span style="font-size:12px;color:{C['text_muted']};min-width:24px;text-align:right;">{ss}</span>
                </div>
            </td>"""
        for ci, c in enumerate(cols[3:]):
            val = str(row[headers[ci + 3]])[:22]
            html_rows += f"<td style='font-size:13px;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px;' title='{val}'>{val}</td>"
        html_rows += "</tr>"

    st.markdown(f"""
    <style>
    .mt {{ width:100%; border-collapse:separate; border-spacing:0;
           font-size:14px; border-radius:12px; overflow:hidden;
           border:1px solid {C['border']}; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
    .mt th {{
        background: linear-gradient(135deg, {C['primary']} 0%, {C['purple']} 100%);
        color:white; padding:11px 14px; text-align:left;
        font-weight:600; font-size:12px; letter-spacing:0.5px;
    }}
    .mt td {{ padding:10px 14px; border-bottom:1px solid {C['border']}; vertical-align:middle;
              background:white; }}
    .mt tr:last-child td {{ border-bottom:none; }}
    .mt tr {{ transition:background 0.15s; }}
    .mt tr:hover td {{ background:{C['bg_hover']}; }}
    </style>
    <table class="mt">
    <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
    <tbody>{html_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)
    return sel_idx


# ---------------------------------------------------------------------------
# 推荐理由
# ---------------------------------------------------------------------------
def render_reason(df, _unused_explains, mode="resume"):
    if df.empty:
        return
    top = df.iloc[0]

    # 直接从文件读取，确保数据完整
    explains_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "output", "explainable_match_result.csv"
    )
    try:
        explains_df = pd.read_csv(explains_path)
    except Exception:
        explains_df = _unused_explains

    matching = explains_df[
        (explains_df["resume_id"] == top["resume_id"])
        & (explains_df["job_id"] == top["job_id"])
    ]
    if matching.empty:
        st.info("暂无推荐理由")
        return

    row = matching.iloc[0]
    reason = row.get("recommendation_reason", "暂无推荐理由")
    matched = row.get("matched_skills", "")
    missing = row.get("missing_skills", "")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**匹配技能**")
        if matched:
            skills_html = "".join(
                f'<span style="display:inline-block;background:{C["green_light"]};color:{C["green_text"]};'
                f'padding:3px 10px;border-radius:20px;font-size:12px;margin:2px 4px 2px 0;">'
                f'+ {s.strip()}</span>'
                for s in str(matched).split(";") if s.strip()
            )
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{C["text_light"]};font-size:13px;">暂无</span>',
                       unsafe_allow_html=True)

    with c2:
        st.markdown("**建议补充**")
        if missing:
            skills_html = "".join(
                f'<span style="display:inline-block;background:{C["red_light"]};color:{C["red_text"]};'
                f'padding:3px 10px;border-radius:20px;font-size:12px;margin:2px 4px 2px 0;">'
                f'- {s.strip()}</span>'
                for s in str(missing).split(";") if s.strip()
            )
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{C["green"]};font-size:13px;">无缺失</span>',
                       unsafe_allow_html=True)

    st.markdown("**推荐理由**")
    for line in str(reason).split("\n"):
        if line.strip():
            st.markdown(
                f'<p style="font-size:14px;color:#334155;margin:4px 0;padding-left:12px;'
                f'border-left:3px solid {C["primary"]};">{line.strip()}</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# 全量统计
# ---------------------------------------------------------------------------
def render_stats_overview(_unused1, _unused2):
    st.subheader("全量匹配统计分析")

    # 始终从文件读取，确保数据完整
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    matches_df = pd.read_csv(os.path.join(data_dir, "top_matches.csv"))
    explains_df = pd.read_csv(os.path.join(data_dir, "explainable_match_result.csv"))

    tabs = st.tabs(["分数分布", "Top 岗位", "Top 候选人", "城市分布", "技能热力"])

    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=matches_df["final_score"],
            nbinsx=12,
            marker_color=C["primary"],
            marker_line_color="white",
            marker_line_width=1,
            hovertemplate="分数 %{x}<br>数量 %{y}<extra></extra>",
        ))
        fig.update_layout(
            title={"text": "综合匹配分分布", "font.size": 15, "font.color": C["text"]},
            xaxis_title="综合分", yaxis_title="匹配对数量",
            height=300,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family=_font(), color=C["text"]),
        )
        fig.add_vline(
            x=matches_df["final_score"].mean(),
            line_dash="dash", line_color=C["red"],
            annotation_text=f"均值 {matches_df['final_score'].mean():.1f}",
            annotation_font_size=11, annotation_font_color=C["red"],
        )
        fig.update_xaxes(gridcolor=C["chart_grid"], tickfont_color=C["text_light"])
        fig.update_yaxes(gridcolor=C["chart_grid"], tickfont_color=C["text_light"])
        st.plotly_chart(fig, width="stretch", key="hist_stat")

    with tabs[1]:
        top_jobs = (
            matches_df.groupby(["job_id", "job_title", "city"])
            .agg(avg_score=("final_score", "mean"), cnt=("final_score", "count"))
            .reset_index().sort_values("avg_score", ascending=False).head(10)
        )
        fig = px.bar(
            top_jobs, x="job_title", y="avg_score",
            color="avg_score", color_continuous_scale="Blues",
            title="各岗位平均综合分 Top-10",
            labels={"job_title": "", "avg_score": "平均综合分"},
            text_auto=True,
        )
        fig.update_layout(
            height=300, xaxis_tickangle=-20, margin=dict(b=80),
            coloraxis_showscale=False,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family=_font(), color=C["text"]),
        )
        fig.update_xaxes(showgrid=False, tickfont_color=C["text_muted"])
        fig.update_yaxes(gridcolor=C["chart_grid"], tickfont_color=C["text_light"], range=[0, 105])
        st.plotly_chart(fig, width="stretch", key="top_jobs_stat")

    with tabs[2]:
        name_col = "resume_name" if "resume_name" in explains_df.columns else "resume_id"
        top_cands = (
            explains_df.groupby(["resume_id", name_col])
            .agg(avg_score=("final_score", "mean"))
            .reset_index().sort_values("avg_score", ascending=False).head(10)
        )
        top_cands.rename(columns={name_col: "candidate_name"}, inplace=True)
        fig = px.bar(
            top_cands, x="candidate_name", y="avg_score",
            color="avg_score", color_continuous_scale="Purples",
            title="候选人平均综合分 Top-10",
            labels={"candidate_name": "", "avg_score": "平均综合分"},
            text_auto=True,
        )
        fig.update_layout(
            height=300, xaxis_tickangle=-20, margin=dict(b=80),
            coloraxis_showscale=False,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family=_font(), color=C["text"]),
        )
        fig.update_xaxes(showgrid=False, tickfont_color=C["text_muted"])
        fig.update_yaxes(gridcolor=C["chart_grid"], tickfont_color=C["text_light"], range=[0, 105])
        st.plotly_chart(fig, width="stretch", key="top_cands_stat")

    with tabs[3]:
        city_stats = (
            matches_df.groupby("city")
            .agg(cnt=("final_score", "count"), avg=("final_score", "mean"))
            .reset_index().sort_values("cnt", ascending=False)
        )
        fig = px.pie(
            city_stats, names="city", values="cnt",
            title="各城市匹配对数量占比",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"city": "", "cnt": "数量"},
        )
        fig.update_layout(
            height=320,
            paper_bgcolor="white",
            font=dict(family=_font(), color=C["text"]),
        )
        fig.update_traces(
            textposition="outside", textfont_size=12,
            hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
        )
        st.plotly_chart(fig, width="stretch", key="city_pie_stat")

    with tabs[4]:
        from collections import Counter
        all_matched = ";".join(explains_df["matched_skills"].dropna().astype(str).tolist())
        all_missing = ";".join(explains_df["missing_skills"].dropna().astype(str).tolist())
        mc = Counter([s.strip() for s in all_matched.split(";") if s.strip()])
        ms = Counter([s.strip() for s in all_missing.split(";") if s.strip()])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**高频匹配技能**")
            for skill, cnt in mc.most_common(12):
                pct = int(cnt / max(mc.values()) * 100)
                st.markdown(
                    f"<div style='margin:6px 0;display:flex;align-items:center;gap:8px;'>"
                    f"<span style='min-width:80px;font-size:13px;color:#334155;'>{skill}</span>"
                    f"<div style='flex:1;height:7px;background:{C['border']};border-radius:4px;'>"
                    f"<div style='width:{pct}%;height:100%;background:linear-gradient(90deg,{C['primary']},{C['purple']});"
                    f"border-radius:4px;'></div></div>"
                    f"<span style='font-size:12px;color:{C['text_light']};min-width:20px;'>{cnt}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown("**高频缺失技能**")
            for skill, cnt in ms.most_common(12):
                pct = int(cnt / max(ms.values()) * 100)
                st.markdown(
                    f"<div style='margin:6px 0;display:flex;align-items:center;gap:8px;'>"
                    f"<span style='min-width:80px;font-size:13px;color:#334155;'>{skill}</span>"
                    f"<div style='flex:1;height:7px;background:{C['border']};border-radius:4px;'>"
                    f"<div style='width:{pct}%;height:100%;background:linear-gradient(90deg,{C['red']},{C['orange']});"
                    f"border-radius:4px;'></div></div>"
                    f"<span style='font-size:12px;color:{C['text_light']};min-width:20px;'>{cnt}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# HDFS 状态
# ---------------------------------------------------------------------------
def render_hdfs_status():
    try:
        import subprocess
        result = subprocess.run(
            ["/bin/bash", "-c",
             "export JAVA_HOME=/usr/lib/jvm/jdk-11 && "
             "export PATH=/usr/local/hadoop/bin:/usr/local/hadoop/sbin:$PATH && "
             "hdfs dfs -ls /resume_matching"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for seg in ["raw_data", "cleaned_data", "results"]:
                st.markdown(
                    f'<span style="font-size:12px;color:{C["green"]};">&#10003; '
                    f'/resume_matching/{seg}</span>',
                    unsafe_allow_html=True,
                )
            return True
    except Exception:
        pass
    st.markdown(
        f'<span style="font-size:12px;color:{C["red"]};">&#10007; '
        'HDFS 未连接（请先启动 Hadoop）</span>',
        unsafe_allow_html=True,
    )
    return False
