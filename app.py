import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Domain mappings (train.py와 반드시 동일)
# =========================
def add_industry_reclass(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "산업군_상위추출" not in out.columns and "산업군" in out.columns:
        out["산업군_상위추출"] = out["산업군"].astype(str).str.split().str[0]

    industry_map = {
        "정부": "공공안정", "국가": "공공안정", "군대": "공공안정",
        "경찰": "공공안정", "학교": "공공안정", "대학교": "공공안정",
        "유치원": "공공안정", "은행": "공공안정",

        "의학": "전문금융", "법률": "전문금융", "보험": "전문금융",

        "자영업": "자영사업", "사업": "자영사업", "부동산": "자영사업",

        "건설": "경기민감", "산업": "경기민감", "무역": "경기민감",
        "운송": "경기민감",

        "레스토랑": "서비스요식", "호텔": "서비스요식", "보안": "서비스요식",
        "서비스": "서비스요식",
    }
    if "산업군_상위추출" in out.columns:
        out["산업군_재분류"] = out["산업군_상위추출"].map(industry_map).fillna("기타")
    else:
        out["산업군_재분류"] = "기타"
    return out


def add_job_ksco_7(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "직업" not in out.columns:
        out["직업_KSCO_7그룹"] = "기타"
        return out

    ksco_job_map = {
        "단순 노동자": "취약_단순",
        "저임금 노동자": "취약_단순",
        "미화원": "취약_단순",

        "조리사": "취약_서비스",
        "가정부": "취약_서비스",
        "요식업 종사자": "취약_서비스",
        "보안 업계 종사자": "취약_서비스",

        "영업직": "취약_판매",
        "부동산중개업자": "취약_판매",

        "운전자": "취약_기계",

        "기술직": "중간_기능",
        "핵심 노동자": "중간_기능",

        "비서": "안정_사무",
        "인사 담당자": "안정_사무",

        "관리직": "안정_전문관리",
        "회계사": "안정_전문관리",
        "의료 업계 종사자": "안정_전문관리",
        "IT 업계 종사자": "안정_전문관리",

        "Unknown": "기타",
    }
    out["직업_KSCO_7그룹"] = out["직업"].map(ksco_job_map).fillna("기타")
    return out


# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    base = Path(__file__).resolve().parent
    art_dir = base / "artifacts"

    with open(art_dir / "model.pkl", "rb") as f:
        pipe = pickle.load(f)

    with open(art_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    combo_woe_table = pd.read_parquet(art_dir / "combo_woe_table.parquet")
    return pipe, meta, combo_woe_table


def score(df_in: pd.DataFrame, pipe, meta):
    """
    Returns:
      proba: np.array
      grade_vec: np.array[str]
      df2: model input dataframe after FE/WOE/drop (for reason codes)
    """
    df2 = df_in.copy()
    df2 = add_industry_reclass(df2)
    df2 = add_job_ksco_7(df2)

    # 산업×직업 combo → WOE 적용
    combo = (df2["산업군_재분류"].astype(str) + " × " + df2["직업_KSCO_7그룹"].astype(str))
    seen = set(meta["woe_map_combo"].keys())
    combo = combo.where(combo.isin(seen), "OTHERS").astype("object")

    df2["고용안정성_WOE"] = combo.map(meta["woe_map_combo"]).fillna(0.0).astype(float)

    # 원본 직업/산업 컬럼 drop (훈련과 동일)
    df2 = df2.drop(columns=meta["drop_cols"], errors="ignore")

    proba = pipe.predict_proba(df2)[:, 1]
    high_cut = meta["high_cut"]
    mid_cut = meta["mid_cut"]

    def grade(p):
        if p >= high_cut:
            return "High"
        elif p >= mid_cut:
            return "Mid"
        else:
            return "Low"

    grade_vec = pd.Series(proba).apply(grade).values
    return proba, grade_vec, df2


def get_or_table(pipe, top_k=15):
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    feat_names = prep.get_feature_names_out()
    coef = model.coef_.ravel()
    or_ = np.exp(coef)

    df_or = pd.DataFrame({"feature": feat_names, "coef": coef, "odds_ratio": or_})
    df_or["feature"] = (df_or["feature"]
                        .str.replace("cat__", "", regex=False)
                        .str.replace("num__", "", regex=False))

    high = df_or.sort_values("odds_ratio", ascending=False).head(top_k)
    low = df_or.sort_values("odds_ratio", ascending=True).head(top_k)
    return df_or, high, low


def top_reason_codes(pipe, X_row: pd.DataFrame, top_k=7):
    """
    영향(impact) < 0: 점수(=좋음)를 깎는 방향 = 리스크 요인
    여기서는 상대적 설명용이므로 스코어 변환 없이 coef*x로 사용.
    """
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    feat_names = prep.get_feature_names_out()
    coef = model.coef_.ravel()

    X_mat = prep.transform(X_row)
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    x_vec = X_mat.ravel()

    impact = -(coef * x_vec)  # 음수로 크게 내려가는 게 리스크 요인
    dfc = pd.DataFrame({"feature": feat_names, "impact": impact})
    dfc["feature"] = (dfc["feature"]
                      .str.replace("cat__", "", regex=False)
                      .str.replace("num__", "", regex=False))
    return dfc.sort_values("impact").head(top_k)


# =========================
# UI
# =========================
st.set_page_config(page_title="CardScore | Delinquency Risk Dashboard", layout="wide")

pipe, meta, combo_woe_table = load_artifacts()
df_or, or_high, or_low = get_or_table(pipe, top_k=15)

st.title("CardScore — 연체 리스크 예측 & 선제적 관리 대시보드")
st.caption("Champion Model: Logistic (설명가능) + KSCO 직업 재분류 + 산업군 재분류 + 산업×직업 WOE(고용안정성)")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("AUC", f"{meta['auc']:.4f}")
k2.metric("KS", f"{meta['ks']:.4f}")
k3.metric("High Cut (Top 20%)", f"{meta['high_cut']:.4f}")
k4.metric("Mid Cut (Top 60%)", f"{meta['mid_cut']:.4f}")

tabs = st.tabs(["① Overview", "② Score Customers", "③ Explain (Reason Codes)", "④ WOE Insights"])


# =========================
# ① Overview
# =========================
with tabs[0]:
    st.subheader("Executive Summary (심사위원용 30초 요약)")
    st.info(
        "본 모델은 고객 기본정보만으로 연체 위험을 조기 식별하고, "
        "High/Mid/Low 3등급으로 분류해 선제적 정책을 적용할 수 있도록 설계했습니다. "
        "특히 직업·산업군을 도메인 기반으로 재분류하고, 산업×직업 조합을 WOE로 수치화해 "
        "‘고용 안정성’ 관점의 리스크를 설명가능하게 반영했습니다."
    )

    st.markdown("### 운영 정책 예시 (Risk Grade Action)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔴 High")
        st.markdown("- 한도/결제조건 조정\n- 사전 안내·콜센터\n- 연체예방 캠페인/리마인드")
    with c2:
        st.markdown("#### 🟠 Mid")
        st.markdown("- 모니터링 강화\n- 자동이체/분할납부 유도\n- 행동기반 알림")
    with c3:
        st.markdown("#### 🟢 Low")
        st.markdown("- 정상 유지\n- 우량 고객 프로모션\n- 과도 제약 최소화")

    st.divider()
    st.markdown("### 모델 해석(요약): Odds Ratio 상위/하위")
    colA, colB = st.columns(2)
    with colA:
        st.write("Top 15 Higher Risk (OR↑)")
        st.dataframe(or_high, use_container_width=True)
    with colB:
        st.write("Top 15 Lower Risk (OR↓)")
        st.dataframe(or_low, use_container_width=True)


# =========================
# ② Score Customers
# =========================
with tabs[1]:
    st.subheader("고객 점수화 (CSV 업로드 → 예측확률/등급 → 다운로드)")
    up = st.file_uploader("CSV 업로드 (TARGET 없어도 됨)", type=["csv"])

    st.caption("※ 업로드 파일은 훈련에 사용한 컬럼 구조(예: 직업, 산업군 등)를 포함해야 합니다.")

    if up is None:
        st.warning("CSV를 업로드하면 결과(확률/등급/요약)가 표시됩니다.")
    else:
        df_new = pd.read_csv(up)

        proba, grade_vec, df2 = score(df_new, pipe, meta)
        out = df_new.copy()
        out["pred_prob"] = proba
        out["risk_grade"] = grade_vec

        # store for Explain tab
        st.session_state["last_out"] = out.reset_index(drop=True)
        st.session_state["last_df2"] = df2.reset_index(drop=True)

        # Summary cards
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows", f"{len(out):,}")
        s2.metric("High", f"{(out['risk_grade']=='High').sum():,}")
        s3.metric("Mid", f"{(out['risk_grade']=='Mid').sum():,}")
        s4.metric("Low", f"{(out['risk_grade']=='Low').sum():,}")

        # charts
        left, right = st.columns([1.2, 1])
        with left:
            fig = px.histogram(out, x="pred_prob", nbins=40, title="Predicted Probability Distribution")
            fig.add_vline(x=meta["mid_cut"], line_dash="dash", annotation_text="Mid cut", annotation_position="top left")
            fig.add_vline(x=meta["high_cut"], line_dash="dash", annotation_text="High cut", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            gcount = out["risk_grade"].value_counts().reindex(["High","Mid","Low"]).fillna(0).astype(int).reset_index()
            gcount.columns = ["risk_grade","count"]
            fig2 = px.bar(gcount, x="risk_grade", y="count", title="Risk Grade Count")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Scoring Results (상위 200행 미리보기)")
        st.dataframe(out.head(200), use_container_width=True)

        # download
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 결과 다운로드 (CSV)",
            data=csv_bytes,
            file_name="scoring_results.csv",
            mime="text/csv"
        )


# =========================
# ③ Explain (Reason Codes)
# =========================
with tabs[2]:
    st.subheader("Explainability: Reason Codes (고객 1명 기준 리스크 요인 Top 7)")
    st.caption("심사위원이 좋아하는 포인트: '왜 이 고객이 High인가?'를 한 눈에 보여주기")

    if "last_out" not in st.session_state:
        st.info("먼저 ② Score Customers 탭에서 CSV를 업로드해 점수화를 수행하세요.")
    else:
        out = st.session_state["last_out"]
        df2 = st.session_state["last_df2"]

        col1, col2 = st.columns([1, 1])
        with col1:
            idx = st.number_input("설명할 고객 행 번호", 0, len(out)-1, 0, step=1)
            st.write("선택 고객 요약")
            st.dataframe(out.loc[[idx]], use_container_width=True)

        with col2:
            st.write("Reason Codes (점수 하락 요인 Top 7)")
            reasons = top_reason_codes(pipe, df2.iloc[[idx]], top_k=7)
            st.dataframe(reasons, use_container_width=True)

            st.write("해석 가이드")
            st.markdown(
                "- impact가 **더 음수(작을수록)** → 해당 항목이 **리스크에 더 크게 기여**\n"
                "- 로지스틱 계수 기반으로 **설명 가능 + 운영 정책 연결 가능**"
            )


# =========================
# ④ WOE Insights
# =========================
with tabs[3]:
    st.subheader("WOE Insights: 산업×직업 조합 기반 ‘고용 안정성’")
    st.caption("조합 WOE는 ‘산업×직업 리스크 프로파일’을 수치화해, 희소 범주도 안정적으로 반영합니다.")

    # filter
    q = st.text_input("조합 검색 (예: '경기민감', '취약_단순')", "")
    wtab = combo_woe_table.reset_index().rename(columns={"index":"combo"}).copy()
    if q.strip():
        wtab = wtab[wtab["combo"].str.contains(q.strip(), na=False)]

    a, b = st.columns(2)
    with a:
        st.write("Most Risky (WOE 낮음) Top 10")
        st.dataframe(wtab.sort_values("woe").head(10), use_container_width=True)
    with b:
        st.write("Most Stable (WOE 높음) Top 10")
        st.dataframe(wtab.sort_values("woe", ascending=False).head(10), use_container_width=True)

    st.markdown("### 전체 WOE 테이블")
    st.dataframe(wtab, use_container_width=True)

    st.markdown("### 시각화 (WOE vs Bad Rate)")
    fig3 = px.scatter(
        wtab,
        x="woe",
        y="bad_rate",
        size="count",
        hover_name="combo",
        title="Combination Risk Map (WOE vs Bad Rate, bubble=count)"
    )
    st.plotly_chart(fig3, use_container_width=True)


# Footer
st.divider()
st.caption("Tip: 발표 때는 ①Overview에서 ‘정책+설명가능성’ → ②에서 ‘실제 점수화 데모’ → ③에서 ‘왜 High인지’ → ④에서 ‘도메인 인사이트’ 순으로 보여주면 설득력이 최고입니다.")
