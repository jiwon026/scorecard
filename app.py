import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Same domain mappings (train.py랑 동일해야 함)
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
    art_dir = "artifacts"
    with open(os.path.join(art_dir, "model.pkl"), "rb") as f:
        pipe = pickle.load(f)
    with open(os.path.join(art_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    woe_table = pd.read_parquet(os.path.join(art_dir, "combo_woe_table.parquet"))
    return pipe, meta, woe_table


def score(df_in: pd.DataFrame, pipe, meta):
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
    return proba, grade_vec


# =========================
# UI
# =========================
st.set_page_config(page_title="연체 리스크 대시보드", layout="wide")
st.title("연체 리스크 대시보드 (PKL 로드형)")
pipe, meta, combo_woe_table = load_artifacts()

c1, c2, c3 = st.columns(3)
c1.metric("AUC (saved)", f"{meta['auc']:.4f}")
c2.metric("KS (saved)", f"{meta['ks']:.4f}")
c3.metric("High cut (80%)", f"{meta['high_cut']:.4f}")

tab1, tab2, tab3 = st.tabs(["Scoring", "WOE Table", "About"])

with tab1:
    st.subheader("고객 파일 업로드 후 스코어링")
    up = st.file_uploader("CSV 업로드 (TARGET 없어도 됨)", type=["csv"])
    if up is not None:
        df_new = pd.read_csv(up)
        proba, grade_vec = score(df_new, pipe, meta)

        out = df_new.copy()
        out["pred_prob"] = proba
        out["risk_grade"] = grade_vec

        st.dataframe(out.head(200))

        st.write("등급 분포")
        st.dataframe(out["risk_grade"].value_counts().rename("count"))

with tab2:
    st.subheader("산업×직업 조합 WOE 테이블 (train 기준)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Most Risky (WOE 낮음) Top 10")
        st.dataframe(combo_woe_table.sort_values("woe").head(10))
    with col2:
        st.write("Most Stable (WOE 높음) Top 10")
        st.dataframe(combo_woe_table.sort_values("woe", ascending=False).head(10))
    st.write("전체 테이블")
    st.dataframe(combo_woe_table)

with tab3:
    st.markdown(
        f"""
- Champion Model: Logistic + 산업/직업 재분류 + 고용안정성_WOE  
- 저장된 cut: High >= {meta['high_cut']:.4f}, Mid >= {meta['mid_cut']:.4f}
"""
    )
