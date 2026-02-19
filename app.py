import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# =========================================================
# Utils
# =========================================================
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def collapse_rare(series: pd.Series, min_count=300, other="OTHERS"):
    vc = series.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return series.where(~series.isin(rare), other=other).astype("object")

def make_woe_map(series: pd.Series, y: pd.Series, smoothing=0.5):
    tmp = pd.DataFrame({"x": series, "y": y})
    g = tmp.groupby("x")["y"]
    bad = g.sum()
    total = g.count()
    good = total - bad

    bad_total = bad.sum()
    good_total = good.sum()

    bad_dist = (bad + smoothing) / (bad_total + smoothing * len(bad))
    good_dist = (good + smoothing) / (good_total + smoothing * len(good))

    woe = np.log(good_dist / bad_dist)
    return woe.to_dict()

def build_combo_woe_table(combo_train: pd.Series, y_train: pd.Series, smoothing=0.5):
    tmp = pd.DataFrame({"combo": combo_train, "y": y_train})
    g = tmp.groupby("combo", observed=True)["y"]
    bad = g.sum()
    total = g.count()
    good = total - bad

    bad_total = bad.sum()
    good_total = good.sum()

    bad_dist = (bad + smoothing) / (bad_total + smoothing * len(bad))
    good_dist = (good + smoothing) / (good_total + smoothing * len(good))

    woe = np.log(good_dist / bad_dist)

    out = pd.DataFrame({
        "count": total,
        "bad": bad,
        "bad_rate": bad / total,
        "woe": woe
    }).sort_values("woe")
    return out

def top_reason_codes(pipe, X_row: pd.DataFrame, factor: float, top_k=7):
    """
    factor = pdo/log(2) 와 같은 scale factor가 아니라,
    여기서는 '포인트 기여도'를 보고 싶으면 factor를 크게 잡아도 되는데,
    우리는 상대적 중요도만 보므로 factor=1.0으로 써도 OK.
    (너는 이미 scorecard meta를 쓰기도 했으니 필요시 연결 가능)
    """
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    feat_names = prep.get_feature_names_out()
    coef = model.coef_.ravel()

    X_mat = prep.transform(X_row)
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    x_vec = X_mat.ravel()

    # "점수" 관점의 기여도(부호는 해석 편의용)
    contrib = -(factor * (coef * x_vec))

    dfc = pd.DataFrame({"feature": feat_names, "point_contrib": contrib})
    dfc["feature"] = (dfc["feature"]
                      .str.replace("cat__", "", regex=False)
                      .str.replace("num__", "", regex=False))

    # 점수를 가장 많이 깎는(음수 큰) 항목이 리스크 요인
    dfc = dfc.sort_values("point_contrib").head(top_k)
    return dfc

# =========================================================
# Domain mappings (너희가 확정한 버전)
# =========================================================
def add_industry_reclass(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "산업군_상위추출" not in out.columns:
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
    out["산업군_재분류"] = out["산업군_상위추출"].map(industry_map).fillna("기타")
    return out

def add_job_ksco_7(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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

# =========================================================
# Train model inside app (demo version)
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    return df

@st.cache_resource(show_spinner=True)
def train_champion(df: pd.DataFrame):
    TARGET = "TARGET"
    df2 = df.copy()

    # Feature engineering
    df2 = add_industry_reclass(df2)
    df2 = add_job_ksco_7(df2)

    y = df2[TARGET].astype(int)
    X = df2.drop(columns=[TARGET], errors="ignore").copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Combo WOE (train 기준)
    combo_train = (X_train["산업군_재분류"].astype(str) + " × " + X_train["직업_KSCO_7그룹"].astype(str))
    combo_test  = (X_test["산업군_재분류"].astype(str)  + " × " + X_test["직업_KSCO_7그룹"].astype(str))

    combo_train = collapse_rare(combo_train, min_count=300, other="OTHERS")
    combo_test  = combo_test.where(combo_test.isin(combo_train.unique()), "OTHERS").astype("object")

    woe_map_combo = make_woe_map(combo_train, y_train, smoothing=0.5)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["고용안정성_WOE"] = combo_train.map(woe_map_combo).fillna(0.0).astype(float)
    X_test["고용안정성_WOE"]  = combo_test.map(woe_map_combo).fillna(0.0).astype(float)

    # WOE 테이블(발표용)
    combo_woe_table = build_combo_woe_table(combo_train, y_train, smoothing=0.5)

    # Drop raw cols to avoid mixing
    drop_cols = ["직업", "산업군", "산업군_상위", "산업군_상위추출"]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test  = X_test.drop(columns=drop_cols, errors="ignore")

    # Preprocess + Logistic
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipe = Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=4000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    ks = ks_stat(y_test, pred)

    # Risk grade (prob 기준)
    res = pd.DataFrame({"pred": pred, "target": y_test.values})
    high_cut = res["pred"].quantile(0.80)
    mid_cut = res["pred"].quantile(0.40)

    def grade(p):
        if p >= high_cut:
            return "High"
        elif p >= mid_cut:
            return "Mid"
        else:
            return "Low"

    res["risk_grade"] = res["pred"].apply(grade)
    grade_summary = (res.groupby("risk_grade")
                     .agg(count=("target","count"), bad_rate=("target","mean"), avg_pred=("pred","mean"))
                     .sort_values("bad_rate", ascending=False))

    overall_bad = res["target"].mean()
    lift = float(grade_summary.loc["High","bad_rate"] / overall_bad) if "High" in grade_summary.index else np.nan

    artifacts = {
        "pipe": pipe,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "pred_test": pred,
        "auc": auc,
        "ks": ks,
        "grade_summary": grade_summary,
        "lift": lift,
        "high_cut": high_cut,
        "mid_cut": mid_cut,
        "woe_map_combo": woe_map_combo,
        "combo_woe_table": combo_woe_table,
        "drop_cols": drop_cols
    }
    return artifacts

def score_new(df_new: pd.DataFrame, artifacts):
    """새 데이터(1행/여러행) 점수화: 동일 FE + 고용안정성_WOE + drop + pipe.predict_proba"""
    out = df_new.copy()
    out = add_industry_reclass(out)
    out = add_job_ksco_7(out)

    # combo 생성 + train 기준 woe_map 적용
    combo = (out["산업군_재분류"].astype(str) + " × " + out["직업_KSCO_7그룹"].astype(str))
    # unseen -> OTHERS로
    seen = set(artifacts["woe_map_combo"].keys())
    combo = combo.where(combo.isin(seen), "OTHERS").astype("object")

    out["고용안정성_WOE"] = combo.map(artifacts["woe_map_combo"]).fillna(0.0).astype(float)

    out = out.drop(columns=artifacts["drop_cols"], errors="ignore")

    # 예측
    proba = artifacts["pipe"].predict_proba(out)[:, 1]

    # 등급
    high_cut = artifacts["high_cut"]
    mid_cut = artifacts["mid_cut"]

    def grade(p):
        if p >= high_cut:
            return "High"
        elif p >= mid_cut:
            return "Mid"
        else:
            return "Low"

    grade_vec = pd.Series(proba).apply(grade).values
    return proba, grade_vec, out

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="연체 리스크 스코어 대시보드", layout="wide")

st.title("카드 고객 연체 리스크 예측 대시보드 (Logistic + 고용안정성_WOE)")
st.caption("Champion Model: 산업군/직업 재분류 + 산업×직업 조합 WOE(고용안정성_WOE) + Logistic Regression")

with st.sidebar:
    st.header("데이터 로드")
    uploaded = st.file_uploader("train_data.csv 업로드", type=["csv"])
    st.divider()
    st.header("메뉴")
    page = st.radio("이동", ["Overview", "Scoring", "Risk Segmentation", "WOE Table"], index=0)

if uploaded is None:
    st.info("왼쪽에서 train_data.csv를 업로드해줘.")
    st.stop()

df = load_data(uploaded)
art = train_champion(df)

# --------------------
# Overview
# --------------------
if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", f"{art['auc']:.4f}")
    c2.metric("KS", f"{art['ks']:.4f}")
    c3.metric("High Lift (vs overall)", f"{art['lift']:.2f}")

    st.subheader("데이터 개요")
    col1, col2 = st.columns(2)
    with col1:
        st.write("TARGET 분포")
        st.dataframe(df["TARGET"].value_counts().rename("count"))
    with col2:
        st.write("직업/산업 재분류 예시")
        tmp = add_job_ksco_7(add_industry_reclass(df.copy()))
        st.dataframe(tmp[["직업","직업_KSCO_7그룹","산업군","산업군_재분류"]].head(10))

    st.subheader("모델 입력 변수(학습 기준)")
    st.write(f"- 수치형: {len(art['X_train'].select_dtypes(exclude='object').columns)}개")
    st.write(f"- 범주형: {len(art['X_train'].select_dtypes(include='object').columns)}개")
    st.dataframe(pd.DataFrame({"columns": art["X_train"].columns}))

# --------------------
# Scoring
# --------------------
elif page == "Scoring":
    st.subheader("고객 점수화(예측 확률/등급 + Reason Codes)")

    st.markdown("**방법 1)** test set에서 샘플 고객 선택 (데모용)")
    idx = st.number_input("샘플 인덱스(0 ~ test_size-1)", min_value=0, max_value=len(art["X_test"])-1, value=0, step=1)

    # 원본 df에서 동일 인덱스를 쓰기 어렵기 때문에: X_test(이미 FE/WOE/Drop 된 상태)가 아니라
    # 다시 df에서 해당 row를 가져와 score_new로 점수화
    # 여기서는 편의상 df 전체에서 랜덤 1행 뽑기 옵션도 제공
    st.divider()
    st.markdown("**방법 2)** 새 고객 데이터 1행 업로드")
    up2 = st.file_uploader("new_customer.csv (헤더 포함, 1행 또는 여러행)", type=["csv"], key="new_customer")

    if up2 is not None:
        new_df = pd.read_csv(up2)
        proba, grade_vec, model_X = score_new(new_df, art)

        st.write("예측 결과")
        out = new_df.copy()
        out["pred_prob"] = proba
        out["risk_grade"] = grade_vec
        st.dataframe(out.head(50))

        # 첫 행 reason codes
        st.write("Reason codes (첫 번째 고객)")
        rc = top_reason_codes(art["pipe"], model_X.iloc[[0]], factor=1.0, top_k=7)
        st.dataframe(rc)

    else:
        # test 샘플: df에서 랜덤/고정 선택이 어려우니, df에서 임의 1행을 보여주고 점수화
        row = df.sample(1, random_state=int(idx)).drop(columns=["TARGET"], errors="ignore")
        proba, grade_vec, model_X = score_new(row, art)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Probability", f"{proba[0]:.4f}")
        c2.metric("Risk Grade", grade_vec[0])
        c3.metric("High Cut (80%)", f"{art['high_cut']:.4f}")

        st.write("입력 고객(원본 일부)")
        st.dataframe(row)

        st.write("Reason Codes (점수 하락 요인 Top 7)")
        rc = top_reason_codes(art["pipe"], model_X.iloc[[0]], factor=1.0, top_k=7)
        st.dataframe(rc)

# --------------------
# Risk Segmentation
# --------------------
elif page == "Risk Segmentation":
    st.subheader("3등급 리스크 분리 결과 (Test set)")
    st.dataframe(art["grade_summary"])

    st.write(f"High Lift (vs overall): **{art['lift']:.2f}**")

    st.markdown("### 발표용 코멘트 예시")
    st.info(
        f"예측 확률 상위 20%를 High로 분류했을 때 실제 연체율은 "
        f"{art['grade_summary'].loc['High','bad_rate']*100:.1f}%로, "
        f"전체 평균 대비 약 {art['lift']:.2f}배 높게 나타났습니다."
    )

# --------------------
# WOE Table
# --------------------
elif page == "WOE Table":
    st.subheader("산업×직업 조합 WOE 테이블 (train 기준)")
    wtab = art["combo_woe_table"].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.write("Most Risky (WOE 낮음) Top 10")
        st.dataframe(wtab.sort_values("woe").head(10))
    with c2:
        st.write("Most Stable (WOE 높음) Top 10")
        st.dataframe(wtab.sort_values("woe", ascending=False).head(10))

    st.write("전체 테이블")
    st.dataframe(wtab)
