import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


# =========================
# Utils
# =========================
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def collapse_rare(series: pd.Series, min_count=300, other="OTHERS"):
    vc = series.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return series.where(~series.isin(rare), other=other).astype("object")

def make_woe_map(series: pd.Series, y: pd.Series, smoothing=0.5):
    tmp = pd.DataFrame({"x": series, "y": y})
    g = tmp.groupby("x", observed=True)["y"]
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

# =========================
# Domain mappings (너희 확정본 반영)
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


def main():
    # ===== 사용자 설정 =====
    CSV_PATH = "train_data.csv"         # 여길 너 파일 경로로 바꾸기
    TARGET = "TARGET"
    ART_DIR = "artifacts"
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Feature engineering
    df = add_industry_reclass(df)
    df = add_job_ksco_7(df)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET], errors="ignore").copy()

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

    # Risk cuts (prob 기반)
    res = pd.DataFrame({"pred": pred, "target": y_test.values})
    high_cut = float(res["pred"].quantile(0.80))
    mid_cut  = float(res["pred"].quantile(0.40))

    # Save artifacts
    with open(os.path.join(ART_DIR, "model.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    meta = {
        "auc": float(auc),
        "ks": float(ks),
        "high_cut": high_cut,
        "mid_cut": mid_cut,
        "woe_map_combo": woe_map_combo,
        "drop_cols": drop_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        # 버전 관리용
        "random_state": 42,
        "min_count_combo": 300,
        "smoothing": 0.5,
    }
    with open(os.path.join(ART_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    combo_woe_table.to_parquet(os.path.join(ART_DIR, "combo_woe_table.parquet"))

    print("✅ Saved:")
    print(f"- {ART_DIR}/model.pkl")
    print(f"- {ART_DIR}/meta.pkl")
    print(f"- {ART_DIR}/combo_woe_table.parquet")
    print(f"✅ Metrics: AUC={auc:.6f}, KS={ks:.6f}, cuts: high={high_cut:.4f}, mid={mid_cut:.4f}")


if __name__ == "__main__":
    main()
