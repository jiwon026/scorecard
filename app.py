import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# ---- 한글 설정 ----
def set_korean_font():
    base_dir = Path(__file__).resolve().parent
    font_path = base_dir / "artifacts" / "NanumGothic.ttf"

    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        mpl.rcParams["font.family"] = font_name
    else:
        # 폰트 없으면 깨질 수 있으니 경고
        mpl.rcParams["font.family"] = "sans-serif"
        st.warning("한글 폰트 파일(artifacts/NanumGothic.ttf)이 없어 한글이 깨질 수 있습니다.")

    mpl.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# 0) Excel Loader
# =========================
def safe_parse(xls: pd.ExcelFile, candidates, required=True) -> pd.DataFrame:
    # 후보 시트명들을 순서대로 시도해서 있으면 parse
    sheet_names = list(xls.sheet_names)

    # 1) 정확 매칭 (대소문자/공백 무시)
    norm = {s.strip().lower(): s for s in sheet_names}
    for c in candidates:
        key = str(c).strip().lower()
        if key in norm:
            return xls.parse(norm[key])

    # 2) 부분 포함 매칭 (예: scorecard_flag_v2 같은 경우)
    for s in sheet_names:
        s_norm = s.strip().lower()
        for c in candidates:
            if str(c).strip().lower() in s_norm:
                return xls.parse(s)

    # 3) 없으면 처리
    if required:
        raise ValueError(f"Worksheet not found. tried={candidates}, available={sheet_names}")
    return pd.DataFrame()

@st.cache_resource
def load_scorecard_excel(_version: str = "v1"):
    base_dir = Path(__file__).resolve().parent
    art_dir = base_dir / "artifacts"
    xlsx_path = art_dir / "scorecard_export.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"scorecard_export.xlsx not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")

    meta_df = xls.parse("meta")
    meta = dict(zip(meta_df["key"], meta_df["value"]))

    coef_df = xls.parse("model_coef")
    intercept = float(coef_df.loc[coef_df["feature"].eq("const"), "beta"].iloc[0])

    factor = float(meta["factor"])
    offset = float(meta["offset"])
    base_points = float(offset - factor * intercept)

    cont = xls.parse("scorecard_cont")
    if "feature" not in cont.columns and "bin" in cont.columns:
        cont = cont.rename(columns={"bin": "feature"})      # feature/points (섹션형)
    cat   = xls.parse("scorecard_cat")       # feature/points (섹션형)
    cross = xls.parse("scorecard_cross")     # feature/points (섹션형)

    binary = xls.parse("scorecard_binary")   # feature/value/points (섹션+행)
    flag = safe_parse(
        xls,
        ["scorecard_flag", "flag", "flags", "scorecard_flags"],
        required=False
    )
    job_ind = xls.parse("job_ind_points")    # 직업/산업군/points

    # meta에 유용 정보 추가
    meta2 = {
        "factor": factor,
        "offset": offset,
        "intercept": intercept,
        "base_points": base_points,
        "base_score": float(meta.get("base_score", np.nan)),
        "PDO": float(meta.get("PDO", np.nan)),
        "base_odds": float(meta.get("base_odds", np.nan)),
    }

    return meta2, cont, cat, cross, binary, flag, job_ind


# =========================
# 1) Section Parser (cont/cat/cross)
# =========================
def parse_section_sheet(df: pd.DataFrame):
    """
    df columns: feature, points
    Rows like:
      [나이_woe]   NaN
      23세 이하     -20.4
      24세 ~ 30세  -12.4
      ...
    return: dict[group_name][label] = points
    """
    out = {}
    current = None
    for _, r in df.iterrows():
        f = r.get("feature")
        p = r.get("points")

        if isinstance(f, str) and f.startswith("[") and f.endswith("]"):
            current = f.strip("[]").strip()
            out.setdefault(current, {})
            continue

        if current is None:
            continue
        if pd.isna(f) or pd.isna(p):
            continue

        label = str(f).strip()
        out[current][label] = float(p)

    # points가 하나도 없는 group은 제거(예: 연간수입_woe가 비어있는 경우)
    out = {g: m for g, m in out.items() if len(m) > 0}
    return out


def build_value_points_map(df: pd.DataFrame):
    """
    df columns: feature, value, points
    섹션 헤더/빈줄 포함됨:
      [한부모 가정] NaN NaN
      한부모 가정 1 -28
      한부모 가정 0  0
    return: dict[feature][value] = points
    """
    if df.empty:
        return {}

    need = {"feature", "value", "points"}
    if not need.issubset(df.columns):
        return {}

    mp = {}
    for _, r in df.iterrows():
        f = r.get("feature")
        v = r.get("value")
        p = r.get("points")
        if pd.isna(f) or pd.isna(v) or pd.isna(p):
            continue

        f = str(f).strip()
        # 헤더 [xxx]는 value/points가 nan이라 이미 걸러짐
        try:
            v_key = int(float(v))
        except Exception:
            continue

        mp.setdefault(f, {})[v_key] = float(p)

    return mp


def build_job_ind_map(job_ind_df: pd.DataFrame):
    """
    columns: 직업, 산업군, points
    return dict[(job, industry)] = points
    """
    m = {}
    for _, r in job_ind_df.iterrows():
        j = r.get("직업")
        i = r.get("산업군")
        p = r.get("points")
        if pd.isna(j) or pd.isna(i) or pd.isna(p):
            continue
        m[(str(j).strip(), str(i).strip())] = float(p)
    return m


# =========================
# 2) Labeling (continuous bins: label로 매칭)
#    - label 텍스트를 보고 숫자에 맞는 구간을 찾아줌
# =========================
def _to_int(s: str):
    # "12,744,000원" -> 12744000
    s = s.replace(",", "")
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s else None


def match_numeric_label(x: float, labels):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    NUM_RE = r"\d+(?:\.\d+)?"
    def _nums(text: str):
        return re.findall(NUM_RE, str(text))

    for lab in labels:
        t = str(lab).strip()

        # "23세 이하"
        if "이하" in t and "~" not in t and "," not in t:
            a = _to_int(t) if ("원" in t) else None
            if a is None:
                nums = _nums(t)
                if nums:
                    a = float(nums[0])
            if a is not None and x <= a:
                return lab

        # "19년 이상"
        if "이상" in t and "~" not in t and "," not in t:
            a = _to_int(t) if ("원" in t) else None
            if a is None:
                nums = _nums(t)
                if nums:
                    a = float(nums[0])
            if a is not None and x >= a:
                return lab

        # "24세 ~ 30세"
        if "~" in t:
            parts = [p.strip() for p in t.split("~")]
            if len(parts) >= 2:
                lo_s, hi_s = parts[0], parts[1]
                lo = _to_int(lo_s) if ("원" in t) else None
                hi = _to_int(hi_s) if ("원" in t) else None

                if lo is None:
                    nums = _nums(lo_s)
                    lo = float(nums[0]) if nums else None
                if hi is None:
                    nums = _nums(hi_s)
                    hi = float(nums[0]) if nums else None

                if lo is not None and hi is not None and lo <= x <= hi:
                    return lab

        # "5, 6년"
        if "," in t and "~" not in t:
            nums = _nums(t)
            if nums:
                vals = set(float(n) for n in nums)
                if float(x) in vals:
                    return lab

    return None


# =========================
# 3) Score functions
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def score_one(row: dict,
              meta,
              cont_map,
              cat_map,
              cross_map,
              bin_map,
              flag_map,
              job_ind_map):
    """
    row: raw input dict
    return: (score, proba, grade, breakdown_df)
    """

    score = meta["base_points"]
    breakdown = []

    # ---- continuous: 근속연수, 나이, 거주지 인구 비율, 가입연수, 가족 구성원 수, (연간수입은 점수 없으면 자동 스킵)
    cont_inputs = {
        "근속연수_woe": float(row.get("근속연수", np.nan)),
        "나이_woe": float(row.get("나이", np.nan)),
        "거주지 인구 비율_woe": float(row.get("거주지 인구 비율", np.nan)),
        "가입연수_woe": float(row.get("가입연수", np.nan)),
        "가족 구성원 수_woe": float(row.get("가족 구성원 수", np.nan)),
        "연간수입_woe": float(row.get("연간 수입", np.nan)),
    }

    for feat, x in cont_inputs.items():
        if feat not in cont_map:
            continue
        labels = list(cont_map[feat].keys())
        lab = match_numeric_label(x, labels)
        if lab is None:
            continue
        pts = cont_map[feat].get(lab, 0.0)
        score += pts
        breakdown.append((feat, lab, pts))

    # ---- categorical: 수입 유형, 최종 학력, 결혼 여부, 주거 형태, 자녀수_구간 등
    cat_inputs = {
        "수입 유형_woe": row.get("수입 유형"),
        "최종 학력_woe": row.get("최종 학력"),
        "결혼 여부_woe": row.get("결혼 여부"),
        "주거 형태_woe": row.get("주거 형태"),
        "자녀수_구간_woe": row.get("자녀수_구간"),
    }

    for feat, val in cat_inputs.items():
        if feat not in cat_map:
            continue
        if val is None:
            continue
        val = str(val).strip()
        pts = cat_map[feat].get(val, 0.0)
        score += pts
        breakdown.append((feat, val, pts))

    # ---- binary: 차량/부동산/업무용폰/이메일 등 (엑셀에 있는 것만 자동 반영)
    # 점수카드 시트의 feature명을 그대로 씀
    for f in bin_map.keys():
        # row에 있는 키를 그대로 사용(예: "차량 소유 여부")
        if f in row:
            v = int(row[f])
            pts = bin_map[f].get(v, 0.0)
            score += pts
            breakdown.append((f, v, pts))

    # ---- cross: 성별x결혼여부_woe
    if "성별x결혼여부_woe" in cross_map:
        sex = row.get("성별", "남성")
        mar = row.get("결혼 여부", "미혼")
        sex_code = 1 if str(sex).strip() == "남성" else 0
        key = f"{sex_code}_{mar}"
        pts = cross_map["성별x결혼여부_woe"].get(key, 0.0)
        score += pts
        breakdown.append(("성별x결혼여부_woe", key, pts))

    # ---- flags: 한부모 가정 / 저소득_부동산 X / 2,30대_저학력,고졸
    if "한부모 가정" in flag_map:
        spouse = int(row.get("배우자유무", 1))
        child_bucket = str(row.get("자녀수_구간", "0")).strip()
        child_n = 3 if child_bucket == "3+" else int(child_bucket)
        is_single_parent = 1 if (spouse == 0 and child_n > 0) else 0
        pts = flag_map["한부모 가정"].get(is_single_parent, 0.0)
        score += pts
        breakdown.append(("한부모 가정", is_single_parent, pts))

    if "저소득_부동산 X" in flag_map:
        income = float(row.get("연간 수입", np.nan))
        real_estate = int(row.get("부동산 소유 여부", 0))
        # 저소득 기준(임시): 37,170,000 이하
        is_low_income = 1 if (not np.isnan(income) and income <= 37_170_000) else 0
        flag_val = 1 if (is_low_income == 1 and real_estate == 0) else 0
        pts = flag_map["저소득_부동산 X"].get(flag_val, 0.0)
        score += pts
        breakdown.append(("저소득_부동산 X", flag_val, pts))

    if "2,30대_저학력,고졸" in flag_map:
        age = float(row.get("나이", np.nan))
        edu = str(row.get("최종 학력", "")).strip()
        flag_val = 1 if ((not np.isnan(age) and age < 40) and (edu in ["저학력자", "고등학교 졸업"])) else 0
        pts = flag_map["2,30대_저학력,고졸"].get(flag_val, 0.0)
        score += pts
        breakdown.append(("2,30대_저학력,고졸", flag_val, pts))

    # ---- job x industry points
    job = str(row.get("직업", "Unknown")).strip()
    ind = str(row.get("산업군", "")).strip()
    pts = job_ind_map.get((job, ind), 0.0)
    score += pts
    breakdown.append(("직업×산업군", f"{job} × {ind}", pts))

    # ---- probability (score -> logit)
    logit = (meta["offset"] - score) / meta["factor"]
    proba = float(sigmoid(logit))

    # ---- grade (점수컷 기준 4등급)
    if score >= 670:
        grade = "우대"
    elif score >= 600:
        grade = "안정"
    elif score >= 550:
        grade = "위험"
    else:
        grade = "고위험"

    bd = pd.DataFrame(breakdown, columns=["feature", "value/bin", "points"]).sort_values("points")
    return score, proba, grade, bd

# =========================
# Portfolio KPI helpers
# =========================
def score_one_fast(row: dict,
                   meta,
                   cont_map,
                   cat_map,
                   cross_map,
                   bin_map,
                   flag_map,
                   job_ind_map):
    """
    포트폴리오 일괄 산출용 경량 버전(안전 캐스팅 포함)
    - breakdown 미생성
    - score, proba만 반환
    - 빈 문자열/쉼표 포함 숫자/NaN 혼재에도 죽지 않음
    """

    def _as_float(v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return np.nan
            v = v.replace(",", "")
        try:
            return float(v)
        except Exception:
            return np.nan

    def _as_int(v, default=0):
        if v is None:
            return default
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return default
            v = v.replace(",", "")
        try:
            return int(float(v))
        except Exception:
            return default

    score = float(meta.get("base_points", 0.0))

    # ---- continuous
    cont_inputs = {
        "근속연수_woe": _as_float(row.get("근속연수")),
        "나이_woe": _as_float(row.get("나이")),
        "거주지 인구 비율_woe": _as_float(row.get("거주지 인구 비율")),
        "가입연수_woe": _as_float(row.get("가입연수")),
        "가족 구성원 수_woe": _as_float(row.get("가족 구성원 수")),
        "연간수입_woe": _as_float(row.get("연간 수입")),
    }

    for feat, x in cont_inputs.items():
        if feat not in cont_map:
            continue
        labels = list(cont_map[feat].keys())
        lab = match_numeric_label(x, labels)
        if lab is None:
            continue
        score += float(cont_map[feat].get(lab, 0.0))

    # ---- categorical
    cat_inputs = {
        "수입 유형_woe": row.get("수입 유형"),
        "최종 학력_woe": row.get("최종 학력"),
        "결혼 여부_woe": row.get("결혼 여부"),
        "주거 형태_woe": row.get("주거 형태"),
        "자녀수_구간_woe": row.get("자녀수_구간"),
    }
    for feat, val in cat_inputs.items():
        if feat not in cat_map or val is None:
            continue
        v = str(val).strip()
        if v == "":
            continue
        score += float(cat_map[feat].get(v, 0.0))

    # ---- binary (엑셀 점수카드 feature명을 그대로 사용)
    for f in bin_map.keys():
        if f in row:
            v = _as_int(row.get(f), default=0)
            score += float(bin_map[f].get(v, 0.0))

    # ---- cross
    if "성별x결혼여부_woe" in cross_map:
        sex = str(row.get("성별", "남성")).strip()
        mar = str(row.get("결혼 여부", "미혼")).strip()
        sex_code = 1 if sex == "남성" else 0
        key = f"{sex_code}_{mar}"
        score += float(cross_map["성별x결혼여부_woe"].get(key, 0.0))

    # ---- flags (flag_map 비어있으면 자동 스킵)
    if flag_map:
        if "한부모 가정" in flag_map:
            spouse = _as_int(row.get("배우자유무"), default=1)
            child_n = _as_int(row.get("자녀 수"), default=0)
            is_single_parent = 1 if (spouse == 0 and child_n > 0) else 0
            score += float(flag_map["한부모 가정"].get(is_single_parent, 0.0))

        if "저소득_부동산 X" in flag_map:
            income = _as_float(row.get("연간 수입"))
            real_estate = _as_int(row.get("부동산 소유 여부"), default=0)
            is_low_income = 1 if (not np.isnan(income) and income <= 37_170_000) else 0
            flag_val = 1 if (is_low_income == 1 and real_estate == 0) else 0
            score += float(flag_map["저소득_부동산 X"].get(flag_val, 0.0))

        if "2,30대_저학력,고졸" in flag_map:
            age = _as_float(row.get("나이"))
            edu = str(row.get("최종 학력", "")).strip()
            flag_val = 1 if ((not np.isnan(age) and age < 40) and (edu in ["저학력자", "고등학교 졸업"])) else 0
            score += float(flag_map["2,30대_저학력,고졸"].get(flag_val, 0.0))

    # ---- job x industry
    job = str(row.get("직업", "Unknown")).strip()
    ind = str(row.get("산업군", "")).strip()
    score += float(job_ind_map.get((job, ind), 0.0))

    # ---- probability (score -> logit)
    factor = float(meta.get("factor", 1.0))
    offset = float(meta.get("offset", 0.0))
    logit = (offset - score) / factor
    proba = float(sigmoid(logit))
    return score, proba
                       
def load_sample_df(art_dir: Path) -> pd.DataFrame:
    """
    artifacts 폴더에서 샘플 데이터를 읽어옴.
    우선순위:
      1) sample_scoring.parquet
      2) sample_scoring.csv
    """
    pqt = art_dir / "sample_scoring.parquet"
    csv = art_dir / "sample_scoring.csv"

    if pqt.exists():
        try:
            return pd.read_parquet(pqt)  # pyarrow/fastparquet 필요
        except Exception as e:
            st.warning(
                "sample_scoring.parquet를 읽지 못해 CSV로 시도합니다. "
                "parquet를 쓰려면 requirements.txt에 pyarrow를 추가하세요."
            )

    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError(
        f"샘플 파일이 없습니다. 다음 중 하나를 artifacts/에 넣어주세요: {pqt.name} 또는 {csv.name}"
    )                       


@st.cache_data(show_spinner=False)
def compute_portfolio_kpis_from_df(df: pd.DataFrame,
                                  meta,
                                  cont_map,
                                  cat_map,
                                  cross_map,
                                  bin_map,
                                  flag_map,
                                  job_ind_map):
    """
    샘플(포트폴리오) KPI 계산 - '점수 컷(우대/안정/위험/고위험)' 기준

    반환 키(Overview/그래프 호환):
      - n
      - avg_pd                (0~1)
      - high_share            (고위험 비중, 0~1)
      - high_avg_pd           (고위험 평균 PD, 0~1)
      - grade_dist            (DataFrame: grade,count,share[%])
      - overall_dr            (0~1, TARGET 있을 때)
      - dr_by_grade           (DataFrame: grade, dr[%])
      - lift_by_grade         (DataFrame: grade, lift_vs_overall)
    """
    df = df.copy()
    df = df.replace({"": np.nan, " ": np.nan})

    # -------------------------
    # 0) 타입/결측 정리(안전)
    # -------------------------
    num_cols = ["나이","근속연수","가입연수","연간 수입","거주지 인구 비율","가족 구성원 수","자녀 수"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bin_cols = ["차량 소유 여부","부동산 소유 여부","배우자유무"]
    for c in bin_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 문자열 컬럼 strip(있으면)
    str_cols = ["성별","결혼 여부","직업","산업군","수입 유형","최종 학력","주거 형태","자녀수_구간"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # -------------------------
    # 1) score / proba 산출
    # -------------------------
    records = df.to_dict("records")
    n = len(records)

    scores = np.empty(n, dtype=float)
    probas = np.empty(n, dtype=float)

    for i, r in enumerate(records):
        s, p = score_one_fast(r, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map)
        scores[i] = s
        probas[i] = p

    # -------------------------
    # 2) 점수 컷으로 등급(4단계) 부여
    #    고위험: ~549
    #    위험  : 550~599
    #    안정  : 600~669
    #    우대  : 670~
    # -------------------------
    def score_to_grade4(s: float) -> str:
        if s <= 549:
            return "고위험"
        elif s <= 599:
            return "위험"
        elif s <= 669:
            return "안정"
        else:
            return "우대"

    grade4 = np.array([score_to_grade4(s) for s in scores])

    # 시각화/보고서용 정렬(좋음 -> 나쁨)
    grade_order = ["우대", "안정", "위험", "고위험"]

    # -------------------------
    # 3) Overview KPI(4개)
    # -------------------------
    avg_pd = float(np.mean(probas)) if n else np.nan

    high_mask = (grade4 == "고위험")
    high_share = float(high_mask.mean()) if n else np.nan
    high_avg_pd = float(np.mean(probas[high_mask])) if np.any(high_mask) else np.nan

    out = {
        "n": int(n),
        "avg_pd": avg_pd,                 # 0~1
        "high_share": high_share,         # 0~1
        "high_avg_pd": high_avg_pd,       # 0~1
    }

    # -------------------------
    # 4) 등급 분포(share[%]) - 그래프 막대용
    # -------------------------
    cnt = pd.Series(grade4).value_counts().reindex(grade_order, fill_value=0)
    out["grade_dist"] = pd.DataFrame({
        "grade": cnt.index,
        "count": cnt.values,
        "share": (cnt.values / n * 100.0) if n else 0.0
    })

    # -------------------------
    # 5) TARGET 있으면 실제 연체율/리프트
    # -------------------------
    if "TARGET" in df.columns:
        y = pd.to_numeric(df["TARGET"], errors="coerce").fillna(0).astype(int).values
        overall_dr = float(y.mean()) if len(y) else np.nan
        out["overall_dr"] = overall_dr  # 0~1

        dr_rows = []
        lift_rows = []

        for g in grade_order:
            m = (grade4 == g)
            dr = float(y[m].mean()) if np.any(m) else np.nan          # 0~1
            dr_pct = dr * 100.0 if not np.isnan(dr) else np.nan      # %
            lift = (dr / overall_dr) if (not np.isnan(dr) and not np.isnan(overall_dr) and overall_dr > 0) else np.nan

            dr_rows.append({"grade": g, "dr": dr_pct})
            lift_rows.append({"grade": g, "lift": lift})

        out["dr_by_grade"] = pd.DataFrame(dr_rows)         # grade, dr(%)
        out["lift_by_grade"] = pd.DataFrame(lift_rows)     # grade, lift
    else:
        out["overall_dr"] = np.nan
        out["dr_by_grade"] = pd.DataFrame({"grade": grade_order, "dr": [np.nan]*len(grade_order)})
        out["lift_by_grade"] = pd.DataFrame({"grade": grade_order, "lift": [np.nan]*len(grade_order)})

    return out
def build_upper_industry_options_and_rep(job_ind_df: pd.DataFrame, sample_path: Path):
    def _upper_from_full(full: str) -> str:
        full = str(full).strip()
        parts = full.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return " ".join(parts[:-1]).strip()
        m = re.match(r"^(.*?)(\d+)$", full)
        if m:
            return m.group(1).strip()
        return full

    sample_path = Path(sample_path)
    df_s = pd.read_parquet(sample_path) if sample_path.suffix.lower() == ".parquet" else pd.read_csv(sample_path)

    if "산업군_상위" not in df_s.columns:
        raise ValueError("샘플 파일에 '산업군_상위' 컬럼이 없습니다.")

    options = df_s["산업군_상위"].dropna().astype(str).str.strip().unique().tolist()

    banned = {"무역", "산업", "운송"}
    options = sorted([x for x in options if x not in banned])

    rep = {}
    if "산업군" in job_ind_df.columns:
        full_list = job_ind_df["산업군"].dropna().astype(str).tolist()
        for full in full_list:
            up = _upper_from_full(full)
            if up not in rep:
                rep[up] = full

    return options, rep

def enrich_with_score(df: pd.DataFrame,
                      meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map):
    df2 = df.copy()

    # score/proba 계산
    recs = df2.to_dict("records")
    scores = np.empty(len(recs), dtype=float)
    probas = np.empty(len(recs), dtype=float)

    for i, r in enumerate(recs):
        s, p = score_one_fast(r, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map)
        scores[i] = s
        probas[i] = p

    df2["score"] = scores
    df2["proba"] = probas

    # ✅ 너희가 정한 등급컷(점수 기반) 반영
    # 고위험: ~549 / 위험: 550~599 / 안정: 600~669 / 우대: 670~
    def grade4(score):
        if score <= 549: return "고위험"
        if score <= 599: return "위험"
        if score <= 669: return "안정"
        return "우대"

    df2["grade4"] = df2["score"].apply(grade4)

    # TARGET 정리(있으면)
    if "TARGET" in df2.columns:
        df2["TARGET"] = pd.to_numeric(df2["TARGET"], errors="coerce").fillna(0).astype(int)

    return df2


def compute_lift_table(df_all: pd.DataFrame, df_seg: pd.DataFrame, col: str):
    """
    col별로: 고위험군 최빈값/구성비/전체비/위험집중도(Lift) 계산
    Lift = seg_share / all_share
    """
    a = df_all[col].dropna().astype(str).str.strip()
    s = df_seg[col].dropna().astype(str).str.strip()

    if len(a) == 0 or len(s) == 0:
        return None

    all_dist = a.value_counts(normalize=True)
    seg_dist = s.value_counts(normalize=True)

    # seg에서 가장 흔한 값(대표 특성)
    top_val = seg_dist.index[0]
    seg_share = float(seg_dist.get(top_val, 0.0))
    all_share = float(all_dist.get(top_val, 0.0))
    lift = (seg_share / all_share) if all_share > 0 else np.nan

    return {
        "구분 기준": col,
        "대표 특성": top_val,
        "위험 집중도": lift,
    }

# =========================
# Policy helper
# =========================
def policy_reco_by_grade(grade):

    if grade == "고위험":
        return (
            "🔴 고위험 고객 – 선제적 리스크 관리 필요",
            [
                "한도/결제조건 조정",
                "콜센터 사전 안내",
                "연체 예방 캠페인",
                "집중 모니터링"
            ],
            "error"
        )

    elif grade == "위험":
        return (
            "🟠 위험 고객 – 관리 강화 필요",
            [
                "자동이체/분할납부 유도",
                "행동기반 알림",
                "한도 점검",
                "리스크 모니터링 강화"
            ],
            "warning"
        )

    elif grade == "안정":
        return (
            "🔵 안정 고객 – 일반 관리",
            [
                "정기 모니터링",
                "리마인드 메시지",
                "관계 유지 관리"
            ],
            "success"
        )

    else:  # 우대
        return (
            "🟢 우대 고객 – 가치 극대화",
            [
                "금리/한도 우대 검토",
                "우량 고객 프로모션",
                "Cross-sell 기회 발굴"
            ],
            "success"
        )

def grade_color(grade: str):
    mp = {
        "우대":   ("#E9F7EF", "#2ECC71", "#1E8449"),
        "안정":   ("#EAF2FF", "#4C78A8", "#1F4E79"),
        "위험":   ("#FFF4E5", "#F39C12", "#8A4B00"),
        "고위험": ("#FDECEC", "#E74C3C", "#922B21"),
    }
    return mp.get(grade, ("#F5F5F5", "#999999", "#333333"))


def metric_card(title: str, value: str, bg: str, border: str, text: str, sub: str = ""):
    sub_html = f"""<div style="margin-top:6px; font-size:12px; color:{text}; opacity:0.75;">{sub}</div>""" if sub else ""
    return f"""
    <div style="
        background:{bg};
        border-left:6px solid {border};
        padding:14px 16px;
        border-radius:12px;
        margin-bottom:10px;
    ">
      <div style="font-size:13px; color:{text}; opacity:0.85; font-weight:700;">{title}</div>
      <div style="font-size:28px; color:{text}; font-weight:800; line-height:1.15;">{value}</div>
      {sub_html}
    </div>
    """
def grade_color(grade: str):
    # 배경/테두리/글자색 (원하면 더 진하게 조정 가능)
    mp = {
        "우대":   ("#E9F7EF", "#2ECC71", "#1E8449"),  # green
        "안정":   ("#EAF2FF", "#4C78A8", "#1F4E79"),  # blue
        "위험":   ("#FFF4E5", "#F39C12", "#8A4B00"),  # orange
        "고위험": ("#FDECEC", "#E74C3C", "#922B21"),  # red
    }
    return mp.get(grade, ("#F5F5F5", "#999999", "#333333"))
# =========================
# 4) Streamlit UI
# =========================
st.set_page_config(page_title="연체 리스크 스코어카드", layout="wide")
st.title("연체 리스크 스코어카드")

meta, cont_df, cat_df, cross_df, bin_df, flag_df, job_ind_df = load_scorecard_excel("v2")

cont_map  = parse_section_sheet(cont_df)
cat_map   = parse_section_sheet(cat_df)
cross_map = parse_section_sheet(cross_df)
bin_map   = build_value_points_map(bin_df)
flag_map  = build_value_points_map(flag_df)
job_ind_map = build_job_ind_map(job_ind_df)



tabs = st.tabs([
    "① Overview",
    "② 고객 입력 & Explain",
    "③ Risk Insight",
])

# ---- Overview
with tabs[0]:
    st.subheader("Executive Summary")
    st.info(
        "이 대시보드는 고객의 기본 정보만으로 연체 가능성을 예측하고, "
        "점수 기반 등급(우대 / 안정 / 위험 / 고위험)에 따라 차별화된 선제적 대응이 가능하도록 만든 도구입니다. "
        "특히 직업과 산업군 정보를 반영하여 고객의 고용 안정성을 고려하고, "
        "각 등급에 맞는 정책(한도 조정, 사전 안내, 프로모션 등)으로 바로 연결할 수 있도록 설계되었습니다."
    )

    # =========================
    # 1) Sample 기반 KPI 산출
    # =========================
    base_dir = Path(__file__).resolve().parent
    art_dir = base_dir / "artifacts"

    # 샘플 파일 우선순위: parquet -> csv
    sample_path = art_dir / "sample_scoring.parquet"
    if not sample_path.exists():
        sample_path = art_dir / "sample_scoring.csv"

    try:
        sample_df = load_sample_df(art_dir)
        kpi = compute_portfolio_kpis_from_df(
            sample_df, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map
        )

        st.markdown("### 📊 Risk Distribution & Business Impact")

        # ---- KPI 4개 (추천)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("신규 고객 수", f"{kpi['n']:,}")
        # 전체 평균 PD (없으면 계산)
        avg_pd = kpi.get("avg_pd", None)
        if avg_pd is None:
            # 구버전 함수 대비: 없으면 대략 계산(grade만 있던 경우)
            # compute_portfolio_kpis_from_df를 최신 버전으로 업데이트하면 이 부분은 자동으로 채워짐
            avg_pd = np.nan
        k2.metric("전체 평균 PD", f"{avg_pd*100:.2f}%" if not np.isnan(avg_pd) else "-")

        # 고위험군 정의: A+B(상위 30%)를 추천. (너희 정책에 맞게 A만(10%)도 가능)
        high_share = kpi.get("high_share", None)
        high_avg_pd = kpi.get("high_avg_pd", None)
        k3.metric("고위험군 비율", f"{high_share*100:.1f}%" if high_share is not None else "-")
        k4.metric("고위험군 평균 PD", f"{high_avg_pd*100:.2f}%" if high_avg_pd is not None and not np.isnan(high_avg_pd) else "-")

        st.caption("※ 위 KPI는 전체 데이터가 아닌 샘플(랜덤 추출) 기준으로 산출되었습니다.")    
        st.caption("Model Performance (Validation) | 모델 성능(AUC): 0.645  |  변별력 성능(KS): 0.215  |  예측 모델의 판별력(Gini): 0.29")

        # =========================
        # 2) 등급별 분포
        # =========================
        st.markdown("### 등급별 고객 비중")
        grade_order = ["우대", "안정", "위험", "고위험"]
        
        dist_df = kpi["grade_dist"]   # grade, count, share(%)
        share_map = dict(zip(dist_df["grade"], dist_df["share"]))
        share = [share_map.get(g, 0.0) for g in grade_order]
        
        x = np.arange(len(grade_order))
        fig1, ax = plt.subplots(figsize=(8, 3.6))
        
        bar_colors = ["#4C78A8"] * len(grade_order)
        # 강조하고 싶은 등급이 있으면 예: 고위험만 빨강
        bar_colors[grade_order.index("고위험")] = "#d62728"
        
        ax.bar(x, share, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(grade_order)
        ax.set_ylabel("고객 비중 (%)")
        ax.set_ylim(0, max(share) * 1.25 if max(share) > 0 else 1)
        
        for i, v in enumerate(share):
            ax.text(i, v + (max(share)*0.03 if max(share)>0 else 0.1), f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
        
        st.pyplot(fig1, use_container_width=True)
        
        
        st.markdown("### 등급별 실제 연체율")
        dr_df = kpi["dr_by_grade"]    # grade, dr(%)
        dr_map = dict(zip(dr_df["grade"], dr_df["dr"]))
        dr = [dr_map.get(g, np.nan) for g in grade_order]
        
        fig2, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(x, dr, color="#E45756", marker="o", linewidth=2, label="등급별 실제 연체율(%)")
        
        ax.set_xticks(x)
        ax.set_xticklabels(grade_order)
        ax.set_ylabel("실제 연체율 (%)")
        ax.set_ylim(0, (np.nanmax(dr) * 1.40) if np.isfinite(np.nanmax(dr)) else 1)
        
        # 값 라벨
        for i, v in enumerate(dr):
            if np.isnan(v):
                continue
        
            # 기본 오프셋
            dy = (np.nanmax(dr) * 0.03) if np.isfinite(np.nanmax(dr)) else 0.3
        
            # ✅ '안정'만 더 띄우기(겹침 방지)
            if grade_order[i] == "안정":
                dy *= 1.8
        
            ax.annotate(
                f"{v:.1f}%",
                (x[i], v),
                textcoords="offset points",
                xytext=(0, 10 if grade_order[i] == "안정" else 6),
                ha="center",
                va="bottom",
                fontsize=10
            )
        
        # 전체 평균 기준선(있을 때만)
        overall = kpi.get("overall_dr", np.nan)  # 0~1
        if not np.isnan(overall):
            y = overall * 100
            ax.axhline(y, color="gray", linestyle="--", linewidth=1.5, label=f"전체 평균 연체율({y:2f}%")
            ax.text(0.02, 0.92, f"전체 평균 연체율: {y:.2f}%", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
            ax.legend(loc="upper right", frameon=True)
        
        st.pyplot(fig2, use_container_width=True)


        # --- enriched df (score/pd/grade + pts_*)
        enriched = compute_portfolio_kpis_from_df(
            sample_df, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map
        )
        
        # =========================
        # 2.5) 고위험군 공통 특징 Top5 (Overview용)
        # =========================
        st.markdown("### 🔍 고위험군 공통 리스크 특징 (Top 5)")
        
        # 분석할 변수 후보(샘플 파일에 실제로 있는 컬럼만 쓰면 됨)
        candidate_cols = [
            "수입 유형", "최종 학력", "결혼 여부", "주거 형태", "자녀수_구간",
            "직업", "산업군_상위",
            "차량 소유 여부", "부동산 소유 여부", "배우자유무",
        ]
        
        # 고위험군 mask (kpi에서 등급이 이미 계산되어 있거나, sample_df에 grade가 있으면 그걸 쓰면 됨)
        if "GRADE" in sample_df.columns:
            high_mask = sample_df["GRADE"].astype(str).eq("고위험")
        else:
            # compute_portfolio_kpis_from_df에서 grade 붙였으면 그 컬럼명에 맞춰 변경
            # 예: sample_df["_grade4"] 등
            high_mask = None
        
        # grade 컬럼이 없으면: kpi 계산 때 썼던 로직대로 grade를 한번 더 붙이는 게 안전
        if high_mask is None:
            # 1) sample_df로 확률 산출
            records = sample_df.to_dict("records")
            probas = np.empty(len(records), dtype=float)
            scores = np.empty(len(records), dtype=float)
        
            for i, r in enumerate(records):
                s, p = score_one_fast(r, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map)
                scores[i] = s
                probas[i] = p
        
            # 2) 점수컷 기반 4등급 (네가 확정한 컷)
            # 고위험: ~549 / 위험: 550~599 / 안정: 600~669 / 우대: 670~
            def to_grade4(score):
                if score <= 549:
                    return "고위험"
                elif score <= 599:
                    return "위험"
                elif score <= 669:
                    return "안정"
                else:
                    return "우대"
        
            grade4 = np.array([to_grade4(s) for s in scores])
            high_mask = (grade4 == "고위험")
        
        # 고위험군 표본이 너무 적으면 안내
        if high_mask.sum() < 20:
            st.info("고위험군 표본이 적어(20명 미만) 특징 요약이 불안정할 수 있어요.")
        else:
            rows = []
            for col in candidate_cols:
                if col not in sample_df.columns:
                    continue
        
                s = sample_df[col].astype(str).fillna("결측")
                overall_top = s.value_counts(normalize=True).head(1)
                high_top = s[high_mask].value_counts(normalize=True).head(1)
        
                if len(overall_top) == 0 or len(high_top) == 0:
                    continue
        
                overall_val, overall_rate = overall_top.index[0], overall_top.iloc[0]
                high_val, high_rate = high_top.index[0], high_top.iloc[0]
        
                # "고위험군에서 더 흔한 정도"를 점수화
                lift = (high_rate / overall_rate) if overall_rate > 0 else np.nan
        
                rows.append({
                    "구분 기준": col,
                    "대표 특성": high_val,
                    "위험 집중도": lift
                })
                
            top_df = pd.DataFrame(rows).dropna()
            if top_df.empty:
                st.warning("특징 요약을 만들 수 있는 컬럼이 부족해요. candidate_cols를 샘플 컬럼에 맞춰 조정해주세요.")
            else:
                top_df["위험 집중도"] = pd.to_numeric(top_df["위험 집중도"], errors="coerce")
                top_df = top_df.dropna(subset=["위험 집중도"])
            
                top_df = top_df.sort_values("위험 집중도", ascending=False).head(5)
            
                # 보기 좋게 반올림
                top_df["위험 집중도"] = top_df["위험 집중도"].round(2)
            
                st.dataframe(top_df, use_container_width=True)
                st.caption("위험 집중도는 전체 대비 해당 특성이 고위험군에 얼마나 더 많이 나타나는지(과대표집)를 의미합니다. 1보다 크면 고위험군에서 더 자주 나타납니다.")
    except Exception as e:
            st.warning(f"샘플 KPI를 계산하지 못했습니다: {e}")
    

    # =========================
    # 4) 운영 정책 예시
    # =========================
    st.divider()
    st.markdown("### 🎯 권장 운영 정책 (Score-based Action)")
    
    a, b, c, d = st.columns(4)
    
    with a:
        st.markdown("#### 🟢 우대 \n<small style='color:gray'>Score ≥ 670</small>", unsafe_allow_html=True)
        st.markdown(
            "- 정상 유지\n"
            "- 우량 고객 프로모션\n"
            "- 금리/한도 우대 검토\n"
            "- 과도 제약 최소화"
        )
    
    with b:
        st.markdown("#### 🔵 안정  \n<small style='color:gray'>600 ≤ Score ≤ 669</small>", unsafe_allow_html=True)
        st.markdown(
            "- 정기 모니터링\n"
            "- 자동이체 유지 안내\n"
            "- 고객 관리 강화\n"
            "- 소프트 리마인드"
        )
    
    with c:
        st.markdown("#### 🟠 위험 \n<small style='color:gray'>550 ≤ Score ≤ 599</small>", unsafe_allow_html=True)
        st.markdown(
            "- 모니터링 강화\n"
            "- 자동이체/분할납부 유도\n"
            "- 행동기반 알림\n"
            "- 한도 점검"
        )
    
    with d:
        st.markdown("#### 🔴 고위험  \n<small style='color:gray'>Score ≤ 549</small>", unsafe_allow_html=True)
        st.markdown(
            "- 한도/결제조건 조정\n"
            "- 사전 안내·콜센터 아웃바운드\n"
            "- 연체예방 캠페인\n"
            "- 선제 리스크 관리"
        )

# ---- 고객 입력 & Explain
with tabs[1]:
    # ---- 산업군 상위 옵션 생성 (form 전에 반드시 위치)
    base_dir = Path(__file__).resolve().parent
    art_dir = base_dir / "artifacts"
    
    sample_path = art_dir / "sample_scoring.parquet"
    if not sample_path.exists():
        sample_path = art_dir / "sample_scoring.csv"
    
    industry_upper_options, upper_to_rep_full = build_upper_industry_options_and_rep(job_ind_df, sample_path)
    st.subheader("고객정보 입력 → 점수/확률/등급 산출")
    st.caption("입력 Tip!\n - 산업군은 '상위 산업군'만 선택합니다. (무역/산업/운송은 제외)")

    # 주거 형태 옵션을 scorecard_cat(=cat_map)에서 그대로 가져오기
    housing_feat = "주거 형태_woe"
    housing_options = list(cat_map.get(housing_feat, {}).keys())
    
    # 혹시 비어있으면(예외) fallback
    if not housing_options:
        housing_options = ["주택 / 아파트", "아파트 임대", "오피스텔", "공공분양", "기타"]

    left, right = st.columns([1.15, 1])
    
    with left:
        with st.form("demo_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                성별 = st.selectbox("성별", ["남성", "여성"])
                나이 = st.number_input("나이", 18, 90, 35, step=1)
                결혼 = st.selectbox("결혼 여부", ["미혼", "기혼", "별거", "사별", "사실혼"])

            with c2:
                # 직업: UI 표기 '기타' -> 내부 'Unknown'
                job_options_ui = [
                    "기타", "단순 노동자","영업직","핵심 노동자","관리직","운전자","기술직","회계사",
                    "의료 업계 종사자","보안 업계 종사자","조리사","미화원","가정부","저임금 노동자",
                    "비서","요식업 종사자","부동산중개업자","인사 담당자","IT 업계 종사자"
                ]
                직업_ui = st.selectbox("직업", job_options_ui, index=0)
                직업 = "Unknown" if 직업_ui == "기타" else 직업_ui

                산업군_상위 = st.selectbox("산업군(상위)", industry_upper_options)  # 너가 만든 옵션 사용
                # 선택된 상위 산업군을 실제 '산업군' 문자열로 매핑(예: 교육 -> 교육 2)
                산업군 = upper_to_rep_full.get(산업군_상위, "")  # 내부 계산용

                근속연수 = st.number_input("근속연수(년)", 0.0, 50.0, 3.0, step=1.0)

            with c3:
                가입연수 = st.number_input("가입연수(년)", 0.0, 50.0, 5.0, step=1.0)
                연간수입 = st.number_input("연간 수입(원)", 0, value=40_000_000, step=1_000_000)
                st.caption(f"입력값: {연간수입:,.0f}원")
                거주지 = st.number_input("거주지 인구 비율", 0.0, 1.0, 0.03, step=0.01)

            # 기타 입력(스코어카드에 쓰일 수 있음)
            수입유형 = st.selectbox("수입 유형", ["근로자", "공무원", "연금수령자", "기타"])
            학력 = st.selectbox("최종 학력", ["저학력자", "고등학교 졸업", "대학교 중퇴", "대학교 졸업 이상"])
            default_idx = housing_options.index("주택 / 아파트") if "주택 / 아파트" in housing_options else 0
            주거 = st.selectbox("주거 형태", housing_options, index=default_idx)
            가족구성 = st.number_input("가족 구성원 수", 1, 10, 2, step=1)
            자녀구간 = st.selectbox("자녀 수", ["0", "1", "2", "3+"])

            차량 = st.selectbox("차량 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
            부동산 = st.selectbox("부동산 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
            배우자유무 = st.selectbox("배우자유무", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")


            submitted = st.form_submit_button("🚀 등급 산출")

        if submitted:
            row = {
                "성별": 성별,
                "나이": float(나이),
                "결혼 여부": 결혼,
                "직업": 직업,
                "산업군": 산업군,  # 내부 계산은 job_ind_points 기준 문자열
                "근속연수": float(근속연수),
                "가입연수": float(가입연수),
                "연간 수입": float(연간수입),
                "거주지 인구 비율": float(거주지),
                "수입 유형": 수입유형,
                "최종 학력": 학력,
                "주거 형태": 주거,
                "가족 구성원 수": int(가족구성),
                "자녀수_구간": str(자녀구간),
                "차량 소유 여부": int(차량),
                "부동산 소유 여부": int(부동산),
                "배우자유무": int(배우자유무),
            }

            score, proba, grade, bd = score_one(
                row, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map
            )

            st.session_state["last_row"] = row
            st.session_state["last_bd"] = bd
            st.session_state["last_result"] = (score, proba, grade)

    with right:
        st.markdown("### 결과")
    
        if "last_result" not in st.session_state:
            st.info("좌측에서 고객 정보를 입력하고 ‘등급 산출’을 눌러주세요.")
        else:
            score, proba, grade = st.session_state["last_result"]
            bg, border, text = grade_color(grade)
    
            # ✅ Score / PD / Grade 카드
            st.markdown(metric_card("Score", f"{score:.1f}", bg, border, text, sub="점수카드 합산 결과"), unsafe_allow_html=True)
            st.markdown(metric_card("연체확률 (PD)", f"{proba:.3f}", bg, border, text, sub="점수 → 확률 변환 결과"), unsafe_allow_html=True)
            st.markdown(metric_card("등급", grade, bg, border, text, sub="점수컷 기준(우대/안정/위험/고위험)"), unsafe_allow_html=True)
    
            st.progress(min(max(proba, 0.0), 1.0))
    
            # ✅ 등급별 정책(맞춤 강조)
            st.markdown("### 🎯 권장 운영 액션")
            title, items, box = policy_reco_by_grade(grade)
            msg = "\n".join([f"- {x}" for x in items])
    
            if box == "error":
                st.error(f"**{title}**\n\n{msg}")
            elif box == "warning":
                st.warning(f"**{title}**\n\n{msg}")
            else:
                st.success(f"**{title}**\n\n{msg}")

    # =========================
    # Explain: 같은 탭 아래에 바로 표시
    # =========================
    st.divider()
    st.subheader("Explainability: Reason Codes (Top 5)")
    
    if "last_bd" not in st.session_state:
        st.info("위에서 고객정보를 입력 후 등급을 산출하면, 왜 그런 결과인지 여기에서 바로 확인할 수 있어요.")
    else:
        bd = st.session_state["last_bd"].copy()
    
        # 안전하게 숫자 변환
        bd["points"] = pd.to_numeric(bd["points"], errors="coerce")
        bd = bd.dropna(subset=["points"])
    
        # ✅ 리스크를 높인 요인: points < 0 만
        bd_risk = bd[bd["points"] < 0].sort_values("points").head(5)
    
        # ✅ 리스크를 낮춘 요인: points >= 0 만
        bd_safe = bd[bd["points"] >= 0].sort_values("points", ascending=False).head(5)
    
        cA, cB = st.columns(2)
    
        with cA:
            st.markdown("#### 🔺 리스크를 높인 요인 (Top 5)")
            if bd_risk.empty:
                st.caption("리스크를 높인 요인(points < 0)이 없습니다.")
            else:
                st.dataframe(bd_risk, use_container_width=True)
    
        with cB:
            st.markdown("#### 🔻 리스크를 낮춘 요인 (Top 5)")
            if bd_safe.empty:
                st.caption("리스크를 낮춘 요인(points ≥ 0)이 없습니다.")
            else:
                st.dataframe(bd_safe, use_container_width=True)
    
        st.caption("points가 음수일수록 점수를 깎아 위험을 높이고, 0 이상이면 점수를 유지/가산하여 위험을 낮춥니다.")

with tabs[2]:
    st.subheader("Risk Insight (고위험군 분석)")

    # 샘플 로드
    base_dir = Path(__file__).resolve().parent
    art_dir = base_dir / "artifacts"
    sample_df = load_sample_df(art_dir)

    df_sc = enrich_with_score(sample_df, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map)

    # -------------------------
    # A) 상위 위험군 vs 전체 평균 비교 (요약 KPI)
    # -------------------------
    st.markdown("### 1) 상위 위험군 vs 전체 평균")
    seg_mode = st.radio(
        "상위 위험군 정의",
        ["고위험(점수≤549)", "위험+고위험(점수≤599)", "PD 상위 20%(확률 기준)"],
        horizontal=True
    )

    if seg_mode == "고위험(점수≤549)":
        df_seg = df_sc[df_sc["grade4"].eq("고위험")]
    elif seg_mode == "위험+고위험(점수≤599)":
        df_seg = df_sc[df_sc["grade4"].isin(["위험", "고위험"])]
    else:
        cut = float(np.quantile(df_sc["proba"].values, 0.80))
        df_seg = df_sc[df_sc["proba"] >= cut]


    # 실제 연체율(타겟 있을 때)
    if "TARGET" in df_sc.columns:
        overall_dr = df_sc["TARGET"].mean()
        seg_dr = df_seg["TARGET"].mean()
    
        lift = seg_dr / overall_dr if overall_dr > 0 else np.nan
    
        k1, k2, k3, k4 = st.columns(4)
    
        k1.metric("전체 평균 연체율", f"{overall_dr*100:.2f}%")
        k2.metric("상위 위험군 연체율", f"{seg_dr*100:.2f}%")
        k3.metric("위험 배율(Lift)", f"{lift:.2f}배")
        k4.metric("상위 위험군 비중", f"{len(df_seg)/len(df_sc)*100:.1f}%")

    st.divider()

    # -------------------------
    # B) 연체율 상위 그룹 변수 분포 + 변수별 Lift (표)
    # -------------------------
    st.markdown("### 2) 변수별 위험 집중도(Lift) — 고위험군 대표 특징")

    # 후보 컬럼: 너희 샘플 컬럼에 맞춰 조정
    default_cols = [c for c in ["수입 유형", "최종 학력", "직업", "산업군_상위", "주거 형태", "차량 소유 여부", "부동산 소유 여부"] if c in df_sc.columns]
    candidate_cols = st.multiselect("분석할 변수 선택", options=list(df_sc.columns), default=default_cols)

    rows = []
    for col in candidate_cols:
        # 너무 연속형은 제외(원하면 bins로 추가 가능)
        if col in ["score", "proba", "TARGET"]:
            continue
        out = compute_lift_table(df_sc, df_seg, col)
        if out is not None:
            rows.append(out)

    top_df = pd.DataFrame(rows)
    if top_df.empty:
        st.warning("선택한 변수로 Lift를 계산할 수 없어요. (결측 많거나 문자열 변환 불가)")
    else:
        top_df["위험 집중도"] = pd.to_numeric(top_df["위험 집중도"], errors="coerce")
        top_df = top_df.dropna(subset=["위험 집중도"]).sort_values("위험 집중도", ascending=False).head(10)
        top_df["위험 집중도"] = top_df["위험 집중도"].round(2)
        st.dataframe(top_df, use_container_width=True)
        st.caption("위험 집중도(Lift)는 ‘상위 위험군에서의 비중 / 전체에서의 비중’입니다. 1보다 크면 고위험군에 더 많이 몰려 있습니다.")

    st.divider()

    # -------------------------
    # C) 연체율 상위 그룹 변수 분포
    # -------------------------
    st.markdown("### 3) 연체율 상위 그룹 변수 분포 (막대 비교)")

    # 분석 변수 선택
    candidate_cols = [
        c for c in df_sc.columns
        if c not in ["score","proba","TARGET"]
    ]
    
    col = st.selectbox("분포를 볼 변수 선택", candidate_cols)
    
    top_n = st.slider("상위 카테고리 개수", 3, 10, 5)
    
    # 데이터 준비
    a = df_sc[col].astype(str).fillna("결측").str.strip()
    s = df_seg[col].astype(str).fillna("결측").str.strip()
    
    # 전체 기준 Top N 카테고리
    top_cats = a.value_counts().head(top_n).index.tolist()
    
    all_share = (
        a.value_counts(normalize=True)
        .reindex(top_cats)
        .fillna(0) * 100
    )
    
    seg_share = (
        s.value_counts(normalize=True)
        .reindex(top_cats)
        .fillna(0) * 100
    )
    
    # ---- 막대그래프 ----
    x = np.arange(len(top_cats))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    ax.bar(x - width/2, all_share.values, width, label="전체", alpha=0.7)
    ax.bar(x + width/2, seg_share.values, width, label="상위 위험군", alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_cats, rotation=30, ha="right")
    ax.set_ylabel("비중 (%)")
    ax.set_title(f"{col} 분포 비교")
    ax.legend()
    
    # 값 라벨
    for i, v in enumerate(all_share.values):
        ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    
    for i, v in enumerate(seg_share.values):
        ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    
    st.pyplot(fig, use_container_width=True)
    
    st.caption("상위 위험군에서 특정 특성이 더 많이 나타나면 두 막대 간 차이가 크게 나타납니다.")

st.divider()
