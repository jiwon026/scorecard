import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================
# 0) Excel Loader
# =========================
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

    cont  = xls.parse("scorecard_cont")      # feature/points (섹션형)
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
    """
    labels: ["23세 이하", "24세 ~ 30세", "19년 이상", "9~14년", "5, 6년", "0.03 이하" ...]
    가능한 범위 패턴들을 파싱해 x가 들어가는 label 반환
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    # 우선: '이하', '이상', 'A ~ B', 'A~B', 'A, B'
    for lab in labels:
        t = str(lab).strip()

        # "23세 이하" / "12,744,000원 이하"
        if "이하" in t and "~" not in t and "," not in t:
            a = _to_int(t) if any(ch.isdigit() for ch in t) and ("원" in t) else None
            if a is None:
                # "23세 이하" 같은 경우
                nums = re.findall(r"\d+(\.\d+)?", t)
                if nums:
                    a = float(nums[0])
            if a is not None and x <= a:
                return lab

        # "100,890,001원 이상" / "19년 이상"
        if "이상" in t and "~" not in t and "," not in t:
            a = _to_int(t) if ("원" in t) else None
            if a is None:
                nums = re.findall(r"\d+(\.\d+)?", t)
                if nums:
                    a = float(nums[0])
            if a is not None and x >= a:
                return lab

        # "24세 ~ 30세", "12,744,001 ~ 37,170,000원 이하"
        if "~" in t:
            parts = [p.strip() for p in t.split("~")]
            if len(parts) >= 2:
                lo_s, hi_s = parts[0], parts[1]
                lo = _to_int(lo_s) if ("원" in t) else None
                hi = _to_int(hi_s) if ("원" in t) else None
                if lo is None:
                    nums = re.findall(r"\d+(\.\d+)?", lo_s)
                    lo = float(nums[0]) if nums else None
                if hi is None:
                    nums = re.findall(r"\d+(\.\d+)?", hi_s)
                    hi = float(nums[0]) if nums else None

                if lo is not None and hi is not None:
                    # "이하"가 있으면 upper inclusive
                    if "이하" in t:
                        if lo <= x <= hi:
                            return lab
                    else:
                        if lo <= x <= hi:
                            return lab

        # "9~14년" (tilde without spaces)
        if re.search(r"\d+\s*~\s*\d+", t):
            nums = re.findall(r"\d+(\.\d+)?", t)
            if len(nums) >= 2:
                lo = float(nums[0])
                hi = float(nums[1])
                if lo <= x <= hi:
                    return lab

        # "5, 6년" 같은 나열
        if "," in t and "~" not in t:
            nums = re.findall(r"\d+(\.\d+)?", t)
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
    # (너희 팀이 확정한 정의로 조정 가능)
    if "한부모 가정" in flag_map:
        spouse = int(row.get("배우자유무", 1))
        child_n = int(row.get("자녀 수", 0))
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

    # ---- grade (3등급: 분위수 대신 cut을 UI에서 조절 가능)
    # 기본 cut은 데모용: High>=0.65, Mid>=0.50
    high_cut = float(row.get("_high_cut", 0.65))
    mid_cut  = float(row.get("_mid_cut", 0.50))
    if proba >= high_cut:
        grade = "High"
    elif proba >= mid_cut:
        grade = "Mid"
    else:
        grade = "Low"

    bd = pd.DataFrame(breakdown, columns=["feature", "value/bin", "points"]).sort_values("points")
    return score, proba, grade, bd


# =========================
# 4) Streamlit UI
# =========================
st.set_page_config(page_title="연체 리스크 스코어카드", layout="wide")
st.title("연체 리스크 스코어카드 대시보드 (Excel Scorecard 기반)")

meta, cont_df, cat_df, cross_df, bin_df, flag_df, job_ind_df = load_scorecard_excel("v2")

cont_map  = parse_section_sheet(cont_df)
cat_map   = parse_section_sheet(cat_df)
cross_map = parse_section_sheet(cross_df)
bin_map   = build_value_points_map(bin_df)
flag_map  = build_value_points_map(flag_df)
job_ind_map = build_job_ind_map(job_ind_df)

tabs = st.tabs([
    "① Overview",
    "② 고객정보 입력(데모)",
    "③ Explain (Reason Codes)",
    "④ WOE/Points Insights",
])

# ---- Overview
with tabs[0]:
    st.subheader("Executive Summary (심사위원 30초 버전)")
    st.info(
        "본 모델은 고객 기본정보만으로 연체 위험을 조기 식별하고 High/Mid/Low 3등급으로 분류합니다. "
        "특히 직업×산업군 조합 점수(고용 안정성 관점)와 WOE 기반 스코어카드를 통해 "
        "‘설명가능 + 운영가능’한 정책 연결을 강조합니다."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Score", f"{meta['base_score']:.0f}" if not np.isnan(meta["base_score"]) else "-")
    c2.metric("PDO", f"{meta['PDO']:.0f}" if not np.isnan(meta["PDO"]) else "-")
    c3.metric("Factor", f"{meta['factor']:.3f}")
    c4.metric("Base Points", f"{meta['base_points']:.2f}")

    st.divider()
    st.markdown("### 운영 정책 예시 (Risk Grade Action)")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🔴 High")
        st.markdown("- 한도/결제조건 조정\n- 사전 안내·콜센터\n- 연체예방 캠페인/리마인드")
    with b:
        st.markdown("#### 🟠 Mid")
        st.markdown("- 모니터링 강화\n- 자동이체/분할납부 유도\n- 행동기반 알림")
    with c:
        st.markdown("#### 🟢 Low")
        st.markdown("- 정상 유지\n- 우량 고객 프로모션\n- 과도 제약 최소화")


# ---- Demo input
with tabs[1]:
    st.subheader("고객정보 입력 → 점수/확률/등급 산출")
    st.caption("※ scorecard_export.xlsx의 점수표를 그대로 사용합니다.")

    left, right = st.columns([1.15, 1])

    with left:
        with st.form("demo_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                성별 = st.selectbox("성별", ["남성", "여성"])
                나이 = st.number_input("나이", 18, 90, 35, step=1)
                결혼 = st.selectbox("결혼 여부", ["미혼", "기혼", "별거", "사별", "사실혼"])

            with c2:
                직업 = st.selectbox("직업", [
                    "Unknown","단순 노동자","영업직","핵심 노동자","관리직","운전자","기술직","회계사",
                    "의료 업계 종사자","보안 업계 종사자","조리사","미화원","가정부","저임금 노동자",
                    "비서","요식업 종사자","부동산중개업자","인사 담당자","IT 업계 종사자"
                ])
                산업군 = st.text_input("산업군 (예: 무역 0, 산업 3, 운송 1 등)", value="무역 0")
                근속연수 = st.number_input("근속연수(년)", 0.0, 50.0, 3.0, step=0.5)

            with c3:
                가입연수 = st.number_input("가입연수(년)", 0.0, 50.0, 5.0, step=0.5)
                연간수입 = st.number_input("연간 수입(원)", 0, value=40_000_000, step=1_000_000)
                거주지 = st.number_input("거주지 인구 비율", 0.0, 1.0, 0.03, step=0.01)

            # 기타 입력(스코어카드에 쓰일 수 있음)
            수입유형 = st.selectbox("수입 유형", ["근로자", "공무원", "연금수령자", "기타"])
            학력 = st.selectbox("최종 학력", ["저학력자", "고등학교 졸업", "대학교 중퇴", "대학교 졸업 이상"])
            주거 = st.selectbox("주거 형태", ["주택 / 아파트", "아파트 임대", "기타"])
            자녀수 = st.number_input("자녀 수", 0, 10, 0, step=1)
            가족구성 = st.number_input("가족 구성원 수", 1, 10, 2, step=1)
            자녀구간 = st.selectbox("자녀수_구간", ["0", "1", "2", "3+"])

            차량 = st.selectbox("차량 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
            부동산 = st.selectbox("부동산 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
            배우자유무 = st.selectbox("배우자유무", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")

            # 등급 컷 조절(발표용)
            st.markdown("##### 등급 컷(확률 기반) 조절")
            mid_cut = st.slider("Mid cut", 0.30, 0.70, 0.50, 0.01)
            high_cut = st.slider("High cut", 0.40, 0.90, 0.65, 0.01)

            submitted = st.form_submit_button("🚀 등급 산출")

        if submitted:
            row = {
                "성별": 성별,
                "나이": float(나이),
                "결혼 여부": 결혼,
                "직업": 직업,
                "산업군": 산업군,
                "근속연수": float(근속연수),
                "가입연수": float(가입연수),
                "연간 수입": float(연간수입),
                "거주지 인구 비율": float(거주지),
                "수입 유형": 수입유형,
                "최종 학력": 학력,
                "주거 형태": 주거,
                "자녀 수": int(자녀수),
                "가족 구성원 수": int(가족구성),
                "자녀수_구간": str(자녀구간),
                "차량 소유 여부": int(차량),
                "부동산 소유 여부": int(부동산),
                "배우자유무": int(배우자유무),
                "_mid_cut": float(mid_cut),
                "_high_cut": float(high_cut),
            }

            score, proba, grade, bd = score_one(
                row, meta, cont_map, cat_map, cross_map, bin_map, flag_map, job_ind_map
            )

            st.session_state["last_row"] = row
            st.session_state["last_bd"] = bd
            st.session_state["last_result"] = (score, proba, grade)

            st.success(f"Score: **{score:.1f}** | 연체확률: **{proba:.3f}** | 등급: **{grade}**")
            st.progress(min(max(proba, 0.0), 1.0))

    with right:
        st.markdown("### 입력 가이드(발표 스토리)")
        st.write(
            "- 본 스코어카드는 월 1회(또는 분기 1회) 최신 고객 데이터로 재학습/검증된다고 가정\n"
            "- 신규/변경 고객 기본정보로 연체 위험을 조기 탐지\n"
            "- 등급별 선제 정책(High/Mid/Low) 적용\n"
            "- 아래 입력은 ‘가장 최근 모델’ 기준 데모"
        )


# ---- Reason Codes
with tabs[2]:
    st.subheader("Explainability: Reason Codes (리스크 요인 Top)")
    if "last_bd" not in st.session_state:
        st.info("② 고객정보 입력(데모) 탭에서 먼저 등급을 산출하세요.")
    else:
        score, proba, grade = st.session_state["last_result"]
        bd = st.session_state["last_bd"].copy()

        st.write(f"결과: Score={score:.1f}, Prob={proba:.3f}, Grade={grade}")

        st.markdown("#### 점수에 가장 불리하게 기여한 항목(Top 10)")
        # points가 음수일수록 점수 깎는 요인(리스크)
        st.dataframe(bd.sort_values("points").head(10), use_container_width=True)

        st.markdown("#### 점수에 유리하게 기여한 항목(Top 10)")
        st.dataframe(bd.sort_values("points", ascending=False).head(10), use_container_width=True)

        st.caption("Tip: 발표 때는 ‘왜 High인가?’를 이 표로 10초 안에 설명 가능하게 보여주면 점수 올라감.")


# ---- Insights
with tabs[3]:
    st.subheader("WOE / Points Insights")
    st.caption("점수카드 테이블 자체가 ‘정책·설명’의 근거입니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 직업×산업군 Points (일부)")
        q = st.text_input("검색(직업 또는 산업군)", "")
        tmp = job_ind_df.copy()
        if q.strip():
            tmp = tmp[tmp["직업"].astype(str).str.contains(q) | tmp["산업군"].astype(str).str.contains(q)]
        st.dataframe(tmp.sort_values("points").head(20), use_container_width=True)

    with col2:
        st.markdown("### Flag/Binary Points")
        st.write("Binary sheet preview")
        st.dataframe(bin_df.head(30), use_container_width=True)
        st.write("Flag sheet preview")
        st.dataframe(flag_df.head(30), use_container_width=True)

st.divider()
st.caption("발표 흐름 추천: ①Overview(정책+설명가능) → ②데모(입력→등급) → ③Reason Codes(왜 High인지) → ④Insights(직업×산업군 점수 테이블).")
