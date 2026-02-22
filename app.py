import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px


# =========================================================
# 0) Load Excel Scorecard (ONLY)
# =========================================================
@st.cache_resource
def load_scorecard_excel(_version: str = "v1"):
    """
    artifacts/scorecard_export.xlsx 로드
    sheets:
      meta, model_coef, scorecard_cont, scorecard_cat, scorecard_cross,
      scorecard_binary, scorecard_flag, job_ind_points
    """
    base_dir = Path(__file__).resolve().parent
    art_dir = base_dir / "artifacts"
    xlsx_path = art_dir / "scorecard_export.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"scorecard_export.xlsx not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)

    meta_df = xls.parse("meta")
    meta = dict(zip(meta_df["key"], meta_df["value"]))

    coef_df = xls.parse("model_coef")
    intercept = float(coef_df.loc[coef_df["feature"].eq("const"), "beta"].iloc[0])

    factor = float(meta["factor"])
    offset = float(meta["offset"])

    # base_points = offset - factor * intercept  (scorecard convention)
    base_points = float(offset - factor * intercept)

    cont = xls.parse("scorecard_cont")      # sections
    cat  = xls.parse("scorecard_cat")       # sections
    cross = xls.parse("scorecard_cross")    # sections
    binary = xls.parse("scorecard_binary")  # feature/value/points
    flag = xls.parse("scorecard_flag")      # feature/value/points
    job_ind = xls.parse("job_ind_points")   # 직업/산업군/points

    return meta, base_points, cont, cat, cross, binary, flag, job_ind


# =========================================================
# 1) Parsing helpers
# =========================================================
def _parse_sections(df: pd.DataFrame):
    """
    DataFrame has rows like:
      [근속연수_woe]
      1년 이하   -14.5
      ...
      blank
      [나이_woe]
      ...
    Return: dict {section_name: [(label, points), ...]}
    """
    sections = {}
    current = None

    for _, row in df.iterrows():
        feat = row.get("feature")
        pts = row.get("points")

        if isinstance(feat, str) and feat.startswith("[") and feat.endswith("]"):
            current = feat.strip()[1:-1].strip()
            sections[current] = []
            continue

        if current is None:
            continue

        if pd.isna(feat) or pd.isna(pts):
            continue

        sections[current].append((str(feat).strip(), float(pts)))

    return sections


def _build_value_points_map(df: pd.DataFrame):
    """
    scorecard_binary / scorecard_flag format:
      [차량 소유 여부]
      차량 소유 여부 0 0
      차량 소유 여부 1 17.2
    Return: {feature: {value: points}}
    """
    m = {}
    for _, r in df.iterrows():
        f = r.get("feature")
        v = r.get("value")
        p = r.get("points")
        if pd.isna(f) or pd.isna(v) or pd.isna(p):
            continue
        f = str(f).strip()
        m.setdefault(f, {})[int(float(v))] = float(p)
    return m


def _build_job_ind_map(job_ind_df: pd.DataFrame):
    """
    job_ind_points columns: 직업, 산업군, points
    Return: {(직업, 산업군): points}
    """
    m = {}
    for _, r in job_ind_df.iterrows():
        job = str(r.get("직업")).strip()
        ind = str(r.get("산업군")).strip()
        pts = r.get("points")
        if pd.isna(pts):
            continue
        m[(job, ind)] = float(pts)
    return m


# =========================================================
# 2) Cont-bin matching (Korean labels)
# =========================================================
def _normalize_num_label(s: str) -> str:
    s = str(s)
    s = s.replace(",", "")
    s = s.replace("원", "").replace("세", "").replace("년", "").replace("명", "")
    s = s.replace(" ", "")
    return s


def _pick_points_from_cont(value: float, rules):
    """
    rules: [(label, points)]
    label examples:
      "23세 이하", "24세 ~ 30세", "65세 이상"
      "0.02516 이하", "0.02517 ~ 0.03579 이하", "0.0358 이상"
      "1년 이하", "5, 6년", "9~14년", "19년 이상"
      "2 ~ 13년 이하", "26~ 35년 이하"
    """
    v = float(value)

    for label, pts in rules:
        raw = str(label).strip()
        s = _normalize_num_label(raw)

        # e.g., "23이하", "0.02516이하"
        m = re.match(r"^([0-9.]+)이하$", s)
        if m and v <= float(m.group(1)):
            return float(pts)

        # e.g., "65이상", "0.0358이상"
        m = re.match(r"^([0-9.]+)이상$", s)
        if m and v >= float(m.group(1)):
            return float(pts)

        # e.g., "24~30", "24~30이하", "0.02517~0.03579이하"
        m = re.match(r"^([0-9.]+)~([0-9.]+)(이하)?$", s)
        if m:
            lo = float(m.group(1))
            hi = float(m.group(2))
            if v >= lo and v <= hi:
                return float(pts)

        # e.g., "9~14" (no 이하)
        m = re.match(r"^([0-9.]+)~([0-9.]+)$", s)
        if m:
            lo = float(m.group(1)); hi = float(m.group(2))
            if v >= lo and v <= hi:
                return float(pts)

        # e.g., "5, 6년" -> "56" after comma removal is ambiguous; handle "5,6" in original
        if "," in raw:
            # "5, 6년" -> [5,6]
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw.replace(" ", ""))
            try:
                nums = [float(x) for x in nums]
                if len(nums) >= 2 and any(abs(v - x) < 1e-9 for x in nums):
                    return float(pts)
            except:
                pass

    return 0.0


# =========================================================
# 3) Feature engineering for flags
# =========================================================
def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scorecard에 필요한 최소 컬럼을 만들어줌(없으면 기본값으로 채움).
    CSV 업로드가 완벽하지 않아도 동작하도록 방어적으로 처리.
    """
    out = df.copy()

    # 기본 카테고리(필수)
    default_cat = {
        "성별": "남성",
        "결혼 여부": "미혼",
        "수입 유형": "기타",
        "최종 학력": "고등학교 졸업",
        "주거 형태": "주택 / 아파트",
        "자녀수_구간": "0",
        "직업": "Unknown",
        "산업군": "기타 0",
    }

    # 기본 숫자(필수)
    default_num = {
        "나이": 35,
        "근속연수": 3,
        "가입연수": 5,
        "거주지 인구 비율": 0.03,
        "가족 구성원 수": 2,
        "연간 수입": 40000000,
        "자녀 수": 0,
    }

    # binary(0/1)
    default_bin = {
        "차량 소유 여부": 0,
        "부동산 소유 여부": 0,
        "업무용 휴대전화 소유 여부": 0,
        "배우자유무": 0,  # 있으면 유용(한부모 가정 계산)
    }

    for c, v in default_cat.items():
        if c not in out.columns:
            out[c] = v

    for c, v in default_num.items():
        if c not in out.columns:
            out[c] = v

    for c, v in default_bin.items():
        if c not in out.columns:
            out[c] = v

    # dtype 안정화
    for c in default_cat.keys():
        out[c] = out[c].astype(str)

    for c in default_num.keys():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v).astype(float)

    for c in default_bin.keys():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # 결혼 여부 normalize (사용자 말: 결혼 여부는 저거 맞아)
    allowed_marriage = {"미혼", "기혼", "별거", "사별", "사실혼"}
    out["결혼 여부"] = out["결혼 여부"].where(out["결혼 여부"].isin(allowed_marriage), "미혼")

    # 성별 normalize
    out["성별"] = out["성별"].replace({"남": "남성", "여": "여성"})
    out["성별"] = out["성별"].where(out["성별"].isin({"남성", "여성"}), "남성")

    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    scorecard_flag에 있는 flag를 데이터에서 재현(가능한 범위에서)
      - 한부모 가정: (배우자유무==0) & (자녀 수>0)
      - 저소득_부동산 X: (부동산 소유 여부==0) & (연간 수입 <= 37,170,000)  (연간수입 구간 컷 참고)
      - 2,30대_저학력,고졸: (20<=나이<40) & (최종 학력 in {저학력자, 고등학교 졸업})
    """
    out = df.copy()

    out["한부모 가정"] = ((out["배우자유무"] == 0) & (out["자녀 수"] > 0)).astype(int)

    low_income_cut = 37170000  # scorecard_cont 연간수입 bin 기준(12,744,001 ~ 37,170,000 이하)
    out["저소득_부동산 X"] = ((out["부동산 소유 여부"] == 0) & (out["연간 수입"] <= low_income_cut)).astype(int)

    out["2,30대_저학력,고졸"] = (
        (out["나이"] >= 20) & (out["나이"] < 40) &
        (out["최종 학력"].isin(["저학력자", "고등학교 졸업"]))
    ).astype(int)

    return out


def add_cross_key_gender_marriage(df: pd.DataFrame) -> pd.DataFrame:
    """
    scorecard_cross의 key는 '1_미혼' 같은 형태.
    여기서 1/0은 성별 이진 인코딩(관행적으로 1=남성, 0=여성)로 가정.
    """
    out = df.copy()
    sex01 = np.where(out["성별"].eq("남성"), 1, 0)
    out["_성별01"] = sex01
    out["성별x결혼여부_key"] = out["_성별01"].astype(str) + "_" + out["결혼 여부"].astype(str)
    return out


# =========================================================
# 4) Scoring with Excel tables
# =========================================================
def score_with_excel(df_in: pd.DataFrame,
                     meta: dict,
                     base_points: float,
                     cont_df: pd.DataFrame,
                     cat_df: pd.DataFrame,
                     cross_df: pd.DataFrame,
                     binary_df: pd.DataFrame,
                     flag_df: pd.DataFrame,
                     job_ind_df: pd.DataFrame):
    """
    Return:
      scored_df (with score, pred_prob, risk_grade),
      reason_list (list of DataFrame for each row)
    """
    factor = float(meta["factor"])
    offset = float(meta["offset"])

    cont_sections = _parse_sections(cont_df)
    cat_sections  = _parse_sections(cat_df)
    cross_sections = _parse_sections(cross_df)

    bin_map = _build_value_points_map(binary_df)
    flag_map = _build_value_points_map(flag_df)
    ji_map  = _build_job_ind_map(job_ind_df)

    df = ensure_base_columns(df_in)
    df = add_flags(df)
    df = add_cross_key_gender_marriage(df)

    scores = []
    reasons_all = []

    # Prebuild cat maps for speed
    cat_maps = {sec: dict(items) for sec, items in cat_sections.items()}
    cross_maps = {sec: dict(items) for sec, items in cross_sections.items()}

    for _, row in df.iterrows():
        s = float(base_points)
        reasons = []

        # ===== Continuous =====
        def add_cont(col, sec):
            nonlocal s, reasons
            if sec in cont_sections and pd.notna(row.get(col, np.nan)):
                pts = _pick_points_from_cont(float(row[col]), cont_sections[sec])
                s += pts
                reasons.append([sec, f"{col}={row[col]}", pts])

        add_cont("근속연수", "근속연수_woe")
        add_cont("나이", "나이_woe")
        add_cont("거주지 인구 비율", "거주지 인구 비율_woe")
        add_cont("가입연수", "가입연수_woe")
        add_cont("가족 구성원 수", "가족 구성원 수_woe")

        # 연간수입_woe는 점수가 NaN인 상태(엑셀 기준)라 실제론 0점 처리됨
        # 그래도 룰이 있는 경우를 대비해 넣어둠.
        if "연간수입_woe" in cont_sections and pd.notna(row.get("연간 수입", np.nan)):
            pts = _pick_points_from_cont(float(row["연간 수입"]), cont_sections["연간수입_woe"])
            if np.isfinite(pts):
                s += pts
                reasons.append(["연간수입_woe", f"연간 수입={row['연간 수입']}", pts])

        # ===== Categorical =====
        def add_cat(col, sec):
            nonlocal s, reasons
            if sec in cat_maps:
                val = str(row.get(col, ""))
                pts = float(cat_maps[sec].get(val, 0.0))
                s += pts
                reasons.append([sec, f"{col}={val}", pts])

        add_cat("수입 유형", "수입 유형_woe")
        add_cat("최종 학력", "최종 학력_woe")
        add_cat("주거 형태", "주거 형태_woe")
        add_cat("자녀수_구간", "자녀수_구간_woe")

        # ===== Cross =====
        if "성별x결혼여부_woe" in cross_maps:
            key = str(row.get("성별x결혼여부_key", ""))
            pts = float(cross_maps["성별x결혼여부_woe"].get(key, 0.0))
            s += pts
            reasons.append(["성별x결혼여부_woe", key, pts])

        # ===== Binary =====
        def add_bin(feat):
            nonlocal s, reasons
            if feat in bin_map:
                v = int(row.get(feat, 0))
                pts = float(bin_map[feat].get(v, 0.0))
                s += pts
                reasons.append([feat, f"{feat}={v}", pts])

        add_bin("차량 소유 여부")
        add_bin("부동산 소유 여부")
        add_bin("업무용 휴대전화 소유 여부")

        # ===== Flags =====
        def add_flag(feat):
            nonlocal s, reasons
            if feat in flag_map:
                v = int(row.get(feat, 0))
                pts = float(flag_map[feat].get(v, 0.0))
                s += pts
                reasons.append([feat, f"{feat}={v}", pts])

        add_flag("한부모 가정")
        add_flag("저소득_부동산 X")
        add_flag("2,30대_저학력,고졸")

        # ===== Job × Industry points =====
        job = str(row.get("직업", "Unknown")).strip()
        ind = str(row.get("산업군", "기타 0")).strip()
        pts = float(ji_map.get((job, ind), 0.0))
        s += pts
        reasons.append(["직업×산업(points)", f"{job} × {ind}", pts])

        scores.append(s)
        reasons_all.append(pd.DataFrame(reasons, columns=["component", "value", "points"]))

    out = df_in.copy()
    out["score"] = scores

    # Score → p_bad (연체확률)
    odds = np.exp((out["score"] - offset) / factor)  # good/bad odds
    out["pred_prob"] = 1.0 / (1.0 + odds)

    return out, reasons_all


def assign_grade_from_proba(proba: pd.Series, high_q=0.80, mid_q=0.40):
    """
    운영 스토리: 상위 20% High, 상위 60% Mid, 나머지 Low
    """
    high_cut = float(proba.quantile(high_q))
    mid_cut = float(proba.quantile(mid_q))

    def grade(p):
        if p >= high_cut:
            return "High"
        elif p >= mid_cut:
            return "Mid"
        else:
            return "Low"

    g = proba.apply(grade)
    return g, high_cut, mid_cut


# =========================================================
# 5) Optional demo data loader (train_data.csv in repo root)
# =========================================================
@st.cache_data
def load_demo_data(max_rows=20000, _version="v1"):
    base_dir = Path(__file__).resolve().parent
    demo_path = base_dir / "train_data.csv"
    if not demo_path.exists():
        return None
    df = pd.read_csv(demo_path)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df


# =========================================================
# 6) UI
# =========================================================
st.set_page_config(page_title="연체 리스크 스코어카드 대시보드", layout="wide")
st.title("연체 리스크 스코어카드 대시보드 (Excel Scorecard 기반)")

meta, base_points, cont, cat, cross, binary, flag, job_ind = load_scorecard_excel("v1")

tabs = st.tabs([
    "① Overview",
    "② Score Customers",
    "③ Explain (Reason Codes)",
    "④ Job×Industry Insights",
    "⑤ 변수별 등급 시각화",
    "⑥ 고객정보 입력(데모)"
])


# -------------------------
# ① Overview
# -------------------------
with tabs[0]:
    st.subheader("Executive Summary (심사위원용 30초 요약)")
    st.info(
        "본 대시보드는 **금융권 표준 Scorecard 방식(점수카드)** 으로 고객 기본정보만으로 연체 위험을 조기 식별합니다.\n"
        "- 입력 → 점수(score) → 연체확률(PD) → High/Mid/Low 등급\n"
        "- Reason Codes로 ‘왜 위험한지’를 항목별 점수 기여로 설명\n"
        "- 직업×산업 점수를 별도 반영해 고용/업종 리스크 프로파일을 운영 관점에서 활용"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDO", f"{float(meta.get('PDO', np.nan)):.0f}")
    c2.metric("Base Score", f"{float(meta.get('base_score', np.nan)):.0f}")
    c3.metric("Base Odds", f"{float(meta.get('base_odds', np.nan)):.4f}")
    c4.metric("Base Points", f"{base_points:.2f}")

    st.markdown("### 운영 정책 예시 (Risk Grade Action)")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🔴 High")
        st.markdown("- 한도/결제조건 조정\n- 사전 안내/콜\n- 연체예방 캠페인")
    with b:
        st.markdown("#### 🟠 Mid")
        st.markdown("- 모니터링 강화\n- 자동이체/분할납부 유도\n- 행동기반 알림")
    with c:
        st.markdown("#### 🟢 Low")
        st.markdown("- 정상 유지\n- 우량 고객 프로모션\n- 과도 제약 최소화")

    st.divider()
    st.markdown("### 점수카드 룰(일부) 미리보기")
    st.caption("각 구간/범주에 부여된 points(점수)가 낮을수록 위험(연체확률↑) 방향입니다.")
    colL, colR = st.columns(2)
    with colL:
        st.write("연속형(예: 나이/근속/가입) 일부")
        st.dataframe(cont.head(35), use_container_width=True)
    with colR:
        st.write("범주형(예: 학력/주거/자녀구간) 일부")
        st.dataframe(cat, use_container_width=True)


# -------------------------
# ② Score Customers
# -------------------------
with tabs[1]:
    st.subheader("고객 점수화 (CSV 업로드 → score/확률/등급 → 다운로드)")
    st.caption("※ 업로드 파일은 가급적 원본 컬럼(성별/결혼 여부/직업/산업군/나이/근속연수/가입연수 등)을 포함해주세요.")

    demo_df = load_demo_data()
    use_demo = st.checkbox("업로드 없이 레포의 train_data.csv(샘플)로 데모 실행", value=(demo_df is not None))

    up = st.file_uploader("CSV 업로드", type=["csv"], disabled=use_demo)

    if use_demo:
        if demo_df is None:
            st.warning("레포 루트에 train_data.csv가 없어서 데모 실행 불가. 업로드로 진행하세요.")
        else:
            df_new = demo_df.copy()
            st.success(f"train_data.csv에서 샘플 {len(df_new):,}행 로드 완료")
    else:
        if up is None:
            st.warning("CSV를 업로드하거나, 데모 실행을 체크하세요.")
            df_new = None
        else:
            df_new = pd.read_csv(up)

    if df_new is not None:
        scored, reasons_all = score_with_excel(df_new, meta, base_points, cont, cat, cross, binary, flag, job_ind)

        grade_vec, high_cut, mid_cut = assign_grade_from_proba(scored["pred_prob"], high_q=0.80, mid_q=0.40)
        scored["risk_grade"] = grade_vec.values

        st.session_state["last_scored"] = scored.reset_index(drop=True)
        st.session_state["last_reasons"] = reasons_all
        st.session_state["cuts"] = {"high": high_cut, "mid": mid_cut}

        # Summary cards
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows", f"{len(scored):,}")
        s2.metric("High", f"{(scored['risk_grade']=='High').sum():,}")
        s3.metric("Mid", f"{(scored['risk_grade']=='Mid').sum():,}")
        s4.metric("Low", f"{(scored['risk_grade']=='Low').sum():,}")

        left, right = st.columns([1.2, 1])
        with left:
            fig = px.histogram(scored, x="pred_prob", nbins=40, title="Predicted Probability Distribution")
            fig.add_vline(x=mid_cut, line_dash="dash", annotation_text="Mid cut", annotation_position="top left")
            fig.add_vline(x=high_cut, line_dash="dash", annotation_text="High cut", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            gcount = scored["risk_grade"].value_counts().reindex(["High","Mid","Low"]).fillna(0).astype(int).reset_index()
            gcount.columns = ["risk_grade","count"]
            fig2 = px.bar(gcount, x="risk_grade", y="count", title="Risk Grade Count")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Scoring Results (상위 200행 미리보기)")
        st.dataframe(scored.head(200), use_container_width=True)

        csv_bytes = scored.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 결과 다운로드 (CSV)",
            data=csv_bytes,
            file_name="scoring_results.csv",
            mime="text/csv"
        )


# -------------------------
# ③ Explain (Reason Codes)
# -------------------------
with tabs[2]:
    st.subheader("Explainability: Reason Codes (고객 1명 기준 점수 하락 요인 Top 7)")
    st.caption("심사위원 포인트: '왜 이 고객이 High인가?'를 Scorecard 항목별 점수 기여로 설명합니다.")

    if "last_scored" not in st.session_state:
        st.info("먼저 ② Score Customers 탭에서 점수화를 수행하세요.")
    else:
        scored = st.session_state["last_scored"]
        reasons_all = st.session_state["last_reasons"]

        col1, col2 = st.columns([1, 1])
        with col1:
            idx = st.number_input("설명할 고객 행 번호", 0, len(scored)-1, 0, step=1)
            st.write("선택 고객 요약")
            st.dataframe(scored.loc[[idx], ["score","pred_prob","risk_grade"]], use_container_width=True)
            st.dataframe(scored.loc[[idx]].head(1), use_container_width=True)

        with col2:
            st.write("Reason Codes (점수 하락 요인 Top 7)")
            r = reasons_all[int(idx)].copy()

            # points가 음수면 score를 깎아서 위험도를 올리는 방향(보통)
            r_sorted = r.sort_values("points", ascending=True).head(7)
            st.dataframe(r_sorted, use_container_width=True)

            st.markdown(
                "- points가 **더 음수(작을수록)** → 그 항목이 **위험에 더 크게 기여**\n"
                "- Scorecard 기반이라 **설명 가능 + 운영 정책 연결**에 매우 유리"
            )


# -------------------------
# ④ Job×Industry Insights
# -------------------------
with tabs[3]:
    st.subheader("직업×산업 점수 인사이트 (job_ind_points)")
    st.caption("직업/산업 조합별 점수(points) 분포를 통해 리스크 프로파일을 확인합니다.")

    dfji = job_ind.copy()
    dfji["직업"] = dfji["직업"].astype(str)
    dfji["산업군"] = dfji["산업군"].astype(str)

    q1, q2 = st.columns(2)
    with q1:
        job_q = st.text_input("직업 검색", "")
    with q2:
        ind_q = st.text_input("산업군 검색 (예: 무역, 산업 3)", "")

    f = dfji.copy()
    if job_q.strip():
        f = f[f["직업"].str.contains(job_q.strip(), na=False)]
    if ind_q.strip():
        f = f[f["산업군"].str.contains(ind_q.strip(), na=False)]

    a, b = st.columns(2)
    with a:
        st.write("Most Risky (points 낮음) Top 15")
        st.dataframe(f.sort_values("points").head(15), use_container_width=True)
    with b:
        st.write("Most Stable (points 높음) Top 15")
        st.dataframe(f.sort_values("points", ascending=False).head(15), use_container_width=True)

    st.markdown("### 전체 테이블")
    st.dataframe(f, use_container_width=True)


# -------------------------
# ⑤ 변수별 등급 시각화
# -------------------------
with tabs[4]:
    st.subheader("변수별 등급 시각화 (Score 결과 기반)")
    st.caption("점수화 결과를 기준으로 High/Mid/Low 별 변수 분포를 시각화합니다.")

    if "last_scored" not in st.session_state:
        st.info("먼저 ② Score Customers 탭에서 점수화를 수행하세요.")
    else:
        scored = st.session_state["last_scored"].copy()
        scored["risk_grade"] = pd.Categorical(scored["risk_grade"], categories=["High","Mid","Low"], ordered=True)

        # 후보 컬럼들(없어도 자동 제외)
        num_candidates = ["나이", "가입연수", "근속연수", "연간 수입", "거주지 인구 비율", "가족 구성원 수", "score", "pred_prob"]
        cat_candidates = ["성별", "결혼 여부", "수입 유형", "최종 학력", "주거 형태", "자녀수_구간", "직업", "산업군"]

        num_cols = [c for c in num_candidates if c in scored.columns]
        cat_cols = [c for c in cat_candidates if c in scored.columns]

        st.markdown("### 수치형 변수 분포 (등급별)")
        for col in num_cols:
            fig = px.box(scored, x="risk_grade", y=col, points="outliers", title=f"{col} by Risk Grade")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 범주형 변수 구성비 (등급별, 100% stacked)")
        for col in cat_cols[:6]:
            tmp = scored.groupby(["risk_grade", col]).size().reset_index(name="count")
            tmp["pct"] = tmp["count"] / tmp.groupby("risk_grade")["count"].transform("sum")

            fig = px.bar(tmp, x="risk_grade", y="pct", color=col,
                         title=f"{col} composition by Risk Grade (100%)",
                         barmode="stack")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# -------------------------
# ⑥ Interactive Demo
# -------------------------
with tabs[5]:
    st.subheader("고객정보 입력 → 등급 확인 (Interactive Demo)")
    st.info(
        "운영 가정\n"
        "- 본 스코어카드는 월 1회(또는 분기 1회) 최신 고객 데이터로 재산출/검증됩니다.\n"
        "- 업데이트된 고객을 대상으로 연체 위험을 조기 탐지하고, 위험 등급별 정책을 적용합니다.\n"
        "- 아래에 기본 정보를 입력해 등급을 확인해보세요."
    )

    # 입력 폼
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            성별 = st.selectbox("성별", ["남성", "여성"])
            결혼여부 = st.selectbox("결혼 여부", ["미혼", "기혼", "별거", "사별", "사실혼"])
            나이 = st.number_input("나이", min_value=18, max_value=90, value=35, step=1)

        with c2:
            직업 = st.selectbox("직업", [
                "Unknown","단순 노동자","영업직","핵심 노동자","관리직","운전자","기술직","회계사",
                "의료 업계 종사자","보안 업계 종사자","조리사","미화원","가정부","저임금 노동자",
                "비서","요식업 종사자","부동산중개업자","인사 담당자","IT 업계 종사자"
            ])
            산업군 = st.text_input("산업군 (예: 무역 0, 산업 3, 운송 1 등)", value="무역 0")
            수입유형 = st.selectbox("수입 유형", ["근로자", "공무원", "연금수령자", "기타"])

        with c3:
            최종학력 = st.selectbox("최종 학력", ["저학력자", "고등학교 졸업", "대학교 중퇴", "대학교 졸업 이상"])
            주거형태 = st.selectbox("주거 형태", ["아파트 임대", "오피스텔", "공공분양", "주택 / 아파트"])
            자녀구간 = st.selectbox("자녀수_구간", ["0", "1", "2", "3+"])

        근속연수 = st.number_input("근속연수(년)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        가입연수 = st.number_input("가입연수(년)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        거주지비율 = st.number_input("거주지 인구 비율", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
        가족수 = st.number_input("가족 구성원 수", min_value=1, max_value=10, value=2, step=1)
        연간수입 = st.number_input("연간 수입(원)", min_value=0, value=40000000, step=1000000)

        차량 = st.selectbox("차량 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
        부동산 = st.selectbox("부동산 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")
        업무폰 = st.selectbox("업무용 휴대전화 소유 여부", [0, 1], format_func=lambda x: "있음" if x==1 else "없음")

        배우자유무 = st.selectbox("배우자 유무(한부모 계산용)", [0, 1], format_func=lambda x: "없음" if x==0 else "있음")
        자녀수 = st.number_input("자녀 수(한부모 계산용)", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("🚀 등급 산출")

    if submitted:
        row = {
            "성별": 성별,
            "결혼 여부": 결혼여부,
            "나이": 나이,
            "직업": 직업,
            "산업군": 산업군,
            "수입 유형": 수입유형,
            "최종 학력": 최종학력,
            "주거 형태": 주거형태,
            "자녀수_구간": 자녀구간,
            "근속연수": 근속연수,
            "가입연수": 가입연수,
            "거주지 인구 비율": 거주지비율,
            "가족 구성원 수": 가족수,
            "연간 수입": 연간수입,
            "차량 소유 여부": 차량,
            "부동산 소유 여부": 부동산,
            "업무용 휴대전화 소유 여부": 업무폰,
            "배우자유무": 배우자유무,
            "자녀 수": 자녀수,
        }
        df_one = pd.DataFrame([row])

        scored_one, reasons_one = score_with_excel(df_one, meta, base_points, cont, cat, cross, binary, flag, job_ind)
        p = float(scored_one["pred_prob"].iloc[0])
        score = float(scored_one["score"].iloc[0])

        # 컷은 마지막 scoring 결과가 있으면 그걸 쓰고, 없으면 임시 컷
        if "cuts" in st.session_state:
            high_cut = st.session_state["cuts"]["high"]
            mid_cut = st.session_state["cuts"]["mid"]
        else:
            # 임시(발표용 fallback)
            high_cut, mid_cut = 0.20, 0.10

        grade = "High" if p >= high_cut else ("Mid" if p >= mid_cut else "Low")

        st.success(f"점수(score): **{score:.1f}** | 예측 연체확률: **{p:.3f}** | 위험등급: **{grade}**")
        st.progress(float(np.clip(p, 0.0, 1.0)))

        st.markdown("### Reason Codes (점수 하락 요인 Top 7)")
        r = reasons_one[0].sort_values("points", ascending=True).head(7)
        st.dataframe(r, use_container_width=True)

        st.markdown("### 추천 선제 정책")
        if grade == "High":
            st.error("High: 한도/결제조건 조정 + 사전 안내(콜) + 연체 예방 캠페인")
        elif grade == "Mid":
            st.warning("Mid: 모니터링 강화 + 자동이체/분할납부 유도 + 리마인드")
        else:
            st.info("Low: 정상 유지 + 우량 고객 프로모션 + 과도 제약 최소화")


st.divider()
st.caption(
    "발표 흐름 추천: ①Overview(정책+표준 스코어카드) → ②(점수화 데모) → ③(Reason Codes) → ④(직업×산업 인사이트) → ⑤(변수 분포)."
)
