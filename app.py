import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="NPB選手年俸予測システム",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&family=Oswald:wght@600&display=swap');

:root {
    --navy:   #0d1b2a;
    --navy2:  #1b2f45;
    --gold:   #f0a500;
    --gold2:  #ffd166;
    --pink:   #ff6b9d;
    --light:  #f5f7fa;
    --white:  #ffffff;
    --gray:   #8899aa;
    --green:  #06d6a0;
    --red:    #ef476f;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    color: var(--light) !important;
}

[data-testid="stMain"] {
    background: var(--navy) !important;
}

.block-container {
    max-width: 900px !important;
    padding: 1.5rem 2rem 3rem 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

h1 {
    font-family: 'Oswald', sans-serif !important;
    font-size: 2.2rem !important;
    color: var(--gold) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
h2, h3 { color: var(--gold2) !important; }
h1 a, h2 a, h3 a, h4 a { display: none !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }

hr { border-color: var(--navy2) !important; }

[data-testid="stMetric"] {
    background: var(--navy2) !important;
    border: 1px solid rgba(240,165,0,0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--gray) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--gold) !important; font-weight: 700 !important; }

[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold), var(--pink)) !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(240,165,0,0.4) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    color: var(--gray) !important;
    border: 1px solid var(--navy2) !important;
    border-radius: 8px !important;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stNumberInput"] input {
    background: var(--navy2) !important;
    color: var(--light) !important;
    border-color: rgba(240,165,0,0.2) !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--gold) !important; }

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    animation: none !important;
    transition: none !important;
}
.stDataFrame { animation: none !important; transition: none !important; }

[data-testid="stExpander"] {
    background: var(--navy2) !important;
    border: 1px solid rgba(240,165,0,0.15) !important;
    border-radius: 10px !important;
}

[data-testid="stSpinner"] { color: var(--gold) !important; }
[data-testid="stRadio"] label { color: var(--light) !important; }
[data-testid="stMultiSelect"] { background: var(--navy2) !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2f45 100%) !important;
    border-right: 1px solid rgba(240,165,0,0.15) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
}
[data-testid="stSidebarUserContent"] { padding-top: 1rem !important; }
[data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    height: 100vh !important;
    padding: 0 0.75rem 2rem 0.75rem !important;
}
[data-testid="stSidebar"] * { cursor: default !important; }
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] input[type="radio"],
[data-testid="stSidebar"] label[data-baseweb="radio"] { cursor: pointer !important; }
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    color: var(--light) !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: rgba(240,165,0,0.1) !important;
}

section[data-testid="stMain"] {
    transition: margin-left 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}

[data-testid="stMarkdownContainer"] p { color: var(--light) !important; }
[data-testid="stMarkdownContainer"] a { color: var(--gold) !important; }
[data-testid="stProgressBar"] > div { background: var(--gold) !important; }
[data-testid="stVerticalBlock"] { gap: 0.5rem; }

[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
@media (max-width: 768px) {
    [data-testid="collapsedControl"] { display: block !important; }
    [data-testid="stSidebarCollapseButton"] { display: block !important; }
    .block-container { max-width: 100% !important; padding: 0.5rem !important; }
}
[data-testid="collapsedControl"] * { color: white !important; fill: white !important; stroke: white !important; }
[data-testid="stSidebarCollapseButton"] * { color: white !important; fill: white !important; stroke: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid rgba(240,165,0,0.15);
    margin-bottom: 1rem;
}
.sidebar-logo .logo-icon { font-size: 2.5rem; }
.sidebar-logo .logo-title {
    font-family: 'Oswald', sans-serif;
    font-size: 0.9rem;
    color: #f0a500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8899aa;
    padding: 0.5rem 0.75rem 0.25rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stDataFrame"] { animation: none !important; transition: none !important; }
.stDataFrame { animation: none !important; transition: none !important; }
[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)

try:
    import matplotlib_fontja
    matplotlib_fontja.japanize()
except ImportError:
    pass

# ============================================================
# ユーティリティ関数
# ============================================================

def normalize_name(name):
    """選手名の正規化（全角スペース・半角スペース統一、前後空白除去）"""
    if pd.isna(name):
        return name
    return str(name).replace('\u3000', ' ').replace('　', ' ').strip()


def parse_innings_pitched(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    val = str(val).strip()
    if chr(10) in val:
        parts = val.split(chr(10))
        try:
            whole = float(parts[0].strip())
            frac_str = parts[1].strip().lstrip('.')
            frac = int(frac_str) if frac_str else 0
            return whole + frac / 3.0
        except Exception:
            return np.nan
    if '\\n' in val:
        val = val.replace('\\n', '.')
    if '.' in val:
        parts = val.split('.')
        try:
            whole = float(parts[0])
            frac = int(parts[1]) if parts[1] else 0
            return whole + frac / 3.0
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def calculate_salary_limit(previous_salary):
    if previous_salary >= 100_000_000:
        reduction_rate = 0.40
        min_salary = previous_salary * 0.60
    else:
        reduction_rate = 0.25
        min_salary = previous_salary * 0.75
    return min_salary, reduction_rate


def check_salary_reduction_limit(predicted_salary, previous_salary):
    min_salary, reduction_rate = calculate_salary_limit(previous_salary)
    if predicted_salary < min_salary:
        return True, min_salary, reduction_rate
    return False, min_salary, reduction_rate


def add_to_history(player_name, predict_year, predicted_salary, actual_salary, previous_salary,
                   stats_dict, model_name, player_type, is_limited=False, limited_salary=None):
    jst_time = datetime.utcnow() + timedelta(hours=9)
    history_item = {
        '予測日時': jst_time.strftime('%Y-%m-%d %H:%M:%S'),
        '選手名': player_name,
        '種別': player_type,
        '予測年度': predict_year,
        '予測年俸': predicted_salary,
        '制限後年俸': limited_salary if is_limited else predicted_salary,
        '実際の年俸': actual_salary,
        '前年年俸': previous_salary,
        '減額制限': is_limited,
        'モデル': model_name,
        '成績': stats_dict
    }
    st.session_state.prediction_history.insert(0, history_item)
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[:20]


# ============================================================
# タイトル
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1b2f45 0%, #0d1b2a 100%);
    border: 1px solid rgba(240,165,0,0.2);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
">
    <div style="font-size:2.8rem;">⚾</div>
    <div>
        <div style="font-family:'Oswald',sans-serif;font-size:1.8rem;color:#f0a500;letter-spacing:2px;text-transform:uppercase;line-height:1.1;">NPB Salary Predictor</div>
        <div style="color:#8899aa;font-size:0.85rem;margin-top:0.2rem;">NPB選手年俸予測システム</div>
    </div>
</div>
""", unsafe_allow_html=True)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================
# データ読み込み
# ============================================================
@st.cache_data
def load_data():
    try:
        salary_df   = pd.read_csv('data/salary_2023&2024&2025.csv')
        stats_2023  = pd.read_csv('data/stats_2023.csv')
        stats_2024  = pd.read_csv('data/stats_2024.csv')
        stats_2025  = pd.read_csv('data/stats_2025.csv')
        titles_df   = pd.read_csv('data/titles_2023&2024&2025.csv')
        pitcher_df  = pd.read_csv('data/npb_pitcher_stats.csv')
        return salary_df, stats_2023, stats_2024, stats_2025, titles_df, pitcher_df, True
    except FileNotFoundError:
        return None, None, None, None, None, None, False

salary_df, stats_2023, stats_2024, stats_2025, titles_df, pitcher_df_raw, data_loaded = load_data()

# ============================================================
# ファイルアップロード処理
# ============================================================
if not data_loaded:
    st.sidebar.markdown("**CSVファイルを選択してアップロード（8つ全て）：**")
    st.sidebar.caption("besmoney_salary_2023/2024/2025.csv, stats_2023/2024/2025.csv, titles.csv, npb_pitcher_stats.csv")
    uploaded_files = st.sidebar.file_uploader(
        "CSVファイルを選択",
        type=['csv'],
        accept_multiple_files=True
    )

    if uploaded_files:
        file_dict = {}
        for file in uploaded_files:
            name = file.name.lower()
            if ('besmoney' in name or 'salary' in name) and '2023' in name:
                file_dict['salary_2023'] = file
            elif ('besmoney' in name or 'salary' in name) and '2024' in name:
                file_dict['salary_2024'] = file
            elif ('besmoney' in name or 'salary' in name) and '2025' in name:
                file_dict['salary_2025'] = file
            elif 'titles' in name or 'タイトル' in name:
                file_dict['titles'] = file
            elif 'pitcher' in name or '投手' in name:
                file_dict['pitcher'] = file
            elif '2023' in name:
                file_dict['stats_2023'] = file
            elif '2024' in name:
                file_dict['stats_2024'] = file
            elif '2025' in name:
                file_dict['stats_2025'] = file

        required = ['salary_2023', 'salary_2024', 'salary_2025', 'titles', 'stats_2023', 'stats_2024', 'stats_2025']
        missing = [k for k in required if k not in file_dict]

        if not missing:
            sal23 = pd.read_csv(file_dict['salary_2023'])
            sal24 = pd.read_csv(file_dict['salary_2024'])
            sal25 = pd.read_csv(file_dict['salary_2025'])
            salary_df = (sal23, sal24, sal25)
            stats_2023  = pd.read_csv(file_dict['stats_2023'])
            stats_2024  = pd.read_csv(file_dict['stats_2024'])
            stats_2025  = pd.read_csv(file_dict['stats_2025'])
            titles_df   = pd.read_csv(file_dict['titles'])
            pitcher_df_raw = pd.read_csv(file_dict['pitcher']) if 'pitcher' in file_dict else None
            data_loaded = True
            st.sidebar.success(f"✅ {len(uploaded_files)}ファイル読み込み完了")
        else:
            st.sidebar.warning(f"⚠️ 不足ファイル: {missing}")

# ============================================================
# データ前処理
# ============================================================
@st.cache_data
def prepare_salary_long(_salary_df):
    """横持ちCSVをlong形式に変換"""
    df = _salary_df.copy()
    rows = []
    for _, row in df.iterrows():
        name23 = row.get('選手名_2023')
        sal23  = row.get('年俸_円_2023')
        name24 = row.get('選手名_2024_2025')
        sal24  = row.get('年俸_円_2024')
        sal25  = row.get('年俸_円_2025')

        if pd.notna(name23) and pd.notna(sal23) and sal23 > 0:
            rows.append({'選手名': normalize_name(name23), '年俸_円': sal23, '年度': 2023})
        if pd.notna(name24) and pd.notna(sal24) and sal24 > 0:
            rows.append({'選手名': normalize_name(name24), '年俸_円': sal24, '年度': 2024})
        if pd.notna(name24) and pd.notna(sal25) and sal25 > 0:
            rows.append({'選手名': normalize_name(name24), '年俸_円': sal25, '年度': 2025})

    salary_long = pd.DataFrame(rows)
    salary_long = salary_long.drop_duplicates(subset=['選手名', '年度'], keep='first')
    return salary_long


@st.cache_data
def prepare_batter_data(_salary_df, _stats_2023, _stats_2024, _stats_2025, _titles_df):
    """野手データの前処理"""
    titles_clean = _titles_df.dropna(subset=['選手名']).copy()
    titles_clean['選手名'] = titles_clean['選手名'].apply(normalize_name)
    title_summary = titles_clean.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')

    s23 = _stats_2023.copy(); s23['年度'] = 2023
    s24 = _stats_2024.copy(); s24['年度'] = 2024
    s25 = _stats_2025.copy(); s25['年度'] = 2025
    stats_all = pd.concat([s23, s24, s25], ignore_index=True)
    stats_all['選手名'] = stats_all['選手名'].apply(normalize_name)

    salary_long = prepare_salary_long(_salary_df)

    age_backup = None
    if '年齢' in stats_all.columns:
        age_backup = stats_all[['選手名', '年度', '年齢']].copy()

    stats_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    stats_with_titles['タイトル数'] = stats_with_titles['タイトル数'].fillna(0)

    stats_with_titles['予測年度'] = stats_with_titles['年度'] + 1
    merged = pd.merge(
        stats_with_titles,
        salary_long,
        left_on=['選手名', '予測年度'],
        right_on=['選手名', '年度'],
        suffixes=('_成績', '_年俸')
    )

    if '年齢' not in merged.columns and age_backup is not None:
        merged = pd.merge(merged, age_backup,
                          left_on=['選手名', '年度_成績'],
                          right_on=['選手名', '年度'], how='left')
        if '年度_y' in merged.columns:
            merged.drop(columns=['年度_y'], inplace=True)
        if '年度_x' in merged.columns:
            merged.rename(columns={'年度_x': '年度_成績'}, inplace=True)

    merged.drop(columns=['年度_年俸', '予測年度'], inplace=True)
    merged.rename(columns={'年度_成績': '成績年度'}, inplace=True)

    return merged, stats_with_titles, salary_long


@st.cache_data
def prepare_pitcher_data(_pitcher_df_raw, _salary_df, _titles_df):
    """投手データの前処理"""
    df = _pitcher_df_raw.copy()
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

    # 選手名の正規化
    df['選手名'] = df['選手名'].apply(normalize_name)

    df['投球回_実数'] = df['投球回'].apply(parse_innings_pitched)
    df['防御率'] = pd.to_numeric(df['防御率'], errors='coerce')
    df['勝率']   = pd.to_numeric(df['勝率'],   errors='coerce')

    # タイトルデータのマージ（年度列が分裂しないよう注意）
    titles_clean = _titles_df.dropna(subset=['選手名']).copy()
    titles_clean['選手名'] = titles_clean['選手名'].apply(normalize_name)
    title_summary = titles_clean.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')
    df = pd.merge(df, title_summary, on=['選手名', '年度'], how='left')
    df['タイトル数'] = df['タイトル数'].fillna(0)

    # マージ後に年度列名が変わっていたら修正
    if '年度_x' in df.columns:
        df.rename(columns={'年度_x': '年度'}, inplace=True)
    if '年度_y' in df.columns:
        df.drop(columns=['年度_y'], inplace=True)

    salary_long = prepare_salary_long(_salary_df)

    # 年俸データは2023〜2025のみなので、投手CSVも該当年度に絞る
    df_filtered = df[df['年度'].isin([2022, 2023, 2024, 2025])].copy()

    # ── マージ戦略1: 翌年度マージ（成績年 → 翌年の年俸） ──
    df_filtered['予測年度'] = df_filtered['年度'] + 1
    merged = pd.merge(
        df_filtered,
        salary_long,
        left_on=['選手名', '予測年度'],
        right_on=['選手名', '年度'],
        suffixes=('_成績', '_年俸')
    )

    # ── マージ戦略2: 結果が少なければ同年度マージも追加 ──
    if len(merged) < 10:
        merged_same = pd.merge(
            df_filtered,
            salary_long,
            left_on=['選手名', '年度'],
            right_on=['選手名', '年度'],
            suffixes=('_成績', '_年俸')
        )
        # 年俸列名を統一
        for alt_col in ['年俸_円_年俸', '年俸_円_成績']:
            if alt_col in merged_same.columns and '年俸_円' not in merged_same.columns:
                merged_same.rename(columns={alt_col: '年俸_円'}, inplace=True)

        if len(merged_same) > len(merged) and '年俸_円' in merged_same.columns:
            merged_same['成績年度'] = merged_same['年度']
            drop_cols = [c for c in ['予測年度', '年度_年俸', '年度_成績'] if c in merged_same.columns]
            merged_same.drop(columns=drop_cols, inplace=True, errors='ignore')
            stats_all_with_titles = df.copy()
            if '年齢' not in stats_all_with_titles.columns:
                stats_all_with_titles['年齢'] = 28
            return merged_same, stats_all_with_titles, salary_long

    # 通常の後処理
    drop_cols = [c for c in ['年度_年俸', '予測年度'] if c in merged.columns]
    merged.drop(columns=drop_cols, inplace=True, errors='ignore')
    if '年度_成績' in merged.columns:
        merged.rename(columns={'年度_成績': '成績年度'}, inplace=True)
    elif '年度' in merged.columns and '成績年度' not in merged.columns:
        merged.rename(columns={'年度': '成績年度'}, inplace=True)

    stats_all_with_titles = df.copy()
    if '年齢' not in stats_all_with_titles.columns:
        stats_all_with_titles['年齢'] = 28

    return merged, stats_all_with_titles, salary_long


# ============================================================
# モデル訓練
# ============================================================
BATTER_FEATURES = [
    '試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', '本塁打',
    '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打',
    '打率', '出塁率', '長打率', '犠打', '犠飛', 'タイトル数'
]

PITCHER_FEATURES = [
    '登板', '勝利', '敗北', 'セーブ', 'H', 'HP', '完投', '完封', '無四球',
    '勝率', '打者', '投球回_実数', '安打', '本塁打', '四球', '死球', '三振',
    '暴投', 'ボーク', '失点', '自責点', '防御率', 'タイトル数'
]


@st.cache_resource
def train_batter_models(_merged_df):
    feature_cols = BATTER_FEATURES.copy()
    ml_df = _merged_df.copy()
    if '年齢' in ml_df.columns:
        feature_cols.append('年齢')
    else:
        ml_df['年齢'] = 28
        feature_cols.append('年齢')

    ml_df = ml_df[feature_cols + ['年俸_円', '選手名', '成績年度']].dropna()

    X = ml_df[feature_cols]
    y_log = np.log1p(ml_df['年俸_円'])

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_test_orig = np.expm1(y_test_log)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        '線形回帰':        LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    results = {}
    for name, model in models.items():
        if name == '線形回帰':
            model.fit(X_train_scaled, y_train_log)
            y_pred = np.expm1(model.predict(X_test_scaled))
        else:
            model.fit(X_train, y_train_log)
            y_pred = np.expm1(model.predict(X_test))
        results[name] = {
            'model': model,
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'R2':  r2_score(y_test_orig, y_pred)
        }

    best_name  = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_name]['model']
    return best_model, best_name, scaler, feature_cols, results, ml_df


@st.cache_resource
def train_pitcher_models(_merged_df):
    feature_cols = PITCHER_FEATURES.copy()
    ml_df = _merged_df.copy()
    if '年齢' in ml_df.columns:
        feature_cols.append('年齢')
    else:
        ml_df['年齢'] = 28
        feature_cols.append('年齢')

    missing = [c for c in feature_cols if c not in ml_df.columns]
    if missing:
        feature_cols = [c for c in feature_cols if c in ml_df.columns]

    ml_df = ml_df[feature_cols + ['年俸_円', '選手名', '成績年度']].dropna()

    # 閾値を5件に緩和
    if len(ml_df) < 5:
        return None, None, None, feature_cols, {}, ml_df

    X = ml_df[feature_cols]
    y_log = np.log1p(ml_df['年俸_円'])

    # サンプル数が少ない場合はtest_sizeを調整
    test_size = 0.2 if len(ml_df) >= 25 else max(1, int(len(ml_df) * 0.2))
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=test_size, random_state=42
    )
    y_test_orig = np.expm1(y_test_log)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        '線形回帰':        LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    results = {}
    for name, model in models.items():
        if name == '線形回帰':
            model.fit(X_train_scaled, y_train_log)
            y_pred = np.expm1(model.predict(X_test_scaled))
        else:
            model.fit(X_train, y_train_log)
            y_pred = np.expm1(model.predict(X_test))
        results[name] = {
            'model': model,
            'MAE': mean_absolute_error(y_test_orig, y_pred),
            'R2':  r2_score(y_test_orig, y_pred)
        }

    best_name  = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_name]['model']
    return best_model, best_name, scaler, feature_cols, results, ml_df


def predict_salary(player_stats_row, feature_cols, best_model, best_model_name, scaler):
    if best_model is None:
        return None

    feat_values = []
    for col in feature_cols:
        if col in player_stats_row.index:
            val = player_stats_row[col]
            feat_values.append(float(val) if pd.notna(val) else 0.0)
        else:
            feat_values.append(28.0 if col == '年齢' else 0.0)

    features = np.array([feat_values])

    if best_model_name == '線形回帰':
        if scaler is None:
            return None
        pred_log = best_model.predict(scaler.transform(features))[0]
    else:
        pred_log = best_model.predict(features)[0]

    salary = round(np.expm1(pred_log) / 100_000) * 100_000
    return salary


# ============================================================
# アプリ本体
# ============================================================
if data_loaded:
    if not st.session_state.model_trained:
        with st.spinner('🤖 野手・投手モデルを訓練中...'):
            # 野手
            batter_merged, batter_stats_all, salary_long = prepare_batter_data(
                salary_df, stats_2023, stats_2024, stats_2025, titles_df
            )
            b_model, b_name, b_scaler, b_fcols, b_results, b_ml_df = train_batter_models(batter_merged)

            # 投手
            if pitcher_df_raw is not None:
                pitcher_merged, pitcher_stats_all, _ = prepare_pitcher_data(
                    pitcher_df_raw, salary_df, titles_df
                )
                p_model, p_name, p_scaler, p_fcols, p_results, p_ml_df = train_pitcher_models(pitcher_merged)

                # マージ結果が少ない場合に警告（エラーではなく続行）
                if p_model is None:
                    st.warning(
                        f"⚠️ 投手モデルの訓練データが不足しています（マージ件数: {len(pitcher_merged)}件）。"
                        "投手データの年度と年俸データの年度が合っているか確認してください。"
                        "投手機能は制限されますが、野手機能は正常に使用できます。"
                    )
            else:
                pitcher_merged = pitcher_stats_all = None
                p_model = p_name = p_scaler = p_fcols = p_results = p_ml_df = None

            st.session_state.update({
                'model_trained': True,
                'b_model': b_model, 'b_name': b_name, 'b_scaler': b_scaler,
                'b_fcols': b_fcols, 'b_results': b_results, 'b_ml_df': b_ml_df,
                'batter_stats_all': batter_stats_all,
                'p_model': p_model, 'p_name': p_name, 'p_scaler': p_scaler,
                'p_fcols': p_fcols, 'p_results': p_results, 'p_ml_df': p_ml_df,
                'pitcher_stats_all': pitcher_stats_all,
                'salary_long': salary_long,
            })

    # ----------------------------------------------------------
    # サイドバー
    # ----------------------------------------------------------
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">⚾</div>
        <div class="logo-title">NPB Salary<br>Predictor</div>
    </div>
    <div class="sidebar-section-label">メニュー</div>
    """, unsafe_allow_html=True)

    menu = st.sidebar.radio(
        "メニュー",
        ["🏠 ホーム", "🔍 選手予測", "📊 選手比較", "🔬 モデル比較",
         "✏️ カスタム", "📈 性能", "📉 要因分析",
         "🏆 精度ランキング", "💰 年俸別予測", "📜 予測履歴"],
        key="main_menu", label_visibility="collapsed"
    )

    # ----------------------------------------------------------
    # ホーム
    # ----------------------------------------------------------
    if menu == "🏠 ホーム":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("野手採用モデル", st.session_state.b_name)
            st.metric("野手R²", f"{st.session_state.b_results[st.session_state.b_name]['R2']:.4f}")
        with col2:
            if st.session_state.p_name:
                st.metric("投手採用モデル", st.session_state.p_name)
            else:
                st.metric("投手採用モデル", "データなし")
            if st.session_state.p_results:
                st.metric("投手R²", f"{st.session_state.p_results[st.session_state.p_name]['R2']:.4f}")
            else:
                st.metric("投手R²", "N/A")

        st.subheader("📖 使い方")
        st.markdown("""
        1. **左サイドバー**のメニューから機能を選択
        2. **野手 / 投手**を切り替えて予測

        ### 機能一覧
        - 🔍 **選手予測**: 野手・投手の年俸予測とレーダーチャート
        - 📊 **選手比較**: 最大5人の選手を比較（同種別）
        - 🔬 **モデル比較**: 全モデルで同時予測し比較
        - ✏️ **カスタム**: オリジナルデータで予測
        - 📈 **性能**: モデルの詳細情報
        - 📉 **要因分析**: 年俸に影響する要因の分析
        - 🏆 **精度ランキング**: 誤差が少ない選手の分析
        - 💰 **年俸別予測**: 年俸レンジ別特化モデルで予測
        - 📜 **予測履歴**: 過去20件の予測履歴

        ### ⚖️ NPB減額制限ルール
        - **1億円以上**: 最大40%まで減額可能（最低60%保証）
        - **1億円未満**: 最大25%まで減額可能（最低75%保証）

        ### 使用したサイトのリンク
        - **成績**: [NPB公式サイト](https://npb.jp/)
        - **年齢**: [Baseball Freak](https://baseball-freak.com/)
        - **年俸**: [Baseball Money](https://www.baseball-money.net/)
        """)

    # ----------------------------------------------------------
    # 選手予測
    # ----------------------------------------------------------
    elif menu == "🔍 選手予測":
        st.header("🔍 選手検索・予測")

        player_type = st.radio("選手種別", ["野手", "投手"], horizontal=True, key="predict_type")

        if player_type == "野手":
            available = sorted(
                st.session_state.batter_stats_all[
                    st.session_state.batter_stats_all['年度'] == 2024
                ]['選手名'].unique()
            )
        else:
            if st.session_state.pitcher_stats_all is None:
                st.error("❌ 投手データが読み込まれていません")
                st.stop()
            available = sorted(
                st.session_state.pitcher_stats_all[
                    st.session_state.pitcher_stats_all['年度'] == 2024
                ]['選手名'].unique()
            )

        search = st.text_input("🔍 絞り込み検索", placeholder="例: 村上、岡本、山本", key="pred_search")
        filtered = [p for p in available if search in p] if search else available
        if not filtered:
            st.warning("⚠️ 該当なし")
            filtered = available

        selected = st.selectbox("選手を選択", filtered, key="pred_player")
        predict_year = st.slider("予測年度", 2024, 2026, 2025, key="pred_year")

        if st.button("🎯 予測実行", type="primary"):
            if player_type == "投手" and st.session_state.p_model is None:
                st.error("❌ 投手モデルの訓練に失敗しました。投手データと年俸データのマージ結果が少なすぎる可能性があります。")
                st.stop()

            stats_year = predict_year - 1
            if player_type == "野手":
                df_stats   = st.session_state.batter_stats_all
                model      = st.session_state.b_model
                model_name = st.session_state.b_name
                scaler     = st.session_state.b_scaler
                fcols      = st.session_state.b_fcols
            else:
                df_stats   = st.session_state.pitcher_stats_all
                model      = st.session_state.p_model
                model_name = st.session_state.p_name
                scaler     = st.session_state.p_scaler
                fcols      = st.session_state.p_fcols

            row = df_stats[(df_stats['選手名'] == selected) & (df_stats['年度'] == stats_year)]
            if row.empty:
                st.error(f"❌ {selected}の{stats_year}年データが見つかりません")
            else:
                row = row.iloc[0]
                predicted = predict_salary(row, fcols, model, model_name, scaler)

                if predicted is None:
                    st.error("❌ 予測に失敗しました。モデルが正常に訓練されていません。")
                    st.stop()

                prev_data = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == selected) &
                    (st.session_state.salary_long['年度'] == stats_year)
                ]
                prev_salary = prev_data['年俸_円'].values[0] if not prev_data.empty else None

                actual_data = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == selected) &
                    (st.session_state.salary_long['年度'] == predict_year)
                ]
                actual_salary = actual_data['年俸_円'].values[0] if not actual_data.empty else None

                is_limited = False
                display_salary = predicted
                if prev_salary:
                    is_limited, min_sal, red_rate = check_salary_reduction_limit(predicted, prev_salary)
                    if is_limited:
                        display_salary = min_sal
                        st.warning(f"""
                        ⚖️ **減額制限に引っかかります**
                        - 前年年俸: {prev_salary/10000:.0f}万円
                        - 予測年俸: {predicted/10000:.0f}万円
                        - 減額制限: {red_rate*100:.0f}%まで（最低{(1-red_rate)*100:.0f}%保証）
                        - **制限後の最低年俸: {min_sal/10000:.0f}万円**
                        """)

                if player_type == "野手":
                    stats_dict = {
                        '試合': int(row['試合']), '安打': int(row['安打']),
                        '本塁打': int(row['本塁打']), '打点': int(row['打点']),
                        '打率': float(row['打率']), '出塁率': float(row['出塁率']),
                        '長打率': float(row['長打率']), 'タイトル数': int(row['タイトル数']),
                        '年齢': int(row['年齢']) if '年齢' in row.index else 28
                    }
                else:
                    stats_dict = {
                        '登板': int(row['登板']), '勝利': int(row['勝利']),
                        '敗北': int(row['敗北']), '防御率': float(row['防御率']),
                        '三振': int(row['三振']), '投球回': float(row['投球回_実数']),
                        'タイトル数': int(row['タイトル数']),
                        '年齢': int(row['年齢']) if '年齢' in row.index else 28
                    }

                add_to_history(selected, predict_year, predicted, actual_salary, prev_salary,
                               stats_dict, model_name, player_type, is_limited,
                               display_salary if is_limited else None)

                st.success("✅ 予測完了！")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("前年年俸", f"{prev_salary/10000:.0f}万円" if prev_salary else "データなし")
                with col2:
                    st.metric("予測年俸", f"{predicted/10000:.0f}万円")
                with col3:
                    st.metric("実際の年俸", f"{actual_salary/10000:.0f}万円" if actual_salary else "データなし")
                with col4:
                    if actual_salary:
                        err = abs(display_salary - actual_salary) / actual_salary * 100
                        st.metric("予測誤差", f"{err:.1f}%")

                st.markdown("---")
                st.subheader(f"{stats_year}年の成績")

                if player_type == "野手":
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("試合", int(row['試合']))
                        st.metric("打率", f"{row['打率']:.3f}")
                    with c2:
                        st.metric("安打", int(row['安打']))
                        st.metric("出塁率", f"{row['出塁率']:.3f}")
                    with c3:
                        st.metric("本塁打", int(row['本塁打']))
                        st.metric("長打率", f"{row['長打率']:.3f}")
                    with c4:
                        st.metric("打点", int(row['打点']))
                        st.metric("タイトル数", int(row['タイトル数']))
                    with c5:
                        st.metric("年齢", int(row['年齢']) if '年齢' in row.index else "N/A")
                else:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("登板", int(row['登板']))
                        st.metric("防御率", f"{row['防御率']:.2f}")
                    with c2:
                        st.metric("勝利", int(row['勝利']))
                        st.metric("投球回", f"{row['投球回_実数']:.1f}")
                    with c3:
                        st.metric("敗北", int(row['敗北']))
                        st.metric("奪三振", int(row['三振']))
                    with c4:
                        st.metric("セーブ", int(row['セーブ']))
                        st.metric("タイトル数", int(row['タイトル数']))
                    with c5:
                        st.metric("年齢", int(row['年齢']) if '年齢' in row.index else "N/A")

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    hist = st.session_state.salary_long[
                        st.session_state.salary_long['選手名'] == selected
                    ].sort_values('年度')
                    if not hist.empty:
                        years    = hist['年度'].astype(int).values
                        salaries = hist['年俸_円'].values / 10000
                        ax1.plot(years, salaries, 'o-', linewidth=2, markersize=8, label='実際の年俸')
                        ax1.plot(int(predict_year), predicted/10000, 'r*', markersize=20, label='予測年俸（制限前）')
                        if prev_salary and is_limited:
                            ax1.plot(int(predict_year), display_salary/10000, 'orange', marker='D', markersize=12, label='制限後年俸')
                        if actual_salary:
                            ax1.plot(int(predict_year), actual_salary/10000, 'go', markersize=12, label=f'実際({predict_year})')
                        ax1.set_xticks([2023, 2024, 2025, 2026])
                        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
                        ax1.set_xlabel('年度', fontweight='bold')
                        ax1.set_ylabel('年俸（万円）', fontweight='bold')
                        ax1.set_title(f'{selected} - 年俸推移', fontweight='bold')
                        ax1.grid(alpha=0.3); ax1.legend()
                    st.pyplot(fig1); plt.close(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(8, 5), subplot_kw=dict(projection='polar'))
                    if player_type == "野手":
                        radar = {
                            '打率':  row['打率'] / 0.4,
                            '出塁率': row['出塁率'] / 0.5,
                            '長打率': row['長打率'] / 0.7,
                            '本塁打': min(row['本塁打'] / 40, 1.0),
                            '打点':  min(row['打点']  / 100, 1.0),
                            '盗塁':  min(row['盗塁']  / 40,  1.0),
                        }
                    else:
                        ip = row['投球回_実数'] if row['投球回_実数'] > 0 else 1
                        radar = {
                            '防御率(逆)': max(0, 1 - row['防御率'] / 6.0),
                            '勝利':     min(row['勝利'] / 20, 1.0),
                            'セーブ':   min(row['セーブ'] / 40, 1.0),
                            '奪三振':   min(row['三振'] / 200, 1.0),
                            '投球回':   min(ip / 200, 1.0),
                            'タイトル': min(row['タイトル数'] / 3, 1.0),
                        }
                    cats = list(radar.keys()); vals = list(radar.values()) + [list(radar.values())[0]]
                    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]
                    ax2.plot(angles, vals, 'o-', linewidth=2, color='#2E86AB')
                    ax2.fill(angles, vals, alpha=0.25, color='#2E86AB')
                    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(cats)
                    ax2.set_ylim(0, 1)
                    ax2.set_title(f'{selected} - 成績レーダー\n({stats_year}年)', fontweight='bold', pad=20)
                    ax2.grid(True)
                    st.pyplot(fig2); plt.close(fig2)

    # ----------------------------------------------------------
    # 選手比較
    # ----------------------------------------------------------
    elif menu == "📊 選手比較":
        st.header("📊 複数選手比較")

        player_type = st.radio("選手種別", ["野手", "投手"], horizontal=True, key="compare_type")
        stats_src = st.session_state.batter_stats_all if player_type == "野手" else st.session_state.pitcher_stats_all

        if stats_src is None:
            st.error("❌ 投手データが読み込まれていません")
        else:
            available = sorted(stats_src[stats_src['年度'] == 2024]['選手名'].unique())
            sel_players = st.multiselect("比較する選手を2〜5人選択", options=available, max_selections=5, key="compare_sel")

            if len(sel_players) >= 2 and st.button("📊 比較実行", type="primary"):
                model      = st.session_state.b_model  if player_type == "野手" else st.session_state.p_model
                model_name = st.session_state.b_name   if player_type == "野手" else st.session_state.p_name
                scaler     = st.session_state.b_scaler if player_type == "野手" else st.session_state.p_scaler
                fcols      = st.session_state.b_fcols  if player_type == "野手" else st.session_state.p_fcols

                rows_list = []
                for player in sel_players:
                    row = stats_src[(stats_src['選手名'] == player) & (stats_src['年度'] == 2024)]
                    if row.empty: continue
                    row = row.iloc[0]
                    pred = predict_salary(row, fcols, model, model_name, scaler)
                    if pred is None: continue

                    prev_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == player) &
                        (st.session_state.salary_long['年度'] == 2024)
                    ]
                    prev = prev_data['年俸_円'].values[0] if not prev_data.empty else None

                    is_lim = False; disp = pred
                    if prev:
                        is_lim, min_sal, _ = check_salary_reduction_limit(pred, prev)
                        if is_lim: disp = min_sal

                    entry = {
                        '選手名': player,
                        '前年年俸': prev/10000 if prev else None,
                        '予測年俸（制限前）': pred/10000,
                        '予測年俸（制限後）': disp/10000,
                        '減額制限': 'あり' if is_lim else 'なし',
                    }
                    if player_type == "野手":
                        entry.update({'打率': row['打率'], '本塁打': int(row['本塁打']), '打点': int(row['打点'])})
                    else:
                        entry.update({'防御率': float(row['防御率']), '勝利': int(row['勝利']), '奪三振': int(row['三振'])})
                    rows_list.append(entry)

                df_res = pd.DataFrame(rows_list)
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                limited = df_res[df_res['減額制限'] == 'あり']
                if not limited.empty:
                    st.warning("⚖️ **減額制限に引っかかった選手:**")
                    for _, r in limited.iterrows():
                        st.write(f"- **{r['選手名']}**: 予測{r['予測年俸（制限前）']:.0f}万円 → 制限後{r['予測年俸（制限後）']:.0f}万円")

                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(len(df_res)); w = 0.35
                ax.barh(x - w/2, df_res['予測年俸（制限前）'], w, label='予測（制限前）', alpha=0.7, color='steelblue')
                ax.barh(x + w/2, df_res['予測年俸（制限後）'], w, label='予測（制限後）', alpha=0.7, color='orange')
                ax.set_yticks(x); ax.set_yticklabels(df_res['選手名'])
                ax.set_xlabel('予測年俸（万円）', fontweight='bold')
                ax.set_title('予測年俸比較', fontweight='bold')
                ax.legend(); ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # モデル比較
    # ----------------------------------------------------------
    elif menu == "🔬 モデル比較":
        st.header("🔬 複数モデル比較")

        player_type = st.radio("選手種別", ["野手", "投手"], horizontal=True, key="mc_type")
        stats_src = st.session_state.batter_stats_all if player_type == "野手" else st.session_state.pitcher_stats_all

        if stats_src is None:
            st.error("❌ 投手データが読み込まれていません")
        else:
            available = sorted(stats_src[stats_src['年度'] == 2024]['選手名'].unique())
            search = st.text_input("🔍 絞り込み検索", key="mc_search")
            filtered = [p for p in available if search in p] if search else available
            selected = st.selectbox("選手を選択", filtered, key="mc_player")
            predict_year = st.slider("予測年度", 2024, 2026, 2025, key="mc_year")

            if st.button("🔬 全モデルで予測", type="primary"):
                stats_year = predict_year - 1
                row = stats_src[(stats_src['選手名'] == selected) & (stats_src['年度'] == stats_year)]
                if row.empty:
                    st.error(f"❌ {selected}の{stats_year}年データが見つかりません")
                else:
                    row = row.iloc[0]
                    results_dict = st.session_state.b_results if player_type == "野手" else st.session_state.p_results
                    scaler       = st.session_state.b_scaler  if player_type == "野手" else st.session_state.p_scaler
                    fcols        = st.session_state.b_fcols   if player_type == "野手" else st.session_state.p_fcols

                    if not results_dict:
                        st.error("❌ 投手モデルが訓練されていません")
                        st.stop()

                    prev_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected) &
                        (st.session_state.salary_long['年度'] == stats_year)
                    ]
                    prev = prev_data['年俸_円'].values[0] if not prev_data.empty else None
                    actual_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected) &
                        (st.session_state.salary_long['年度'] == predict_year)
                    ]
                    actual = actual_data['年俸_円'].values[0] if not actual_data.empty else None

                    feat_values = []
                    for col in fcols:
                        if col in row.index:
                            val = row[col]
                            feat_values.append(float(val) if pd.notna(val) else 0.0)
                        else:
                            feat_values.append(28.0 if col == '年齢' else 0.0)
                    feats = np.array([feat_values])

                    preds = []
                    for mname, minfo in results_dict.items():
                        mdl = minfo['model']
                        if mname == '線形回帰':
                            pred_log = mdl.predict(scaler.transform(feats))[0]
                        else:
                            pred_log = mdl.predict(feats)[0]
                        pred = round(np.expm1(pred_log) / 100_000) * 100_000
                        is_lim = False; disp = pred
                        if prev:
                            is_lim, min_sal, _ = check_salary_reduction_limit(pred, prev)
                            if is_lim: disp = min_sal
                        err = abs(disp - actual) / actual * 100 if actual else None
                        preds.append({'モデル': mname, '予測（制限前）万円': pred/10000,
                                      '予測（制限後）万円': disp/10000,
                                      '減額制限': 'あり' if is_lim else 'なし',
                                      'MAE万円': minfo['MAE']/10000, 'R²': minfo['R2'],
                                      '誤差率(%)': err})

                    df_preds = pd.DataFrame(preds)
                    st.success("✅ 全モデルでの予測完了！")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("前年年俸", f"{prev/10000:.0f}万円" if prev else "データなし")
                    c2.metric("実際の年俸", f"{actual/10000:.0f}万円" if actual else "データなし")
                    c3.metric("平均予測", f"{df_preds['予測（制限後）万円'].mean():.0f}万円")
                    st.markdown("---")
                    st.dataframe(df_preds, use_container_width=True, hide_index=True)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(df_preds)); w = 0.35
                    ax.barh(x - w/2, df_preds['予測（制限前）万円'], w, label='制限前', alpha=0.7, color='steelblue')
                    ax.barh(x + w/2, df_preds['予測（制限後）万円'], w, label='制限後', alpha=0.7, color='orange')
                    if actual:
                        ax.axvline(actual/10000, color='green', linestyle='--', linewidth=2, label='実際の年俸')
                    ax.set_yticks(x); ax.set_yticklabels(df_preds['モデル'])
                    ax.set_xlabel('年俸（万円）', fontweight='bold')
                    ax.set_title(f'{selected} - モデル別予測年俸', fontweight='bold')
                    ax.legend(); ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # カスタム入力予測
    # ----------------------------------------------------------
    elif menu == "✏️ カスタム":
        st.header("✏️ カスタム入力予測")

        player_type = st.radio("選手種別", ["野手", "投手"], horizontal=True, key="custom_type")
        player_name = st.text_input("選手名（任意）", placeholder="例: 山田太郎", key="custom_name")

        if player_type == "野手":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**基本成績**")
                games = st.number_input("試合数", 0, 200, 143, key="cg")
                pa    = st.number_input("打席",   0, 800, 600, key="cpa")
                ab    = st.number_input("打数",   0, 700, 520, key="cab")
                runs  = st.number_input("得点",   0, 200, 80,  key="cr")
                hits  = st.number_input("安打",   0, 300, 150, key="ch")
            with c2:
                st.markdown("**長打成績**")
                d2b  = st.number_input("二塁打", 0, 100, 30, key="c2b")
                d3b  = st.number_input("三塁打", 0, 30,  3,  key="c3b")
                hr   = st.number_input("本塁打", 0, 70,  25, key="chr")
                rbi  = st.number_input("打点",   0, 200, 90, key="crbi")
            with c3:
                st.markdown("**走塁・選球眼**")
                sb  = st.number_input("盗塁",   0, 100, 10,  key="csb")
                cs  = st.number_input("盗塁刺", 0, 50,  3,   key="ccs")
                bb  = st.number_input("四球",   0, 200, 60,  key="cbb")
                hbp = st.number_input("死球",   0, 50,  5,   key="chbp")
                so  = st.number_input("三振",   0, 300, 120, key="cso")

            c1, c2, c3 = st.columns(3)
            with c1:
                gdp = st.number_input("併殺打", 0, 50, 10, key="cgdp")
                sh  = st.number_input("犠打",   0, 50, 2,  key="csh")
                sf  = st.number_input("犠飛",   0, 30, 5,  key="csf")
            with c2:
                titles   = st.number_input("タイトル数",       0, 10,     0, key="ctit")
                prev_sal = st.number_input("前年年俸（万円）", 0, 100000, 0, key="cprev")
                age      = st.number_input("年齢",             18, 50,   28, key="cage")
            with c3:
                avg = hits / ab if ab > 0 else 0.0
                obp = (hits + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0.0
                slg = (hits + d2b + d3b*2 + hr*3) / ab if ab > 0 else 0.0
                st.metric("打率",  f"{avg:.3f}")
                st.metric("出塁率", f"{obp:.3f}")
                st.metric("長打率", f"{slg:.3f}")

            if st.button("🎯 年俸予測実行", type="primary", key="custom_pred"):
                tb = hits + d2b + d3b*2 + hr*3
                custom_feats = np.array([[
                    games, pa, ab, runs, hits, d2b, d3b, hr, tb, rbi,
                    sb, cs, bb, hbp, so, gdp, avg, obp, slg, sh, sf, titles, age
                ]])
                _results = st.session_state.b_results
                _scaler  = st.session_state.b_scaler

                preds = []
                for mname, minfo in _results.items():
                    mdl = minfo['model']
                    if mname == '線形回帰':
                        pred_log = mdl.predict(_scaler.transform(custom_feats))[0]
                    else:
                        pred_log = mdl.predict(custom_feats)[0]
                    pred = round(np.expm1(pred_log) / 100_000) * 100_000
                    is_lim = False; disp = pred
                    if prev_sal > 0:
                        is_lim, ms, _ = check_salary_reduction_limit(pred, prev_sal * 10000)
                        if is_lim: disp = ms
                    preds.append({'モデル': mname, '予測年俸万円': pred/10000,
                                  '制限後万円': disp/10000, '減額制限': 'あり' if is_lim else 'なし'})

                df_p = pd.DataFrame(preds)
                st.success("✅ 予測完了！")
                c1, c2, c3 = st.columns(3)
                c1.metric("平均予測", f"{df_p['制限後万円'].mean():.0f}万円")
                c2.metric("最大予測", f"{df_p['制限後万円'].max():.0f}万円")
                c3.metric("最小予測", f"{df_p['制限後万円'].min():.0f}万円")
                st.dataframe(df_p, use_container_width=True, hide_index=True)

        else:  # 投手
            if st.session_state.p_model is None:
                st.error("❌ 投手モデルが訓練されていません")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**基本成績**")
                    games  = st.number_input("登板",   0, 80,  25, key="pg")
                    wins   = st.number_input("勝利",   0, 30,  10, key="pw")
                    losses = st.number_input("敗北",   0, 30,  8,  key="pl")
                    saves  = st.number_input("セーブ", 0, 60,  0,  key="psv")
                    holds  = st.number_input("H",      0, 60,  0,  key="ph")
                    hp     = st.number_input("HP",     0, 60,  0,  key="php")
                with c2:
                    st.markdown("**投球内容**")
                    cg    = st.number_input("完投",   0, 30, 5,     key="pcg")
                    sho   = st.number_input("完封",   0, 10, 2,     key="psho")
                    nhit  = st.number_input("無四球", 0, 10, 0,     key="pnhit")
                    bfp   = st.number_input("打者",   0, 1000, 700, key="pbfp")
                    ip    = st.number_input("投球回（小数）", 0.0, 300.0, 180.0, step=0.1, key="pip")
                with c3:
                    st.markdown("**被打・三振**")
                    hits_a = st.number_input("被安打",   0, 300, 160, key="pha")
                    hr_a   = st.number_input("被本塁打", 0, 40,  15,  key="phra")
                    bb_a   = st.number_input("四球",     0, 150, 50,  key="pbb")
                    hbp_a  = st.number_input("死球",     0, 30,  5,   key="phbp")
                    so_a   = st.number_input("奪三振",   0, 300, 160, key="pso")

                c1, c2, c3 = st.columns(3)
                with c1:
                    wp     = st.number_input("暴投",   0, 20, 2, key="pwp")
                    bk     = st.number_input("ボーク", 0, 10, 0, key="pbk")
                    runs_a = st.number_input("失点",   0, 150, 60, key="pra")
                    er     = st.number_input("自責点", 0, 150, 55, key="per")
                with c2:
                    era   = round(er * 9 / ip, 2) if ip > 0 else 0.0
                    wlpct = round(wins / (wins + losses), 3) if (wins + losses) > 0 else 0.0
                    st.metric("防御率（自動計算）", f"{era:.2f}")
                    st.metric("勝率（自動計算）",   f"{wlpct:.3f}")
                with c3:
                    titles   = st.number_input("タイトル数",       0, 10,     0, key="ptit")
                    prev_sal = st.number_input("前年年俸（万円）", 0, 100000, 0, key="pprev")
                    age      = st.number_input("年齢",             18, 50,   28, key="page")

                if st.button("🎯 年俸予測実行", type="primary", key="pitcher_custom_pred"):
                    custom_feats = np.array([[
                        games, wins, losses, saves, holds, hp, cg, sho, nhit,
                        wlpct, bfp, ip, hits_a, hr_a, bb_a, hbp_a, so_a,
                        wp, bk, runs_a, er, era, titles, age
                    ]])
                    _results = st.session_state.p_results
                    _scaler  = st.session_state.p_scaler

                    preds = []
                    for mname, minfo in _results.items():
                        mdl = minfo['model']
                        if mname == '線形回帰':
                            pred_log = mdl.predict(_scaler.transform(custom_feats))[0]
                        else:
                            pred_log = mdl.predict(custom_feats)[0]
                        pred = round(np.expm1(pred_log) / 100_000) * 100_000
                        is_lim = False; disp = pred
                        if prev_sal > 0:
                            is_lim, ms, _ = check_salary_reduction_limit(pred, prev_sal * 10000)
                            if is_lim: disp = ms
                        preds.append({'モデル': mname, '予測年俸万円': pred/10000,
                                      '制限後万円': disp/10000, '減額制限': 'あり' if is_lim else 'なし'})

                    df_p = pd.DataFrame(preds)
                    st.success("✅ 投手年俸予測完了！")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("平均予測", f"{df_p['制限後万円'].mean():.0f}万円")
                    c2.metric("最大予測", f"{df_p['制限後万円'].max():.0f}万円")
                    c3.metric("最小予測", f"{df_p['制限後万円'].min():.0f}万円")
                    st.dataframe(df_p, use_container_width=True, hide_index=True)

                    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(projection='polar'))
                    radar = {
                        '防御率(逆)': max(0, 1 - era / 6.0),
                        '勝利':     min(wins / 20, 1.0),
                        'セーブ':   min(saves / 40, 1.0),
                        '奪三振':   min(so_a / 200, 1.0),
                        '投球回':   min(ip / 200, 1.0),
                        'タイトル': min(titles / 3, 1.0),
                    }
                    cats = list(radar.keys()); vals = list(radar.values()) + [list(radar.values())[0]]
                    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]
                    ax.plot(angles, vals, 'o-', linewidth=2, color='#E74C3C')
                    ax.fill(angles, vals, alpha=0.25, color='#E74C3C')
                    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats)
                    ax.set_ylim(0, 1)
                    title_str = player_name if player_name else "カスタム投手"
                    ax.set_title(f'{title_str} - 成績レーダー', fontweight='bold', pad=20)
                    ax.grid(True)
                    st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # モデル性能
    # ----------------------------------------------------------
    elif menu == "📈 性能":
        st.header("📈 モデル性能")

        player_type  = st.radio("種別", ["野手", "投手"], horizontal=True, key="perf_type")
        results_dict = st.session_state.b_results if player_type == "野手" else st.session_state.p_results
        best_name    = st.session_state.b_name    if player_type == "野手" else st.session_state.p_name
        best_model   = st.session_state.b_model   if player_type == "野手" else st.session_state.p_model
        fcols        = st.session_state.b_fcols   if player_type == "野手" else st.session_state.p_fcols

        if not results_dict:
            st.error("❌ 投手モデルが訓練されていません")
        else:
            rows = [{'モデル': n, 'MAE（万円）': f"{v['MAE']/10000:.2f}", 'R²スコア': f"{v['R2']:.4f}"}
                    for n, v in results_dict.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)
            st.success(f"🏆 最良モデル: {best_name}")

            if best_name in ['ランダムフォレスト', '勾配ブースティング'] and hasattr(best_model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("特徴量重要度 Top 10")
                fi = pd.DataFrame({'特徴量': fcols, '重要度': best_model.feature_importances_})
                fi = fi.sort_values('重要度', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(fi)), fi['重要度'], color='#9b59b6', alpha=0.7)
                ax.set_yticks(range(len(fi))); ax.set_yticklabels(fi['特徴量'])
                ax.set_xlabel('重要度', fontweight='bold')
                ax.set_title(f'特徴量重要度 Top 10（{player_type}）', fontweight='bold')
                ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
                st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # 要因分析
    # ----------------------------------------------------------
    elif menu == "📉 要因分析":
        st.header("📉 要因分析")

        player_type = st.radio("種別", ["野手", "投手"], horizontal=True, key="factor_type")
        ml_df = st.session_state.b_ml_df if player_type == "野手" else st.session_state.p_ml_df

        if ml_df is None:
            st.error("❌ データなし")
        else:
            st.subheader("タイトル獲得の影響")
            tg = ml_df.groupby(ml_df['タイトル数'] > 0)['年俸_円'].agg(['count', 'mean', 'median'])
            tg['mean']   = round(tg['mean'] / 10000)
            tg['median'] = tg['median'] / 10000
            tg.index   = ['タイトル無し', 'タイトル有り']
            tg.columns = ['選手数', '平均年俸（万円）', '中央値（万円）']
            st.dataframe(tg, use_container_width=False)

            st.markdown("---")
            st.subheader("主要指標との相関")
            if player_type == "野手":
                corr_cols = ['打率', '本塁打', '打点', '出塁率', '長打率', 'タイトル数', '年齢', '試合', '年俸_円']
            else:
                corr_cols = ['防御率', '勝利', '三振', '投球回_実数', 'タイトル数', '年齢', '登板', '年俸_円']

            avail = [c for c in corr_cols if c in ml_df.columns]
            corr = ml_df[avail].corr()['年俸_円'].sort_values(ascending=False)
            corr_data = [{'指標': i, '相関係数': f"{v:.4f}"} for i, v in corr.items() if i != '年俸_円']
            st.dataframe(pd.DataFrame(corr_data), use_container_width=False, hide_index=True)

            if player_type == "野手":
                pairs = [('打率', '打率と年俸'), ('本塁打', '本塁打と年俸')]
            else:
                pairs = [('防御率', '防御率と年俸'), ('三振', '奪三振と年俸')]

            c1, c2 = st.columns(2)
            for col_ax, (feat, title) in zip([c1, c2], pairs):
                if feat in ml_df.columns:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(ml_df[feat], ml_df['年俸_円']/10000, alpha=0.5)
                    ax.set_xlabel(feat, fontweight='bold')
                    ax.set_ylabel('年俸（万円）', fontweight='bold')
                    ax.set_title(title, fontweight='bold')
                    ax.grid(alpha=0.3)
                    with col_ax:
                        st.pyplot(fig)
                    plt.close(fig)

            if '年齢' in ml_df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(ml_df['年齢'], ml_df['年俸_円']/10000, alpha=0.5, color='green')
                ax.set_xlabel('年齢', fontweight='bold')
                ax.set_ylabel('年俸（万円）', fontweight='bold')
                ax.set_title('年齢と年俸の関係', fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # 精度ランキング
    # ----------------------------------------------------------
    elif menu == "🏆 精度ランキング":
        st.header("🏆 予測精度ランキング")

        player_type = st.radio("種別", ["野手", "投手"], horizontal=True, key="rank_type")
        stats_src  = st.session_state.batter_stats_all if player_type == "野手" else st.session_state.pitcher_stats_all
        model      = st.session_state.b_model  if player_type == "野手" else st.session_state.p_model
        model_name = st.session_state.b_name   if player_type == "野手" else st.session_state.p_name
        scaler     = st.session_state.b_scaler if player_type == "野手" else st.session_state.p_scaler
        fcols      = st.session_state.b_fcols  if player_type == "野手" else st.session_state.p_fcols

        if stats_src is None or model is None:
            st.error("❌ データなし（投手モデルが訓練されていません）")
        else:
            rank_year  = st.selectbox("対象年度", [2024, 2025], index=1, key="rank_year")
            sort_col   = st.selectbox("ソート", ["誤差率", "誤差額", "予測年俸(万円)"], key="rank_sort")
            sort_order = st.radio("並び順", ["昇順（小→大）", "降順（大→小）"], key="rank_order")
            top_n      = st.slider("表示件数", 10, 100, 30, 10, key="rank_n")

            if st.button("📊 ランキング作成", type="primary"):
                stats_year = rank_year - 1
                actual_players = st.session_state.salary_long[
                    st.session_state.salary_long['年度'] == rank_year
                ]['選手名'].unique()
                stats_players = stats_src[stats_src['年度'] == stats_year]['選手名'].unique()
                targets = list(set(actual_players) & set(stats_players))

                with st.spinner('🔄 全選手の予測を計算中...'):
                    ranking = []
                    pb = st.progress(0)
                    for i, player in enumerate(targets):
                        pb.progress((i+1)/len(targets))
                        row = stats_src[(stats_src['選手名'] == player) & (stats_src['年度'] == stats_year)]
                        if row.empty: continue
                        row = row.iloc[0]
                        pred = predict_salary(row, fcols, model, model_name, scaler)
                        if pred is None: continue

                        prev_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == player) &
                            (st.session_state.salary_long['年度'] == stats_year)
                        ]
                        prev = prev_data['年俸_円'].values[0] if not prev_data.empty else None
                        disp = pred; is_lim = False
                        if prev:
                            is_lim, ms, _ = check_salary_reduction_limit(pred, prev)
                            if is_lim: disp = ms

                        actual_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == player) &
                            (st.session_state.salary_long['年度'] == rank_year)
                        ]
                        actual = actual_data['年俸_円'].values[0] if not actual_data.empty else None
                        if actual is None: continue

                        err_amt  = abs(disp - actual)
                        err_rate = err_amt / actual * 100
                        entry = {
                            '順位': 0, '選手名': player,
                            '実際の年俸(万円)': actual/10000,
                            '予測年俸(万円)': disp/10000,
                            '誤差額': err_amt/10000,
                            '誤差率': err_rate,
                            '減額制限': 'あり' if is_lim else 'なし',
                        }
                        if player_type == "野手":
                            entry.update({'打率': row['打率'], '本塁打': int(row['本塁打']), '打点': int(row['打点'])})
                        else:
                            entry.update({'防御率': float(row['防御率']), '勝利': int(row['勝利']), '奪三振': int(row['三振'])})
                        ranking.append(entry)
                    pb.empty()

                if ranking:
                    df_rank = pd.DataFrame(ranking)
                    asc = sort_order == "昇順（小→大）"
                    df_rank = df_rank.sort_values(sort_col, ascending=asc)
                    df_rank['順位'] = range(1, len(df_rank)+1)
                    df_top = df_rank.head(top_n)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("平均誤差率",  f"{df_rank['誤差率'].mean():.1f}%")
                    c2.metric("中央値誤差率", f"{df_rank['誤差率'].median():.1f}%")
                    c3.metric("最小誤差率",  f"{df_rank['誤差率'].min():.1f}%")
                    c4.metric("最大誤差率",  f"{df_rank['誤差率'].max():.1f}%")
                    st.markdown("---")
                    st.dataframe(df_top, use_container_width=True, hide_index=True, height=600)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df_rank['誤差率'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(df_rank['誤差率'].mean(),   color='red',   linestyle='--', linewidth=2, label=f'平均: {df_rank["誤差率"].mean():.1f}%')
                    ax.axvline(df_rank['誤差率'].median(), color='green', linestyle='--', linewidth=2, label=f'中央値: {df_rank["誤差率"].median():.1f}%')
                    ax.set_xlabel('誤差率 (%)', fontweight='bold')
                    ax.set_ylabel('選手数', fontweight='bold')
                    ax.set_title(f'予測誤差率の分布（{player_type}）', fontweight='bold')
                    ax.legend(); ax.grid(alpha=0.3)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.warning("⚠️ 対象選手が見つかりませんでした")

    # ----------------------------------------------------------
    # 年俸別予測
    # ----------------------------------------------------------
    elif menu == "💰 年俸別予測":
        st.header("💰 年俸レンジ別特化モデルで予測")

        player_type = st.radio("選手種別", ["野手", "投手"], horizontal=True, key="ranged_type")
        ml_df     = st.session_state.b_ml_df  if player_type == "野手" else st.session_state.p_ml_df
        fcols     = st.session_state.b_fcols  if player_type == "野手" else st.session_state.p_fcols
        stats_src = st.session_state.batter_stats_all if player_type == "野手" else st.session_state.pitcher_stats_all

        if ml_df is None:
            st.error("❌ 投手データなし")
        else:
            st.markdown("---")
            st.subheader("⚙️ ステップ1: 年俸レンジを設定")
            preset = st.radio(
                "設定方法",
                ["おすすめ（5分割）", "簡単（3分割）", "詳細（7分割）"],
                horizontal=True, key="rv_preset"
            )
            if preset == "簡単（3分割）":
                range_values = [0, 30_000_000, 80_000_000, 10_000_000_000]
            elif preset == "おすすめ（5分割）":
                range_values = [0, 20_000_000, 40_000_000, 70_000_000, 100_000_000, 10_000_000_000]
            else:
                range_values = [0, 15_000_000, 30_000_000, 50_000_000, 70_000_000, 100_000_000, 200_000_000, 10_000_000_000]

            st.markdown("---")
            st.subheader("⚙️ ステップ2: モデルを訓練")
            ranged_key = f"ranged_models_{player_type}"

            if st.button("🔧 モデルを訓練する", type="primary", key="rv_train"):
                with st.spinner("各レンジ用モデルを訓練中..."):
                    ranged = {}
                    for i in range(len(range_values) - 1):
                        mn = range_values[i]; mx = range_values[i+1]
                        rname = f"{mn/10000:.0f}万～{mx/10000:.0f}万円"
                        sub = ml_df[(ml_df['年俸_円'] >= mn) & (ml_df['年俸_円'] < mx)].copy()
                        if len(sub) < 10:
                            st.warning(f"⚠️ {rname}: データ不足({len(sub)}件)でスキップ")
                            continue
                        X = sub[fcols]; y_log = np.log1p(sub['年俸_円'])
                        X_tr, X_te, yt_tr, yt_te = train_test_split(X, y_log, test_size=0.2, random_state=42)
                        mdl = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                        mdl.fit(X_tr, yt_tr)
                        yp = np.expm1(mdl.predict(X_te)); yt_o = np.expm1(yt_te)
                        ranged[rname] = {
                            'model': mdl, 'MAE': mean_absolute_error(yt_o, yp),
                            'R2': r2_score(yt_o, yp),
                            'min_salary': mn, 'max_salary': mx,
                            'n_samples': len(sub), 'feature_cols': fcols
                        }
                    st.session_state[ranged_key] = ranged
                    st.success("✅ モデル訓練完了！")

            if ranged_key in st.session_state:
                ranged = st.session_state[ranged_key]
                perf = [{'年俸レンジ': k, '選手数': v['n_samples'],
                         '平均誤差': f"{v['MAE']/10000:.0f}万円", '精度(R²)': f"{v['R2']:.3f}"}
                        for k, v in ranged.items()]
                st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("⚙️ ステップ3: 選手を選んで予測")
                available = sorted(stats_src[stats_src['年度'] == 2024]['選手名'].unique())
                search = st.text_input("🔍 選手名で検索", key="rv_search")
                filtered = [p for p in available if search in p] if search else available
                c1, c2 = st.columns([3, 1])
                with c1:
                    sel = st.selectbox("選手を選択", filtered, key="rv_player")
                with c2:
                    pred_yr = st.selectbox("予測年度", [2024, 2025, 2026], index=1, key="rv_year")

                if st.button("🎯 予測する！", type="primary"):
                    sy = pred_yr - 1
                    row = stats_src[(stats_src['選手名'] == sel) & (stats_src['年度'] == sy)]
                    if row.empty:
                        st.error(f"❌ {sel}の{sy}年データが見つかりません")
                    else:
                        row = row.iloc[0]
                        prev_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == sel) &
                            (st.session_state.salary_long['年度'] == sy)
                        ]
                        prev = prev_data['年俸_円'].values[0] if not prev_data.empty else None
                        actual_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == sel) &
                            (st.session_state.salary_long['年度'] == pred_yr)
                        ]
                        actual = actual_data['年俸_円'].values[0] if not actual_data.empty else None

                        all_preds = []
                        model      = st.session_state.b_model  if player_type == "野手" else st.session_state.p_model
                        model_name = st.session_state.b_name   if player_type == "野手" else st.session_state.p_name
                        scaler     = st.session_state.b_scaler if player_type == "野手" else st.session_state.p_scaler
                        uni_pred = predict_salary(row, fcols, model, model_name, scaler)
                        if uni_pred is not None:
                            uni_disp = uni_pred; uni_lim = False
                            if prev:
                                uni_lim, ms, _ = check_salary_reduction_limit(uni_pred, prev)
                                if uni_lim: uni_disp = ms
                            all_preds.append({
                                'モデル': '📊 通常モデル',
                                '予測年俸(万円)': uni_disp/10000,
                                '誤差(万円)': abs(uni_disp - actual)/10000 if actual else None
                            })

                        for rname, rinfo in ranged.items():
                            r_pred = predict_salary(row, rinfo['feature_cols'],
                                                    rinfo['model'], 'ランダムフォレスト', None)
                            if r_pred is None: continue
                            r_disp = r_pred; r_lim = False
                            if prev:
                                r_lim, ms, _ = check_salary_reduction_limit(r_pred, prev)
                                if r_lim: r_disp = ms
                            all_preds.append({
                                'モデル': f'🎯 {rname}用',
                                '予測年俸(万円)': r_disp/10000,
                                '誤差(万円)': abs(r_disp - actual)/10000 if actual else None
                            })

                        if all_preds:
                            df_all = pd.DataFrame(all_preds)
                            if actual:
                                df_all = df_all.sort_values('誤差(万円)')
                            st.success("✅ 予測完了！")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("前年年俸", f"{prev/10000:.0f}万円" if prev else "データなし")
                            c2.metric("実際の年俸", f"{actual/10000:.0f}万円" if actual else "データなし")
                            c3.metric("最良予測", f"{df_all.iloc[0]['予測年俸(万円)']:.0f}万円")
                            st.dataframe(df_all, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------
    # 予測履歴
    # ----------------------------------------------------------
    elif menu == "📜 予測履歴":
        st.header("📜 予測履歴")

        if not st.session_state.prediction_history:
            st.info("📭 予測履歴がありません。選手予測を実行すると履歴が保存されます。")
        else:
            st.markdown(f"**保存件数**: {len(st.session_state.prediction_history)} / 20件")
            c1, c2 = st.columns([3, 1])
            with c2:
                if st.button("🗑️ 履歴をクリア", type="secondary"):
                    st.session_state.prediction_history = []
                    st.rerun()

            for idx, item in enumerate(st.session_state.prediction_history):
                badge = "🏃 野手" if item.get('種別') == '野手' else "⚾ 投手"
                with st.expander(
                    f"#{idx+1} {badge} {item['選手名']} - {item['予測年度']}年予測 ({item['予測日時']})",
                    expanded=(idx == 0)
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("前年年俸",  f"{item['前年年俸']/10000:.1f}万円" if item['前年年俸'] else "データなし")
                    c2.metric("予測年俸",  f"{item['予測年俸']/10000:.1f}万円")
                    c3.metric("制限後年俸", f"{item['制限後年俸']/10000:.1f}万円" if item['減額制限'] else "制限なし")
                    if item['実際の年俸']:
                        c4.metric("実際の年俸", f"{item['実際の年俸']/10000:.1f}万円")
                        err = abs(item['制限後年俸'] - item['実際の年俸']) / item['実際の年俸'] * 100
                        st.metric("誤差率", f"{err:.1f}%")
                    else:
                        c4.metric("実際の年俸", "データなし")

                    st.markdown(f"**使用モデル**: {item['モデル']} | **種別**: {item.get('種別', '野手')}")
                    stats = item['成績']
                    if item.get('種別') == '投手':
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("登板",   stats.get('登板', '-'))
                        c2.metric("勝利",   stats.get('勝利', '-'))
                        c3.metric("防御率", stats.get('防御率', '-'))
                        c4.metric("奪三振", stats.get('三振', '-'))
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("試合",   stats.get('試合', '-'))
                        c2.metric("本塁打", stats.get('本塁打', '-'))
                        c3.metric("打点",   stats.get('打点', '-'))
                        c4.metric("打率",   f"{stats.get('打率', 0):.3f}")

# ============================================================
# データなし時の案内
# ============================================================
else:
    st.info("📁 CSVファイルが見つかりませんでした")
    st.markdown("""
    ### データ配置方法

    **方法1: dataフォルダに配置**
    ```
    data/
    ├── salary_2023&2024&2025.csv
    ├── stats_2023.csv
    ├── stats_2024.csv
    ├── stats_2025.csv
    ├── titles_2023&2024&2025.csv
    └── npb_pitcher_stats.csv
    ```

    **方法2: 左サイドバーから手動アップロード（8ファイル）**
    - besmoney_salary_2023/2024/2025.csv（年俸）
    - stats_2023/2024/2025.csv（野手成績）
    - titles_2023&2024&2025.csv（タイトル）
    - npb_pitcher_stats.csv（投手成績）
    """)

st.markdown("---")
st.markdown("*NPB選手年俸予測システム - made by Sato&Kurokawa - Powered by Streamlit*")

st.cache_data.clear()
st.cache_resource.clear()
