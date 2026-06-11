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
    layout="centered",
)

st.markdown("""
<style>

/* ====== サイドバー固定 ====== */
[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0;
    left: 0;
    width: 280px !important;
    height: 100vh !important;
    background-color: #ffe4e9 !important;
    border-right: 1px solid #e0e0e0;
    padding: 0 !important;
    margin: 0 !important;
    z-index: 1000000;
    overflow: hidden;
    border-radius: 0px 30px 30px 0;
}

/* サイドバーのユーザーコンテンツエリア */
[data-testid="stSidebarUserContent"] {
    padding-top: 3rem !important;
    margin-top: 0 !important;
}

/* スクロールコンテンツ */
[data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    height: 100vh !important;
    padding: 0 0.5rem 1rem 0.5rem !important;
    margin: 0 !important;
}

/* サイドバー内の最初の要素の上余白を削除 */
[data-testid="stSidebarContent"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* すべてのVerticalBlock */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* すべてのelement-container */
[data-testid="stSidebar"] .element-container {
    margin-top: 0 !important;
}

[data-testid="stSidebar"] .element-container:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* サイドバー内のカーソルを標準化 */
[data-testid="stSidebar"] * {
    cursor: default !important;
}

/* ボタンやリンクなど、クリック可能な要素のみポインターカーソル */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] input[type="radio"],
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    cursor: pointer !important;
}

/* サイドバーのラジオボタンラベルの文字サイズを小さく */
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    font-size: 13px !important;
    line-height: 1.2 !important;
}

/* サイドバーのラジオボタン全体を小さく */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    padding: 0.2rem 0 !important;
}

/* ====== メインエリア ====== */
.main {
    margin-left: 280px !important;
    transition: margin-left 0.3s ease !important;
}

/* サイドバーが閉じている時はメインエリアを全幅に */
[data-testid="stSidebar"][aria-expanded="false"] ~ .main,
[data-testid="collapsedControl"] ~ .main {
    margin-left: 0 !important;
}

/* メインの最大幅を固定（揺れ防止） */
.block-container {
    max-width: 1400px !important;
    padding-top: 2rem !important;
}

/* サイドバーが閉じている時はブロックコンテナを広く */
[data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container,
body:not(:has([data-testid="stSidebar"])) .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ====== 表（テーブル）の揺れ対策 ====== */
.stDataFrame, .stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}

table {
    table-layout: auto !important;
    width: 100% !important;
    min-width: 600px !important;
}

thead tr th {
    background-color: #f8f8f8 !important;
    white-space: nowrap !important;
    padding: 8px 12px !important;
}

tbody td {
    white-space: nowrap !important;
    padding: 6px 10px !important;
}

/* ====== 見出しの縦線（カーソル）を非表示 ====== */
h1::before, h2::before, h3::before, h4::before, h5::before, h6::before {
    content: none !important;
    display: none !important;
}

/* Markdownの見出しも対象 */
.element-container h1::before,
.element-container h2::before,
.element-container h3::before,
.element-container h4::before {
    display: none !important;
}

/* ====== 見出しのアンカーリンクを完全に非表示 ====== */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
    pointer-events: none !important;
}

/* Streamlitの見出しアンカー */
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* 見出しのホバー時のリンク表示も消す */
h1:hover a, h2:hover a, h3:hover a, h4:hover a, h5:hover a, h6:hover a {
    display: none !important;
}

/* ====== スマホ対応 ====== */
@media (max-width: 900px) {
    [data-testid="stSidebar"] {
        position: relative !important;
        width: 100% !important;
        height: auto !important;
        border-right: none !important;
    }
    .main {
        margin-left: 0 !important;
    }
    .block-container {
        max-width: 100% !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        font-size: 12px !important;
    }
    
    /* テーブルのスクロール対応 */
    .stDataFrame {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    
    /* テーブルのフォントサイズ縮小 */
    .stDataFrame table {
        font-size: 11px !important;
    }
    
    /* メトリクスの縮小 */
    [data-testid="stMetric"] {
        font-size: 0.85rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    
    /* 見出しのサイズ調整 */
    h1 {
        font-size: 1.5rem !important;
    }
    
    h2 {
        font-size: 1.3rem !important;
    }
    
    h3 {
        font-size: 1.1rem !important;
    }
    
    /* ボタンの調整 */
    .stButton button {
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* セレクトボックスの調整 */
    .stSelectbox label {
        font-size: 0.9rem !important;
    }
    
    /* 数値入力の調整 */
    .stNumberInput label {
        font-size: 0.85rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

# CSSでアニメーションを無効化
st.markdown("""
<style>
    /* データフレームの震えを防止 */
    [data-testid="stDataFrame"] {
        animation: none !important;
        transition: none !important;
    }
    
    /* テーブル全体の震えを防止 */
    .stDataFrame {
        animation: none !important;
        transition: none !important;
    }
    
    /* 全体的なアニメーション抑制 */
    * {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
    }
    
    /* モバイル用：テーブルをスクロール可能に */
    @media (max-width: 768px) {
        .dataframe-container {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            width: 100% !important;
        }
        
        /* 小さい画面用の文字サイズ */
        .stDataFrame table {
            font-size: 10px !important;
        }
        
        .stDataFrame th,
        .stDataFrame td {
            padding: 4px 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ====== ダークモード全体 ====== */
@media (prefers-color-scheme: dark) {

    /* メイン背景 */
    .main, .block-container {
        background-color: #1e1e1e !important;
        color: #f2f2f2 !important;
    }

    /* サイドバー */
    [data-testid="stSidebar"] {
        background-color: #2a2a2a !important;
        border-right: 1px solid #444 !important;
    }

    /* テキスト色 */
    [data-testid="stSidebar"] *, .main * {
        color: #f2f2f2 !important;
    }

    /* テーブルヘッダー */
    thead tr th {
        background-color: #333 !important;
        color: #fff !important;
    }

    /* テーブル本体 */
    tbody tr {
        background-color: #2b2b2b !important;
        color: #fff !important;
    }

    /* ボタン */
    button[kind="primary"], .stButton button {
        background-color: #444 !important;
        color: #fff !important;
        border-radius: 8px;
        border: 1px solid #666 !important;
    }
    button[kind="primary"]:hover, .stButton button:hover {
        background-color: #555 !important;
    }

    /* 入力フォーム */
    input, textarea, select, .stTextInput input {
        background-color: #2b2b2b !important;
        color: #fff !important;
        border: 1px solid #666 !important;
    }

    /* プロット周り（Matplotlib） */
    .stPlotlyChart, .stPyplot {
        background-color: #1e1e1e !important;
    }
}

</style>
""", unsafe_allow_html=True)

# 日本語フォント設定
try:
    import japanize_matplotlib
    plt.rcParams["font.family"] = "IPAexGothic"
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

# 減額制限計算関数
def calculate_salary_limit(previous_salary):
    """
    NPBの減額制限を計算する
    1億円以上: 40%まで減額可能（最低60%）
    1億円未満: 25%まで減額可能（最低75%）
    """
    if previous_salary >= 100_000_000:  # 1億円以上
        reduction_rate = 0.40
        min_salary = previous_salary * 0.60
    else:  # 1億円未満
        reduction_rate = 0.25
        min_salary = previous_salary * 0.75
    
    return min_salary, reduction_rate

def check_salary_reduction_limit(predicted_salary, previous_salary):
    """
    予測年俸が減額制限に引っかかるかチェック
    """
    min_salary, reduction_rate = calculate_salary_limit(previous_salary)
    
    if predicted_salary < min_salary:
        return True, min_salary, reduction_rate
    else:
        return False, min_salary, reduction_rate

# タイトル
st.title("⚾ NPB選手年俸予測システム")
st.markdown("---")

# セッション状態の初期化
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# 予測履歴の初期化
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def add_to_history(player_name, predict_year, predicted_salary, actual_salary, previous_salary, 
                   stats_dict, model_name, is_limited=False, limited_salary=None):
    """予測履歴に追加（最大20件）"""
    
    # 日本時間を取得（UTC+9時間）
    jst_time = datetime.utcnow() + timedelta(hours=9)
    
    history_item = {
        '予測日時': jst_time.strftime('%Y-%m-%d %H:%M:%S'),
        '選手名': player_name,
        '予測年度': predict_year,
        '予測年俸': predicted_salary,
        '制限後年俸': limited_salary if is_limited else predicted_salary,
        '実際の年俸': actual_salary,
        '前年年俸': previous_salary,
        '減額制限': is_limited,
        'モデル': model_name,
        '成績': stats_dict
    }
    
    # 先頭に追加
    st.session_state.prediction_history.insert(0, history_item)
    
    # 20件を超えたら古いものを削除
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[:20]
# データ読み込み処理
@st.cache_data
def load_data():
    """データを読み込んでキャッシュする"""
    try:
        salary_df = pd.read_csv('data/salary_2023&2024&2025.csv')
        stats_2023 = pd.read_csv('data/stats_2023.csv')
        stats_2024 = pd.read_csv('data/stats_2024.csv')
        stats_2025 = pd.read_csv('data/stats_2025.csv')
        titles_df = pd.read_csv('data/titles_2023&2024&2025.csv')
        return salary_df, stats_2023, stats_2024, stats_2025, titles_df, True
    except FileNotFoundError:
        return None, None, None, None, None, False

salary_df, stats_2023, stats_2024, stats_2025, titles_df, data_loaded = load_data()

# ファイルアップロード処理
if not data_loaded:
    st.sidebar.markdown("**5つのCSVファイルを一度に選択してアップロード：**")
    uploaded_files = st.sidebar.file_uploader(
        "CSVファイルを選択（5つ全て選択してください）",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 5:
        file_dict = {}
        for file in uploaded_files:
            if 'salary' in file.name or '年俸' in file.name:
                file_dict['salary'] = file
            elif 'titles' in file.name or 'タイトル' in file.name:
                file_dict['titles'] = file
            elif '2023' in file.name:
                file_dict['stats_2023'] = file
            elif '2024' in file.name:
                file_dict['stats_2024'] = file
            elif '2025' in file.name:
                file_dict['stats_2025'] = file
        
        if len(file_dict) == 5:
            salary_df = pd.read_csv(file_dict['salary'])
            stats_2023 = pd.read_csv(file_dict['stats_2023'])
            stats_2024 = pd.read_csv(file_dict['stats_2024'])
            stats_2025 = pd.read_csv(file_dict['stats_2025'])
            titles_df = pd.read_csv(file_dict['titles'])
            data_loaded = True
        else:
            st.sidebar.error("❌ ファイル名が正しくありません")
    elif uploaded_files:
        st.sidebar.warning(f"⚠️ {len(uploaded_files)}個のファイルが選択されています。5つ必要です。")

# データ前処理関数
@st.cache_data
def prepare_data(_salary_df, _stats_2023, _stats_2024, _stats_2025, _titles_df):
    """データの前処理を行う"""
    titles_df_clean = _titles_df.dropna(subset=['選手名'])
    title_summary = titles_df_clean.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')
    
    stats_2023_copy = _stats_2023.copy()
    stats_2024_copy = _stats_2024.copy()
    stats_2025_copy = _stats_2025.copy()
    
    stats_2023_copy['年度'] = 2023
    stats_2024_copy['年度'] = 2024
    stats_2025_copy['年度'] = 2025
    
    stats_all = pd.concat([stats_2023_copy, stats_2024_copy, stats_2025_copy], ignore_index=True)
    
    df_2023 = _salary_df[['選手名_2023', '年俸_円_2023']].copy()
    df_2023['年度'] = 2023
    df_2023.rename(columns={'選手名_2023': '選手名', '年俸_円_2023': '年俸_円'}, inplace=True)
    
    df_2024 = _salary_df[['選手名_2024_2025', '年俸_円_2024']].copy()
    df_2024['年度'] = 2024
    df_2024.rename(columns={'選手名_2024_2025': '選手名', '年俸_円_2024': '年俸_円'}, inplace=True)
    
    df_2025 = _salary_df[['選手名_2024_2025', '年俸_円_2025']].copy()
    df_2025['年度'] = 2025
    df_2025.rename(columns={'選手名_2024_2025': '選手名', '年俸_円_2025': '年俸_円'}, inplace=True)
    
    salary_long = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
    salary_long = salary_long.dropna(subset=['年俸_円'])
    salary_long = salary_long[salary_long['年俸_円'] > 0]
    salary_long = salary_long.sort_values('年俸_円', ascending=False)
    salary_long = salary_long.drop_duplicates(subset=['選手名', '年度'], keep='first')
    
    stats_all['予測年度'] = stats_all['年度'] + 1
    merged_df = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    merged_df['タイトル数'] = merged_df['タイトル数'].fillna(0)
    
    # ★ 年齢データを保存 ★
    if '年齢' in merged_df.columns:
        age_backup = merged_df[['選手名', '年度', '年齢']].copy()
    
    merged_df = pd.merge(
        merged_df,
        salary_long,
        left_on=['選手名', '予測年度'],
        right_on=['選手名', '年度'],
        suffixes=('_成績', '_年俸')
    )
    
    # ★ 年齢列が消えた場合は復元 ★
    if '年齢' not in merged_df.columns and 'age_backup' in locals():
        merged_df = pd.merge(
            merged_df,
            age_backup,
            left_on=['選手名', '年度_成績'],
            right_on=['選手名', '年度'],
            how='left'
        )
        # 重複列を削除
        if '年度_y' in merged_df.columns:
            merged_df = merged_df.drop(columns=['年度_y'])
        if '年度_x' in merged_df.columns:
            merged_df = merged_df.rename(columns={'年度_x': '年度_成績'})
    
    merged_df = merged_df.drop(columns=['年度_年俸', '予測年度'])
    merged_df.rename(columns={'年度_成績': '成績年度'}, inplace=True)
    
    stats_all_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    stats_all_with_titles['タイトル数'] = stats_all_with_titles['タイトル数'].fillna(0)
    
    return merged_df, stats_all_with_titles, salary_long

# モデル訓練関数（対数変換版・年齢追加）
@st.cache_resource
def train_models(_merged_df):
    """モデルを訓練する（対数変換適用・年齢を特徴量に追加）"""
    feature_cols = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', '本塁打', 
                   '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打', 
                   '打率', '出塁率', '長打率', '犠打', '犠飛', 'タイトル数']
    
    # 年齢列が存在する場合は特徴量に追加
    if '年齢' in _merged_df.columns:
        feature_cols.append('年齢')
        ml_df = _merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
    else:
        # 年齢データがない場合は平均年齢（28歳）で補完
        ml_df = _merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
        ml_df['年齢'] = 28  # デフォルト年齢
        feature_cols.append('年齢')
    
    ml_df = ml_df.dropna()
    
    X = ml_df[feature_cols]
    y = ml_df['年俸_円']
    
    y_log = np.log1p(y)
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        '線形回帰': LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    for name, model in models.items():
        if name == '線形回帰':
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
        
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'R2': r2
        }
    
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_model = results[best_model_name]['model']
    
    return best_model, best_model_name, scaler, feature_cols, results, ml_df

# データ読み込みとモデル訓練
if data_loaded:
    if not st.session_state.model_trained:
        with st.spinner('🤖 モデルを訓練中...'):
            merged_df, stats_all_with_titles, salary_long = prepare_data(
                salary_df, stats_2023, stats_2024, stats_2025, titles_df
            )
            
            best_model, best_model_name, scaler, feature_cols, results, ml_df = train_models(merged_df)
            
            st.session_state.model_trained = True
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.stats_all_with_titles = stats_all_with_titles
            st.session_state.salary_long = salary_long
            st.session_state.results = results
            st.session_state.ml_df = ml_df
    
    # メインコンテンツ
    st.sidebar.markdown("### 🎯 機能選択")
    menu = st.sidebar.radio(
        "メニュー",
        ["🏠 ホーム", "🔍 選手予測", "📊 選手比較", "🔬 モデル比較", "✏️ カスタム", "📈 性能", "📉 要因分析", "🏆 精度ランキング", "💰 年俸別予測", "📜 予測履歴"],
        key="main_menu",
        label_visibility="collapsed"
    )
    
    # ホーム
    if menu == "🏠 ホーム":
        col1, col2= st.columns([1,1])
        with col1:
            st.metric("採用モデル", st.session_state.best_model_name)
        with col2:
            st.metric("R²スコア", f"{st.session_state.results[st.session_state.best_model_name]['R2']:.4f}")

        st.subheader("📖 使い方")
        st.markdown("""
        1. **左サイドバー**のメニューから機能を選択
        2. **選手名**を入力して年俸を予測
        
        ### 機能一覧
        - 🔍 **選手予測**: 個別選手の年俸予測とレーダーチャート
        - 📊 **選手比較**: 最大5人の選手を比較
        - 🔬 **モデル比較**: 全モデルで同時予測して比較
        - ✏️ **カスタム**: オリジナル選手データで予測
        - 📈 **性能**: 予測モデルの詳細情報
        - 📉 **要因分析**: 年俸に影響を与える要因の分析
        - 🏆 **精度ランキング**: 誤差が少ない選手の分析
        - 💰 **年俸別予測**: 年俸レンジ別に特化したモデルで予測
        - 📜 **予測履歴**: 過去20件の予測履歴を表示
        
        ### ⚖️ NPB減額制限ルール
        - **1億円以上**: 最大40%まで減額可能（最低60%保証）
        - **1億円未満**: 最大25%まで減額可能（最低75%保証）

        ### 使用したサイトのリンク
        - **成績**: [NPB公式サイト](https://npb.jp/)
        - **年齢**: [Baseball Freak](https://baseball-freak.com/)
        - **年俸**: [Baseball Money](https://www.baseball-money.net/)

        """)
    
    # 選手検索・予測
    elif menu == "🔍 選手予測":
        st.header("🔍 選手検索・予測")
        
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        sorted_players = sorted(available_players)
        
        st.markdown("### 選手を選択")
        
        search_filter = st.text_input(
            "🔍 絞り込み検索（オプション）",
            placeholder="例: 村上、岡本、近藤",
            key="player_search_filter",
            help="選手名の一部を入力すると候補が絞り込まれます"
        )
        
        if search_filter:
            filtered_players = [p for p in sorted_players if search_filter in p]
            if not filtered_players:
                st.warning("⚠️ 該当する選手が見つかりません")
                filtered_players = sorted_players
        else:
            filtered_players = sorted_players
        
        selected_player = st.selectbox(
            f"選手を選択してください",
            options=filtered_players,
            index=0,
            key="player_select_main"
        )
        
        predict_year = st.slider("予測年度", 2024, 2026, 2025, key="predict_year_slider")
        
        if st.button("🎯 予測実行", type="primary", key="predict_button"):
            if not selected_player:
                st.error("❌ 選手を選択してください")
            else:
                stats_year = predict_year - 1
                player_stats = st.session_state.stats_all_with_titles[
                    (st.session_state.stats_all_with_titles['選手名'] == selected_player) &
                    (st.session_state.stats_all_with_titles['年度'] == stats_year)
                ]
                
                if player_stats.empty:
                    st.error(f"❌ {selected_player}の{stats_year}年のデータが見つかりません")
                else:
                    player_stats = player_stats.iloc[0]
                    
                    # 年齢データがない場合は28歳（平均）を使用
                    if '年齢' not in st.session_state.feature_cols:
                        features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                    else:
                        # 年齢データがある場合
                        if '年齢' in player_stats.index:
                            features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        else:
                            # 選手データに年齢がない場合は28歳で補完
                            features_list = player_stats[st.session_state.feature_cols[:-1]].values.tolist()
                            features_list.append(28)  # デフォルト年齢
                            features = np.array([features_list])
                    
                    # 予測（対数変換版）
                    if st.session_state.best_model_name == '線形回帰':
                        features_scaled = st.session_state.scaler.transform(features)
                        predicted_salary_log = st.session_state.best_model.predict(features_scaled)[0]
                    else:
                        predicted_salary_log = st.session_state.best_model.predict(features)[0]
                    
                    predicted_salary = np.expm1(predicted_salary_log)
                    
                    # 十万円単位で四捨五入
                    predicted_salary = round(predicted_salary / 100000) * 100000
                    
                    # 前年の年俸を取得
                    previous_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == stats_year)
                    ]
                    previous_salary = previous_salary_data['年俸_円'].values[0] if not previous_salary_data.empty else None
                    
                    # 実際の年俸を取得
                    actual_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == predict_year)
                    ]
                    actual_salary = actual_salary_data['年俸_円'].values[0] if not actual_salary_data.empty else None
                    
                    st.success("✅ 予測完了！")
                    
                    # 履歴に保存
                    stats_dict = {
                        '試合': int(player_stats['試合']),
                        '安打': int(player_stats['安打']),
                        '本塁打': int(player_stats['本塁打']),
                        '打点': int(player_stats['打点']),
                        '打率': float(player_stats['打率']),
                        '出塁率': float(player_stats['出塁率']),
                        '長打率': float(player_stats['長打率']),
                        'タイトル数': int(player_stats['タイトル数']),
                        '年齢': int(player_stats['年齢']) if '年齢' in player_stats.index else 28
                    }
                    
                    # 減額制限チェック
                    if previous_salary is not None:
                        is_limited, min_salary, reduction_rate = check_salary_reduction_limit(predicted_salary, previous_salary)
                        
                        if is_limited:
                            st.warning(f"""
                            ⚖️ **減額制限に引っかかります**
                            - 前年年俸: {previous_salary/10000:.0f}万円
                            - 予測年俸: {predicted_salary/10000:.0f}万円
                            - 減額制限: {reduction_rate*100:.0f}%まで（最低{(1-reduction_rate)*100:.0f}%保証）
                            - **制限後の最低年俸: {min_salary/10000:.0f}万円**
                            """)
                            display_salary = min_salary
                        else:
                            display_salary = predicted_salary
                            is_limited = False
                    else:
                        display_salary = predicted_salary
                        is_limited = False
                    
                    add_to_history(
                        player_name=selected_player,
                        predict_year=predict_year,
                        predicted_salary=predicted_salary,
                        actual_salary=actual_salary,
                        previous_salary=previous_salary,
                        stats_dict=stats_dict,
                        model_name=st.session_state.best_model_name,
                        is_limited=is_limited,
                        limited_salary=display_salary if is_limited else None
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if previous_salary is not None:
                            st.metric("前年年俸", f"{previous_salary/10000:.0f}万円")
                        else:
                            st.metric("前年年俸", "データなし")
                    with col2:
                        st.metric("予測年俸", f"{predicted_salary/10000:.0f}万円")
                    with col3:
                        if actual_salary:
                            st.metric("実際の年俸", f"{actual_salary/10000:.0f}万円")
                        else:
                            st.metric("実際の年俸", "データなし")
                    with col4:
                        if actual_salary:
                            error = abs(display_salary - actual_salary) / actual_salary * 100
                            st.metric("予測誤差", f"{error:.1f}%")
                    
                    st.markdown("---")
                    st.subheader(f"{stats_year}年の成績")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("試合", int(player_stats['試合']))
                        st.metric("打率", f"{player_stats['打率']:.3f}")
                    with col2:
                        st.metric("安打", int(player_stats['安打']))
                        st.metric("出塁率", f"{player_stats['出塁率']:.3f}")
                    with col3:
                        st.metric("本塁打", int(player_stats['本塁打']))
                        st.metric("長打率", f"{player_stats['長打率']:.3f}")
                    with col4:
                        st.metric("打点", int(player_stats['打点']))
                        st.metric("タイトル数", int(player_stats['タイトル数']))
                    with col5:
                        st.metric("年齢", int(player_stats['年齢']))
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        player_salary_history = st.session_state.salary_long[
                            st.session_state.salary_long['選手名'] == selected_player
                        ].sort_values('年度')
                        
                        if not player_salary_history.empty:
                            # 年度でソート
                            player_salary_history = player_salary_history.sort_values('年度')

                            # 年度を整数化
                            years = player_salary_history['年度'].astype(int).values
                            salaries = player_salary_history['年俸_円'].values / 10000  # 万円に変換

                            # 実際の年俸
                            ax1.plot(years, salaries, 'o-', linewidth=2, markersize=8, label='実際の年俸')

                            # 予測年
                            ax1.plot(int(predict_year), predicted_salary/10000, 'r*', markersize=20, label='予測年俸（制限前）')

                            # 制限後
                            if previous_salary is not None and is_limited:
                                ax1.plot(int(predict_year), display_salary/10000, 'orange', marker='D', markersize=12, label='制限後年俸')

                            # 実際の年俸（当年）
                            if actual_salary:
                                ax1.plot(int(predict_year), actual_salary/10000, 'go', markersize=12, 
                                    label=f'実際の年俸({int(predict_year)})')

                            # ★ X軸を固定で「2023〜2026」の4つにする ★
                            ax1.set_xticks([2023, 2024, 2025, 2026])

                            # 表示を整数に
                            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

                            ax1.set_xlabel('年度', fontweight='bold')
                            ax1.set_ylabel('年俸（万円）', fontweight='bold')
                            ax1.set_title(f'{selected_player} - 年俸推移', fontweight='bold')
                            ax1.grid(alpha=0.3)
                            ax1.legend()

                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 5), subplot_kw=dict(projection='polar'))
                        
                        radar_stats = {
                            '打率': player_stats['打率'] / 0.4,
                            '出塁率': player_stats['出塁率'] / 0.5,
                            '長打率': player_stats['長打率'] / 0.7,
                            '本塁打': min(player_stats['本塁打'] / 40, 1.0),
                            '打点': min(player_stats['打点'] / 100, 1.0),
                            '盗塁': min(player_stats['盗塁'] / 40, 1.0),
                        }
                        
                        categories = list(radar_stats.keys())
                        values = list(radar_stats.values())
                        values += values[:1]
                        
                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        angles += angles[:1]
                        
                        ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
                        ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
                        ax2.set_xticks(angles[:-1])
                        ax2.set_xticklabels(categories)
                        ax2.set_ylim(0, 1)
                        ax2.set_title(f'{selected_player} - 成績レーダー\n({stats_year}年)', fontweight='bold', pad=20)
                        ax2.grid(True)
                        
                        st.pyplot(fig2)
                        plt.close(fig2)
    
     # 複数選手比較
    elif menu == "📊 選手比較":
        st.header("📊 複数選手比較")
        
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        
        selected_players = st.multiselect(
            "比較する選手を2人以上選択してください（最大5人）",
            options=sorted(available_players),
            max_selections=5,
            key="compare_players_multiselect"
        )
        
        if len(selected_players) >= 2:
            if st.button("📊 比較実行", type="primary", key="compare_button"):
                results_list = []
                
                for player in selected_players:
                    player_stats = st.session_state.stats_all_with_titles[
                        (st.session_state.stats_all_with_titles['選手名'] == player) &
                        (st.session_state.stats_all_with_titles['年度'] == 2024)
                    ]
                    
                    if not player_stats.empty:
                        player_stats = player_stats.iloc[0]
                        
                        # 年齢データがない場合は28歳（平均）を使用
                        if '年齢' not in st.session_state.feature_cols:
                            features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        else:
                            # 年齢データがある場合
                            if '年齢' in player_stats.index:
                                features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                            else:
                                # 選手データに年齢がない場合は28歳で補完
                                features_list = player_stats[st.session_state.feature_cols[:-1]].values.tolist()
                                features_list.append(28)  # デフォルト年齢
                                features = np.array([features_list])
                        
                        # 予測（対数変換版）
                        if st.session_state.best_model_name == '線形回帰':
                            features_scaled = st.session_state.scaler.transform(features)
                            predicted_salary_log = st.session_state.best_model.predict(features_scaled)[0]
                        else:
                            predicted_salary_log = st.session_state.best_model.predict(features)[0]
                        
                        predicted_salary = np.expm1(predicted_salary_log)
                        
                        # 十万円単位で四捨五入
                        predicted_salary = round(predicted_salary / 100000) * 100000
                        
                        # 前年（2024年）の年俸を取得
                        previous_salary_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == player) &
                            (st.session_state.salary_long['年度'] == 2024)
                        ]
                        previous_salary = previous_salary_data['年俸_円'].values[0] if not previous_salary_data.empty else None
                        
                        # 減額制限チェック
                        is_limited = False
                        display_salary = predicted_salary
                        if previous_salary is not None:
                            is_limited, min_salary, reduction_rate = check_salary_reduction_limit(predicted_salary, previous_salary)
                            if is_limited:
                                display_salary = min_salary
                        
                        results_list.append({
                            '選手名': player,
                            '前年年俸': previous_salary / 10000 if previous_salary else None,
                            '予測年俸（制限前）': predicted_salary / 10000,
                            '予測年俸（制限後）': display_salary / 10000,
                            '減額制限': 'あり' if is_limited else 'なし',
                            '打率': player_stats['打率'],
                            '本塁打': int(player_stats['本塁打']),
                            '打点': int(player_stats['打点']),
                            'タイトル数': int(player_stats['タイトル数'])
                        })
                
                if results_list:
                    df_results = pd.DataFrame(results_list)
                    
                    st.dataframe(
                        df_results,
                        use_container_width=True,
                        hide_index=True,
                        height=None
                    )
                    
                    # 減額制限に引っかかった選手を表示
                    limited_players = df_results[df_results['減額制限'] == 'あり']
                    if not limited_players.empty:
                        st.warning("⚖️ **減額制限に引っかかった選手:**")
                        for _, row in limited_players.iterrows():
                            st.write(f"- **{row['選手名']}**: 予測{row['予測年俸（制限前）']:.0f}万円 → 制限後{row['予測年俸（制限後）']:.0f}万円")
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        
                        x = np.arange(len(df_results))
                        width = 0.35
                        
                        ax1.barh(x - width/2, df_results['予測年俸（制限前）'], width, label='予測年俸（制限前）', alpha=0.7, color='steelblue')
                        ax1.barh(x + width/2, df_results['予測年俸（制限後）'], width, label='予測年俸（制限後）', alpha=0.7, color='orange')
                        
                        ax1.set_yticks(x)
                        ax1.set_yticklabels(df_results['選手名'])
                        ax1.set_xlabel('予測年俸（万円）', fontweight='bold')
                        ax1.set_title('予測年俸比較', fontweight='bold')
                        ax1.legend()
                        ax1.grid(axis='x', alpha=0.3)
                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        x = np.arange(len(df_results))
                        width = 0.25
                        
                        ax2.bar(x - width, df_results['打率']*100, width, label='打率 x100', alpha=0.8)
                        ax2.bar(x, df_results['本塁打'], width, label='本塁打', alpha=0.8)
                        ax2.bar(x + width, df_results['打点']/10, width, label='打点 /10', alpha=0.8)
                        
                        ax2.set_xlabel('選手', fontweight='bold')
                        ax2.set_ylabel('値（正規化）', fontweight='bold')
                        ax2.set_title('成績比較', fontweight='bold')
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(df_results['選手名'], rotation=45, ha='right')
                        ax2.legend()
                        ax2.grid(axis='y', alpha=0.3)
                        st.pyplot(fig2)
                        plt.close(fig2)

    
    # 複数モデル比較
    elif menu == "🔬 モデル比較":
        st.header("🔬 複数モデル比較")
        st.markdown("同じ選手の年俸を全モデルで予測し、結果を比較します")
        
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        sorted_players = sorted(available_players)
        
        st.markdown("### 選手を選択")
        
        search_filter = st.text_input(
            "🔍 絞り込み検索（オプション）",
            placeholder="例: 村上、岡本、近藤",
            key="model_compare_search",
            help="選手名の一部を入力すると候補が絞り込まれます"
        )
        
        if search_filter:
            filtered_players = [p for p in sorted_players if search_filter in p]
            if not filtered_players:
                st.warning("⚠️ 該当する選手が見つかりません")
                filtered_players = sorted_players
        else:
            filtered_players = sorted_players
        
        selected_player = st.selectbox(
            f"選手を選択してください",
            options=filtered_players,
            index=0,
            key="model_compare_player_select"
        )
        
        predict_year = st.slider("予測年度", 2024, 2026, 2025, key="model_compare_year")
        
        if st.button("🔬 全モデルで予測", type="primary", key="model_compare_button"):
            if not selected_player:
                st.error("❌ 選手を選択してください")
            else:
                stats_year = predict_year - 1
                player_stats = st.session_state.stats_all_with_titles[
                    (st.session_state.stats_all_with_titles['選手名'] == selected_player) &
                    (st.session_state.stats_all_with_titles['年度'] == stats_year)
                ]
                
                if player_stats.empty:
                    st.error(f"❌ {selected_player}の{stats_year}年のデータが見つかりません")
                else:
                    player_stats = player_stats.iloc[0]
                    
                    # 年齢データがない場合は28歳（平均）を使用
                    if '年齢' not in st.session_state.feature_cols:
                        features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                    else:
                        # 年齢データがある場合
                        if '年齢' in player_stats.index:
                            features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        else:
                            # 選手データに年齢がない場合は28歳で補完
                            features_list = player_stats[st.session_state.feature_cols[:-1]].values.tolist()
                            features_list.append(28)  # デフォルト年齢
                            features = np.array([features_list])
                    
                    # 前年の年俸と実際の年俸を取得
                    previous_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == stats_year)
                    ]
                    previous_salary = previous_salary_data['年俸_円'].values[0] if not previous_salary_data.empty else None
                    
                    actual_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == predict_year)
                    ]
                    actual_salary = actual_salary_data['年俸_円'].values[0] if not actual_salary_data.empty else None
                    
                    # 全モデルで予測
                    model_predictions = []
                    for model_name, model_info in st.session_state.results.items():
                        model = model_info['model']
                        
                        if model_name == '線形回帰':
                            features_scaled = st.session_state.scaler.transform(features)
                            pred_log = model.predict(features_scaled)[0]
                        else:
                            pred_log = model.predict(features)[0]
                        
                        pred_salary = np.expm1(pred_log)
                        
                        # 十万円単位で四捨五入
                        pred_salary = round(pred_salary / 100000) * 100000
                        
                        # 減額制限チェック
                        is_limited = False
                        display_salary = pred_salary
                        if previous_salary is not None:
                            is_limited, min_salary, reduction_rate = check_salary_reduction_limit(pred_salary, previous_salary)
                            if is_limited:
                                display_salary = min_salary
                        
                        # 実際の年俸との誤差計算
                        error_pct = None
                        if actual_salary:
                            error_pct = abs(display_salary - actual_salary) / actual_salary * 100
                        
                        model_predictions.append({
                            'モデル': model_name,
                            '予測年俸（制限前）': pred_salary / 10000,
                            '予測年俸（制限後）': display_salary / 10000,
                            '減額制限': 'あり' if is_limited else 'なし',
                            'MAE': model_info['MAE'] / 10000,
                            'R²': model_info['R2'],
                            '誤差率(%)': error_pct if error_pct is not None else None
                        })
                    
                    df_predictions = pd.DataFrame(model_predictions)
                    
                    st.success("✅ 全モデルでの予測完了！")
                    
                    # メトリクス表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if previous_salary:
                            st.metric("前年年俸", f"{previous_salary/10000:.1f}万円")
                        else:
                            st.metric("前年年俸", "データなし")
                    with col2:
                        if actual_salary:
                            st.metric("実際の年俸", f"{actual_salary/10000:.1f}万円")
                        else:
                            st.metric("実際の年俸", "データなし")
                    with col3:
                        avg_pred = df_predictions['予測年俸（制限後）'].mean()
                        st.metric("平均予測値", f"{avg_pred:.1f}万円")
                    
                    st.markdown("---")
                    st.subheader("📊 モデル別予測結果")
                    
                    # 表示用にフォーマット
                    df_display = df_predictions.copy()
                    df_display['予測年俸（制限前）'] = df_display['予測年俸（制限前）'].apply(lambda x: f"{x:.1f}")
                    df_display['予測年俸（制限後）'] = df_display['予測年俸（制限後）'].apply(lambda x: f"{x:.1f}")
                    df_display['MAE'] = df_display['MAE'].apply(lambda x: f"{x:.2f}")
                    df_display['R²'] = df_display['R²'].apply(lambda x: f"{x:.4f}")
                    if actual_salary:
                        df_display['誤差率(%)'] = df_display['誤差率(%)'].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
                    else:
                        df_display = df_display.drop(columns=['誤差率(%)'])
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # グラフ表示
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        
                        x = np.arange(len(df_predictions))
                        width = 0.35
                        
                        bars1 = ax1.barh(x - width/2, df_predictions['予測年俸（制限前）'], 
                                        width, label='予測年俸（制限前）', alpha=0.7, color='steelblue')
                        bars2 = ax1.barh(x + width/2, df_predictions['予測年俸（制限後）'], 
                                        width, label='予測年俸（制限後）', alpha=0.7, color='orange')
                        
                        # 実際の年俸がある場合は線を追加
                        if actual_salary:
                            ax1.axvline(x=actual_salary/10000, color='green', linestyle='--', 
                                       linewidth=2, label='実際の年俸', alpha=0.8)
                        
                        ax1.set_yticks(x)
                        ax1.set_yticklabels(df_predictions['モデル'])
                        ax1.set_xlabel('年俸（万円）', fontweight='bold')
                        ax1.set_title(f'{selected_player} - モデル別予測年俸', fontweight='bold')
                        ax1.legend()
                        ax1.grid(axis='x', alpha=0.3)
                        
                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col2:
                        if actual_salary:
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            
                            errors = df_predictions['誤差率(%)'].dropna()
                            models = df_predictions[df_predictions['誤差率(%)'].notna()]['モデル']
                            
                            colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in errors]
                            
                            ax2.barh(range(len(errors)), errors, color=colors, alpha=0.7)
                            ax2.set_yticks(range(len(errors)))
                            ax2.set_yticklabels(models)
                            ax2.set_xlabel('誤差率 (%)', fontweight='bold')
                            ax2.set_title('モデル別予測誤差', fontweight='bold')
                            ax2.grid(axis='x', alpha=0.3)
                            
                            # 誤差率の目安線
                            ax2.axvline(x=10, color='green', linestyle=':', alpha=0.5, label='10%')
                            ax2.axvline(x=20, color='orange', linestyle=':', alpha=0.5, label='20%')
                            ax2.legend()
                            
                            st.pyplot(fig2)
                            plt.close(fig2)
                        else:
                            st.info("📊 実際の年俸データがないため、誤差グラフは表示されません")
                    
                    # モデル性能との関連性分析
                    st.markdown("---")
                    st.subheader("📈 予測精度とモデル性能の関係")
                    
                    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # MAEとの関係
                    ax3.scatter(df_predictions['MAE'], df_predictions['予測年俸（制限後）'], 
                               s=100, alpha=0.6, c=range(len(df_predictions)), cmap='viridis')
                    for i, model in enumerate(df_predictions['モデル']):
                        ax3.annotate(model, 
                                    (df_predictions.iloc[i]['MAE'], 
                                     df_predictions.iloc[i]['予測年俸（制限後）']),
                                    fontsize=9, alpha=0.8)
                    ax3.set_xlabel('MAE（万円）', fontweight='bold')
                    ax3.set_ylabel('予測年俸（万円）', fontweight='bold')
                    ax3.set_title('モデルMAEと予測年俸の関係', fontweight='bold')
                    ax3.grid(alpha=0.3)
                    
                    # R²との関係
                    ax4.scatter(df_predictions['R²'], df_predictions['予測年俸（制限後）'], 
                               s=100, alpha=0.6, c=range(len(df_predictions)), cmap='viridis')
                    for i, model in enumerate(df_predictions['モデル']):
                        ax4.annotate(model, 
                                    (df_predictions.iloc[i]['R²'], 
                                     df_predictions.iloc[i]['予測年俸（制限後）']),
                                    fontsize=9, alpha=0.8)
                    ax4.set_xlabel('R² スコア', fontweight='bold')
                    ax4.set_ylabel('予測年俸（万円）', fontweight='bold')
                    ax4.set_title('モデルR²と予測年俸の関係', fontweight='bold')
                    ax4.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
                    # 統計サマリー
                    st.markdown("---")
                    st.subheader("📊 予測統計サマリー")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        max_pred = df_predictions['予測年俸（制限後）'].max()
                        st.metric("最大予測", f"{max_pred:.1f}万円")
                    with col2:
                        min_pred = df_predictions['予測年俸（制限後）'].min()
                        st.metric("最小予測", f"{min_pred:.1f}万円")
                    with col3:
                        std_pred = df_predictions['予測年俸（制限後）'].std()
                        st.metric("標準偏差", f"{std_pred:.1f}万円")
                    with col4:
                        range_pred = max_pred - min_pred
                        st.metric("予測幅", f"{range_pred:.1f}万円")
    
    # カスタム入力予測
    elif menu == "✏️ カスタム":
        st.header("✏️ カスタム入力予測")
        st.markdown("オリジナルの選手データを入力して年俸を予測します")
        # 入力フォーム
        st.subheader("📝 選手情報入力")
        
        player_name = st.text_input("選手名（任意）", placeholder="例: 山田太郎", key="custom_player_name")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**基本成績**")
            games = st.number_input("試合数", min_value=0, max_value=200, value=143, key="custom_games")
            plate_appearances = st.number_input("打席", min_value=0, max_value=800, value=600, key="custom_pa")
            at_bats = st.number_input("打数", min_value=0, max_value=700, value=520, key="custom_ab")
            runs = st.number_input("得点", min_value=0, max_value=200, value=80, key="custom_runs")
            hits = st.number_input("安打", min_value=0, max_value=300, value=150, key="custom_hits")
            
        with col2:
            st.markdown("**長打成績**")
            doubles = st.number_input("二塁打", min_value=0, max_value=100, value=30, key="custom_2b")
            triples = st.number_input("三塁打", min_value=0, max_value=30, value=3, key="custom_3b")
            home_runs = st.number_input("本塁打", min_value=0, max_value=70, value=25, key="custom_hr")
            rbis = st.number_input("打点", min_value=0, max_value=200, value=90, key="custom_rbi")
            
        with col3:
            st.markdown("**走塁・選球眼**")
            stolen_bases = st.number_input("盗塁", min_value=0, max_value=100, value=10, key="custom_sb")
            caught_stealing = st.number_input("盗塁刺", min_value=0, max_value=50, value=3, key="custom_cs")
            walks = st.number_input("四球", min_value=0, max_value=200, value=60, key="custom_bb")
            hit_by_pitch = st.number_input("死球", min_value=0, max_value=50, value=5, key="custom_hbp")
            strikeouts = st.number_input("三振", min_value=0, max_value=300, value=120, key="custom_so")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**その他**")
            double_plays = st.number_input("併殺打", min_value=0, max_value=50, value=10, key="custom_gdp")
            sac_hits = st.number_input("犠打", min_value=0, max_value=50, value=2, key="custom_sh")
            sac_flies = st.number_input("犠飛", min_value=0, max_value=30, value=5, key="custom_sf")
        
        with col2:
            st.markdown("**タイトル・前年年俸・年齢**")
            titles = st.number_input("タイトル数", min_value=0, max_value=10, value=0, key="custom_titles")
            previous_salary = st.number_input("前年年俸（万円）", min_value=0, max_value=100000, value=0, 
                                            help="減額制限チェック用。0の場合はチェックなし", key="custom_prev_salary")
            age = st.number_input("年齢", min_value=18, max_value=50, value=28, key="custom_age")
            
        with col3:
            st.markdown("**指標（自動計算）**")
            # 打率・出塁率・長打率は自動計算
            avg = hits / at_bats if at_bats > 0 else 0.0
            obp = (hits + walks + hit_by_pitch) / (at_bats + walks + hit_by_pitch + sac_flies) if (at_bats + walks + hit_by_pitch + sac_flies) > 0 else 0.0
            slg = (hits+doubles+triples*2+home_runs*3) / at_bats if at_bats > 0 else 0.0
            
            st.metric("打率", f"{avg:.3f}")
            st.metric("出塁率", f"{obp:.3f}")
            st.metric("長打率", f"{slg:.3f}")
        st.markdown("---")
        
        if st.button("🎯 年俸予測実行", type="primary", key="custom_predict_button"):
            # 入力データの検証
            if at_bats == 0:
                st.error("❌ 打数は0より大きい値を入力してください")
            elif hits > at_bats:
                st.error("❌ 安打は打数以下にしてください")
            else:
                # ★ 塁打を計算 ★
                total_bases = hits + doubles + triples * 2 + home_runs * 3
        
                # 特徴量を作成（年齢を含む23項目）
                custom_features = np.array([[
                    games, plate_appearances, at_bats, runs, hits, 
                    doubles, triples, home_runs, total_bases, rbis,
                    stolen_bases, caught_stealing, walks, hit_by_pitch, strikeouts,
                    double_plays, avg, obp, slg, sac_hits, sac_flies, titles, age
                ]])
        
                # 全モデルで予測
                st.success("✅ 予測完了！")
                
                st.subheader("📊 予測結果")
                
                predictions = []
                for model_name, model_info in st.session_state.results.items():
                    model = model_info['model']
                    
                    if model_name == '線形回帰':
                        features_scaled = st.session_state.scaler.transform(custom_features)
                        pred_log = model.predict(features_scaled)[0]
                    else:
                        pred_log = model.predict(custom_features)[0]
                    
                    pred_salary = np.expm1(pred_log)
                    
                    # 十万円単位で四捨五入
                    pred_salary = round(pred_salary / 100000) * 100000
                    
                    # 減額制限チェック
                    is_limited = False
                    display_salary = pred_salary
                    if previous_salary > 0:
                        prev_salary_yen = previous_salary * 10000
                        is_limited, min_salary, reduction_rate = check_salary_reduction_limit(pred_salary, prev_salary_yen)
                        if is_limited:
                            display_salary = min_salary
                    
                    predictions.append({
                        'モデル': model_name,
                        '予測年俸': pred_salary / 10000,
                        '制限後年俸': display_salary / 10000,
                        '減額制限': 'あり' if is_limited else 'なし'
                    })
                
                df_pred = pd.DataFrame(predictions)
                
                # メトリクス表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pred = df_pred['制限後年俸'].mean()
                    st.metric("平均予測年俸", f"{avg_pred:.1f}万円")
                with col2:
                    max_pred = df_pred['制限後年俸'].max()
                    st.metric("最大予測", f"{max_pred:.1f}万円")
                with col3:
                    min_pred = df_pred['制限後年俸'].min()
                    st.metric("最小予測", f"{min_pred:.1f}万円")
                
                # 減額制限の警告
                if any(df_pred['減額制限'] == 'あり'):
                    st.warning(f"""
                    ⚖️ **減額制限が適用されました**
                    - 前年年俸: {previous_salary:.1f}万円
                    - 一部のモデルで減額制限に該当しています
                    """)
                
                st.markdown("---")
                
                # 予測結果テーブル
                df_display = df_pred.copy()
                df_display['予測年俸'] = df_display['予測年俸'].apply(lambda x: f"{x:.1f}")
                df_display['制限後年俸'] = df_display['制限後年俸'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # グラフ表示
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    
                    x = np.arange(len(df_pred))
                    width = 0.35
                    
                    ax1.barh(x - width/2, df_pred['予測年俸'], width, 
                            label='予測年俸（制限前）', alpha=0.7, color='steelblue')
                    ax1.barh(x + width/2, df_pred['制限後年俸'], width, 
                            label='予測年俸（制限後）', alpha=0.7, color='orange')
                    
                    ax1.set_yticks(x)
                    ax1.set_yticklabels(df_pred['モデル'])
                    ax1.set_xlabel('予測年俸（万円）', fontweight='bold')
                    player_title = player_name if player_name else "カスタム選手"
                    ax1.set_title(f'{player_title} - モデル別予測年俸', fontweight='bold')
                    ax1.legend()
                    ax1.grid(axis='x', alpha=0.3)
                    
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
                    
                    radar_stats = {
                        '打率': avg / 0.4,
                        '出塁率': obp / 0.5,
                        '長打率': slg / 0.7,
                        '本塁打': min(home_runs / 40, 1.0),
                        '打点': min(rbis / 100, 1.0),
                        '盗塁': min(stolen_bases / 40, 1.0),
                    }
                    
                    categories = list(radar_stats.keys())
                    values = list(radar_stats.values())
                    values += values[:1]
                    
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]
                    
                    ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
                    ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(categories)
                    ax2.set_ylim(0, 1)
                    player_title = player_name if player_name else "カスタム選手"
                    ax2.set_title(f'{player_title} - 成績レーダー', fontweight='bold', pad=20)
                    ax2.grid(True)
                    
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                # 成績サマリー
                st.markdown("---")
                st.subheader("📈 入力成績サマリー")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("試合", games)
                with col2:
                    st.metric("安打", hits)
                with col3:
                    st.metric("本塁打", home_runs)
                with col4:
                    st.metric("打点", rbis)
                with col5:
                    st.metric("年齢", f"{age}歳")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("打率", f"{avg:.3f}")
                with col2:
                    st.metric("出塁率", f"{obp:.3f}")
                with col3:
                    st.metric("長打率", f"{slg:.3f}")
                with col4:
                    st.metric("タイトル数", titles)
                
                # データセットとの比較
                st.markdown("---")
                st.subheader("📊 データセットとの比較")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 打率の分布と入力値の位置
                    fig3, ax3 = plt.subplots(figsize=(8, 5))
                    ax3.hist(st.session_state.ml_df['打率'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'入力値: {avg:.3f}')
                    ax3.set_xlabel('打率', fontweight='bold')
                    ax3.set_ylabel('選手数', fontweight='bold')
                    ax3.set_title('打率の分布', fontweight='bold')
                    ax3.legend()
                    ax3.grid(alpha=0.3)
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                with col2:
                    # 本塁打の分布と入力値の位置
                    fig4, ax4 = plt.subplots(figsize=(8, 5))
                    ax4.hist(st.session_state.ml_df['本塁打'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                    ax4.axvline(home_runs, color='red', linestyle='--', linewidth=2, label=f'入力値: {home_runs}')
                    ax4.set_xlabel('本塁打', fontweight='bold')
                    ax4.set_ylabel('選手数', fontweight='bold')
                    ax4.set_title('本塁打の分布', fontweight='bold')
                    ax4.legend()
                    ax4.grid(alpha=0.3)
                    st.pyplot(fig4)
                    plt.close(fig4)
                
                # 年齢の分布
                if '年齢' in st.session_state.ml_df.columns:
                    st.markdown("---")
                    fig5, ax5 = plt.subplots(figsize=(10, 5))
                    ax5.hist(st.session_state.ml_df['年齢'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                    ax5.axvline(age, color='red', linestyle='--', linewidth=2, label=f'入力値: {age}歳')
                    ax5.set_xlabel('年齢', fontweight='bold')
                    ax5.set_ylabel('選手数', fontweight='bold')
                    ax5.set_title('年齢の分布', fontweight='bold')
                    ax5.legend()
                    ax5.grid(alpha=0.3)
                    st.pyplot(fig5)
                    plt.close(fig5)
    
    # モデル性能
    elif menu == "📈 性能":
        st.header("📈 モデル性能")
        
        model_data = []
        for name, result in st.session_state.results.items():
            model_data.append({
                'モデル': name,
                'MAE（万円）': f"{result['MAE']/10000:.2f}",
                'R²スコア': f"{result['R2']:.4f}"
            })
        
        df_models = pd.DataFrame(model_data).sort_values('R²スコア', ascending=False)
        st.dataframe(
            df_models,
            use_container_width=False,
            hide_index=True
        )
        st.success(f"🏆 最良モデル: {st.session_state.best_model_name}")
        
        if st.session_state.best_model_name == 'ランダムフォレスト':
            st.markdown("---")
            st.subheader("特徴量重要度 Top 10")
            
            feature_importance = pd.DataFrame({
                '特徴量': st.session_state.feature_cols,
                '重要度': st.session_state.best_model.feature_importances_
            }).sort_values('重要度', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(feature_importance)), feature_importance['重要度'], color='#9b59b6', alpha=0.7)
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['特徴量'])
            ax.set_xlabel('重要度', fontweight='bold')
            ax.set_title('特徴量重要度 Top 10', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)
    
    # 要因分析
    elif menu == "📉 要因分析":
        st.header("📉 要因分析")
        
        st.subheader("タイトル獲得の影響")
        title_groups = st.session_state.ml_df.groupby(
            st.session_state.ml_df['タイトル数'] > 0
        )['年俸_円'].agg(['count', 'mean', 'median'])
        
        title_groups['mean'] = round(title_groups['mean'] / 10000)
        title_groups['median'] = title_groups['median'] / 10000
        title_groups.index = ['タイトル無し', 'タイトル有り']
        title_groups.columns = ['選手数', '平均年俸（万円）', '中央値（万円）']
        
        st.dataframe(
            title_groups,
            use_container_width=False
        )
        
        if len(title_groups) == 2:
            diff = title_groups.loc['タイトル有り', '平均年俸（万円）'] - title_groups.loc['タイトル無し', '平均年俸（万円）']
            diff_rounded = round(diff)
            st.metric("タイトル獲得による年俸増加", f"{diff_rounded}万円")
        
        st.markdown("---")
        st.subheader("主要指標との相関")
        
        correlations = st.session_state.ml_df[
            ['打率', '本塁打', '打点', '出塁率', '長打率', 'タイトル数','年齢','試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', 
                '塁打', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打', '犠打', '犠飛', '年俸_円']
        ].corr()['年俸_円'].sort_values(ascending=False)
        
        corr_data = []
        for idx, val in correlations.items():
            if idx != '年俸_円':
                corr_data.append({'指標': idx, '相関係数': f"{val:.4f}"})
        
        st.dataframe(
            pd.DataFrame(corr_data),
            use_container_width=False,
            hide_index=True
        )
        
        # 1つ目
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(st.session_state.ml_df['打率'], st.session_state.ml_df['年俸_円']/10000, alpha=0.5)
        ax1.set_xlabel('打率', fontweight='bold')
        ax1.set_ylabel('年俸（万円）', fontweight='bold')
        ax1.set_title('打率と年俸の関係', fontweight='bold')
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

        # 2つ目
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(st.session_state.ml_df['本塁打'], st.session_state.ml_df['年俸_円']/10000, alpha=0.5, color='orange')
        ax2.set_xlabel('本塁打', fontweight='bold')
        ax2.set_ylabel('年俸（万円）', fontweight='bold')
        ax2.set_title('本塁打と年俸の関係', fontweight='bold')
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

        # 3つ目
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.scatter(st.session_state.ml_df['年齢'], st.session_state.ml_df['年俸_円']/10000, alpha=0.5, color='green')
        ax3.set_xlabel('年齢', fontweight='bold')
        ax3.set_ylabel('年俸（万円）', fontweight='bold')
        ax3.set_title('年齢と年俸の関係', fontweight='bold')
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    # 予測精度ランキング
    elif menu == "🏆 精度ランキング":
        st.header("🏆 予測精度ランキング")
        st.markdown("実際の年俸データがある選手の予測精度を分析し、ランキング表示します")
        
        # 予測年度を選択
        rank_year = st.selectbox("ランキング対象年度", [2024, 2025], index=1, key="rank_year_select")
        
        # ランキングのソート基準を選択
        col1, col2 = st.columns([2, 1])
        with col1:
            sort_column = st.selectbox(
                "ソート項目",
                ["誤差率", "誤差額", "予測年俸(万円)"],
                key="rank_sort_column"
            )
        with col2:
            sort_order = st.radio(
                "並び順",
                ["昇順（小→大）", "降順（大→小）"],
                key="rank_sort_order"
            )
        
        # 表示件数を選択
        top_n = st.slider("表示件数", min_value=10, max_value=100, value=30, step=10, key="rank_top_n")
        
        if st.button("📊 ランキング作成", type="primary", key="rank_create_button"):
            with st.spinner('🔄 全選手の予測を計算中...'):
                stats_year = rank_year - 1
                
                # 対象選手を取得（実際の年俸データがある選手のみ）
                actual_salary_players = st.session_state.salary_long[
                    st.session_state.salary_long['年度'] == rank_year
                ]['選手名'].unique()
                
                # 成績データがある選手を取得
                stats_players = st.session_state.stats_all_with_titles[
                    st.session_state.stats_all_with_titles['年度'] == stats_year
                ]['選手名'].unique()
                
                # 両方のデータがある選手のみを対象
                target_players = list(set(actual_salary_players) & set(stats_players))
                
                if not target_players:
                    st.error(f"❌ {rank_year}年の実際の年俸データと{stats_year}年の成績データが両方ある選手が見つかりません")
                else:
                    ranking_data = []
                    
                    # プログレスバー
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, player in enumerate(target_players):
                        # プログレス更新
                        progress = (idx + 1) / len(target_players)
                        progress_bar.progress(progress)
                        status_text.text(f"処理中: {player} ({idx + 1}/{len(target_players)})")
                        
                        # 選手の成績データを取得
                        player_stats = st.session_state.stats_all_with_titles[
                            (st.session_state.stats_all_with_titles['選手名'] == player) &
                            (st.session_state.stats_all_with_titles['年度'] == stats_year)
                        ]
                        
                        if player_stats.empty:
                            continue
                        
                        player_stats = player_stats.iloc[0]
                        
                        # 特徴量を作成
                        if '年齢' not in st.session_state.feature_cols:
                            features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        else:
                            if '年齢' in player_stats.index:
                                features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                            else:
                                features_list = player_stats[st.session_state.feature_cols[:-1]].values.tolist()
                                features_list.append(28)
                                features = np.array([features_list])
                        
                        # 予測
                        if st.session_state.best_model_name == '線形回帰':
                            features_scaled = st.session_state.scaler.transform(features)
                            predicted_salary_log = st.session_state.best_model.predict(features_scaled)[0]
                        else:
                            predicted_salary_log = st.session_state.best_model.predict(features)[0]
                        
                        predicted_salary = np.expm1(predicted_salary_log)
                        
                        # 十万円単位で四捨五入
                        predicted_salary = round(predicted_salary / 100000) * 100000
                        
                        # 前年の年俸を取得
                        previous_salary_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == player) &
                            (st.session_state.salary_long['年度'] == stats_year)
                        ]
                        previous_salary = previous_salary_data['年俸_円'].values[0] if not previous_salary_data.empty else None
                        
                        # 減額制限チェック
                        display_salary = predicted_salary
                        is_limited = False
                        if previous_salary is not None:
                            is_limited, min_salary, reduction_rate = check_salary_reduction_limit(predicted_salary, previous_salary)
                            if is_limited:
                                display_salary = min_salary
                        
                        # 実際の年俸を取得
                        actual_salary_data = st.session_state.salary_long[
                            (st.session_state.salary_long['選手名'] == player) &
                            (st.session_state.salary_long['年度'] == rank_year)
                        ]
                        actual_salary = actual_salary_data['年俸_円'].values[0] if not actual_salary_data.empty else None
                        
                        if actual_salary is not None:
                            # 誤差を計算
                            error_amount = abs(display_salary - actual_salary)
                            error_rate = (error_amount / actual_salary) * 100
                            
                            ranking_data.append({
                                '順位': 0,  # 後で設定
                                '選手名': player,
                                '実際の年俸(万円)': actual_salary / 10000,
                                '予測年俸(万円)': display_salary / 10000,
                                '誤差額': error_amount / 10000,
                                '誤差率': error_rate,
                                '減額制限': 'あり' if is_limited else 'なし',
                                '打率': player_stats['打率'],
                                '本塁打': int(player_stats['本塁打']),
                                '打点': int(player_stats['打点']),
                                'タイトル数': int(player_stats['タイトル数'])
                            })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if ranking_data:
                        df_ranking = pd.DataFrame(ranking_data)
                        
                        # ソート実行（昇順/降順を判定）
                        ascending = (sort_order == "昇順（小→大）")
                        df_ranking = df_ranking.sort_values(sort_column, ascending=ascending)
                        
                        # 順位を設定
                        df_ranking['順位'] = range(1, len(df_ranking) + 1)
                        
                        # Top N のみ表示
                        df_top = df_ranking.head(top_n)
                        
                        # 統計サマリー
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_error_rate = df_ranking['誤差率'].mean()
                            st.metric("平均誤差率", f"{avg_error_rate:.1f}%")
                        with col2:
                            median_error_rate = df_ranking['誤差率'].median()
                            st.metric("中央値誤差率", f"{median_error_rate:.1f}%")
                        with col3:
                            best_error_rate = df_ranking['誤差率'].min()
                            st.metric("最小誤差率", f"{best_error_rate:.1f}%")
                        with col4:
                            worst_error_rate = df_ranking['誤差率'].max()
                            st.metric("最大誤差率", f"{worst_error_rate:.1f}%")
                        
                        st.markdown("---")
                        sort_label = f"{sort_column}（{'小→大' if ascending else '大→小'}）"
                        st.subheader(f"📊 Top {top_n} ランキング ({rank_year}年) - {sort_label}")
                        
                        # データフレーム表示
                        df_display = df_top.copy()
                        df_display['実際の年俸(万円)'] = df_display['実際の年俸(万円)'].apply(lambda x: f"{x:.1f}")
                        df_display['予測年俸(万円)'] = df_display['予測年俸(万円)'].apply(lambda x: f"{x:.1f}")
                        df_display['誤差額'] = df_display['誤差額'].apply(lambda x: f"{x:.1f}")
                        df_display['誤差率'] = df_display['誤差率'].apply(lambda x: f"{x:.2f}%")
                        df_display['打率'] = df_display['打率'].apply(lambda x: f"{x:.3f}")
                        
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            height=600
                        )
                        
                        # Top 10のハイライト
                        st.markdown("---")
                        st.subheader("🌟 トップ10選手")
                        
                        top_10 = df_ranking.head(10)
                        
                        for idx, row in top_10.iterrows():
                            with st.expander(f"#{row['順位']} {row['選手名']} - 誤差率: {row['誤差率']:.2f}%"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("実際の年俸(万円)", f"{row['実際の年俸(万円)']:.1f}万円")
                                with col2:
                                    st.metric("予測年俸(万円)", f"{row['予測年俸(万円)']:.1f}万円")
                                with col3:
                                    st.metric("誤差額", f"{row['誤差額']:.1f}万円")
                                with col4:
                                    st.metric("誤差率", f"{row['誤差率']:.2f}%")
                                
                                st.markdown(f"**{stats_year}年成績**: 打率{row['打率']:.3f} / {row['本塁打']}本塁打 / {row['打点']}打点 / タイトル{row['タイトル数']}個")
                        
                        # グラフ表示
                        st.markdown("---")
                        st.subheader("📈 分析グラフ")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 誤差率分布
                            fig1, ax1 = plt.subplots(figsize=(10, 6))
                            ax1.hist(df_ranking['誤差率'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                            ax1.axvline(df_ranking['誤差率'].mean(), color='red', linestyle='--', 
                                       linewidth=2, label=f'平均: {df_ranking["誤差率"].mean():.1f}%')
                            ax1.axvline(df_ranking['誤差率'].median(), color='green', linestyle='--', 
                                       linewidth=2, label=f'中央値: {df_ranking["誤差率"].median():.1f}%')
                            ax1.set_xlabel('誤差率 (%)', fontweight='bold')
                            ax1.set_ylabel('選手数', fontweight='bold')
                            ax1.set_title('予測誤差率の分布', fontweight='bold')
                            ax1.legend()
                            ax1.grid(alpha=0.3)
                            st.pyplot(fig1)
                            plt.close(fig1)
                        
                        with col2:
                            # Top 20 の誤差率
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            top_20 = df_ranking.head(20)
                            colors = ['green' if e < 5 else 'orange' if e < 10 else 'lightcoral' 
                                     for e in top_20['誤差率']]
                            ax2.barh(range(len(top_20)), top_20['誤差率'], color=colors, alpha=0.7)
                            ax2.set_yticks(range(len(top_20)))
                            ax2.set_yticklabels(top_20['選手名'], fontsize=8)
                            ax2.set_xlabel('誤差率 (%)', fontweight='bold')
                            ax2.set_title('Top 20 選手の誤差率', fontweight='bold')
                            ax2.axvline(x=5, color='green', linestyle=':', alpha=0.5, label='5%')
                            ax2.axvline(x=10, color='orange', linestyle=':', alpha=0.5, label='10%')
                            ax2.legend()
                            ax2.grid(axis='x', alpha=0.3)
                            ax2.invert_yaxis()
                            st.pyplot(fig2)
                            plt.close(fig2)
                        
                        # 実際の年俸 vs 予測年俸 散布図
                        st.markdown("---")
                        fig3, ax3 = plt.subplots(figsize=(12, 8))
                        
                        # 誤差率でカラーマップ
                        scatter = ax3.scatter(df_ranking['実際の年俸(万円)'], 
                                            df_ranking['予測年俸(万円)'],
                                            c=df_ranking['誤差率'], 
                                            cmap='RdYlGn_r',
                                            s=100, 
                                            alpha=0.6,
                                            edgecolors='black',
                                            linewidth=0.5)
                        
                        # 完全一致の線
                        max_val = max(df_ranking['実際の年俸(万円)'].max(), df_ranking['予測年俸(万円)'].max())
                        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='完全一致')
                        
                        ax3.set_xlabel('実際の年俸（万円）', fontweight='bold')
                        ax3.set_ylabel('予測年俸（万円）', fontweight='bold')
                        ax3.set_title(f'{rank_year}年 実際の年俸 vs 予測年俸', fontweight='bold')
                        ax3.legend()
                        ax3.grid(alpha=0.3)
                        
                        # カラーバー
                        cbar = plt.colorbar(scatter, ax=ax3)
                        cbar.set_label('誤差率 (%)', fontweight='bold')
                        
                        st.pyplot(fig3)
                        plt.close(fig3)
                        
                        # 誤差率別の選手数
                        st.markdown("---")
                        st.subheader("📊 誤差率別の選手分布")
                        
                        bins = [0, 5, 10, 15, 20, 25, 30, 100]
                        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30%以上']
                        df_ranking['誤差率区分'] = pd.cut(df_ranking['誤差率'], bins=bins, labels=labels)
                        
                        error_dist = df_ranking['誤差率区分'].value_counts().sort_index()
                        
                        st.dataframe(
                                pd.DataFrame({
                                    '誤差率区分': error_dist.index,
                                    '選手数': error_dist.values,
                                    '割合': [f"{(v/len(df_ranking)*100):.1f}%" for v in error_dist.values]
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        st.error("❌ ランキングを作成できませんでした")

    # 年俸別予測
    elif menu == "💰 年俸別予測":
        st.header("💰 年俸レンジ別特化モデルで予測")
        st.markdown("""
        年俸を**複数のレンジに分けて**、それぞれに特化したモデルで予測します。
        
        💡 **メリット**: 高年俸選手は高年俸用モデル、低年俸選手は低年俸用モデルで予測するため、精度が向上します！
        """)
        
        # レンジ設定セクション
        st.markdown("---")
        st.subheader("⚙️ ステップ1: 年俸レンジを設定")
        
        # プリセット選択（わかりやすく）
        preset = st.radio(
            "設定方法を選んでください",
            ["おすすめ設定（5分割）", "簡単設定（3分割）", "詳細設定（7分割）", "高年俸対応（6分割）", "自分で設定"],
            horizontal=True,
            key="range_preset"
        )

        if preset == "簡単設定（3分割）":
            range_values = [0, 30_000_000, 80_000_000, 10_000_000_000]
            st.info("📊 **3つのレンジ**: ～3000万円 / 3000万～8000万円 / 8000万円～")
        elif preset == "おすすめ設定（5分割）":
            range_values = [0, 20_000_000, 40_000_000, 70_000_000, 100_000_000, 10_000_000_000]
            st.info("📊 **5つのレンジ**: ～2000万 / 2000-4000万 / 4000-7000万 / 7000万-1億 / 1億～")
        elif preset == "詳細設定（7分割）":
            range_values = [0, 15_000_000, 30_000_000, 50_000_000, 70_000_000, 100_000_000, 200_000_000, 10_000_000_000]
            st.info("📊 **7つのレンジ**: より細かく分けて予測精度アップ！")
        elif preset == "高年俸対応（6分割）":
            range_values = [0, 30_000_000, 70_000_000, 100_000_000, 200_000_000, 400_000_000, 10_000_000_000]
            st.info("📊 **高年俸特化**: ～3000万 / 3000-7000万 / 7000万-1億 / 1-2億 / 2-4億 / 4億～")
        else:
    # カスタム設定（わかりやすく改善）
            st.markdown("### 自分で年俸の区切りを設定")
            
            # 単位選択を追加
            unit = st.radio("入力単位を選んでください", ["万円", "億円"], horizontal=True, key="salary_unit")
            
            num_ranges = st.slider("いくつに分けますか？", min_value=2, max_value=8, value=5, key="num_ranges")
            
            st.markdown(f"**{num_ranges}個の区切り位置を設定**（低い順に入力）")
            
            range_values = [0]
            cols = st.columns(min(num_ranges - 1, 4))  # 最大4列表示
            
            if unit == "万円":
                default_values = [2000, 4000, 7000, 10000, 20000, 40000, 80000]
                for i in range(num_ranges - 1):
                    with cols[i % 4]:
                        val = st.number_input(
                            f"区切り{i+1}",
                            min_value=100,
                            max_value=100000,  # ← 10億円（100000万円）まで
                            value=default_values[i] if i < len(default_values) else 10000,
                            step=500,
                            key=f"range_{i}",
                            help=f"{i+1}番目の区切り位置（万円）"
                        )
                        range_values.append(val * 10000)
            else:  # 億円
                default_values = [0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 8.0]
                for i in range(num_ranges - 1):
                    with cols[i % 4]:
                        val = st.number_input(
                            f"区切り{i+1}（億円）",
                            min_value=0.01,
                            max_value=10.0,  # ← 10億円まで
                            value=default_values[i] if i < len(default_values) else 1.0,
                            step=0.1,
                            format="%.2f",
                            key=f"range_{i}",
                            help=f"{i+1}番目の区切り位置（億円）"
                        )
                        range_values.append(int(val * 100_000_000))
            
            range_values.append(10_000_000_000)  # ← 最終上限を10億円に変更
            
            # 設定内容をプレビュー
            st.markdown("**設定したレンジ:**")
            preview_ranges = []
            for i in range(len(range_values) - 1):
                min_val = range_values[i] / 10000
                max_val = range_values[i + 1] / 10000
                
                # 表示を見やすく（1億以上は億単位表示）
                if min_val >= 10000:
                    min_display = f"{min_val/10000:.1f}億円"
                else:
                    min_display = f"{min_val:.0f}万円"
                
                if max_val >= 10000:
                    max_display = f"{max_val/10000:.1f}億円"
                else:
                    max_display = f"{max_val:.0f}万円"
                
                preview_ranges.append(f"レンジ{i+1}: {min_display}～{max_display}")
            st.markdown("  \n".join(preview_ranges))
        
        # モデル訓練ボタン
        st.markdown("---")
        st.subheader("⚙️ ステップ2: モデルを訓練")
        
        if st.button("🔧 モデルを訓練する", type="primary", use_container_width=True, key="train_ranged_model"):
            with st.spinner("🤖 各レンジ用のモデルを訓練中... 少々お待ちください"):
                # レンジ別モデル訓練
                feature_cols = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', '本塁打', 
                               '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打', 
                               '打率', '出塁率', '長打率', '犠打', '犠飛', 'タイトル数']
                
                merged_df = st.session_state.ml_df.copy()
                
                if '年齢' in merged_df.columns:
                    feature_cols.append('年齢')
                else:
                    merged_df['年齢'] = 28
                    feature_cols.append('年齢')
                
                ranged_models = {}
                
                for i in range(len(range_values) - 1):
                    min_sal = range_values[i]
                    max_sal = range_values[i + 1]
                    range_name = f"{min_sal/10000:.0f}万～{max_sal/10000:.0f}万円"
                    
                    range_df = merged_df[(merged_df['年俸_円'] >= min_sal) & (merged_df['年俸_円'] < max_sal)].copy()
                    
                    if len(range_df) < 10:
                        st.warning(f"⚠️ {range_name}: データが少ないためスキップしました（{len(range_df)}人）")
                        continue
                    
                    X = range_df[feature_cols]
                    y = range_df['年俸_円']
                    y_log = np.log1p(y)
                    
                    test_size = min(0.2, max(0.1, len(range_df) * 0.2 / len(range_df)))
                    X_train, X_test, y_train_log, y_test_log = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )
                    
                    y_test_original = np.expm1(y_test_log)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    model.fit(X_train, y_train_log)
                    y_pred_log = model.predict(X_test)
                    y_pred = np.expm1(y_pred_log)
                    
                    mae = mean_absolute_error(y_test_original, y_pred)
                    r2 = r2_score(y_test_original, y_pred)
                    
                    ranged_models[range_name] = {
                        'model': model,
                        'scaler': scaler,
                        'MAE': mae,
                        'R2': r2,
                        'min_salary': min_sal,
                        'max_salary': max_sal,
                        'n_samples': len(range_df),
                        'feature_cols': feature_cols
                    }
                
                st.session_state.custom_ranged_models = ranged_models
                st.success("✅ モデル訓練完了！下にスクロールして性能を確認してください")
        
        # モデル性能表示
        if 'custom_ranged_models' in st.session_state:
            st.markdown("---")
            st.subheader("📊 訓練したモデルの性能")
            st.markdown("各レンジごとのモデルがどれくらい正確か確認できます")
            
            range_performance = []
            for range_name, model_info in st.session_state.custom_ranged_models.items():
                range_performance.append({
                    '年俸レンジ': range_name,
                    '選手数': model_info['n_samples'],
                    '平均誤差': f"{model_info['MAE']/10000:.0f}万円",
                    '精度(R²)': f"{model_info['R2']:.3f}"
                })
            
            df_range_perf = pd.DataFrame(range_performance)
            st.dataframe(df_range_perf, use_container_width=True, hide_index=True)
            
            st.info("💡 **精度(R²)が高いほど正確**です。0.8以上なら優秀！")
            
            # 選手選択
            st.markdown("---")
            st.subheader("⚙️ ステップ3: 選手を選んで予測")
            
            available_players = st.session_state.stats_all_with_titles[
                st.session_state.stats_all_with_titles['年度'] == 2024
            ]['選手名'].unique()
            sorted_players = sorted(available_players)
            
            # 検索フィルター
            col1, col2 = st.columns([3, 1])
            with col1:
                search_filter = st.text_input(
                    "🔍 選手名で検索",
                    placeholder="選手名を入力（例: 村上、岡本、近藤）",
                    key="ranged_search_filter"
                )
            
            if search_filter:
                filtered_players = [p for p in sorted_players if search_filter in p]
                if not filtered_players:
                    st.warning("⚠️ 該当する選手が見つかりません")
                    filtered_players = sorted_players
            else:
                filtered_players = sorted_players
            
            selected_player = st.selectbox(
                "予測する選手を選んでください",
                options=filtered_players,
                key="ranged_player_select"
            )
            
            with col2:
                predict_year = st.selectbox("予測年度", [2024, 2025, 2026], index=1, key="ranged_predict_year")
            
            if st.button("🎯 予測する！", type="primary", use_container_width=True, key="ranged_predict_button"):
                stats_year = predict_year - 1
                player_stats = st.session_state.stats_all_with_titles[
                    (st.session_state.stats_all_with_titles['選手名'] == selected_player) &
                    (st.session_state.stats_all_with_titles['年度'] == stats_year)
                ]
                
                if player_stats.empty:
                    st.error(f"❌ {selected_player}の{stats_year}年のデータが見つかりません")
                else:
                    player_stats = player_stats.iloc[0]
                    
                    # 前年年俸取得
                    previous_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == stats_year)
                    ]
                    previous_salary = previous_salary_data['年俸_円'].values[0] if not previous_salary_data.empty else None
                    
                    # 実際の年俸取得
                    actual_salary_data = st.session_state.salary_long[
                        (st.session_state.salary_long['選手名'] == selected_player) &
                        (st.session_state.salary_long['年度'] == predict_year)
                    ]
                    actual_salary = actual_salary_data['年俸_円'].values[0] if not actual_salary_data.empty else None
                    
                    all_predictions = []
                    best_model_info = None
                    best_error = float('inf')
                    
                    # 統一モデルで予測
                    feature_cols = st.session_state.feature_cols
                    if '年齢' in player_stats.index:
                        features = player_stats[feature_cols].values.reshape(1, -1)
                    else:
                        features_list = player_stats[feature_cols[:-1]].values.tolist()
                        features_list.append(28)
                        features = np.array([features_list])
                    
                    if st.session_state.best_model_name == '線形回帰':
                        features_scaled = st.session_state.scaler.transform(features)
                        unified_pred_log = st.session_state.best_model.predict(features_scaled)[0]
                    else:
                        unified_pred_log = st.session_state.best_model.predict(features)[0]
                    
                    unified_pred = np.expm1(unified_pred_log)
                    unified_pred = round(unified_pred / 100000) * 100000
                    
                    unified_display = unified_pred
                    unified_limited = False
                    if previous_salary:
                        unified_limited, min_sal, _ = check_salary_reduction_limit(unified_pred, previous_salary)
                        if unified_limited:
                            unified_display = min_sal
                    
                    unified_error = abs(unified_display - actual_salary) if actual_salary else None
                    
                    all_predictions.append({
                        'モデル': '📊 通常モデル',
                        '予測年俸(万円)': unified_display / 10000,
                        '減額制限': 'あり' if unified_limited else 'なし',
                        '誤差(万円)': unified_error / 10000 if unified_error else None
                    })
                    
                    if unified_error and unified_error < best_error:
                        best_error = unified_error
                        best_model_info = ('通常モデル', unified_display)
                    
                    # レンジ別モデルで予測
                    for range_name, model_info in st.session_state.custom_ranged_models.items():
                        range_features = player_stats[model_info['feature_cols']].values.reshape(1, -1) if '年齢' in player_stats.index else np.array([player_stats[model_info['feature_cols'][:-1]].values.tolist() + [28]])
                        
                        range_pred_log = model_info['model'].predict(range_features)[0]
                        range_pred = np.expm1(range_pred_log)
                        range_pred = round(range_pred / 100000) * 100000
                        
                        range_display = range_pred
                        range_limited = False
                        if previous_salary:
                            range_limited, min_sal, _ = check_salary_reduction_limit(range_pred, previous_salary)
                            if range_limited:
                                range_display = min_sal
                        
                        range_error = abs(range_display - actual_salary) if actual_salary else None
                        
                        all_predictions.append({
                            'モデル': f'🎯 {range_name}用',
                            '予測年俸(万円)': range_display / 10000,
                            '減額制限': 'あり' if range_limited else 'なし',
                            '誤差(万円)': range_error / 10000 if range_error else None
                        })
                        
                        if range_error and range_error < best_error:
                            best_error = range_error
                            best_model_info = (range_name, range_display)
                    
                    df_predictions = pd.DataFrame(all_predictions)
                    
                    st.success("✅ 予測完了！")
                    
                    # メトリクス表示（最も正確だったモデルを強調）
                    if best_model_info and actual_salary:
                        # 最良モデルを大きく表示
                        error_rate = (best_error / actual_salary) * 100
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 20px;
                                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; margin: 0; font-size: 28px;'>🏆 最も正確だったモデル</h2>
                            <p style='color: #f0f0f0; margin: 10px 0 5px 0; font-size: 18px;'>{best_model_info[0]}</p>
                            <h1 style='color: #ffd700; margin: 10px 0; font-size: 48px; font-weight: bold;'>
                                {best_model_info[1]/10000:.0f}万円
                            </h1>
                            <p style='color: #90ee90; margin: 5px 0 0 0; font-size: 20px;'>
                                誤差: {best_error/10000:.0f}万円 ({error_rate:.1f}%)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 参考情報を小さく表示
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if previous_salary:
                                st.metric("前年年俸", f"{previous_salary/10000:.0f}万円")
                            else:
                                st.metric("前年年俸", "データなし")
                        with col2:
                            st.metric("実際の年俸", f"{actual_salary/10000:.0f}万円")
                        with col3:
                            st.metric("最良モデル予測", f"{best_model_info[1]/10000:.0f}万円", 
                                     delta=f"誤差 {best_error/10000:.0f}万円")
                    else:
                        # 実際の年俸がない場合は従来通り
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if previous_salary:
                                st.metric("前年年俸", f"{previous_salary/10000:.0f}万円")
                            else:
                                st.metric("前年年俸", "データなし")
                        with col2:
                            st.metric("実際の年俸", "データなし")
                        with col3:
                            st.metric("通常モデル予測", f"{df_predictions.iloc[0]['予測年俸(万円)']:.0f}万円")
                    
                    st.markdown("---")
                    st.subheader("📊 全モデルの予測結果")
                    
                    # 表示用フォーマット
                    df_display = df_predictions.copy()
                    if actual_salary:
                        df_display['誤差率'] = df_display['誤差(万円)'].apply(
                            lambda x: f"{(x/(actual_salary/10000))*100:.1f}%" if x is not None else "N/A"
                        )
                        df_display = df_display.sort_values('誤差(万円)')
                    
                    df_display['予測年俸(万円)'] = df_display['予測年俸(万円)'].apply(lambda x: f"{x:.0f}万円")
                    if '誤差(万円)' in df_display.columns:
                        df_display['誤差(万円)'] = df_display['誤差(万円)'].apply(
                            lambda x: f"{x:.0f}万円" if x is not None else "N/A"
                        )
                    
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # グラフ（わかりやすく改善）
                    st.markdown("---")
                    st.subheader("📈 予測結果の比較")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 最良モデルを赤色で強調
                    colors = ['red' if df_predictions.iloc[i]['誤差(万円)'] == df_predictions['誤差(万円)'].min() and actual_salary 
                             else 'steelblue' for i in range(len(df_predictions))]
                    
                    ax.barh(range(len(df_predictions)), df_predictions['予測年俸(万円)'], alpha=0.7, color=colors)
                    
                    # 実際の年俸に線を追加
                    if actual_salary:
                        ax.axvline(x=actual_salary/10000, color='green', linestyle='--', linewidth=2, label='実際の年俸')
                    
                    ax.set_yticks(range(len(df_predictions)))
                    ax.set_yticklabels(df_predictions['モデル'])
                    ax.set_xlabel('予測年俸（万円）', fontweight='bold', fontsize=12)
                    ax.set_title(f'{selected_player}の{predict_year}年予測', fontweight='bold', fontsize=14)
                    ax.legend(fontsize=11)
                    ax.grid(axis='x', alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.info("⬆️ まず「モデルを訓練する」ボタンを押してください")
    # 予測履歴
    elif menu == "📜 予測履歴":
        st.header("📜 予測履歴")
        
        if not st.session_state.prediction_history:
            st.info("📭 予測履歴がありません。選手予測を実行すると履歴が保存されます。")
        else:
            st.markdown(f"**保存件数**: {len(st.session_state.prediction_history)} / 100件")
            
            # 履歴をクリアするボタン
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ 履歴をクリア", type="secondary", key="clear_history"):
                    st.session_state.prediction_history = []
                    st.rerun()
            
            st.markdown("---")
            
            # 履歴を表示
            for idx, item in enumerate(st.session_state.prediction_history):
                with st.expander(
                    f"#{idx+1} {item['選手名']} - {item['予測年度']}年予測 ({item['予測日時']})",
                    expanded=(idx == 0)  # 最新のものだけ展開
                ):
                    # 基本情報
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if item['前年年俸']:
                            st.metric("前年年俸", f"{item['前年年俸']/10000:.1f}万円")
                        else:
                            st.metric("前年年俸", "データなし")
                    
                    with col2:
                        st.metric("予測年俸", f"{item['予測年俸']/10000:.1f}万円")
                    
                    with col3:
                        if item['減額制限']:
                            st.metric("制限後年俸", f"{item['制限後年俸']/10000:.1f}万円")
                        else:
                            st.metric("制限後年俸", "制限なし")
                    
                    with col4:
                        if item['実際の年俸']:
                            st.metric("実際の年俸", f"{item['実際の年俸']/10000:.1f}万円")
                            error = abs(item['制限後年俸'] - item['実際の年俸']) / item['実際の年俸'] * 100
                            st.metric("誤差率", f"{error:.1f}%")
                        else:
                            st.metric("実際の年俸", "データなし")
                    
                    # 減額制限の表示
                    if item['減額制限']:
                        st.warning(f"⚖️ 減額制限が適用されました")
                    
                    # 使用モデル
                    st.markdown(f"**使用モデル**: {item['モデル']}")
                    
                    # 成績サマリー
                    st.markdown("---")
                    st.markdown("**成績サマリー**")
                    
                    stats = item['成績']
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("試合", stats['試合'])
                    with col2:
                        st.metric("安打", stats['安打'])
                    with col3:
                        st.metric("本塁打", stats['本塁打'])
                    with col4:
                        st.metric("打点", stats['打点'])
                    with col5:
                        st.metric("年齢", f"{stats['年齢']}歳")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("打率", f"{stats['打率']:.3f}")
                    with col2:
                        st.metric("出塁率", f"{stats['出塁率']:.3f}")
                    with col3:
                        st.metric("長打率", f"{stats['長打率']:.3f}")
                    with col4:
                        st.metric("タイトル", stats['タイトル数'])
            
            # 履歴の統計情報
            st.markdown("---")
            st.subheader("📊 履歴統計")
            
            # 予測回数が多い選手トップ5
            from collections import Counter
            player_counts = Counter([item['選手名'] for item in st.session_state.prediction_history])
            
            if player_counts:
                st.markdown("**予測回数が多い選手 Top 5**")
                top_players = player_counts.most_common(5)
                
                for rank, (player, count) in enumerate(top_players, 1):
                    st.write(f"{rank}. {player}: {count}回")
            
            # 平均予測年俸
            avg_predicted = sum([item['予測年俸'] for item in st.session_state.prediction_history]) / len(st.session_state.prediction_history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均予測年俸", f"{avg_predicted/10000:.1f}万円")
            with col2:
                limited_count = sum([1 for item in st.session_state.prediction_history if item['減額制限']])
                st.metric("減額制限適用", f"{limited_count}件")
            with col3:
                # 誤差率を計算できるものの平均
                errors = []
                for item in st.session_state.prediction_history:
                    if item['実際の年俸']:
                        error = abs(item['制限後年俸'] - item['実際の年俸']) / item['実際の年俸'] * 100
                        errors.append(error)
                
                if errors:
                    avg_error = sum(errors) / len(errors)
                    st.metric("平均誤差率", f"{avg_error:.1f}%")
                else:
                    st.metric("平均誤差率", "計算不可")
            
            # 履歴の推移グラフ
            st.markdown("---")
            st.subheader("📈 履歴の推移")
            
            # 予測年俸の推移グラフ
            history_df = pd.DataFrame([
                {
                    '日時': item['予測日時'],
                    '選手名': item['選手名'],
                    '予測年俸': item['予測年俸'] / 10000,
                    '実際の年俸': item['実際の年俸'] / 10000 if item['実際の年俸'] else None
                }
                for item in reversed(st.session_state.prediction_history)
            ])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 予測年俸の推移
            ax.plot(range(len(history_df)), history_df['予測年俸'], 
                   'o-', label='予測年俸', linewidth=2, markersize=6)
            
            # 実際の年俸がある場合
            actual_mask = history_df['実際の年俸'].notna()
            if actual_mask.any():
                actual_indices = [i for i, mask in enumerate(actual_mask) if mask]
                actual_values = history_df.loc[actual_mask, '実際の年俸'].values
                ax.plot(actual_indices, actual_values, 
                       's', label='実際の年俸', markersize=8, color='green', alpha=0.7)
            
            ax.set_xlabel('予測順序（古→新）', fontweight='bold')
            ax.set_ylabel('年俸（万円）', fontweight='bold')
            ax.set_title('予測年俸の推移', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # X軸のラベルを選手名に（スペースの都合で一部のみ）
            step = max(1, len(history_df) // 10)
            ax.set_xticks(range(0, len(history_df), step))
            ax.set_xticklabels([history_df.iloc[i]['選手名'][:4] for i in range(0, len(history_df), step)], 
                              rotation=45, ha='right')
            
            st.pyplot(fig)
            plt.close(fig)
            
            # 誤差率の分布（実際の年俸がある場合のみ）
            if errors:
                st.markdown("---")
                st.subheader("📊 誤差率の分布")
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                ax2.hist(errors, bins=min(len(errors), 15), alpha=0.7, 
                        color='steelblue', edgecolor='black')
                ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                           linewidth=2, label=f'平均: {np.mean(errors):.1f}%')
                ax2.axvline(np.median(errors), color='green', linestyle='--', 
                           linewidth=2, label=f'中央値: {np.median(errors):.1f}%')
                
                ax2.set_xlabel('誤差率 (%)', fontweight='bold')
                ax2.set_ylabel('件数', fontweight='bold')
                ax2.set_title('予測誤差率の分布', fontweight='bold')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                st.pyplot(fig2)
                plt.close(fig2)
            
            # 履歴データのエクスポート機能
            st.markdown("---")
            st.subheader("💾 データエクスポート")
            
            # DataFrameに変換
            export_data = []
            for item in st.session_state.prediction_history:
                export_data.append({
                    '予測日時': item['予測日時'],
                    '選手名': item['選手名'],
                    '予測年度': item['予測年度'],
                    '予測年俸(万円)': round(item['予測年俸']/10000, 1),
                    '制限後年俸(万円)': round(item['制限後年俸']/10000, 1) if item['制限後年俸'] else None,
                    '実際の年俸(万円)': round(item['実際の年俸']/10000, 1) if item['実際の年俸'] else None,
                    '前年年俸(万円)': round(item['前年年俸']/10000, 1) if item['前年年俸'] else None,
                    '減額制限': '有' if item['減額制限'] else '無',
                    'モデル': item['モデル'],
                    '試合': item['成績']['試合'],
                    '本塁打': item['成績']['本塁打'],
                    '打点': item['成績']['打点'],
                    '打率': item['成績']['打率'],
                    '出塁率': item['成績']['出塁率'],
                    '長打率': item['成績']['長打率'],
                    '年齢': item['成績']['年齢'],
                    'タイトル数': item['成績']['タイトル数']
                })
            
            df_export = pd.DataFrame(export_data)
            
            # プレビュー表示
            st.markdown("**エクスポートデータプレビュー（最新5件）**")
            st.dataframe(
                df_export.head(5),
                use_container_width=True,
                hide_index=True
            )
            
            # CSVダウンロード
            csv = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSVでダウンロード",
                data=csv,
                file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_history_csv"
            )
            
            # フィルタリング機能
            st.markdown("---")
            st.subheader("🔍 履歴フィルタリング")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 選手名でフィルタ
                unique_players = sorted(list(set([item['選手名'] for item in st.session_state.prediction_history])))
                selected_filter_player = st.selectbox(
                    "選手名で絞り込み",
                    options=["すべて"] + unique_players,
                    key="filter_player"
                )
            
            with col2:
                # 減額制限でフィルタ
                filter_limited = st.selectbox(
                    "減額制限で絞り込み",
                    options=["すべて", "制限あり", "制限なし"],
                    key="filter_limited"
                )
            
            # フィルタリング実行
            filtered_history = st.session_state.prediction_history.copy()
            
            if selected_filter_player != "すべて":
                filtered_history = [item for item in filtered_history if item['選手名'] == selected_filter_player]
            
            if filter_limited == "制限あり":
                filtered_history = [item for item in filtered_history if item['減額制限']]
            elif filter_limited == "制限なし":
                filtered_history = [item for item in filtered_history if not item['減額制限']]
            
            if filtered_history:
                st.success(f"✅ {len(filtered_history)}件の履歴が見つかりました")
                
                # フィルタ結果を表示
                filter_export_data = []
                for item in filtered_history:
                    filter_export_data.append({
                        '予測日時': item['予測日時'],
                        '選手名': item['選手名'],
                        '予測年度': item['予測年度'],
                        '予測年俸(万円)': f"{item['予測年俸']/10000:.1f}",
                        '制限後年俸(万円)': f"{item['制限後年俸']/10000:.1f}" if item['制限後年俸'] else "N/A",
                        '実際の年俸(万円)': f"{item['実際の年俸']/10000:.1f}" if item['実際の年俸'] else "N/A",
                        '減額制限': '有' if item['減額制限'] else '無',
                        'モデル': item['モデル']
                    })
                
                df_filtered = pd.DataFrame(filter_export_data)
                st.dataframe(
                    df_filtered,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("⚠️ フィルタ条件に一致する履歴がありません")
else:
    # ファイル未アップロード時
    st.info("📁 CSVファイルが見つかりませんでした")
    st.markdown("""
    ### データ配置方法
    
    以下のいずれかの方法でデータを用意してください：
    
    **方法1: dataフォルダに配置**
    ```
    data/
    ├── salary_2023&2024&2025.csv
    ├── stats_2023.csv
    ├── stats_2024.csv
    ├── stats_2025.csv
    └── titles_2023&2024&2025.csv
    ```
    
    **方法2: 左サイドバーから手動アップロード**
    
    ### 🚀 機能
    - ⚾ 選手個別の年俸予測（対数変換による精度向上）
    - 📊 複数選手の比較分析
    - 🔬 複数モデルでの同時予測と比較
    - ✏️ オリジナル選手データでの予測
    - 📈 予測モデルの性能評価
    - 📉 年俸影響要因の分析
    - 🏆 予測精度ランキング（誤差の少ない選手分析）
    - ⚖️ NPB減額制限ルールの適用
    """)

# フッター
st.markdown("---")
st.markdown("*NPB選手年俸予測システム - made by Sato&Kurokawa - Powered by Streamlit*")

# Streamlitアプリを再起動するか、以下のコマンドを実行
st.cache_data.clear()
st.cache_resource.clear()
