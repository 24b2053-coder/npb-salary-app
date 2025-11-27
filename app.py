import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
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
    padding: 0 1rem 1rem 1rem !important;
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

/* ====== メインエリア ====== */
.main {
    margin-left: 280px !important;
}

/* メインの最大幅を固定（揺れ防止） */
.block-container {
    max-width: 1400px !important;
    padding-top: 2rem !important;
}

/* ====== 表（テーブル）の揺れ対策 ====== */
.stDataFrame, .stTable {
    max-width: 100% !important;
}

table {
    table-layout: fixed !important;
    width: 100% !important;
}

thead tr th {
    background-color: #f8f8f8 !important;
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
        padding: 1rem !important;
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
if 'auto_weight' not in st.session_state:
    st.session_state.auto_weight = True
if 'feature_weights' not in st.session_state:
    st.session_state.feature_weights = {}

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
    merged_df = pd.merge(
        merged_df,
        salary_long,
        left_on=['選手名', '予測年度'],
        right_on=['選手名', '年度'],
        suffixes=('_成績', '_年俸')
    )
    merged_df = merged_df.drop(columns=['年度_年俸', '予測年度'])
    merged_df.rename(columns={'年度_成績': '成績年度'}, inplace=True)
    
    stats_all_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    stats_all_with_titles['タイトル数'] = stats_all_with_titles['タイトル数'].fillna(0)
    
    return merged_df, stats_all_with_titles, salary_long

# 自動重み付け関数
def calculate_auto_weights(X, y):
    """
    Lasso回帰を使って自動的に特徴量の重要度（重み）を計算
    """
    # Lasso回帰で重要な特徴量を抽出
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X, y)
    
    # 係数の絶対値を重要度として使用
    weights = np.abs(lasso.coef_)
    
    # 正規化（合計が1になるように）
    weights = weights / np.sum(weights)
    
    return weights

# モデル訓練関数（対数変換版 + 重み付け対応）
def train_models(_merged_df, use_auto_weight=True, manual_weights=None):
    """モデルを訓練する（対数変換適用 + 重み付け）"""
    feature_cols = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', '本塁打', 
                   '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打', 
                   '打率', '出塁率', '長打率', '犠打', '犠飛', 'タイトル数']
    
    ml_df = _merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
    ml_df = ml_df.dropna()
    
    X = ml_df[feature_cols]
    y = ml_df['年俸_円']
    
    y_log = np.log1p(y)
    
    # 重み付けの適用
    if use_auto_weight:
        # 自動重み付け
        weights = calculate_auto_weights(X, y_log)
        feature_weights = dict(zip(feature_cols, weights))
        X_weighted = X * weights
    elif manual_weights is not None:
        # 手動重み付け
        weights = np.array([manual_weights.get(col, 1.0) for col in feature_cols])
        weights = weights / np.sum(weights)  # 正規化
        feature_weights = dict(zip(feature_cols, weights))
        X_weighted = X * weights
    else:
        # 重み付けなし
        X_weighted = X.copy()
        feature_weights = dict(zip(feature_cols, [1.0/len(feature_cols)] * len(feature_cols)))
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X_weighted, y_log, test_size=0.2, random_state=42
    )
    
    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        '線形回帰': LinearRegression(),
        'Ridge回帰': Ridge(alpha=1.0, random_state=42),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    for name, model in models.items():
        if 'Ridge' in name or '線形回帰' in name:
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
    
    return best_model, best_model_name, scaler, feature_cols, results, ml_df, feature_weights

# データ読み込みとモデル訓練
if data_loaded:
    # サイドバーに重み付け設定を追加
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 重み付け設定")
    
    weight_mode = st.sidebar.radio(
        "重み付けモード",
        ["自動最適化", "手動調整", "重み付けなし"],
        key="weight_mode",
        help="自動最適化: Lasso回帰で自動的に重要な特徴量を抽出\n手動調整: 各特徴量の重要度を手動で設定"
    )
    
    use_auto_weight = (weight_mode == "自動最適化")
    use_manual_weight = (weight_mode == "手動調整")
    
    manual_weights = None
    if use_manual_weight:
        st.sidebar.markdown("#### 特徴量の重み調整")
        st.sidebar.markdown("*重要度が高い項目の値を大きくしてください*")
        
        # 主要な特徴量のみ手動調整可能に
        key_features = ['打率', '本塁打', '打点', '出塁率', '長打率', 'タイトル数', '安打', '試合']
        manual_weights = {}
        
        for feature in key_features:
            manual_weights[feature] = st.sidebar.slider(
                feature,
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key=f"weight_{feature}"
            )
    
    # モデル訓練フラグの変更検知
    weight_changed = False
    if 'last_weight_mode' not in st.session_state:
        st.session_state.last_weight_mode = weight_mode
        weight_changed = True
    elif st.session_state.last_weight_mode != weight_mode:
        st.session_state.last_weight_mode = weight_mode
        weight_changed = True
    
    if not st.session_state.model_trained or weight_changed:
        with st.spinner('🤖 モデルを訓練中...'):
            merged_df, stats_all_with_titles, salary_long = prepare_data(
                salary_df, stats_2023, stats_2024, stats_2025, titles_df
            )
            
            best_model, best_model_name, scaler, feature_cols, results, ml_df, feature_weights = train_models(
                merged_df, 
                use_auto_weight=use_auto_weight,
                manual_weights=manual_weights if use_manual_weight else None
            )
            
            st.session_state.model_trained = True
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.stats_all_with_titles = stats_all_with_titles
            st.session_state.salary_long = salary_long
            st.session_state.results = results
            st.session_state.ml_df = ml_df
            st.session_state.feature_weights = feature_weights
            st.session_state.weight_mode = weight_mode
    
    # メインコンテンツ
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 機能選択")
    menu = st.sidebar.radio(
        "メニュー",
        ["🏠 ホーム", "🔍 選手検索・予測", "📊 複数選手比較", "📈 モデル性能", "📉 要因分析", "⚖️ 重み付け詳細"],
        key="main_menu",
        label_visibility="collapsed"
    )
    
    # ホーム
    if menu == "🏠 ホーム":
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.metric("訓練データ数", f"{len(st.session_state.ml_df)}人")
        with col2:
            st.metric("採用モデル", st.session_state.best_model_name)
        with col3:
            st.metric("R²スコア", f"{st.session_state.results[st.session_state.best_model_name]['R2']:.4f}")
        
        # 現在の重み付けモードを表示
        if st.session_state.weight_mode == "自動最適化":
            st.info("🤖 **重み付けモード**: 自動最適化（Lasso回帰による特徴量選択）")
        elif st.session_state.weight_mode == "手動調整":
            st.info("✋ **重み付けモード**: 手動調整")
        else:
            st.info("📊 **重み付けモード**: 重み付けなし（全特徴量を均等に使用）")
        
        st.subheader("📖 使い方")
        st.markdown("""
        1. **左サイドバー**のメニューから機能を選択
        2. **選手名**を入力して年俸を予測
        
        ### 機能一覧
        - 🔍 **選手検索・予測**: 個別選手の年俸予測とレーダーチャート
        - 📊 **複数選手比較**: 最大5人の選手を比較
        - 📈 **モデル性能**: 予測モデルの詳細情報
        - 📉 **要因分析**: 年俸に影響を与える要因の分析
        - ⚖️ **重み付け詳細**: 各特徴量の重要度を確認
        
        ### 🎯 重み付け機能
        - **自動最適化**: Lasso回帰で自動的に重要な特徴量を抽出
        - **手動調整**: 各特徴量の重要度を手動で設定
        - **重み付けなし**: 全特徴量を均等に使用
        
        ### ⚖️ NPB減額制限ルール
        - **1億円以上**: 最大40%まで減額可能（最低60%保証）
        - **1億円未満**: 最大25%まで減額可能（最低75%保証）
        """)
    
    # 選手検索・予測
    elif menu == "🔍 選手検索・予測":
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
            f"選手を選択してください ({len(filtered_players)}人)",
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
                    features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                    
                    # 重み付けを適用
                    weights = np.array([st.session_state.feature_weights.get(col, 1.0) for col in st.session_state.feature_cols])
                    features_weighted = features * weights
                    
                    # 予測（対数変換版）
                    if 'Ridge' in st.session_state.best_model_name or '線形回帰' in st.session_state.best_model_name:
                        features_scaled = st.session_state.scaler.transform(features_weighted)
                        predicted_salary_log = st.session_state.best_model.predict(features_scaled)[0]
                    else:
                        predicted_salary_log = st.session_state.best_model.predict(features_weighted)[0]
                    
                    predicted_salary = np.expm1(predicted_salary_log)
                    
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
                    
                    # 減額制限チェック
                    if previous_salary is not None:
                        is_limited, min_salary, reduction_rate = check_salary_reduction_limit(predicted_salary, previous_salary)
                        
                        if is_limited:
                            st.warning(f"""
                            ⚖️ **減額制限に引っかかります**
                            - 前年年俸: {previous_salary/1e6:.1f}百万円
                            - 予測年俸: {predicted_salary/1e6:.1f}百万円
                            - 減額制限: {reduction_rate*100:.0f}%まで（最低{(1-reduction_rate)*100:.0f}%保証）
                            - **制限後の最低年俸: {min_salary/1e6:.1f}百万円**
                            """)
                            display_salary = min_salary
                        else:
                            display_salary = predicted_salary
                    else:
                        display_salary = predicted_salary
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if previous_salary is not None:
                            st.metric("前年年俸", f"{previous_salary/1e6:.1f}百万円")
                        else:
                            st.metric("前年年俸", "データなし")
                    with col2:
                        st.metric("予測年俸", f"{predicted_salary/1e6:.1f}百万円")
                    with col3:
                        if actual_salary:
                            st.metric("実際の年俸", f"{actual_salary/1e6:.1f}百万円")
                        else:
                            st.metric("実際の年俸", "データなし")
                    with col4:
                        if actual_salary:
                            error = abs(display_salary - actual_salary) / actual_salary * 100
                            st.metric("予測誤差", f"{error:.1f}%")
                    
                    st.markdown("---")
                    st.subheader(f"{stats_year}年の成績")
                    
                    col1, col2, col3, col4 = st.columns(4)
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
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        player_salary_history = st.session_state.salary_long[
                            st.session_state.salary_long['選手名'] == selected_player
                        ].sort_values('年度')
                        
                        if not player_salary_history.empty:
                            years = player_salary_history['年度'].values
                            salaries = player_salary_history['年俸_円'].values / 1e6
                            ax1.plot(years, salaries, 'o-', linewidth=2, markersize=8, label='実際の年俸')
                            ax1.plot(predict_year, predicted_salary/1e6, 'r*', markersize=20, label='予測年俸（制限前）')
                            
                            if previous_salary is not None and is_limited:
                                ax1.plot(predict_year, display_salary/1e6, 'orange', marker='D', markersize=12, label='制限後年俸')
                            
                            if actual_salary:
                                ax1.plot(predict_year, actual_salary/1e6, 'go', markersize=12, label=f'実際の年俸({predict_year})')
                            
                            ax1.set_xlabel('年度', fontweight='bold')
                            ax1.set_ylabel('年俸（百万円）', fontweight='bold')
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
    elif menu == "📊 複数選手比較":
        st.header("📊 複数選手比較")
        
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        
        selected_players = st.multiselect(
            "比較する選手を選択してください（最大5人）",
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
                        features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        
                        # 重み付けを適用
                        weights = np.array([st.session_state.feature_weights.get(col, 1.0) for col in st.session_state.feature_cols])
                        features_weighted = features * weights
                        
                        # 予測（対数変換版）
                        if 'Ridge' in st.session_state.best_model_name or '線形回帰' in st.session_state.best_model_name:
                            features_scaled = st.session_state.scaler.transform(features_weighted)
                            predicted_salary_log = st.session_state.best_model.predict(features_scaled)[0]
                        else:
                            predicted_salary_log = st.session_state.best_model.predict(features_weighted)[0]
                        
                        predicted_salary = np.expm1(predicted_salary_log)
                        
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
                            '前年年俸': previous_salary / 1e6 if previous_salary else None,
                            '予測年俸（制限前）': predicted_salary / 1e6,
                            '予測年俸（制限後）': display_salary / 1e6,
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
                            st.write(f"- **{row['選手名']}**: 予測{row['予測年俸（制限前）']:.1f}百万円 → 制限後{row['予測年俸（制限後）']:.1f}百万円")
                    
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
                        ax1.set_xlabel('予測年俸（百万円）', fontweight='bold')
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
        else:
            st.info("👆 2人以上の選手を選択してください")
    
    # モデル性能
    elif menu == "📈 モデル性能":
        st.header("📈 モデル性能")
        
        model_data = []
        for name, result in st.session_state.results.items():
            model_data.append({
                'モデル': name,
                'MAE（百万円）': f"{result['MAE']/1e6:.2f}",
                'R²スコア': f"{result['R2']:.4f}"
            })
        
        df_models = pd.DataFrame(model_data).sort_values('R²スコア', ascending=False)
        st.dataframe(
            df_models,
            use_container_width=False,
            hide_index=True
        )
        st.success(f"🏆 最良モデル: {st.session_state.best_model_name}")
        st.info(f"🎯 重み付けモード: {st.session_state.weight_mode}")
        
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
        
        title_groups['mean'] = title_groups['mean'] / 1e6
        title_groups['median'] = title_groups['median'] / 1e6
        title_groups.index = ['タイトル無し', 'タイトル有り']
        title_groups.columns = ['選手数', '平均年俸（百万円）', '中央値（百万円）']
        
        st.dataframe(
            title_groups,
            use_container_width=False
        )
        
        if len(title_groups) == 2:
            diff = title_groups.loc['タイトル有り', '平均年俸（百万円）'] - title_groups.loc['タイトル無し', '平均年俸（百万円）']
            st.metric("タイトル獲得による年俸増加", f"{diff:.1f}百万円")
        
        st.markdown("---")
        st.subheader("主要指標との相関")
        
        correlations = st.session_state.ml_df[
            ['打率', '本塁打', '打点', '出塁率', '長打率', 'タイトル数', '年俸_円']
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.scatter(st.session_state.ml_df['打率'], st.session_state.ml_df['年俸_円']/1e6, alpha=0.5)
            ax1.set_xlabel('打率', fontweight='bold')
            ax1.set_ylabel('年俸（百万円）', fontweight='bold')
            ax1.set_title('打率と年俸の関係', fontweight='bold')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.scatter(st.session_state.ml_df['本塁打'], st.session_state.ml_df['年俸_円']/1e6, alpha=0.5, color='orange')
            ax2.set_xlabel('本塁打', fontweight='bold')
            ax2.set_ylabel('年俸（百万円）', fontweight='bold')
            ax2.set_title('本塁打と年俸の関係', fontweight='bold')
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)
    
    # 重み付け詳細
    elif menu == "⚖️ 重み付け詳細":
        st.header("⚖️ 特徴量の重み付け詳細")
        
        st.info(f"**現在の重み付けモード**: {st.session_state.weight_mode}")
        
        # 重み付けデータを表示
        weights_df = pd.DataFrame({
            '特徴量': list(st.session_state.feature_weights.keys()),
            '重み': list(st.session_state.feature_weights.values())
        }).sort_values('重み', ascending=False)
        
        st.subheader("全特徴量の重み")
        st.dataframe(
            weights_df,
            use_container_width=True,
            hide_index=True
        )
        
        # 重み付けTop 10を可視化
        st.markdown("---")
        st.subheader("重要度 Top 10")
        
        top_weights = weights_df.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_weights)), top_weights['重み'], color='#3498db', alpha=0.7)
        ax.set_yticks(range(len(top_weights)))
        ax.set_yticklabels(top_weights['特徴量'])
        ax.set_xlabel('重み', fontweight='bold')
        ax.set_title('特徴量の重み付け Top 10', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)
        
        # 重み付けモードの説明
        st.markdown("---")
        st.subheader("📖 重み付けモードについて")
        
        st.markdown("""
        ### 🤖 自動最適化
        Lasso回帰を使用して、年俸予測に最も影響を与える特徴量を自動的に抽出します。
        - **メリット**: データに基づいた客観的な重み付け
        - **用途**: 一般的な予測に最適
        
        ### ✋ 手動調整
        ユーザーが各特徴量の重要度を自由に設定できます。
        - **メリット**: ドメイン知識を反映可能
        - **用途**: 特定の指標を重視したい場合
        
        ### 📊 重み付けなし
        全ての特徴量を均等に扱います。
        - **メリット**: シンプルで解釈が容易
        - **用途**: ベースラインとしての比較
        """)

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
    - 📈 予測モデルの性能評価
    - 📉 年俸影響要因の分析
    - ⚖️ NPB減額制限ルールの適用
    - 🎯 特徴量の重み付け（自動・手動）
    """)

# フッター
st.markdown("---")
st.markdown("*NPB選手年俸予測システム（対数変換 + 減額制限 + 重み付け対応） - Powered by Streamlit*")
