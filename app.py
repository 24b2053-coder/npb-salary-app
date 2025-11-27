import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 基本設定
# ===========================
st.set_page_config(page_title="NPB選手年俸予測システム", page_icon="⚾", layout="centered")

try:
    import japanize_matplotlib
    plt.rcParams["font.family"] = "IPAexGothic"
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial Unicode MS','sans-serif']


# ===========================
# CSV読み込み（merged版）
# ===========================
@st.cache_data
def load_data():
    try:
        merged_df = pd.read_csv("data/merged_stats_salary_age.csv")
        titles_df = pd.read_csv("data/titles_2023&2024&2025.csv")
        return merged_df, titles_df, True
    except FileNotFoundError:
        return None, None, False


merged_df, titles_df, data_loaded = load_data()


# ===========================
# 年俸減額制限
# ===========================
def calculate_salary_limit(previous_salary):
    if previous_salary >= 100_000_000:
        min_salary = previous_salary * 0.60
        reduction_rate = 0.40
    else:
        min_salary = previous_salary * 0.75
        reduction_rate = 0.25
    return min_salary, reduction_rate


def check_salary_reduction_limit(predicted, previous):
    min_salary, reduction_rate = calculate_salary_limit(previous)
    return predicted < min_salary, min_salary, reduction_rate


# ===========================
# データ前処理（merged にタイトル列追加）
# ===========================
@st.cache_data
def prepare_data(merged_df, titles_df):

    # タイトル集計
    titles_df_clean = titles_df.dropna(subset=['選手名'])
    title_summary = titles_df_clean.groupby(
        ['選手名', '年度']
    ).size().reset_index(name='タイトル数')

    # mergedと結合
    merged_df = pd.merge(
        merged_df,
        title_summary,
        on=['選手名', '年度'],
        how='left'
    )
    merged_df['タイトル数'] = merged_df['タイトル数'].fillna(0)

    # stats_all_with_titles としてそのまま利用
    stats_all_with_titles = merged_df.copy()

    # salary_long を merged から生成
    salary_long = merged_df[['選手名', '年度', '年俸_円']].dropna()

    return merged_df, stats_all_with_titles, salary_long


# ===========================
# モデル訓練（対数変換）
# ===========================
@st.cache_resource
def train_models(merged_df):

    feature_cols = [
        '試合','打席','打数','得点','安打','二塁打','三塁打','本塁打',
        '塁打','打点','盗塁','盗塁刺','四球','死球','三振','併殺打',
        '打率','出塁率','長打率','犠打','犠飛','タイトル数','年齢'
    ]

    ml_df = merged_df.copy()
    ml_df = ml_df.dropna(subset=feature_cols + ['年俸_円'])

    X = ml_df[feature_cols]
    y = ml_df['年俸_円']

    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        '線形回帰': LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=200, max_depth=5)
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
        y_test_original = np.expm1(y_test_log)

        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        results[name] = {'model': model, 'MAE': mae, 'R2': r2}

    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_model = results[best_model_name]['model']

    return best_model, best_model_name, scaler, feature_cols, results, ml_df


# ===========================
# モデルの準備
# ===========================
if data_loaded:

    merged_df, stats_all_with_titles, salary_long = prepare_data(merged_df, titles_df)

    best_model, best_model_name, scaler, feature_cols, results, ml_df = train_models(merged_df)


# ===========================
# UI
# ===========================
st.title("⚾ NPB選手年俸予測システム（Merged版）")
st.markdown("---")

menu = st.sidebar.radio(
    "メニューを選択",
    ["🏠 ホーム","🔍 選手検索・予測","📊 複数選手比較","📈 モデル性能","📉 要因分析"]
)


# ===========================
# ホーム
# ===========================
if menu == "🏠 ホーム":

    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("データ数", len(ml_df))
    with col2:
        st.metric("モデル", best_model_name)
    with col3:
        st.metric("R²", f"{results[best_model_name]['R2']:.4f}")

    st.info("merged CSV に対応した最新版です")


# ===========================
# 🔍 選手検索・予測
# ===========================
elif menu == "🔍 選手検索・予測":

    st.header("🔍 選手検索・予測")

    available_players = stats_all_with_titles['選手名'].unique()
    available_players = sorted(available_players)

    selected_player = st.selectbox("選手名を選択", available_players)

    predict_year = st.slider("予測年度", 2024, 2026, 2025)

    if st.button("予測実行"):

        stats_year = predict_year - 1
        
        player_stats = stats_all_with_titles[
            (stats_all_with_titles['選手名']==selected_player) &
            (stats_all_with_titles['年度']==stats_year)
        ]

        if player_stats.empty:
            st.error(f"{stats_year}年のデータなし")
        else:
            row = player_stats.iloc[0]
            features = row[feature_cols].values.reshape(1,-1)

            if best_model_name=="線形回帰":
                features_scaled = scaler.transform(features)
                pred_log = best_model.predict(features_scaled)[0]
            else:
                pred_log = best_model.predict(features)[0]

            predicted_salary = np.expm1(pred_log)

            # 前年年俸
            ps = salary_long[
                (salary_long['選手名']==selected_player)&
                (salary_long['年度']==stats_year)
            ]
            previous_salary = ps['年俸_円'].values[0] if not ps.empty else None

            # 減額制限処理
            display_salary = predicted_salary
            if previous_salary is not None:
                is_limit, min_salary, rate = check_salary_reduction_limit(predicted_salary, previous_salary)
                if is_limit:
                    display_salary = min_salary
                    st.warning(f"⚠️ 減額制限により {min_salary/1e6:.1f}百万円 に調整")

            st.success("予測完了！")

            col1, col2 = st.columns(2)
            col1.metric("予測年俸", f"{display_salary/1e6:.1f}百万円")
            if previous_salary:
                col2.metric("前年年俸", f"{previous_salary/1e6:.1f}百万円")

            # レーダー
            st.markdown("---")
            st.subheader(f"{stats_year}年 成績レーダー")

            radar_stats = {
                '打率': row['打率']/0.35,
                '出塁率': row['出塁率']/0.45,
                '長打率': row['長打率']/0.60,
                '本塁打': min(row['本塁打']/40,1),
                '打点': min(row['打点']/100,1),
                '盗塁': min(row['盗塁']/40,1)
            }

            categories = list(radar_stats.keys())
            values = list(radar_stats.values()) + [list(radar_stats.values())[0]]

            angles = np.linspace(0,2*np.pi,len(categories),endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values,'o-', linewidth=2)
            ax.fill(angles, values,alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            st.pyplot(fig)
            plt.close(fig)


# ===========================
# 📊 複数選手比較
# ===========================
elif menu == "📊 複数選手比較":

    st.header("📊 複数選手比較")

    available_players = sorted(stats_all_with_titles['選手名'].unique())

    selected_players = st.multiselect("比較選手（最大5）", available_players)

    if len(selected_players)>=2 and st.button("比較実行"):

        results_list = []

        for p in selected_players:
            row = stats_all_with_titles[
                (stats_all_with_titles['選手名']==p)&
                (stats_all_with_titles['年度']==2024)
            ]
            if row.empty: continue
            row = row.iloc[0]

            features = row[feature_cols].values.reshape(1,-1)

            if best_model_name=="線形回帰":
                pred_log = best_model.predict(scaler.transform(features))[0]
            else:
                pred_log = best_model.predict(features)[0]

            pred = np.expm1(pred_log)

            prev = salary_long[
                (salary_long['選手名']==p)&(salary_long['年度']==2024)
            ]
            prev_salary = prev['年俸_円'].values[0] if not prev.empty else None

            disp = pred
            is_limit=False
            if prev_salary is not None:
                is_limit, ms, rate = check_salary_reduction_limit(pred, prev_salary)
                if is_limit:
                    disp = ms

            results_list.append({
                "選手名":p,
                "予測（制限前）": pred/1e6,
                "予測（制限後）": disp/1e6,
                "前年年俸": prev_salary/1e6 if prev_salary else None,
                "本塁打":row["本塁打"],
                "打点":row["打点"],
                "打率":row["打率"],
            })

        df_results = pd.DataFrame(results_list)
        st.dataframe(df_results, use_container_width=True)


# ===========================
# 📈 モデル性能
# ===========================
elif menu == "📈 モデル性能":

    st.header("📈 モデル性能")

    model_rows = []
    for name, res in results.items():
        model_rows.append({
            "モデル": name,
            "MAE（百万円）": res["MAE"]/1e6,
            "R²": res["R2"],
        })

    st.dataframe(pd.DataFrame(model_rows))


# ===========================
# 📉 要因分析
# ===========================
elif menu == "📉 要因分析":

    st.header("📉 要因分析")

    corr = ml_df[['打率','本塁打','打点','出塁率','長打率','タイトル数','年齢','年俸_円']].corr()['年俸_円']
    st.write(corr)


# ===========================
# 終わり
# ===========================
st.markdown("---")
st.caption("NPB 年俸予測（merged CSV 対応版）")
