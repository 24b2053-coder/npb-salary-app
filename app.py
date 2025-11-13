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
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="NPB選手年俸予測システム",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント設定
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams["font.family"] = "IPAexGothic"

# タイトル
st.title("⚾ NPB選手年俸予測システム")
st.markdown("---")

# サイドバー
st.sidebar.header("📁 データアップロード")
st.sidebar.markdown("以下の5つのCSVファイルをアップロードしてください：")

# ファイルアップロード部分を以下に変更
try:
    salary_df = pd.read_csv('data/salary_2023&2024&2025.csv')
    stats_2023 = pd.read_csv('data/stats_2023.csv')
    stats_2023 = pd.read_csv('data/stats_2024.csv')
    stats_2023 = pd.read_csv('data/stats_2025.csv')
    stats_2023 = pd.read_csv('data/titles_2023&2024&2025.csv')
    st.sidebar.success("✅ データ読み込み完了！")
except:
    # ファイルアップロード処理
    salary_file = st.sidebar.file_uploader("1. 年俸データ (salary_2023&2024&2025.csv)", type=['csv'])
    stats_2023_file = st.sidebar.file_uploader("2. 2023年成績 (stats_2023.csv)", type=['csv'])
    stats_2024_file = st.sidebar.file_uploader("3. 2024年成績 (stats_2024.csv)", type=['csv'])
    stats_2025_file = st.sidebar.file_uploader("4. 2025年成績 (stats_2025.csv)", type=['csv'])
    titles_file = st.sidebar.file_uploader("5. タイトルデータ (titles_2023&2024&2025.csv)", type=['csv'])


# セッション状態の初期化
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# データ読み込みとモデル訓練
if salary_file and stats_2023_file and stats_2024_file and stats_2025_file and titles_file:
    
    if not st.session_state.model_trained:
        with st.spinner('📊 データを読み込み中...'):
            # データ読み込み
            salary_df = pd.read_csv(salary_file)
            stats_2023 = pd.read_csv(stats_2023_file)
            stats_2024 = pd.read_csv(stats_2024_file)
            stats_2025 = pd.read_csv(stats_2025_file)
            titles_df = pd.read_csv(titles_file)
            
            st.sidebar.success("✅ データ読み込み完了！")
        
        with st.spinner('🤖 モデルを訓練中...'):
            # タイトルデータの前処理
            titles_df = titles_df.dropna(subset=['選手名'])
            title_summary = titles_df.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')
            
            # 成績データの統合
            stats_2023['年度'] = 2023
            stats_2024['年度'] = 2024
            stats_2025['年度'] = 2025
            stats_all = pd.concat([stats_2023, stats_2024, stats_2025], ignore_index=True)
            
            # 年俸データの整形
            df_2023 = salary_df[['選手名_2023', '年俸_円_2023']].copy()
            df_2023['年度'] = 2023
            df_2023.rename(columns={'選手名_2023': '選手名', '年俸_円_2023': '年俸_円'}, inplace=True)

            df_2024 = salary_df[['選手名_2024_2025', '年俸_円_2024']].copy()
            df_2024['年度'] = 2024
            df_2024.rename(columns={'選手名_2024_2025': '選手名', '年俸_円_2024': '年俸_円'}, inplace=True)

            df_2025 = salary_df[['選手名_2024_2025', '年俸_円_2025']].copy()
            df_2025['年度'] = 2025
            df_2025.rename(columns={'選手名_2024_2025': '選手名', '年俸_円_2025': '年俸_円'}, inplace=True)
            
            salary_long = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
            salary_long = salary_long.dropna(subset=['年俸_円'])
            salary_long = salary_long[salary_long['年俸_円'] > 0]
            salary_long = salary_long.sort_values('年俸_円', ascending=False)
            salary_long = salary_long.drop_duplicates(subset=['選手名', '年度'], keep='first')
            
            # データ結合
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
            
            # 予測用データ
            stats_all_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
            stats_all_with_titles['タイトル数'] = stats_all_with_titles['タイトル数'].fillna(0)
            
            # 特徴量選択
            feature_cols = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', 
                            '本塁打', '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', 
                            '三振', '併殺打', '打率', '出塁率', '長打率', '犠打', '犠飛',
                            'タイトル数']
            
            ml_df = merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
            ml_df = ml_df.dropna()
            
            X = ml_df[feature_cols]
            y = ml_df['年俸_円']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # モデル訓練
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
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'MAE': mae,
                    'R2': r2
                }
            
            best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
            best_model = results[best_model_name]['model']
            
            # セッション状態に保存
            st.session_state.model_trained = True
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.stats_all_with_titles = stats_all_with_titles
            st.session_state.salary_long = salary_long
            st.session_state.results = results
            st.session_state.ml_df = ml_df
            
            st.sidebar.success(f"✅ モデル訓練完了！\n採用モデル: {best_model_name}")
    
    # メインコンテンツ
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 機能選択")
    menu = st.sidebar.radio(
        "メニュー",
        ["🏠 ホーム", "🔍 選手検索・予測", "📊 複数選手比較", "📈 モデル性能", "📉 要因分析"]
    )
    
    # ホーム
    if menu == "🏠 ホーム":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("訓練データ数", f"{len(st.session_state.ml_df)}人")
        with col2:
            st.metric("採用モデル", st.session_state.best_model_name)
        with col3:
            st.metric("R²スコア", f"{st.session_state.results[st.session_state.best_model_name]['R2']:.4f}")
        
        st.markdown("---")
        st.subheader("📖 使い方")
        st.markdown("""
        1. **左サイドバー**から5つのCSVファイルをアップロード
        2. **メニュー**から機能を選択
        3. **選手名**を入力して年俸を予測
        
        ### 機能一覧
        - 🔍 **選手検索・予測**: 個別選手の年俸予測とレーダーチャート
        - 📊 **複数選手比較**: 最大5人の選手を比較
        - 📈 **モデル性能**: 予測モデルの詳細情報
        - 📉 **要因分析**: 年俸に影響を与える要因の分析
        """)
    
    # 選手検索・予測
    elif menu == "🔍 選手検索・予測":
        st.header("🔍 選手検索・予測")
        
        # 選手選択
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        
        selected_player = st.selectbox(
            "選手を選択してください",
            options=sorted(available_players),
            index=0
        )
        
        predict_year = st.slider("予測年度", 2024, 2026, 2025)
        
        if st.button("🎯 予測実行", type="primary"):
            stats_year = predict_year - 1
            player_stats = st.session_state.stats_all_with_titles[
                (st.session_state.stats_all_with_titles['選手名'] == selected_player) & 
                (st.session_state.stats_all_with_titles['年度'] == stats_year)
            ]
            
            if not player_stats.empty:
                player_stats = player_stats.iloc[0]
                features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                
                if st.session_state.best_model_name == '線形回帰':
                    features_scaled = st.session_state.scaler.transform(features)
                    predicted_salary = st.session_state.best_model.predict(features_scaled)[0]
                else:
                    predicted_salary = st.session_state.best_model.predict(features)[0]
                
                # 実際の年俸取得
                actual_salary_data = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == selected_player) & 
                    (st.session_state.salary_long['年度'] == predict_year)
                ]
                actual_salary = actual_salary_data['年俸_円'].values[0] if not actual_salary_data.empty else None
                
                # 結果表示
                st.success(f"✅ 予測完了！")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("予測年俸", f"{predicted_salary/1e6:.1f}百万円")
                with col2:
                    if actual_salary:
                        st.metric("実際の年俸", f"{actual_salary/1e6:.1f}百万円")
                    else:
                        st.metric("実際の年俸", "データなし")
                with col3:
                    if actual_salary:
                        error = abs(predicted_salary - actual_salary) / actual_salary * 100
                        st.metric("予測誤差", f"{error:.1f}%")
                
                # 成績表示
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
                
                # グラフ表示
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 年俸推移
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    player_salary_history = st.session_state.salary_long[
                        st.session_state.salary_long['選手名'] == selected_player
                    ].sort_values('年度')
                    
                    if not player_salary_history.empty:
                        years = player_salary_history['年度'].values
                        salaries = player_salary_history['年俸_円'].values / 1e6
                        
                        ax1.plot(years, salaries, 'o-', linewidth=2, markersize=8, label='実際の年俸')
                        ax1.plot(predict_year, predicted_salary/1e6, 'r*', markersize=20, label='予測年俸')
                        
                        if actual_salary:
                            ax1.plot(predict_year, actual_salary/1e6, 'go', markersize=12, label='実際の年俸(2025)')
                    
                    ax1.set_xlabel('年度', fontweight='bold')
                    ax1.set_ylabel('年俸（百万円）', fontweight='bold')
                    ax1.set_title(f'{selected_player} - 年俸推移', fontweight='bold')
                    ax1.grid(alpha=0.3)
                    ax1.legend()
                    st.pyplot(fig1)
                
                with col2:
                    # レーダーチャート
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
    
    # 複数選手比較
    elif menu == "📊 複数選手比較":
        st.header("📊 複数選手比較")
        
        available_players = st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique()
        
        selected_players = st.multiselect(
            "比較する選手を選択してください（最大5人）",
            options=sorted(available_players),
            max_selections=5
        )
        
        if len(selected_players) >= 2:
            if st.button("📊 比較実行", type="primary"):
                results_list = []
                
                for player in selected_players:
                    player_stats = st.session_state.stats_all_with_titles[
                        (st.session_state.stats_all_with_titles['選手名'] == player) & 
                        (st.session_state.stats_all_with_titles['年度'] == 2024)
                    ]
                    
                    if not player_stats.empty:
                        player_stats = player_stats.iloc[0]
                        features = player_stats[st.session_state.feature_cols].values.reshape(1, -1)
                        
                        if st.session_state.best_model_name == '線形回帰':
                            features_scaled = st.session_state.scaler.transform(features)
                            predicted_salary = st.session_state.best_model.predict(features_scaled)[0]
                        else:
                            predicted_salary = st.session_state.best_model.predict(features)[0]
                        
                        results_list.append({
                            '選手名': player,
                            '予測年俸': predicted_salary / 1e6,
                            '打率': player_stats['打率'],
                            '本塁打': int(player_stats['本塁打']),
                            '打点': int(player_stats['打点']),
                            'タイトル数': int(player_stats['タイトル数'])
                        })
                
                if results_list:
                    df_results = pd.DataFrame(results_list)
                    
                    # 比較表示
                    st.dataframe(df_results, use_container_width=True)
                    
                    # グラフ表示
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        ax1.barh(df_results['選手名'], df_results['予測年俸'], alpha=0.7, color='steelblue')
                        ax1.set_xlabel('予測年俸（百万円）', fontweight='bold')
                        ax1.set_title('予測年俸比較', fontweight='bold')
                        ax1.grid(axis='x', alpha=0.3)
                        st.pyplot(fig1)
                    
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
        else:
            st.info("👆 2人以上の選手を選択してください")
    
    # モデル性能
    elif menu == "📈 モデル性能":
        st.header("📈 モデル性能")
        
        # モデル比較表
        model_data = []
        for name, result in st.session_state.results.items():
            model_data.append({
                'モデル': name,
                'MAE（百万円）': f"{result['MAE']/1e6:.2f}",
                'R²スコア': f"{result['R2']:.4f}"
            })
        
        df_models = pd.DataFrame(model_data).sort_values('R²スコア', ascending=False)
        st.dataframe(df_models, use_container_width=True)
        
        st.success(f"🏆 最良モデル: {st.session_state.best_model_name}")
        
        # 特徴量重要度
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
    
    # 要因分析
    elif menu == "📉 要因分析":
        st.header("📉 要因分析")
        
        # タイトル獲得の影響
        st.subheader("タイトル獲得の影響")
        title_groups = st.session_state.ml_df.groupby(
            st.session_state.ml_df['タイトル数'] > 0
        )['年俸_円'].agg(['count', 'mean', 'median'])
        title_groups['mean'] = title_groups['mean'] / 1e6
        title_groups['median'] = title_groups['median'] / 1e6
        title_groups.index = ['タイトル無し', 'タイトル有り']
        title_groups.columns = ['選手数', '平均年俸（百万円）', '中央値（百万円）']
        
        st.dataframe(title_groups, use_container_width=True)
        
        if len(title_groups) == 2:
            diff = title_groups.loc['タイトル有り', '平均年俸（百万円）'] - title_groups.loc['タイトル無し', '平均年俸（百万円）']
            st.metric("タイトル獲得による年俸増加", f"{diff:.1f}百万円")
        
        # 相関分析
        st.markdown("---")
        st.subheader("主要指標との相関")
        
        correlations = st.session_state.ml_df[
            ['打率', '本塁打', '打点', '出塁率', '長打率', 'タイトル数', '年俸_円']
        ].corr()['年俸_円'].sort_values(ascending=False)
        
        corr_data = []
        for idx, val in correlations.items():
            if idx != '年俸_円':
                corr_data.append({'指標': idx, '相関係数': f"{val:.4f}"})
        
        st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
        
        # グラフ
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.scatter(st.session_state.ml_df['打率'], st.session_state.ml_df['年俸_円']/1e6, alpha=0.5)
            ax1.set_xlabel('打率', fontweight='bold')
            ax1.set_ylabel('年俸（百万円）', fontweight='bold')
            ax1.set_title('打率と年俸の関係', fontweight='bold')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.scatter(st.session_state.ml_df['本塁打'], st.session_state.ml_df['年俸_円']/1e6, alpha=0.5, color='orange')
            ax2.set_xlabel('本塁打', fontweight='bold')
            ax2.set_ylabel('年俸（百万円）', fontweight='bold')
            ax2.set_title('本塁打と年俸の関係', fontweight='bold')
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

else:
    # ファイル未アップロード時
    st.info("👈 左サイドバーから5つのCSVファイルをアップロードしてください")
    
    st.markdown("""
    ### 📁 必要なファイル
    1. `salary_2023&2024&2025.csv` - 年俸データ
    2. `stats_2023.csv` - 2023年成績
    3. `stats_2024.csv` - 2024年成績
    4. `stats_2025.csv` - 2025年成績
    5. `titles_2023&2024&2025.csv` - タイトルデータ
    
    ### 🚀 機能
    - ⚾ 選手個別の年俸予測
    - 📊 複数選手の比較分析
    - 📈 予測モデルの性能評価
    - 📉 年俸影響要因の分析
    """)

# フッター
st.markdown("---")




