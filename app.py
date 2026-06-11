import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 日本語フォント ──────────────────────────────────────────
try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass
plt.rcParams['axes.unicode_minus'] = False

# ── ページ設定 ───────────────────────────────────────────────
st.set_page_config(page_title="NPB選手年俸予測", page_icon="⚾", layout="wide")
st.title("⚾ NPB選手年俸予測システム")
st.markdown("---")

# ════════════════════════════════════════════════════════════
# データ読み込み＆前処理
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    salary_df = pd.read_csv('data/salary_2023&2024&2025.csv')
    s23       = pd.read_csv('data/stats_2023.csv')
    s24       = pd.read_csv('data/stats_2024.csv')
    s25       = pd.read_csv('data/stats_2025.csv')
    titles_df = pd.read_csv('data/titles_2023&2024&2025.csv')
    for df in [salary_df, s23, s24, s25, titles_df]:
        df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return _process(salary_df, s23, s24, s25, titles_df)

# ════════════════════════════════════════════════════════════
# モデル訓練
# ════════════════════════════════════════════════════════════
FEATURE_COLS = ['試合','打席','打数','得点','安打','二塁打','三塁打',
                '本塁打','塁打','打点','盗塁','盗塁刺','四球','死球',
                '三振','併殺打','打率','出塁率','長打率','犠打','犠飛','タイトル数']

@st.cache_resource
def train_models(merged):
    ml = merged[FEATURE_COLS + ['年俸_円']].dropna()
    X, y = ml[FEATURE_COLS], ml['年俸_円']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(Xtr)
    Xte_sc   = scaler.transform(Xte)

    model_defs = {
        '線形回帰':          LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    }
    res = {}
    for name, mdl in model_defs.items():
        if name == '線形回帰':
            mdl.fit(Xtr_sc, ytr); pred = mdl.predict(Xte_sc)
        else:
            mdl.fit(Xtr, ytr);    pred = mdl.predict(Xte)
        res[name] = {'model': mdl,
                     'MAE':  mean_absolute_error(yte, pred),
                     'RMSE': np.sqrt(mean_squared_error(yte, pred)),
                     'R2':   r2_score(yte, pred)}

    best = max(res, key=lambda k: res[k]['R2'])
    return res, best, res[best]['model'], scaler, ml

# ════════════════════════════════════════════════════════════
# データ読み込み実行
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data_from_uploads(sal, s23, s24, s25, ttl):
    """アップロードされたファイルからデータを読み込む"""
    salary_df = pd.read_csv(sal)
    s23_df    = pd.read_csv(s23)
    s24_df    = pd.read_csv(s24)
    s25_df    = pd.read_csv(s25)
    titles_df = pd.read_csv(ttl)
    for df in [salary_df, s23_df, s24_df, s25_df, titles_df]:
        df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return _process(salary_df, s23_df, s24_df, s25_df, titles_df)

def _process(salary_df, s23, s24, s25, titles_df):
    """前処理の共通処理"""
    titles_df = titles_df.dropna(subset=['選手名'])
    title_summary = (titles_df.groupby(['選手名','年度'])
                               .size().reset_index(name='タイトル数'))
    s23['年度'], s24['年度'], s25['年度'] = 2023, 2024, 2025
    stats_all = pd.concat([s23, s24, s25], ignore_index=True)

    frames = []
    for col, yr in [('年俸_円_2023',2023),('年俸_円_2024',2024),('年俸_円_2025',2025)]:
        if col in salary_df.columns:
            d = salary_df[['選手名', col]].copy()
            d.columns = ['選手名','年俸_円']; d['年度'] = yr
            frames.append(d)
    salary_long = pd.concat(frames, ignore_index=True)
    salary_long = salary_long.dropna(subset=['年俸_円'])
    salary_long = salary_long[salary_long['年俸_円'] > 0]
    salary_long = (salary_long.sort_values('年俸_円', ascending=False)
                               .drop_duplicates(subset=['選手名','年度'], keep='first'))

    stats_all['予測年度'] = stats_all['年度'] + 1
    merged = pd.merge(stats_all, title_summary, on=['選手名','年度'], how='left')
    merged['タイトル数'] = merged['タイトル数'].fillna(0)
    merged = pd.merge(merged, salary_long,
                      left_on=['選手名','予測年度'],
                      right_on=['選手名','年度'],
                      suffixes=('_成績','_年俸'))
    merged = merged.drop(columns=['年度_年俸','予測年度'])
    merged.rename(columns={'年度_成績':'成績年度'}, inplace=True)

    stats_wt = pd.merge(stats_all, title_summary, on=['選手名','年度'], how='left')
    stats_wt['タイトル数'] = stats_wt['タイトル数'].fillna(0)
    return merged, stats_wt, salary_long

# ── データ読み込み：data/ フォルダ → なければアップロード ──
data_ready = False
try:
    merged, stats_wt, salary_long = load_data()
    data_ready = True
except FileNotFoundError:
    st.sidebar.markdown("### 📁 CSVファイルをアップロード")
    st.sidebar.caption("data/ フォルダが見つからないため手動アップロードが必要です")
    sal = st.sidebar.file_uploader("salary_2023&2024&2025.csv", type="csv", key="sal")
    s23 = st.sidebar.file_uploader("stats_2023.csv",            type="csv", key="s23")
    s24 = st.sidebar.file_uploader("stats_2024.csv",            type="csv", key="s24")
    s25 = st.sidebar.file_uploader("stats_2025.csv",            type="csv", key="s25")
    ttl = st.sidebar.file_uploader("titles_2023&2024&2025.csv", type="csv", key="ttl")

    if all([sal, s23, s24, s25, ttl]):
        merged, stats_wt, salary_long = load_data_from_uploads(sal, s23, s24, s25, ttl)
        data_ready = True
    else:
        uploaded = sum(1 for f in [sal,s23,s24,s25,ttl] if f)
        st.info(f"📁 {uploaded}/5 ファイルがアップロードされています。5つ全てアップロードしてください。")
        st.stop()

results, best_name, best_model, scaler, ml_df = train_models(merged)

# ════════════════════════════════════════════════════════════
# 予測ヘルパー
# ════════════════════════════════════════════════════════════
def predict(player_name, year=2025):
    sy  = year - 1
    row = stats_wt[(stats_wt['選手名'] == player_name) & (stats_wt['年度'] == sy)]
    if row.empty:
        return None
    row  = row.iloc[0]
    feat = row[FEATURE_COLS].values.reshape(1, -1)

    if best_name == '線形回帰':
        pred_sal = best_model.predict(scaler.transform(feat))[0]
    else:
        pred_sal = best_model.predict(feat)[0]

    act_row = salary_long[(salary_long['選手名'] == player_name) &
                          (salary_long['年度'] == year)]
    actual  = act_row['年俸_円'].values[0] if not act_row.empty else None
    return {'name': player_name, 'sy': sy, 'yr': year,
            'pred': pred_sal, 'actual': actual, 'stats': row}

# ════════════════════════════════════════════════════════════
# サイドバーメニュー
# ════════════════════════════════════════════════════════════
menu = st.sidebar.radio("🎯 メニュー", [
    "🏠 ホーム",
    "🔍 選手予測",
    "📊 選手比較",
    "📈 モデル性能",
    "📉 要因分析",
])

# ════════════════════════════════════════════════════════════
# 🏠 ホーム
# ════════════════════════════════════════════════════════════
if menu == "🏠 ホーム":
    c1, c2, c3 = st.columns(3)
    c1.metric("採用モデル",  best_name)
    c2.metric("R²スコア",   f"{results[best_name]['R2']:.4f}")
    c3.metric("平均誤差",   f"{results[best_name]['MAE']/1e6:.1f}百万円")

    st.markdown("---")
    st.subheader("📖 使い方")
    st.markdown("""
    | メニュー | 内容 |
    |---|---|
    | 🔍 選手予測 | 選手名を選んで年俸を予測、年俸推移・レーダーチャート表示 |
    | 📊 選手比較 | 最大5人を並べて比較 |
    | 📈 モデル性能 | 3モデルの精度比較・特徴量重要度 |
    | 📉 要因分析 | 打率・本塁打・タイトルと年俸の関係 |
    """)

# ════════════════════════════════════════════════════════════
# 🔍 選手予測
# ════════════════════════════════════════════════════════════
elif menu == "🔍 選手予測":
    st.header("🔍 選手検索・年俸予測")

    players = sorted(stats_wt[stats_wt['年度'] == 2024]['選手名'].unique())
    kw = st.text_input("絞り込み検索", placeholder="例: 村上、岡本")
    filtered = [p for p in players if kw in p] if kw else players
    sel = st.selectbox("選手を選択", filtered)
    yr  = st.slider("予測年度", 2024, 2026, 2025)

    if st.button("🎯 予測実行", type="primary"):
        r = predict(sel, yr)
        if r is None:
            st.error(f"❌ {sel} の {yr-1} 年成績が見つかりません")
        else:
            st.success("✅ 予測完了！")
            c1, c2, c3 = st.columns(3)
            c1.metric("予測年俸", f"{r['pred']/1e6:.1f}百万円")
            if r['actual']:
                c2.metric("実際の年俸", f"{r['actual']/1e6:.1f}百万円")
                c3.metric("予測誤差",   f"{abs(r['pred']-r['actual'])/r['actual']*100:.1f}%")

            st.markdown("---")
            st.subheader(f"{r['sy']}年の成績")
            s = r['stats']
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("試合",  int(s['試合']));  c1.metric("打率",  f"{s['打率']:.3f}")
            c2.metric("安打",  int(s['安打']));  c2.metric("出塁率", f"{s['出塁率']:.3f}")
            c3.metric("本塁打", int(s['本塁打'])); c3.metric("長打率", f"{s['長打率']:.3f}")
            c4.metric("打点",  int(s['打点']));  c4.metric("タイトル", int(s['タイトル数']))

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            # ① 年俸推移
            with col1:
                hist = salary_long[salary_long['選手名'] == sel].sort_values('年度')
                fig, ax = plt.subplots(figsize=(5, 4))
                if not hist.empty:
                    ax.plot(hist['年度'], hist['年俸_円']/1e6, 'o-', lw=2, ms=8, label='実績')
                ax.plot(yr, r['pred']/1e6, 'r*', ms=18, label='予測')
                if r['actual']:
                    ax.plot(yr, r['actual']/1e6, 'go', ms=10, label=f'実際({yr})')
                ax.set_xlabel('年度'); ax.set_ylabel('年俸（百万円）')
                ax.set_title(f'{sel} 年俸推移'); ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)

            # ② レーダーチャート
            with col2:
                radar = {
                    '打率':  s['打率']/0.4,
                    '出塁率': s['出塁率']/0.5,
                    '長打率': s['長打率']/0.7,
                    '本塁打': min(s['本塁打']/40, 1.0),
                    '打点':  min(s['打点']/100,  1.0),
                    '盗塁':  min(s['盗塁']/40,   1.0),
                }
                cats = list(radar.keys())
                vals = list(radar.values()) + [list(radar.values())[0]]
                angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(projection='polar'))
                ax.plot(angles, vals, 'o-', lw=2, color='#2E86AB')
                ax.fill(angles, vals, alpha=0.25, color='#2E86AB')
                ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats)
                ax.set_ylim(0, 1)
                ax.set_title(f'{sel}\n成績レーダー({r["sy"]})', pad=15)
                st.pyplot(fig); plt.close(fig)

            # ③ リーグ内分布
            with col3:
                preds = []
                for _, row in stats_wt[stats_wt['年度'] == r['sy']].head(50).iterrows():
                    try:
                        ft = row[FEATURE_COLS].values.reshape(1, -1)
                        p  = (best_model.predict(scaler.transform(ft))[0]
                              if best_name == '線形回帰'
                              else best_model.predict(ft)[0])
                        preds.append(p/1e6)
                    except Exception:
                        pass
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(preds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(r['pred']/1e6, color='red', lw=2, ls='--', label=sel)
                ax.set_xlabel('予測年俸（百万円）'); ax.set_ylabel('選手数')
                ax.set_title('リーグ内年俸分布'); ax.legend(); ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════
# 📊 選手比較
# ════════════════════════════════════════════════════════════
elif menu == "📊 選手比較":
    st.header("📊 複数選手比較")
    players = sorted(stats_wt[stats_wt['年度'] == 2024]['選手名'].unique())
    sels    = st.multiselect("比較する選手（2〜5人）", players, max_selections=5)

    if len(sels) >= 2 and st.button("📊 比較実行", type="primary"):
        rows = []
        for p in sels:
            r = predict(p, 2025)
            if r:
                rows.append({'選手名': p,
                             '予測年俸(百万円)': round(r['pred']/1e6, 1),
                             '実際の年俸': round(r['actual']/1e6,1) if r['actual'] else None,
                             '打率': r['stats']['打率'],
                             '本塁打': int(r['stats']['本塁打']),
                             '打点': int(r['stats']['打点']),
                             'タイトル数': int(r['stats']['タイトル数'])})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(df)); w = 0.35
            ax.bar(x - w/2, df['予測年俸(百万円)'], w, label='予測', alpha=0.8, color='skyblue')
            if df['実際の年俸'].notna().any():
                ax.bar(x + w/2, df['実際の年俸'].fillna(0), w, label='実際', alpha=0.8, color='orange')
            ax.set_xticks(x); ax.set_xticklabels(df['選手名'], rotation=30, ha='right')
            ax.set_ylabel('年俸（百万円）'); ax.set_title('年俸比較')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(df)); w = 0.2
            ax.bar(x-w*1.5, df['打率']*100, w, label='打率×100', alpha=0.8)
            ax.bar(x-w*0.5, df['本塁打'],   w, label='本塁打',   alpha=0.8)
            ax.bar(x+w*0.5, df['打点']/10,  w, label='打点÷10',  alpha=0.8)
            ax.bar(x+w*1.5, df['タイトル数']*10, w, label='タイトル×10', alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(df['選手名'], rotation=30, ha='right')
            ax.set_title('成績比較'); ax.legend(); ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════
# 📈 モデル性能
# ════════════════════════════════════════════════════════════
elif menu == "📈 モデル性能":
    st.header("📈 モデル性能")
    rows = [{'モデル': n, 'MAE（百万円）': f"{v['MAE']/1e6:.2f}",
             'RMSE（百万円）': f"{v['RMSE']/1e6:.2f}", 'R²': f"{v['R2']:.4f}"}
            for n, v in results.items()]
    st.dataframe(pd.DataFrame(rows).sort_values('R²', ascending=False),
                 use_container_width=False, hide_index=True)
    st.success(f"🏆 採用モデル: **{best_name}**　R²={results[best_name]['R2']:.4f}")

    if best_name == 'ランダムフォレスト':
        st.markdown("---")
        st.subheader("特徴量重要度 Top 10")
        fi = (pd.DataFrame({'特徴量': FEATURE_COLS,
                            '重要度': best_model.feature_importances_})
               .sort_values('重要度', ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(fi)), fi['重要度'], color='#9b59b6', alpha=0.7)
        ax.set_yticks(range(len(fi))); ax.set_yticklabels(fi['特徴量'])
        ax.set_xlabel('重要度'); ax.set_title('特徴量重要度 Top 10')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig); plt.close(fig)

# ════════════════════════════════════════════════════════════
# 📉 要因分析
# ════════════════════════════════════════════════════════════
elif menu == "📉 要因分析":
    st.header("📉 年俸影響要因分析")

    tg = ml_df.groupby(ml_df['タイトル数'] > 0)['年俸_円'].agg(['count','mean','median'])
    tg['mean']   /= 1e6; tg['median'] /= 1e6
    tg.index   = ['タイトル無し','タイトル有り']
    tg.columns = ['選手数','平均年俸(百万円)','中央値(百万円)']
    st.subheader("タイトル獲得の影響")
    st.dataframe(tg, use_container_width=False)
    diff = tg.loc['タイトル有り','平均年俸(百万円)'] - tg.loc['タイトル無し','平均年俸(百万円)']
    st.metric("タイトル獲得による平均年俸増加", f"{diff:.1f}百万円")

    st.markdown("---")
    st.subheader("主要指標との相関")
    corr = (ml_df[['打率','本塁打','打点','出塁率','長打率','タイトル数','年俸_円']]
            .corr()['年俸_円'].sort_values(ascending=False).drop('年俸_円'))
    st.dataframe(corr.rename('相関係数').reset_index().rename(columns={'index':'指標'}),
                 use_container_width=False, hide_index=True)

    st.markdown("---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].scatter(ml_df['打率'],   ml_df['年俸_円']/1e6, alpha=0.5)
    axes[0,0].set(xlabel='打率', ylabel='年俸（百万円）', title='打率 vs 年俸')
    axes[0,0].grid(alpha=0.3)

    axes[0,1].scatter(ml_df['本塁打'], ml_df['年俸_円']/1e6, alpha=0.5, color='orange')
    axes[0,1].set(xlabel='本塁打', ylabel='年俸（百万円）', title='本塁打 vs 年俸')
    axes[0,1].grid(alpha=0.3)

    tsd = ml_df.groupby('タイトル数')['年俸_円'].apply(list)
    axes[1,0].boxplot([tsd.get(i, []) for i in range(int(ml_df['タイトル数'].max())+1)],
                      labels=range(int(ml_df['タイトル数'].max())+1))
    axes[1,0].set(xlabel='タイトル数', ylabel='年俸（円）', title='タイトル数 vs 年俸')
    axes[1,0].grid(alpha=0.3)

    axes[1,1].hist(ml_df['年俸_円']/1e6, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].axvline(ml_df['年俸_円'].mean()/1e6,   color='red',  ls='--', lw=2,
                      label=f"平均: {ml_df['年俸_円'].mean()/1e6:.0f}M")
    axes[1,1].axvline(ml_df['年俸_円'].median()/1e6, color='blue', ls='--', lw=2,
                      label=f"中央値: {ml_df['年俸_円'].median()/1e6:.0f}M")
    axes[1,1].set(xlabel='年俸（百万円）', ylabel='人数', title='年俸分布')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── フッター ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("*NPB選手年俸予測システム - Powered by Streamlit ⚾*")
