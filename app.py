import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="投手データ診断ツール", page_icon="🔍", layout="wide")
st.title("🔍 投手データ診断ツール")
st.markdown("このツールで投手CSVと年俸CSVの選手名がどうなっているか確認できます。")

def normalize_name(name):
    if pd.isna(name):
        return name
    return str(name).replace('\u3000', '').replace(' ', '').replace('\u00a0', '').strip()

col1, col2 = st.columns(2)

with col1:
    st.subheader("① 投手成績CSV (npb_pitcher_stats.csv)")
    pitcher_file = st.file_uploader("投手CSVをアップロード", type=['csv'], key="pitcher")

with col2:
    st.subheader("② 年俸CSV (salary_2023&2024&2025.csv)")
    salary_file = st.file_uploader("年俸CSVをアップロード", type=['csv'], key="salary")

if pitcher_file and salary_file:
    pitcher_df = pd.read_csv(pitcher_file)
    salary_df  = pd.read_csv(salary_file)

    # BOMクリーニング
    pitcher_df.columns = [c.lstrip('\ufeff').strip() for c in pitcher_df.columns]
    salary_df.columns  = [c.lstrip('\ufeff').strip() for c in salary_df.columns]

    st.markdown("---")
    st.subheader("📋 投手CSV カラム一覧")
    st.write(list(pitcher_df.columns))

    st.subheader("📋 年俸CSV カラム一覧")
    st.write(list(salary_df.columns))

    st.markdown("---")
    st.subheader("🔎 投手CSV 選手名サンプル（先頭10件）")
    if '選手名' in pitcher_df.columns:
        sample = pitcher_df['選手名'].head(10).tolist()
        for name in sample:
            encoded = repr(name)
            st.code(f"{name}  →  {encoded}")
    else:
        st.error("'選手名'列が見つかりません。列名を確認してください。")

    st.subheader("🔎 年俸CSV 選手名サンプル（先頭10件）")
    name_cols = [c for c in salary_df.columns if '選手名' in c]
    st.write(f"選手名関連列: {name_cols}")
    for col in name_cols[:2]:
        st.markdown(f"**{col}**")
        for name in salary_df[col].dropna().head(10).tolist():
            encoded = repr(name)
            st.code(f"{name}  →  {encoded}")

    st.markdown("---")
    st.subheader("🔗 正規化後のマージ結果")

    # 投手の選手名を正規化
    pitcher_df['選手名_normalized'] = pitcher_df['選手名'].apply(normalize_name)
    pitcher_2024 = pitcher_df[pitcher_df['年度'] == 2024]['選手名_normalized'].unique() if '年度' in pitcher_df.columns else []
    st.write(f"投手CSV 2024年選手数: {len(pitcher_2024)}")

    # 年俸の選手名を正規化（salary_longを作成）
    rows = []
    for _, row in salary_df.iterrows():
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

    salary_long = pd.DataFrame(rows).drop_duplicates(subset=['選手名', '年度'], keep='first')
    salary_2025 = salary_long[salary_long['年度'] == 2025]['選手名'].unique()
    st.write(f"年俸CSV 2025年選手数: {len(salary_2025)}")

    # マージテスト
    pitcher_df['選手名'] = pitcher_df['選手名_normalized']
    pitcher_df['予測年度'] = pitcher_df['年度'] + 1
    merged = pd.merge(
        pitcher_df,
        salary_long,
        left_on=['選手名', '予測年度'],
        right_on=['選手名', '年度'],
        suffixes=('_成績', '_年俸')
    )
    st.metric("マージ後のレコード数", len(merged))

    if len(merged) == 0:
        st.error("❌ マージ結果が0件です！")
        st.subheader("投手CSV 正規化後 選手名（2024年）")
        st.write(sorted(pitcher_df[pitcher_df['年度'] == 2024]['選手名'].unique())[:20])
        st.subheader("年俸CSV 正規化後 選手名（2025年）")
        st.write(sorted(salary_2025)[:20])
    else:
        st.success(f"✅ {len(merged)}件マージ成功！")
        st.dataframe(merged[['選手名', '年度_成績', '年俸_円']].head(10))
else:
    st.info("👆 両方のファイルをアップロードしてください")
