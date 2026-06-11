# ============================================================
# セル1: ライブラリのインポートと設定
# ============================================================

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

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("NPB選手年俸予測システム（完全版）")
print("=" * 60)
print("\nライブラリのインポート完了！")


# ============================================================
# セル2: データのアップロードと読み込み
# ============================================================

from google.colab import files

print("\n[1] データをアップロードしてください\n")
print("以下の5つのファイルをアップロードしてください：")
print("  1. salary_2023&2024&2025.csv")
print("  2. stats_2023.csv")
print("  3. stats_2024.csv")
print("  4. stats_2025.csv")
print("  5. titles_2023&2024&2025.csv")
print("\n" + "=" * 60)

uploaded = files.upload()

print("\nデータ読み込み中...")

# データ読み込み
salary_df = pd.read_csv('salary_2023&2024&2025.csv')
stats_2023 = pd.read_csv('stats_2023.csv')
stats_2024 = pd.read_csv('stats_2024.csv')
stats_2025 = pd.read_csv('stats_2025.csv')
titles_df = pd.read_csv('titles_2023&2024&2025.csv')

print(f"\n✓ 年俸データ: {salary_df.shape[0]}行 × {salary_df.shape[1]}列")
print(f"✓ 2023年成績: {stats_2023.shape[0]}行 × {stats_2023.shape[1]}列")
print(f"✓ 2024年成績: {stats_2024.shape[0]}行 × {stats_2024.shape[1]}列")
print(f"✓ 2025年成績: {stats_2025.shape[0]}行 × {stats_2025.shape[1]}列")
print(f"✓ タイトルデータ: {titles_df.shape[0]}行 × {titles_df.shape[1]}列")

print("\n" + "=" * 60)
print("データの読み込み完了！")


# ============================================================
# セル3: タイトルデータの前処理
# ============================================================

print("\n[2] タイトルデータを前処理中...")

titles_df = titles_df.dropna(subset=['選手名'])
title_summary = titles_df.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')
print(f"\n✓ タイトル数集計: {title_summary.shape}")


# ============================================================
# セル4: データの前処理と結合
# ============================================================

print("\n[3] データを前処理・結合中...")

stats_2023['年度'] = 2023
stats_2024['年度'] = 2024
stats_2025['年度'] = 2025

stats_all = pd.concat([stats_2023, stats_2024, stats_2025], ignore_index=True)
print(f"\n✓ 成績データ統合完了: {stats_all.shape}")

df_2023 = salary_df[['選手名', '年俸_円_2023']].copy()
df_2023['年度'] = 2023
df_2023.rename(columns={'年俸_円_2023': '年俸_円'}, inplace=True)

df_2024 = salary_df[['選手名', '年俸_円_2024']].copy()
df_2024['年度'] = 2024
df_2024.rename(columns={'年俸_円_2024': '年俸_円'}, inplace=True)

df_2025 = salary_df[['選手名', '年俸_円_2025']].copy()
df_2025['年度'] = 2025
df_2025.rename(columns={'年俸_円_2025': '年俸_円'}, inplace=True)

salary_long = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
salary_long = salary_long.dropna(subset=['年俸_円'])
salary_long = salary_long[salary_long['年俸_円'] > 0]
salary_long = salary_long.sort_values('年俸_円', ascending=False)
salary_long = salary_long.drop_duplicates(subset=['選手名', '年度'], keep='first')

print(f"✓ 年俸データ整形完了: {salary_long.shape}")

stats_all['予測年度'] = stats_all['年度'] + 1
merged_df = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
merged_df['タイトル数'] = merged_df['タイトル数'].fillna(0)

merged_df = pd.merge(
    merged_df, salary_long,
    left_on=['選手名', '予測年度'],
    right_on=['選手名', '年度'],
    suffixes=('_成績', '_年俸')
)

merged_df = merged_df.drop(columns=['年度_年俸', '予測年度'])
merged_df.rename(columns={'年度_成績': '成績年度'}, inplace=True)

print(f"✓ 全データ結合完了: {merged_df.shape}")

stats_all_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
stats_all_with_titles['タイトル数'] = stats_all_with_titles['タイトル数'].fillna(0)


# ============================================================
# セル5: 特徴量選択とデータ分割
# ============================================================

print("\n[4] 特徴量を選択してデータ分割中...")

feature_cols = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打',
                '本塁打', '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球',
                '三振', '併殺打', '打率', '出塁率', '長打率', '犠打', '犠飛',
                'タイトル数']

ml_df = merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
ml_df = ml_df.dropna()

print(f"\n✓ 機械学習用データ: {ml_df.shape}")

X = ml_df[feature_cols]
y = ml_df['年俸_円']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ 訓練データ: {X_train.shape[0]}サンプル")
print(f"✓ テストデータ: {X_test.shape[0]}サンプル")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# セル6: モデルの訓練
# ============================================================

print("\n[5] モデルを訓練中...")

models = {
    '線形回帰': LinearRegression(),
    'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\n{name}を訓練中...")

    if name == '線形回帰':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {'model': model, 'predictions': y_pred,
                     'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"  MAE: {mae/1e6:.2f}百万円, R²: {r2:.4f}")

best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_model      = results[best_model_name]['model']

print(f"\n✓ 最良モデル: {best_model_name} (R²={results[best_model_name]['R2']:.4f})")


# ============================================================
# セル7: 選手名予測関数の定義
# ============================================================

def predict_player_salary(player_name, year=2025):
    stats_year   = year - 1
    player_stats = stats_all_with_titles[
        (stats_all_with_titles['選手名'] == player_name) &
        (stats_all_with_titles['年度']   == stats_year)
    ]

    if player_stats.empty:
        return None

    player_stats = player_stats.iloc[0]
    features     = player_stats[feature_cols].values.reshape(1, -1)

    if best_model_name == '線形回帰':
        predicted_salary = best_model.predict(scaler.transform(features))[0]
    else:
        predicted_salary = best_model.predict(features)[0]

    actual_salary_data = salary_long[
        (salary_long['選手名'] == player_name) &
        (salary_long['年度']   == year)
    ]
    actual_salary = actual_salary_data['年俸_円'].values[0] \
                    if not actual_salary_data.empty else None

    return {
        'player_name':      player_name,
        'stats_year':       stats_year,
        'predicted_year':   year,
        'predicted_salary': predicted_salary,
        'actual_salary':    actual_salary,
        'stats':            player_stats,
        'features':         features[0]
    }

print("\n✓ 予測関数を定義しました")


# ============================================================
# セル8: 選手情報表示関数の定義
# ============================================================

def display_player_prediction(player_name, year=2025):
    result = predict_player_salary(player_name, year)

    if result is None:
        print(f"\n❌ エラー: 選手「{player_name}」の{year-1}年成績が見つかりません")
        available_players = stats_all_with_titles[
            stats_all_with_titles['年度'] == year-1
        ]['選手名'].unique()[:20]
        for i, p in enumerate(available_players, 1):
            print(f"  {i}. {p}")
        return

    print("\n" + "=" * 70)
    print("選手年俸予測結果")
    print("=" * 70)
    print(f"\n選手名: {result['player_name']}")
    print(f"成績年度: {result['stats_year']}年")
    print(f"予測年度: {result['predicted_year']}年")
    print(f"\n予測年俸: {result['predicted_salary']/1e6:.1f}百万円")

    if result['actual_salary'] is not None:
        print(f"実際の年俸: {result['actual_salary']/1e6:.1f}百万円")
        error = abs(result['predicted_salary'] - result['actual_salary'])
        print(f"予測誤差: {error/1e6:.1f}百万円 ({error/result['actual_salary']*100:.1f}%)")

    stats = result['stats']
    print(f"\n  試合: {int(stats['試合'])}  打率: {stats['打率']:.3f}  出塁率: {stats['出塁率']:.3f}")
    print(f"  安打: {int(stats['安打'])}  本塁打: {int(stats['本塁打'])}  打点: {int(stats['打点'])}  盗塁: {int(stats['盗塁'])}")
    print(f"  タイトル数: {int(stats['タイトル数'])}個")

    fig = plt.figure(figsize=(16, 5))

    # ── 年俸推移 ──────────────────────────────
    ax1 = plt.subplot(1, 3, 1)
    player_salary_history = salary_long[
        salary_long['選手名'] == player_name
    ].sort_values('年度')

    if not player_salary_history.empty:
        yrs  = player_salary_history['年度'].values
        sals = player_salary_history['年俸_円'].values / 1e6
        ax1.plot(yrs, sals, 'o-', linewidth=2, markersize=8, label='Actual Salary')
        ax1.plot(result['predicted_year'], result['predicted_salary']/1e6,
                 'r*', markersize=20, label='Predicted Salary')
        if result['actual_salary'] is not None:
            ax1.plot(result['predicted_year'], result['actual_salary']/1e6,
                     'go', markersize=12, label='Actual Salary (2025)')

    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Salary (million yen)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{player_name} - Salary Trend', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ── レーダーチャート ───────────────────────
    ax2 = plt.subplot(1, 3, 2, projection='polar')
    radar_stats = {
        'BA':  stats['打率']   / 0.4,
        'OBP': stats['出塁率'] / 0.5,
        'SLG': stats['長打率'] / 0.7,
        'HR':  min(stats['本塁打'] / 40,  1.0),
        'RBI': min(stats['打点']   / 100, 1.0),
        'SB':  min(stats['盗塁']   / 40,  1.0),
    }
    categories = list(radar_stats.keys())
    values     = list(radar_stats.values()) + [list(radar_stats.values())[0]]
    angles     = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles    += angles[:1]

    ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{player_name} - Performance Radar\n({result["stats_year"]})',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True)

    # ── リーグ内年俸分布 ─────────────────────
    ax3 = plt.subplot(1, 3, 3)
    sample_preds = []
    for _, row in stats_all_with_titles[
        stats_all_with_titles['年度'] == result['stats_year']
    ].head(50).iterrows():
        try:
            ft = row[feature_cols].values.reshape(1, -1)
            p  = best_model.predict(scaler.transform(ft))[0] \
                 if best_model_name == '線形回帰' else best_model.predict(ft)[0]
            sample_preds.append(p / 1e6)
        except:
            continue

    ax3.hist(sample_preds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(result['predicted_salary']/1e6, color='red', linestyle='--',
                linewidth=2, label=f'{player_name}')
    ax3.set_xlabel('Predicted Salary (million yen)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
    ax3.set_title('Salary Distribution\n(League Comparison)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("\n" + "=" * 70)

print("\n✓ 表示関数を定義しました")


# ============================================================
# セル9: 選手一覧表示関数
# ============================================================

def show_available_players(year=2024, limit=50):
    available = stats_all_with_titles[
        stats_all_with_titles['年度'] == year
    ].copy().sort_values('打率', ascending=False)

    print(f"\n予測可能な選手一覧（{year}年成績）:")
    print("=" * 70)
    print(f"{'No.':<5} {'選手名':<20} {'所属':<15} {'打率':<8} {'本塁打':<6} {'タイトル'}")
    print("-" * 70)
    for i, (_, row) in enumerate(available.head(limit).iterrows(), 1):
        team = row.get('所属球団', 'N/A')
        print(f"{i:<5} {row['選手名']:<20} {team:<15} "
              f"{row['打率']:.3f}  {int(row['本塁打']):<6} {int(row['タイトル数'])}")
    print("=" * 70)
    print(f"\n※ 全{len(available)}人が予測対象です")

print("\n✓ 選手一覧表示関数を定義しました")


# ============================================================
# セル10: 起動確認
# ============================================================

show_available_players(year=2024, limit=30)

print("\n" + "=" * 70)
print("使用例:")
print("  display_player_prediction('村上　宗隆', 2025)")
print("  display_player_prediction('岡本　和真', 2025)")
print("  display_player_prediction('山田　哲人', 2025)")
print("※ 選手名は完全一致・全角スペースで入力してください")
print("=" * 70)


# ============================================================
# セル11: サンプル予測
# ============================================================

sample_players = ['村上　宗隆', '岡本　和真', '佐藤　輝明']

for player in sample_players:
    if player in stats_all_with_titles[
        stats_all_with_titles['年度'] == 2024
    ]['選手名'].values:
        print(f"\n{'='*70}\n予測実行: {player}\n{'='*70}")
        display_player_prediction(player, 2025)


# ============================================================
# セル12: カスタム予測
# ============================================================

player_name = '近本　光司'   # ← ここを変更してください
year        = 2025
display_player_prediction(player_name, year)


# ============================================================
# セル13: 複数選手の比較
# ============================================================

def compare_players(player_names, year=2025):
    results_list = []
    for name in player_names:
        r = predict_player_salary(name, year)
        if r is not None:
            results_list.append(r)

    if not results_list:
        print("予測できる選手がいません")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    names     = [r['player_name'] for r in results_list]
    predicted = [r['predicted_salary']/1e6 for r in results_list]
    actual    = [r['actual_salary']/1e6 if r['actual_salary'] else 0
                 for r in results_list]

    x, w = np.arange(len(names)), 0.35

    # 年俸比較
    ax1 = axes[0]
    ax1.bar(x - w/2, predicted, w, label='Predicted', alpha=0.8, color='skyblue')
    if any(actual):
        ax1.bar(x + w/2, actual, w, label='Actual', alpha=0.8, color='orange')
    ax1.set_xlabel('Player', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Salary (million yen)', fontsize=12, fontweight='bold')
    ax1.set_title('Salary Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 成績比較
    ax2  = axes[1]
    sc   = pd.DataFrame([
        {'Player': r['player_name'],
         'BA':     r['stats']['打率'],
         'HR':     r['stats']['本塁打'],
         'RBI':    r['stats']['打点'],
         'Titles': r['stats']['タイトル数']}
        for r in results_list
    ])
    w2 = 0.2
    ax2.bar(x - w2*1.5, sc['BA']*100,    w2, label='BA x100',    alpha=0.8)
    ax2.bar(x - w2*0.5, sc['HR'],        w2, label='HR',         alpha=0.8)
    ax2.bar(x + w2*0.5, sc['RBI']/10,    w2, label='RBI /10',    alpha=0.8)
    ax2.bar(x + w2*1.5, sc['Titles']*10, w2, label='Titles x10', alpha=0.8)
    ax2.set_xlabel('Player', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value (normalized)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 90)
    print("選手比較サマリー")
    print("=" * 90)
    print(f"{'選手名':<15} {'予測年俸':<12} {'実際の年俸':<12} "
          f"{'打率':<8} {'本塁打':<6} {'打点':<6} {'タイトル'}")
    print("-" * 90)
    for r in results_list:
        a_str = f"{r['actual_salary']/1e6:.1f}M" if r['actual_salary'] else "N/A"
        print(f"{r['player_name']:<15} {r['predicted_salary']/1e6:>10.1f}M {a_str:>12} "
              f"{r['stats']['打率']:>7.3f} {int(r['stats']['本塁打']):>6} "
              f"{int(r['stats']['打点']):>6} {int(r['stats']['タイトル数']):>6}")
    print("=" * 90)

print("\n✓ 比較関数を定義しました")
compare_players(['村上　宗隆', '岡本　和真', '佐藤　輝明', '牧　秀悟'], 2025)


# ============================================================
# セル14: インタラクティブ入力
# ============================================================

try:
    user_input = input("予測したい選手名を入力してください: ")
    if user_input.strip():
        matches = stats_all_with_titles[
            stats_all_with_titles['選手名'].str.contains(user_input, na=False)
        ]['選手名'].unique()

        if len(matches) == 0:
            print(f"\n「{user_input}」に一致する選手が見つかりません")
            show_available_players(2024, 20)
        elif len(matches) == 1:
            display_player_prediction(matches[0], 2025)
        else:
            for i, p in enumerate(matches, 1):
                print(f"  {i}. {p}")
            print("\n完全な選手名で再度実行してください")
    else:
        print("\n入力がスキップされました")
except:
    print("\n（インタラクティブ入力はColabでのみ動作します）")


# ============================================================
# セル15: モデル性能サマリー
# ============================================================

summary_df = pd.DataFrame({
    'モデル':        list(results.keys()),
    'MAE (百万円)':  [results[m]['MAE']/1e6  for m in results],
    'RMSE (百万円)': [results[m]['RMSE']/1e6 for m in results],
    'R²スコア':      [results[m]['R2']        for m in results]
}).sort_values('R²スコア', ascending=False)

print("\n" + "=" * 70)
print("モデル性能サマリー")
print("=" * 70)
print(summary_df.to_string(index=False))
print(f"\n🏆 採用モデル: {best_model_name}")
print(f"   R²スコア: {results[best_model_name]['R2']:.4f}")
print(f"   平均誤差: {results[best_model_name]['MAE']/1e6:.2f}百万円")

if best_model_name == 'ランダムフォレスト':
    print("\n【特徴量重要度 Top 10】")
    fi = pd.DataFrame({'feature': feature_cols,
                       'importance': best_model.feature_importances_}
                     ).sort_values('importance', ascending=False).head(10)
    for _, row in fi.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<12} {bar} {row['importance']:.4f}")


# ============================================================
# セル16: 年俸影響要因分析
# ============================================================

def analyze_salary_factors():
    print("\n" + "=" * 70)
    print("年俸影響要因分析")
    print("=" * 70)

    tg = ml_df.groupby(ml_df['タイトル数'] > 0)['年俸_円'].agg(['count','mean','median'])
    tg['mean']   /= 1e6
    tg['median'] /= 1e6
    tg.index   = ['タイトル無し', 'タイトル有り']
    tg.columns = ['選手数', '平均年俸(百万円)', '中央値(百万円)']
    print("\n【タイトル獲得の影響】")
    print(tg)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    ax1.scatter(ml_df['打率'], ml_df['年俸_円']/1e6, alpha=0.5)
    ax1.set_xlabel('Batting Average', fontweight='bold')
    ax1.set_ylabel('Salary (million yen)', fontweight='bold')
    ax1.set_title('Batting Average vs Salary', fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    ax2.scatter(ml_df['本塁打'], ml_df['年俸_円']/1e6, alpha=0.5, color='orange')
    ax2.set_xlabel('Home Runs', fontweight='bold')
    ax2.set_ylabel('Salary (million yen)', fontweight='bold')
    ax2.set_title('Home Runs vs Salary', fontweight='bold')
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    tsd = ml_df.groupby('タイトル数')['年俸_円'].apply(list)
    ax3.boxplot([tsd[i] if i in tsd.index else []
                 for i in range(int(ml_df['タイトル数'].max())+1)],
                labels=range(int(ml_df['タイトル数'].max())+1))
    ax3.set_xlabel('Number of Titles', fontweight='bold')
    ax3.set_ylabel('Salary (yen)', fontweight='bold')
    ax3.set_title('Titles vs Salary', fontweight='bold')
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(ml_df['年俸_円']/1e6, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(ml_df['年俸_円'].mean()/1e6,   color='red',  linestyle='--', linewidth=2,
                label=f'Mean: {ml_df["年俸_円"].mean()/1e6:.1f}M')
    ax4.axvline(ml_df['年俸_円'].median()/1e6, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {ml_df["年俸_円"].median()/1e6:.1f}M')
    ax4.set_xlabel('Salary (million yen)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Salary Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

analyze_salary_factors()


# ============================================================
# セル17: チーム別分析
# ============================================================

def analyze_by_team(year=2024):
    print("\n" + "=" * 70)
    print(f"チーム別分析 ({year}年)")
    print("=" * 70)

    team_data = stats_all_with_titles[stats_all_with_titles['年度'] == year].copy()
    if '所属球団' not in team_data.columns:
        print("\n※ 所属球団の情報がデータに含まれていません")
        return

    team_preds = []
    for team in team_data['所属球団'].unique():
        preds = []
        for _, player in team_data[team_data['所属球団'] == team].iterrows():
            try:
                ft = player[feature_cols].values.reshape(1, -1)
                p  = best_model.predict(scaler.transform(ft))[0] \
                     if best_model_name == '線形回帰' else best_model.predict(ft)[0]
                preds.append(p)
            except:
                continue
        if preds:
            team_preds.append({
                'チーム': team,
                '選手数': len(preds),
                '平均予測年俸': np.mean(preds) / 1e6,
                '総予測年俸':   np.sum(preds)  / 1e6
            })

    team_df = pd.DataFrame(team_preds).sort_values('平均予測年俸', ascending=False)
    print(team_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax1 = axes[0]
    ax1.barh(team_df['チーム'], team_df['平均予測年俸'], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Average Predicted Salary (million yen)', fontweight='bold')
    ax1.set_title('Average Salary by Team', fontweight='bold', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    ax2.barh(team_df['チーム'], team_df['総予測年俸'], alpha=0.7, color='coral')
    ax2.set_xlabel('Total Predicted Salary (million yen)', fontweight='bold')
    ax2.set_title('Total Salary by Team', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

analyze_by_team(2024)


# ============================================================
# セル18: 最終まとめ
# ============================================================

print("\n" + "=" * 70)
print("✅ 全ての分析が完了しました！")
print("=" * 70)
print(f"\n  成績データ: {len(stats_all)}件 / 年俸データ: {len(salary_long)}件")
print(f"  タイトルデータ: {len(titles_df)}件 / 予測対象選手: {len(ml_df)}人")
print(f"\n  採用モデル: {best_model_name}")
print(f"  R²スコア: {results[best_model_name]['R2']:.4f}")
print(f"  平均誤差: {results[best_model_name]['MAE']/1e6:.2f}百万円")
print("\n" + "=" * 70)
print("🎯 display_player_prediction('選手名', 2025) で予測できます！")
print("=" * 70)
