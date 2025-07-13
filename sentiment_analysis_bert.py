# sentiment_analysis_bert.py
# 国会会議録APIからのリアルタイムデータ取得と、BERTによる高度な分析を統合した最終版です。

import os
import sys
import pandas as pd
from tqdm import tqdm
import requests
import time
import re
from datetime import date, timedelta

# Hugging FaceのTransformersライブラリをインポート
try:
    from transformers import pipeline
except ImportError:
    print("エラー: 必要なライブラリがインストールされていません。")
    print("ターミナルで以下のコマンドを実行してください:")
    print("pip install torch transformers pandas fugashi ipadic requests")
    sys.exit()

# --- 設定項目 ---
# 使用する日本語BERTモデル
MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"

def fetch_data_from_api():
    """
    国会会議録APIに接続し、発言データを取得して政党ごとにグループ化します。
    """
    print("\n[ステップ 1/4] 検索条件を指定してAPIからデータを取得します...")
    base_url = "https://kokkai.ndl.go.jp/api/speech"
    params = {}

    any_keyword = input('検索キーワードを入力してください (必須) >> ')
    if not any_keyword:
        print("キーワードは必須です。プログラムを終了します。")
        return None
    params['any'] = any_keyword

    from_input = input('検索開始日を入力 (YYYY-MM-DD, Enterで1年前) >> ')
    if from_input and re.match(r'^\d{4}-\d{2}-\d{2}$', from_input):
        params['from'] = from_input
    else:
        one_year_ago = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        params['from'] = one_year_ago
        print(f"  - 開始日が不正または未入力のため、1年前に設定: {one_year_ago}")

    until_input = input('検索終了日を入力 (YYYY-MM-DD, Enterで本日) >> ')
    if until_input and re.match(r'^\d{4}-\d{2}-\d{2}$', until_input):
        params['until'] = until_input
    else:
        today = date.today().strftime("%Y-%m-%d")
        params['until'] = today
        print(f"  - 終了日が不正または未入力のため、本日に設定: {today}")

    # ヒット件数確認
    params_count = params.copy()
    params_count['maximumRecords'] = 1
    params_count['recordPacking'] = "json"

    try:
        print("  - ヒット件数をサーバーに問い合わせています...")
        response_count = requests.get(base_url, params=params_count)
        response_count.raise_for_status()
        total_num = response_count.json().get("numberOfRecords", 0)
    except Exception as e:
        print(f"APIへのリクエスト中にエラーが発生しました: {e}")
        return None
    
    if int(total_num) == 0:
        print("検索結果は0件でした。")
        return None

    next_input = input(f"検索結果は {total_num} 件です。分析しますか？ (キャンセルする場合は 'n' を入力) >> ")
    if next_input.lower() == "n":
        return None
    
    # 全件取得とデータ整形
    all_records = []
    max_records_per_request = 100
    if int(total_num) > 1000: # あまりに多い場合は件数を制限
        print("  - 注意: 検索結果が多すぎるため、最新の1000件を分析対象とします。")
        total_num = 1000
    
    pages = (int(total_num) + max_records_per_request - 1) // max_records_per_request
    
    params_data = params.copy()
    params_data['maximumRecords'] = max_records_per_request
    params_data['recordPacking'] = "json"

    for i in range(pages):
        params_data['startRecord'] = 1 + (i * max_records_per_request)
        try:
            response_data = requests.get(base_url, params=params_data)
            response_data.raise_for_status()
            records = response_data.json().get('speechRecord', [])
            for record in records:
                all_records.append({
                    'party': record.get('speakerGroup', '所属なし'),
                    'speech': record.get('speech', '')
                })
            sys.stdout.write(f"\r  - データを取得中... {i + 1}/{pages} ページ完了")
            sys.stdout.flush()
            time.sleep(1)
        except Exception as e:
            print(f"\nデータ取得中にエラーが発生しました: {e}")
            return None
    
    print("\nすべてのデータ取得が完了しました。")
    
    # Pandas DataFrameに変換し、政党ごとにグループ化
    df = pd.DataFrame(all_records)
    df.dropna(subset=['speech'], inplace=True)
    return df.groupby('party')['speech'].apply(list)


def analyze_speeches_with_bert(speeches_by_party, sentiment_analyzer):
    """
    BERTモデルを使い、政党ごとにネガポジ分析を行います。
    """
    analysis_results = []
    
    for party, speeches in tqdm(speeches_by_party.items(), desc="政党ごとに分析中"):
        total_score = 0
        analyzed_speech_count = 0
        
        for speech in speeches:
            sentences = str(speech).split('。')
            speech_score = 0
            analyzed_sentence_count = 0
            
            for sentence in sentences:
                if not sentence.strip() or len(sentence) < 5:
                    continue
                try:
                    result = sentiment_analyzer(sentence, truncation=True, max_length=512)[0]
                    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                    speech_score += score
                    analyzed_sentence_count += 1
                except Exception:
                    continue
            
            if analyzed_sentence_count > 0:
                total_score += (speech_score / analyzed_sentence_count)
                analyzed_speech_count += 1

        if analyzed_speech_count > 0:
            final_score = total_score / analyzed_speech_count
            analysis_results.append({
                'party': party,
                'score': final_score,
                'speech_count': len(speeches)
            })
    return analysis_results

def main():
    """
    メイン処理
    """
    print("--- 【BERT+API連携版】政党別ネガポジ判定プログラム ---")
    
    # 1. APIからデータを取得
    speeches_by_party = fetch_data_from_api()
    if speeches_by_party is None:
        print("データ取得を中止、または失敗したためプログラムを終了します。")
        sys.exit()
    print("データの準備が完了しました。")

    # 2. BERTモデルの準備
    print("\n[ステップ 2/4] LLM(BERTモデル)を読み込んでいます...")
    print(f"（モデル: {MODEL_NAME}）")
    print("※初回実行時はモデルのダウンロードに時間がかかることがあります。")
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        sys.exit()
    print("モデルの準備が完了しました。")

    # 3. BERTによる分析の実行
    print("\n[ステップ 3/4] BERTによるネガポジ分析を実行します...")
    results = analyze_speeches_with_bert(speeches_by_party, sentiment_analyzer)

    if not results:
        print("分析結果がありません。")
        sys.exit()

    # 4. 結果の表示
    print("\n[ステップ 4/4] 分析結果を表示します...")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print("\n--- 分析結果 (スコア順) ---")
    print("※スコアは-1.0(ネガティブ)から+1.0(ポジティブ)の範囲で示されます。")
    
    for result in sorted_results:
        score = result['score']
        bar_length = 20
        if score > 0:
            pos_bar = '#' * int(bar_length * score)
            bar = f"[ {' ' * bar_length} | {pos_bar:<{bar_length}} ]"
        else:
            neg_bar = '#' * int(bar_length * abs(score))
            bar = f"[ {neg_bar:>{bar_length}} | {' ' * bar_length} ]"

        print(f"\n【{result['party']}】 (分析発言数: {result['speech_count']})")
        print(f"  スコア: {score:+.4f}")
        print(f"  グラフ: {bar}")

    print("\n\n--- プログラム終了 ---")

if __name__ == '__main__':
    main()
