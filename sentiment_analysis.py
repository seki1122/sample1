# sentiment_analysis.py
# お客様の成功ロジックとエラー報告を反映した最終修正版です。

import requests
import time
import json
import sys
import os
import re # 正規表現モジュールをインポート
from janome.tokenizer import Tokenizer
from datetime import date, timedelta # 日付を扱うために追加

# --- 設定項目 ---
DIC_FILE = 'pn_ja.dic.txt'

def load_sentiment_dictionary(filepath):
    """
    日本語評価極性辞書を読み込みます。
    """
    sentiment_dic = {}
    if not os.path.exists(filepath):
        print(f"エラー: 辞書ファイルが見つかりません: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) >= 4:
                    word, polarity = parts[0], parts[3]
                    if polarity in ['p', 'n']:
                        sentiment_dic[word] = polarity
    except Exception as e:
        print(f"辞書ファイルの読み込み中にエラーが発生しました: {e}")
        return None
    print(f"辞書の読み込み完了。{len(sentiment_dic)}語を登録しました。")
    return sentiment_dic

def analyze_sentiment(text, sentiment_dic):
    """
    与えられたテキストのネガポジを判定します。
    """
    print("\n[ステップ 3/3] ネガポジ分析を実行します...")
    tokenizer = Tokenizer()
    
    # ★★★ 修正点 ★★★
    # TypeErrorを解消するため、非対応の引数 'stream=True' を削除します。
    tokens = tokenizer.tokenize(text)
    # ★★★ 修正ここまで ★★★
    
    score = 0
    positive_words = []
    negative_words = []
    for token in tokens:
        word = token.surface
        if word in sentiment_dic:
            if sentiment_dic[word] == 'p':
                score += 1
                positive_words.append(word)
            elif sentiment_dic[word] == 'n':
                score -= 1
                negative_words.append(word)
    return score, positive_words, negative_words

def main():
    """
    メイン処理
    """
    print("--- 国会会議録ネガポジ判定プログラム (成功ロジック再現版) ---")
    
    # 1. 感情辞書の読み込み
    print("\n[ステップ 1/3] 感情辞書を読み込んでいます...")
    sentiment_dic = load_sentiment_dictionary(DIC_FILE)
    if sentiment_dic is None:
        sys.exit()

    # 2. 検索パラメータの入力 (お客様のコードのロジックを忠実に再現)
    print("\n[ステップ 2/3] 検索条件を指定してデータを取得します...")
    
    # ★★★ 修正点 ★★★
    # お客様のご報告に基づき、動作実績のあるURLに修正します。
    base_url = "https://kokkai.ndl.go.jp/api/speech"
    # ★★★ 修正ここまで ★★★
    
    params = {}

    any_keyword = input('検索キーワードを入力してください (必須) >> ')
    if not any_keyword:
        print("キーワードは必須です。プログラムを終了します。")
        sys.exit()
    params['any'] = any_keyword

    speaker = input('発言者名を入力してください (Enterでスキップ) >> ')
    if speaker:
        params['speaker'] = speaker

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

    # ヒット件数確認用のリクエスト
    params_count = params.copy()
    params_count['maximumRecords'] = 1
    params_count['recordPacking'] = "json"

    try:
        print("  - ヒット件数をサーバーに問い合わせています...")
        response_count = requests.get(base_url, params=params_count)
        response_count.raise_for_status()
        json_data_count = response_count.json()
        total_num = json_data_count.get("numberOfRecords", 0)
    except Exception as e:
        print(f"APIへのリクエスト中にエラーが発生しました: {e}")
        sys.exit()
    
    if int(total_num) == 0:
        print("検索結果は0件でした。プログラムを終了します。")
        sys.exit()

    # キャンセルオプション
    next_input = input(f"検索結果は {total_num} 件です。分析しますか？\n(キャンセルする場合は 'n' を入力) >> ")
    if next_input.lower() == "n":
        print('プログラムをキャンセルしました')
        sys.exit()
    
    # 全件取得
    max_records_per_request = 100
    if int(total_num) > 30000:
        print("  - 注意: 検索結果が30000件を超えています。最新の30000件を分析対象とします。")
        total_num = 30000
    
    pages = (int(total_num) + max_records_per_request - 1) // max_records_per_request
    
    all_speech_text = ""
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
                all_speech_text += record.get('speech', '') + "\n"
            
            sys.stdout.write(f"\r  - データを取得中... {i + 1}/{pages} ページ完了")
            sys.stdout.flush()
            
            # ★重要なインターバル
            time.sleep(1)

        except Exception as e:
            print(f"\nデータ取得中にエラーが発生しました: {e}")
            sys.exit()
    
    print("\nすべてのデータ取得が完了しました。")

    if not all_speech_text:
        print("分析対象の発言が取得できませんでした。プログラムを終了します。")
        sys.exit()

    # 3. ネガポジ分析
    score, positive_words, negative_words = analyze_sentiment(all_speech_text, sentiment_dic)
    
    # 4. 結果表示
    print("\n--- 分析結果 ---")
    print(f"検索キーワード: '{params.get('any')}'")
    if params.get('speaker'):
        print(f"発言者名: '{params.get('speaker')}'")
    print(f"検索期間: {params.get('from')} ~ {params.get('until')}")
    print(f"分析対象の発言数: {total_num} 件")
    print(f"総合スコア: {score}")
    
    if score > 0:
        print("判定: ポジティブな内容の可能性が高いです。")
    elif score < 0:
        print("判定: ネガティブな内容の可能性が高いです。")
    else:
        print("判定: 中立的な内容、またはポジティブとネガティブが均衡しています。")

    print("\n検出されたポジティブ単語 (上位20種):")
    if positive_words:
        print(f"  {list(set(positive_words))[:20]}")
    else:
        print("  (なし)")
        
    print("\n検出されたネガティブ単語 (上位20種):")
    if negative_words:
        print(f"  {list(set(negative_words))[:20]}")
    else:
        print("  (なし)")
    
    print("\n--- プログラム終了 ---")

if __name__ == '__main__':
    main()
