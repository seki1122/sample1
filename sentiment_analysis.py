# sentiment_analysis.py
# 発言者の所属政党ごとにネガポジスコアを算出し、ランキング形式で表示する機能を追加しました。

import requests
import time
import json
import sys
import os
import re
from janome.tokenizer import Tokenizer
from datetime import date, timedelta

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

def analyze_sentiment_for_party(text, sentiment_dic):
    """
    与えられたテキストのネガポジを判定します。（政党分析用）
    """
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    
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
    # 重複を除いた単語リストを返す
    return score, list(set(positive_words)), list(set(negative_words))

def main():
    """
    メイン処理
    """
    print("--- 【政党別スコア対応版】国会会議録ネガポジ判定プログラム ---")
    
    # 1. 感情辞書の読み込み
    print("\n[ステップ 1/3] 感情辞書を読み込んでいます...")
    sentiment_dic = load_sentiment_dictionary(DIC_FILE)
    if sentiment_dic is None:
        sys.exit()

    # 2. 検索パラメータの入力とデータ取得
    print("\n[ステップ 2/3] 検索条件を指定してデータを取得します...")
    base_url = "https://kokkai.ndl.go.jp/api/speech"
    params = {}

    any_keyword = input('検索キーワードを入力してください (必須) >> ')
    if not any_keyword:
        print("キーワードは必須です。プログラムを終了します。")
        sys.exit()
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
        sys.exit()
    
    if int(total_num) == 0:
        print("検索結果は0件でした。")
        sys.exit()

    next_input = input(f"検索結果は {total_num} 件です。分析しますか？ (キャンセルする場合は 'n' を入力) >> ")
    if next_input.lower() == "n":
        print('プログラムをキャンセルしました')
        sys.exit()
    
    # 全件取得と政党ごとのデータ分類
    party_speeches = {} # { '政党名': '発言1 発言2 ...', ... } という形式の辞書
    
    max_records_per_request = 100
    if int(total_num) > 30000:
        print("  - 注意: 検索結果が30000件を超えています。最新の30000件を分析対象とします。")
        total_num = 30000
    
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
                # 'speakerGroup'がない場合は「所属なし」とする
                party_name = record.get('speakerGroup', '所属なし')
                speech_text = record.get('speech', '')
                
                if party_name not in party_speeches:
                    party_speeches[party_name] = "" # 辞書に新しい政党を追加
                party_speeches[party_name] += speech_text + "\n"
            
            sys.stdout.write(f"\r  - データを取得中... {i + 1}/{pages} ページ完了")
            sys.stdout.flush()
            
            time.sleep(1)

        except Exception as e:
            print(f"\nデータ取得中にエラーが発生しました: {e}")
            sys.exit()
    
    print("\nすべてのデータ取得が完了しました。")

    # 3. 政党ごとにネガポジ分析を実行
    print("\n[ステップ 3/3] 政党ごとのネガポジ分析を実行します...")
    analysis_results = []
    for party, speeches in party_speeches.items():
        score, pos_words, neg_words = analyze_sentiment_for_party(speeches, sentiment_dic)
        # 発言がない政党は結果に追加しない
        if pos_words or neg_words:
            analysis_results.append({
                'party': party,
                'score': score,
                'positive_words': pos_words,
                'negative_words': neg_words
            })
    
    # スコアの高い順に並び替え
    sorted_results = sorted(analysis_results, key=lambda x: x['score'], reverse=True)
    
    # 4. 結果表示
    print("\n--- 政党別 分析結果 (スコア順) ---")
    
    for result in sorted_results:
        print(f"\n【{result['party']}】")
        print(f"  スコア: {result['score']}")
        print(f"  ポジティブ単語: {result['positive_words'][:10]}") # 上位10単語を表示
        print(f"  ネガティブ単語: {result['negative_words'][:10]}") # 上位10単語を表示
    
    print("\n\n--- プログラム終了 ---")

if __name__ == '__main__':
    main()
