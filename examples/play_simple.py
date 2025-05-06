#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import random

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hit_and_blow.core.environment import Environment

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow Game')
    parser.add_argument('--digits', type=int, default=3, help='桁数')
    parser.add_argument('--range', type=int, default=10, help='数字の範囲（0〜range-1）')
    parser.add_argument('--turns', type=int, default=10, help='最大ターン数')
    args = parser.parse_args()
    
    # 環境の初期化
    env = Environment(digits=args.digits, number_range=args.range, max_turns=args.turns)
    env.reset()
    
    # デバッグ用に正解を表示
    answer = env.get_secret()
    print(f"====== Hit and Blow ゲーム ======")
    print(f"桁数: {args.digits}, 範囲: 0-{args.range-1}, 最大ターン数: {args.turns}")
    print(f"デバッグ用: 正解は {answer} です。")
    print("入力形式: スペース区切り（例: 1 4 5）または連続数字（例: 145）")
    print("=====================================")
    
    turn = 0
    game_history = []
    
    while turn < args.turns:
        turn += 1
        print(f"\nターン {turn}/{args.turns}")
        
        # ユーザー入力を取得
        valid_input = False
        while not valid_input:
            try:
                user_input = input("予測を入力: ").strip()
                
                # スペース区切りと連続数字の両方に対応
                if ' ' in user_input:
                    prediction = [int(x) for x in user_input.split()]
                else:
                    prediction = [int(x) for x in user_input]
                
                if len(prediction) != args.digits:
                    print(f"予測は{args.digits}桁でなければなりません。")
                    continue
                    
                for digit in prediction:
                    if digit < 0 or digit >= args.range:
                        print(f"数字は0から{args.range-1}の範囲内でなければなりません。")
                        valid_input = False
                        break
                else:
                    valid_input = True
                    
            except ValueError:
                print("無効な入力です。数字のみを入力してください。")
        
        # 環境にステップを実行
        state, reward, done, info = env.step(prediction)
        hits = info['hits']
        blows = info['blows']
        
        print(f"結果: {hits} Hit, {blows} Blow")
        game_history.append((prediction, hits, blows))
        
        # ゲーム終了条件をチェック
        if done:
            if hits == args.digits:
                print(f"\nおめでとうございます！{turn}ターンで正解しました。")
            else:
                print(f"\nゲーム終了。正解は {answer} でした。")
            break
    
    # ゲーム履歴を表示
    print("\n===== ゲーム履歴 =====")
    for i, (pred, hits, blows) in enumerate(game_history, 1):
        print(f"ターン {i}: 予測 {pred} -> {hits} Hit, {blows} Blow")

if __name__ == "__main__":
    main() 