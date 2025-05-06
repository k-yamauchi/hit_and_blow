#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hit_and_blow.core.advisor import HitAndBlowAdvisor

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow Advisor')
    parser.add_argument('--digits', type=int, default=3, help='桁数')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲（0〜range-1）')
    args = parser.parse_args()
    
    advisor = HitAndBlowAdvisor(digits=args.digits, number_range=args.range)
    
    print(f"====== Hit and Blow アドバイザー ======")
    print(f"桁数: {args.digits}, 範囲: 0-{args.range-1}")
    print("入力形式: 予測した数字 ヒット数 ブロー数")
    print("例: 123 1 1 （123を予測して、1ヒット1ブローだった場合）")
    print("コマンド:")
    print("  hint: 次の予測候補を表示")
    print("  reset: ゲームをリセット")
    print("  exit: 終了")
    print("=====================================")
    
    while True:
        user_input = input("\n予測と結果を入力（またはコマンド）: ").strip().lower()
        
        if user_input == 'exit':
            break
        elif user_input == 'reset':
            advisor.reset()
            print("ゲームをリセットしました。すべての候補が再初期化されました。")
            continue
        elif user_input == 'hint':
            suggestion = advisor.suggest_next_guess()
            print(f"おすすめの予測: {suggestion}")
            continue
        
        # 入力の解析
        try:
            parts = user_input.split()
            if len(parts) < 3:
                print("無効な入力です。予測、ヒット数、ブロー数を入力してください。")
                continue
                
            guess_str = parts[0]
            hits = int(parts[1])
            blows = int(parts[2])
            
            # 予測を整数のリストに変換
            guess = [int(digit) for digit in guess_str]
            
            if len(guess) != args.digits:
                print(f"予測は{args.digits}桁でなければなりません。")
                continue
                
            # アドバイザーを結果で更新
            advisor.update_with_result(guess, hits, blows)
            
            # 状況を表示
            candidates_count = len(advisor.candidates)
            print(f"残りの候補数: {candidates_count}")
            
            if candidates_count <= 5:
                print("可能性のある答え:")
                for candidate in advisor.candidates:
                    print(f"  {candidate}")
            
            if candidates_count == 1:
                print(f"答えは: {advisor.candidates[0]} です！")
            elif candidates_count == 0:
                print("有効な候補がありません。入力した結果に誤りがある可能性があります。")
                print("ゲームをリセットします...")
                advisor.reset()
            else:
                suggestion = advisor.suggest_next_guess()
                print(f"次におすすめの予測: {suggestion}")
                
        except ValueError:
            print("入力形式が無効です。数字のみを入力してください。")
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    main() 