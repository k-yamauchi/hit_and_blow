#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os

from hit_and_blow.core.advisor import HitAndBlowAdvisor

def main():
    """アドバイザーモードのエントリーポイント"""
    parser = argparse.ArgumentParser(description='Hit and Blow Advisor')
    parser.add_argument('--digits', type=int, default=3, help='桁数')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲（0〜range-1）')
    parser.add_argument('--lang', choices=['ja', 'en'], default='ja', help='言語（日本語/英語）')
    args = parser.parse_args()
    
    advisor = HitAndBlowAdvisor(digits=args.digits, number_range=args.range)
    
    # 言語に応じたメッセージを設定
    if args.lang == 'ja':
        welcome_msg = f"====== Hit and Blow アドバイザー ======"
        setup_msg = f"桁数: {args.digits}, 範囲: 0-{args.range-1}"
        input_format_msg = "入力形式: 予測した数字 ヒット数 ブロー数"
        example_msg = "例: 123 1 1 （123を予測して、1ヒット1ブローだった場合）"
        commands_msg = "コマンド:"
        hint_cmd_msg = "  hint: 次の予測候補を表示"
        reset_cmd_msg = "  reset: ゲームをリセット"
        exit_cmd_msg = "  exit: 終了"
        input_prompt = "\n予測と結果を入力（またはコマンド）: "
        reset_msg = "ゲームをリセットしました。すべての候補が再初期化されました。"
        hint_msg = "おすすめの予測: "
        invalid_input_msg = "無効な入力です。予測、ヒット数、ブロー数を入力してください。"
        invalid_digits_msg = f"予測は{args.digits}桁でなければなりません。"
        remaining_msg = "残りの候補数: "
        possible_answers_msg = "可能性のある答え:"
        answer_found_msg = "答えは: "
        no_candidates_msg = "有効な候補がありません。入力した結果に誤りがある可能性があります。"
        resetting_msg = "ゲームをリセットします..."
        next_suggestion_msg = "次におすすめの予測: "
        invalid_format_msg = "入力形式が無効です。数字のみを入力してください。"
        error_msg = "エラー: "
    else:
        welcome_msg = f"====== Hit and Blow Advisor ======"
        setup_msg = f"Digits: {args.digits}, Range: 0-{args.range-1}"
        input_format_msg = "Input format: Enter your guess followed by the result (hits blows)"
        example_msg = "Example: 123 1 1 (means you guessed 123 and got 1 hit, 1 blow)"
        commands_msg = "Commands:"
        hint_cmd_msg = "  hint: Get a suggestion"
        reset_cmd_msg = "  reset: Reset the game"
        exit_cmd_msg = "  exit: Quit"
        input_prompt = "\nEnter your guess and result (or command): "
        reset_msg = "Game reset. All candidates reinitialized."
        hint_msg = "Suggestion: "
        invalid_input_msg = "Invalid input. Please provide guess, hits, and blows."
        invalid_digits_msg = f"Guess must have exactly {args.digits} digits."
        remaining_msg = "Remaining candidates: "
        possible_answers_msg = "Possible answers:"
        answer_found_msg = "The answer is: "
        no_candidates_msg = "No valid candidates found. There might be an error in the provided results."
        resetting_msg = "Resetting the game..."
        next_suggestion_msg = "Suggested next guess: "
        invalid_format_msg = "Invalid input format. Please enter numbers only."
        error_msg = "Error: "
    
    print(welcome_msg)
    print(setup_msg)
    print(input_format_msg)
    print(example_msg)
    print(commands_msg)
    print(hint_cmd_msg)
    print(reset_cmd_msg)
    print(exit_cmd_msg)
    print("=====================================")
    
    while True:
        user_input = input(input_prompt).strip().lower()
        
        if user_input == 'exit':
            break
        elif user_input == 'reset':
            advisor.reset()
            print(reset_msg)
            continue
        elif user_input == 'hint':
            suggestion = advisor.suggest_next_guess()
            print(f"{hint_msg}{suggestion}")
            continue
        
        # 入力の解析
        try:
            parts = user_input.split()
            if len(parts) < 3:
                print(invalid_input_msg)
                continue
                
            guess_str = parts[0]
            hits = int(parts[1])
            blows = int(parts[2])
            
            # 予測を整数のリストに変換
            guess = [int(digit) for digit in guess_str]
            
            if len(guess) != args.digits:
                print(invalid_digits_msg)
                continue
                
            # アドバイザーを結果で更新
            advisor.update_with_result(guess, hits, blows)
            
            # 状況を表示
            candidates_count = len(advisor.candidates)
            print(f"{remaining_msg}{candidates_count}")
            
            if candidates_count <= 5:
                print(possible_answers_msg)
                for candidate in advisor.candidates:
                    print(f"  {candidate}")
            
            if candidates_count == 1:
                print(f"{answer_found_msg}{advisor.candidates[0]}{'です！' if args.lang == 'ja' else ''}")
            elif candidates_count == 0:
                print(no_candidates_msg)
                print(resetting_msg)
                advisor.reset()
            else:
                suggestion = advisor.suggest_next_guess()
                print(f"{next_suggestion_msg}{suggestion}")
                
        except ValueError:
            print(invalid_format_msg)
        except Exception as e:
            print(f"{error_msg}{e}")

if __name__ == "__main__":
    main() 