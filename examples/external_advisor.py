#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from typing import List, Tuple

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hit_and_blow.core.advice import AdviceGenerator


def validate_input(input_str: str, digits: int, number_range: int) -> List[int]:
    """
    ユーザー入力を検証して正しい形式に変換
    
    Args:
        input_str: ユーザーの入力文字列
        digits: 桁数
        number_range: 数字の範囲
        
    Returns:
        変換された数字リスト、または無効な入力の場合はNone
    """
    try:
        # 入力を整数のリストに変換
        if ',' in input_str:
            # カンマ区切りの場合
            numbers = [int(n.strip()) for n in input_str.split(',')]
        else:
            # スペース区切りまたは連続した数字の場合
            numbers = [int(n) for n in input_str.split()] if ' ' in input_str else [int(c) for c in input_str]
        
        # 桁数のチェック
        if len(numbers) != digits:
            print(f"エラー: {digits}桁の数字を入力してください")
            return None
        
        # 範囲のチェック
        if any(n < 0 or n >= number_range for n in numbers):
            print(f"エラー: 数字は0から{number_range-1}の範囲である必要があります")
            return None
        
        return numbers
    
    except ValueError:
        print("エラー: 有効な数字を入力してください")
        return None


def validate_result(input_str: str, digits: int) -> Tuple[int, int]:
    """
    ヒット・ブロー結果の入力を検証
    
    Args:
        input_str: ユーザーの入力文字列（ヒット数,ブロー数）
        digits: 桁数
    
    Returns:
        (ヒット数, ブロー数)のタプル、または無効な入力の場合はNone
    """
    try:
        # カンマ区切りで入力
        if ',' in input_str:
            hits_str, blows_str = input_str.split(',')
            hits = int(hits_str.strip())
            blows = int(blows_str.strip())
        # スペース区切りで入力
        elif ' ' in input_str:
            hits_str, blows_str = input_str.split()
            hits = int(hits_str.strip())
            blows = int(blows_str.strip())
        # 2桁の数字として入力（例：「21」→ヒット2, ブロー1）
        elif len(input_str) == 2:
            hits = int(input_str[0])
            blows = int(input_str[1])
        else:
            print("エラー: 結果は 'ヒット数,ブロー数' の形式で入力してください")
            return None
        
        if hits < 0 or blows < 0 or hits + blows > digits:
            print(f"エラー: 無効な結果です。ヒットとブローの合計は最大{digits}です。")
            return None
        
        return (hits, blows)
    
    except ValueError:
        print("エラー: 結果は 'ヒット数,ブロー数' の形式で入力してください")
        return None


def show_help(digits: int):
    """ヘルプメッセージを表示"""
    print("\n----- ヘルプ -----")
    print("コマンド:")
    print("  q, quit, exit - 外部アドバイザーを終了")
    print("  h, help - このヘルプを表示")
    print("  reset - ゲームをリセット")
    print("  show - 現在の状態を表示")
    print("\n入力方法:")
    print(f"  予測: {digits}桁の数字を入力（例: '012' または '0,1,2' または '0 1 2'）")
    print("  結果: 以下のいずれかの形式で入力")
    print("    - カンマ区切り（例: '1,2'）")
    print("    - スペース区切り（例: '1 2'）")
    print("    - 連続した数字（例: '12' → ヒット1, ブロー2）")
    print("-----------------\n")


def run_external_advisor(model_path: str, digits: int, number_range: int):
    """
    外部アドバイザーモードの実行
    
    Args:
        model_path: Q学習モデルのパス
        digits: 桁数
        number_range: 数字の範囲
    """
    print("\n===== Hit and Blow 外部アドバイザー =====")
    print(f"桁数: {digits}, 数字の範囲: 0-{number_range-1}")
    print("Q学習モデルがプレイのアドバイスを提供します。")
    print("終了するには 'q' または 'quit' と入力してください。")
    print("ヘルプを表示するには 'h' または 'help' と入力してください。")
    
    # アドバイス生成器の初期化
    advisor = AdviceGenerator(model_path=model_path)
    advisor.digits = digits
    advisor.number_range = number_range
    
    # ゲーム状態の初期化
    guesses = []
    results = []
    turn = 1
    
    while True:
        print(f"\n----- ターン {turn} -----")
        
        # 現在の状態に基づくアドバイスを表示
        if not guesses:
            advice_text = advisor.explain_advice([], [])
            print("\nアドバイス:")
            print(advice_text)
        else:
            advice_text = advisor.explain_advice(guesses, results)
            print("\nアドバイス:")
            print(advice_text)
        
        print("\n予測を入力してください:")
        while True:
            guess_input = input("> ").strip().lower()
            
            # 終了コマンド
            if guess_input in ['q', 'quit', 'exit']:
                print("外部アドバイザーを終了します。")
                return
            
            # ヘルプコマンド
            if guess_input in ['h', 'help']:
                show_help(digits)
                continue
            
            # リセットコマンド
            if guess_input == 'reset':
                print("\nゲームをリセットします。")
                guesses = []
                results = []
                turn = 1
                break
            
            # 状態表示コマンド
            if guess_input == 'show':
                if not guesses:
                    print("まだ予測は行われていません。")
                else:
                    print("\n現在までの予測と結果:")
                    for i, (g, r) in enumerate(zip(guesses, results), 1):
                        print(f"ターン {i}: 予測={g}, 結果={r[0]}ヒット,{r[1]}ブロー")
                continue
            
            # ユーザー入力を検証
            guess = validate_input(guess_input, digits, number_range)
            if guess:
                break
        
        # リセットコマンドの場合、次のターンへ
        if guess_input == 'reset':
            continue
        
        print("\n結果を入力してください（ヒット数,ブロー数）:")
        while True:
            result_input = input("> ").strip().lower()
            
            # 終了コマンド
            if result_input in ['q', 'quit', 'exit']:
                print("外部アドバイザーを終了します。")
                return
            
            # ヘルプコマンド
            if result_input in ['h', 'help']:
                show_help(digits)
                continue
            
            # 結果を検証
            result = validate_result(result_input, digits)
            if result:
                hits, blows = result
                break
        
        # 予測と結果を記録
        guesses.append(guess)
        results.append((hits, blows))
        
        # 正解した場合
        if hits == digits:
            print(f"\nおめでとうございます！{turn}ターンで正解しました！")
            
            print("\nもう一度プレイしますか？ (y/n)")
            while True:
                choice = input("> ").strip().lower()
                if choice in ['y', 'yes']:
                    guesses = []
                    results = []
                    turn = 0
                    break
                elif choice in ['n', 'no']:
                    print("外部アドバイザーを終了します。")
                    return
                else:
                    print("'y' または 'n' を入力してください。")
        
        turn += 1


def main():
    parser = argparse.ArgumentParser(description='Hit and Blow 外部アドバイザー')
    parser.add_argument('--model', type=str, default='models/multi_q_learning_model.pkl',
                        help='使用するQ学習モデルのパス（デフォルト: models/multi_q_learning_model.pkl）')
    parser.add_argument('--digits', type=int, default=3,
                        help='桁数（デフォルト: 3）')
    parser.add_argument('--range', type=int, default=6,
                        help='数字の範囲 0-(range-1)（デフォルト: 6）')
    
    args = parser.parse_args()
    
    run_external_advisor(args.model, args.digits, args.range)


if __name__ == '__main__':
    main() 