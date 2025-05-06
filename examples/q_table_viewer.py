#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_q_table(model_path):
    """モデルファイルからQ値テーブルをロード"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if 'q_table' in model_data:
            q_table = model_data['q_table']
            return q_table, model_data
        else:
            print(f"エラー: モデルにQ値テーブルが含まれていません: {model_path}")
            return None, None
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return None, None

def print_q_table_stats(q_table):
    """Q値テーブルの統計情報を表示"""
    if not q_table:
        print("Q値テーブルが空です")
        return
    
    state_action_pairs = len(q_table)
    unique_states = len(set([state for state, _ in q_table.keys()]))
    
    q_values = list(q_table.values())
    max_q = max(q_values)
    min_q = min(q_values)
    avg_q = sum(q_values) / len(q_values)
    
    print(f"\n===== Q値テーブル統計 =====")
    print(f"状態-行動ペア数: {state_action_pairs}")
    print(f"ユニーク状態数: {unique_states}")
    print(f"最大Q値: {max_q:.4f}")
    print(f"最小Q値: {min_q:.4f}")
    print(f"平均Q値: {avg_q:.4f}")
    
    # Q値の分布
    q_ranges = {
        "10.0以上": 0,
        "5.0-10.0": 0,
        "1.0-5.0": 0,
        "0.5-1.0": 0,
        "0.0-0.5": 0,
        "-0.5-0.0": 0,
        "-1.0--0.5": 0,
        "-5.0--1.0": 0,
        "-10.0--5.0": 0,
        "-10.0未満": 0
    }
    
    for q_value in q_values:
        if q_value >= 10.0:
            q_ranges["10.0以上"] += 1
        elif q_value >= 5.0:
            q_ranges["5.0-10.0"] += 1
        elif q_value >= 1.0:
            q_ranges["1.0-5.0"] += 1
        elif q_value >= 0.5:
            q_ranges["0.5-1.0"] += 1
        elif q_value >= 0.0:
            q_ranges["0.0-0.5"] += 1
        elif q_value >= -0.5:
            q_ranges["-0.5-0.0"] += 1
        elif q_value >= -1.0:
            q_ranges["-1.0--0.5"] += 1
        elif q_value >= -5.0:
            q_ranges["-5.0--1.0"] += 1
        elif q_value >= -10.0:
            q_ranges["-10.0--5.0"] += 1
        else:
            q_ranges["-10.0未満"] += 1
    
    print("\nQ値の分布:")
    for range_name, count in q_ranges.items():
        percentage = count / state_action_pairs * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")

def print_top_actions_for_state(q_table, state_key, top_n=10):
    """特定の状態に対するトップNのアクションを表示"""
    actions = {}
    
    for (state, action), q_value in q_table.items():
        if state == state_key:
            actions[action] = q_value
    
    if not actions:
        print(f"状態 '{state_key}' に対するアクションが見つかりません")
        return
    
    # Q値で降順にソート
    sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n===== 状態 '{state_key}' のトップ{min(top_n, len(sorted_actions))}アクション =====")
    for i, (action, q_value) in enumerate(sorted_actions[:top_n], 1):
        # タプル形式の文字列からリスト形式に変換
        action_str = action.strip('()').replace(' ', '')
        action_list = [int(x) for x in action_str.split(',') if x]
        
        print(f"{i}. アクション: {action_list}, Q値: {q_value:.4f}")

def print_initial_state_actions(q_table, top_n=20):
    """初期状態に対するアクションを表示"""
    print_top_actions_for_state(q_table, "initial", top_n)

def find_states_with_pattern(q_table, pattern):
    """指定したパターンを含む状態キーを検索"""
    matching_states = []
    
    for state, _ in q_table.keys():
        if pattern in state:
            matching_states.append(state)
    
    # 重複を削除
    unique_states = list(set(matching_states))
    return unique_states

def main():
    parser = argparse.ArgumentParser(description='Q値テーブルビューア')
    parser.add_argument('--model', type=str, default='models/multi_q_learning_model.pkl',
                      help='表示するモデルのパス')
    parser.add_argument('--opponent-model', type=str, 
                      help='比較する対戦相手モデルのパス（指定した場合は両方表示）')
    parser.add_argument('--state', type=str, default='initial',
                      help='表示する状態キー（デフォルト: initial）')
    parser.add_argument('--search', type=str,
                      help='状態キーで検索するパターン')
    parser.add_argument('--top-n', type=int, default=10,
                      help='表示するトップNのアクション数')
    
    args = parser.parse_args()
    
    # メインモデルのロード
    q_table, model_data = load_q_table(args.model)
    if q_table is None:
        return
    
    print(f"\n===== モデル: {args.model} =====")
    print(f"モデル情報:")
    for key, value in model_data.items():
        if key != 'q_table':
            print(f"  {key}: {value}")
    
    # Q値テーブルの統計情報を表示
    print_q_table_stats(q_table)
    
    # 初期状態のアクションを表示
    if args.state == 'initial':
        print_initial_state_actions(q_table, args.top_n)
    else:
        print_top_actions_for_state(q_table, args.state, args.top_n)
    
    # パターン検索
    if args.search:
        matching_states = find_states_with_pattern(q_table, args.search)
        print(f"\n===== パターン '{args.search}' を含む状態 =====")
        print(f"一致する状態数: {len(matching_states)}")
        
        if len(matching_states) > 0:
            print("\n最初の10状態:")
            for i, state in enumerate(matching_states[:10], 1):
                print(f"{i}. {state}")
            
            if len(matching_states) > 10:
                print(f"...他 {len(matching_states) - 10} 個の状態があります")
            
            # 最初の状態についてアクションを表示
            if matching_states:
                print_top_actions_for_state(q_table, matching_states[0], args.top_n)
    
    # 対戦相手モデルとの比較
    if args.opponent_model:
        opponent_q_table, opponent_model_data = load_q_table(args.opponent_model)
        if opponent_q_table is not None:
            print(f"\n===== 対戦相手モデル: {args.opponent_model} =====")
            print_q_table_stats(opponent_q_table)
            
            # 初期状態の比較
            print("\n===== 初期状態での両モデルの比較 =====")
            
            main_actions = {}
            opponent_actions = {}
            
            for (state, action), q_value in q_table.items():
                if state == "initial":
                    main_actions[action] = q_value
            
            for (state, action), q_value in opponent_q_table.items():
                if state == "initial":
                    opponent_actions[action] = q_value
            
            # 共通するアクション
            common_actions = set(main_actions.keys()) & set(opponent_actions.keys())
            print(f"初期状態での共通アクション数: {len(common_actions)}")
            
            if common_actions:
                print("\n共通する上位5アクション（Q値の差分順）:")
                
                # Q値の差が大きいものから表示
                diff_actions = [(action, main_actions[action], opponent_actions[action], 
                               abs(main_actions[action] - opponent_actions[action])) 
                              for action in common_actions]
                
                sorted_diff = sorted(diff_actions, key=lambda x: x[3], reverse=True)
                
                for i, (action, main_q, opponent_q, diff) in enumerate(sorted_diff[:5], 1):
                    action_str = action.strip('()').replace(' ', '')
                    action_list = [int(x) for x in action_str.split(',') if x]
                    
                    print(f"{i}. アクション: {action_list}")
                    print(f"   メインQ値: {main_q:.4f}, 対戦相手Q値: {opponent_q:.4f}, 差分: {diff:.4f}")

if __name__ == "__main__":
    main() 