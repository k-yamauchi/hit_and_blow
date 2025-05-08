#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import argparse
import numpy as np
from collections import defaultdict
import pprint

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model(model_path):
    """
    Load the model file and return the whole model
    """
    if not os.path.exists(model_path):
        print(f"モデルファイル {model_path} が見つかりません")
        exit(1)
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        exit(1)

def print_model_structure(model):
    """
    Print the structure of the model
    """
    print("\n===== モデル構造 =====")
    
    # モデルの型を表示
    print(f"モデルのタイプ: {type(model)}")
    
    # クラスの場合、属性を表示
    if hasattr(model, "__dict__"):
        print("\nモデル属性:")
        for attr_name in dir(model):
            # 特殊メソッドと組み込みメソッドをスキップ
            if attr_name.startswith('__') or callable(getattr(model, attr_name)):
                continue
            
            attr_value = getattr(model, attr_name)
            attr_type = type(attr_value).__name__
            
            # 値の概要を表示（長すぎる場合は省略）
            value_summary = str(attr_value)
            if len(value_summary) > 100:
                value_summary = value_summary[:100] + "..."
            
            print(f"  {attr_name} ({attr_type}): {value_summary}")
    
    # 辞書の場合、キーを表示
    elif isinstance(model, dict):
        print("\nモデルキー:")
        for key in model.keys():
            value = model[key]
            value_type = type(value).__name__
            
            # 値の概要を表示
            value_summary = str(value)
            if len(value_summary) > 100:
                value_summary = value_summary[:100] + "..."
            
            print(f"  {key} ({value_type}): {value_summary}")
    
    else:
        print(f"モデルデータ: {model}")

def load_q_table(model):
    """
    Try to extract Q-table from the model
    """
    # クラスインスタンスの場合
    if hasattr(model, "q_table") and model.q_table is not None:
        return model.q_table
    
    # 辞書の場合
    elif isinstance(model, dict) and "q_table" in model:
        return model["q_table"]
    
    # HybridQLearningAgentの場合、q_dictの可能性
    elif hasattr(model, "q_dict") and model.q_dict is not None:
        return model.q_dict
    
    print("モデルからQ値テーブルを見つけられませんでした")
    return None

def print_q_table_stats(q_table):
    """
    Print statistics about the Q-table
    """
    if q_table is None:
        return
    
    # Check if keys are in the format (state, action)
    first_key = next(iter(q_table.keys()), None)
    is_tuple_format = isinstance(first_key, tuple) and len(first_key) == 2
    
    if is_tuple_format:
        # Old format: (state, action) -> q_value
        states = set(state for state, _ in q_table.keys())
        num_states = len(states)
        q_values = list(q_table.values())
    else:
        # New format: state -> {action -> q_value}
        num_states = len(q_table)
        # Flatten the dictionary to get all Q-values
        q_values = []
        for actions in q_table.values():
            if isinstance(actions, dict):
                q_values.extend(actions.values())
            else:
                q_values.append(actions)
    
    print(f"\n===== Q値テーブル統計 =====")
    print(f"状態-行動ペア数: {len(q_table)}")
    print(f"ユニーク状態数: {num_states}")
    
    if q_values:
        print(f"最大Q値: {max(q_values):.4f}")
        print(f"最小Q値: {min(q_values):.4f}")
        print(f"平均Q値: {sum(q_values) / len(q_values):.4f}")
        
        # Q-value distribution
        print("\nQ値の分布:")
        thresholds = [10.0, 5.0, 1.0, 0.5, 0.0, -0.5, -1.0, -5.0, -10.0]
        counts = [0] * (len(thresholds) + 1)
        
        for q in q_values:
            for i, t in enumerate(thresholds):
                if q >= t:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        
        labels = [
            "10.0以上",
            "5.0-10.0",
            "1.0-5.0",
            "0.5-1.0",
            "0.0-0.5",
            "-0.5-0.0",
            "-1.0--0.5",
            "-5.0--1.0",
            "-10.0--5.0",
            "-10.0未満"
        ]
        
        for i, (label, count) in enumerate(zip(labels, counts)):
            percentage = (count / len(q_values)) * 100 if q_values else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
    else:
        print("Q値テーブルに値が存在しません")

def print_state_actions(q_table, state):
    """
    Print top actions for a specific state
    """
    if q_table is None:
        return
    
    # Check if keys are in the format (state, action)
    first_key = next(iter(q_table.keys()), None)
    is_tuple_format = isinstance(first_key, tuple) and len(first_key) == 2
    
    if is_tuple_format:
        # Old format: (state, action) -> q_value
        actions = {}
        for (s, a), q in q_table.items():
            if s == state:
                actions[a] = q
    else:
        # New format: state -> {action -> q_value}
        actions = q_table.get(state, {})
    
    if not actions:
        print(f"状態 '{state}' に対するアクションが見つかりません")
        return
    
    sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n===== 状態 '{state}' のトップ10アクション =====")
    for i, (action, q_value) in enumerate(sorted_actions[:10], 1):
        print(f"{i}. アクション: {action}, Q値: {q_value:.4f}")

def find_states(q_table, pattern):
    """
    Find states matching a pattern
    """
    if q_table is None:
        return
    
    # Check if keys are in the format (state, action)
    first_key = next(iter(q_table.keys()), None)
    is_tuple_format = isinstance(first_key, tuple) and len(first_key) == 2
    
    if is_tuple_format:
        # Old format: (state, action) -> q_value
        states = set(state for state, _ in q_table.keys())
    else:
        # New format: state -> {action -> q_value}
        states = q_table.keys()
    
    matching_states = [state for state in states if pattern in str(state)]
    
    print(f"\n===== パターン '{pattern}' を含む状態 =====")
    print(f"一致する状態数: {len(matching_states)}")
    
    if matching_states:
        print("\n最初の10状態:")
        for i, state in enumerate(matching_states[:10], 1):
            print(f"{i}. {state}")
        
        if len(matching_states) > 10:
            print(f"...他 {len(matching_states) - 10} 個の状態があります")
        
        # Print actions for the first state
        if matching_states:
            print_state_actions(q_table, matching_states[0])
    else:
        print("一致する状態が見つかりません")

def find_highest_q_value_states(q_table, n=10):
    """
    Find states with the highest Q-values
    """
    if q_table is None:
        return
    
    # Check if keys are in the format (state, action)
    first_key = next(iter(q_table.keys()), None)
    is_tuple_format = isinstance(first_key, tuple) and len(first_key) == 2
    
    state_max_q_values = {}
    
    if is_tuple_format:
        # Old format: (state, action) -> q_value
        for (state, action), q_value in q_table.items():
            current_max = state_max_q_values.get(state, -float('inf'))
            state_max_q_values[state] = max(current_max, q_value)
    else:
        # New format: state -> {action -> q_value}
        for state, actions in q_table.items():
            if isinstance(actions, dict) and actions:
                state_max_q_values[state] = max(actions.values())
            elif isinstance(actions, (int, float)):
                state_max_q_values[state] = actions
    
    # Sort states by their max Q-value
    sorted_states = sorted(state_max_q_values.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n===== 最も高いQ値を持つ上位{n}状態 =====")
    for i, (state, max_q) in enumerate(sorted_states[:n], 1):
        print(f"{i}. 状態: {state}, 最大Q値: {max_q:.4f}")
        
        # Show the best action for this state
        if is_tuple_format:
            # Find the action with this Q-value
            best_actions = []
            for (s, a), q in q_table.items():
                if s == state and q == max_q:
                    best_actions.append((a, q))
            
            if best_actions:
                action, q = best_actions[0]
                print(f"   最適アクション: {action}, Q値: {q:.4f}")
        else:
            # Find the action with this Q-value
            actions = q_table.get(state, {})
            if isinstance(actions, dict):
                best_actions = [(a, q) for a, q in actions.items() if q == max_q]
                
                if best_actions:
                    action, q = best_actions[0]
                    print(f"   最適アクション: {action}, Q値: {q:.4f}")

def print_model_info(model):
    """
    Print information about the model
    """
    print(f"\n===== モデル情報: =====")
    
    # Print attributes that most models should have
    attrs_to_print = [
        'digits', 'number_range', 'allow_repetition',
        'learning_rate', 'discount_factor', 'exploration_rate',
        'optimal_first_guesses', 'training_stats'
    ]
    
    for attr in attrs_to_print:
        if hasattr(model, attr):
            value = getattr(model, attr)
            print(f"  {attr}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Q値テーブル解析ツール')
    parser.add_argument('--model', required=True, help='モデルファイルのパス')
    parser.add_argument('--state', help='アクションを表示する状態')
    parser.add_argument('--search', help='指定したパターンに一致する状態を検索')
    parser.add_argument('--list-states', action='store_true', help='最初の10状態を表示')
    parser.add_argument('--highest', action='store_true', help='最も高いQ値を持つ状態を表示')
    parser.add_argument('--show-structure', action='store_true', help='モデルの構造を詳細に表示')
    
    args = parser.parse_args()
    
    print(f"\n===== モデル: {args.model} =====")
    model = load_model(args.model)
    
    if args.show_structure:
        print_model_structure(model)
    
    q_table = load_q_table(model)
    
    if hasattr(model, 'digits'):
        print(f"モデル情報:")
        print_model_info(model)
    
    print_q_table_stats(q_table)
    
    if q_table is None:
        return
    
    # Check if keys are in the format (state, action)
    first_key = next(iter(q_table.keys()), None)
    is_tuple_format = isinstance(first_key, tuple) and len(first_key) == 2
    
    if args.state:
        print_state_actions(q_table, args.state)
    
    if args.search:
        find_states(q_table, args.search)
    
    # Special case for the initial state
    print_state_actions(q_table, 'initial')
    
    if args.list_states:
        # List first 10 states
        if is_tuple_format:
            states = sorted(set(state for state, _ in q_table.keys()))
        else:
            states = sorted(q_table.keys())
        
        print("\n===== 最初の10状態 =====")
        for i, state in enumerate(states[:10], 1):
            print(f"{i}. {state}")
        
        if len(states) > 10:
            print(f"...他 {len(states) - 10} 個の状態があります")
    
    if args.highest:
        find_highest_q_value_states(q_table)

if __name__ == '__main__':
    main() 