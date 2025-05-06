#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import unittest

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hit_and_blow.core.environment import Environment

class TestEnvironment(unittest.TestCase):
    """Environmentクラスのテスト"""
    
    def test_initialization(self):
        """環境の初期化テスト"""
        env = Environment(digits=3, number_range=6, max_turns=10)
        self.assertEqual(env.digits, 3)
        self.assertEqual(env.number_range, 6)
        self.assertEqual(env.max_turns, 10)
        
        # 初期化時に答えはまだ生成されていない
        self.assertIsNone(env.answer)
        
        # 履歴は空
        self.assertEqual(env.history, [])
        
        # ターン数は0
        self.assertEqual(env.turn, 0)
    
    def test_reset(self):
        """リセット機能のテスト"""
        env = Environment(digits=3, number_range=6)
        state = env.reset()
        
        # 答えが生成されている
        self.assertIsNotNone(env.answer)
        self.assertEqual(len(env.answer), 3)
        
        # 答えの各桁が範囲内
        for digit in env.answer:
            self.assertTrue(0 <= digit < 6)
        
        # 履歴が空でターン数は0
        self.assertEqual(env.history, [])
        self.assertEqual(env.turn, 0)
        
        # 状態が返される
        self.assertEqual(state, {'turn': 0, 'history': []})
    
    def test_calculate_hits_blows(self):
        """ヒットとブローの計算テスト"""
        env = Environment(digits=3, number_range=6)
        
        # 答えを固定
        env.reset()
        env.answer = [1, 2, 3]
        
        # テストケース: 予測 -> 期待されるヒット数とブロー数
        test_cases = [
            ([1, 2, 3], (3, 0)),  # 完全一致
            ([4, 5, 0], (0, 0)),  # 完全不一致
            ([1, 3, 2], (1, 2)),  # 1つの数字と位置が一致、2つの数字が含まれるが位置が異なる
            ([3, 2, 1], (0, 3)),  # すべての数字が含まれるが位置が異なる
            ([1, 4, 5], (1, 0)),  # 1つの数字と位置が一致、他は不一致
        ]
        
        for prediction, expected in test_cases:
            hits, blows = env._calculate_hits_blows(prediction)
            self.assertEqual((hits, blows), expected, f"Failed for prediction={prediction}")
    
    def test_step(self):
        """ステップ実行のテスト"""
        env = Environment(digits=3, number_range=6, max_turns=10)
        env.reset()
        env.answer = [1, 2, 3]  # 答えを固定
        
        # 正しくない予測
        state, reward, done, info = env.step([4, 5, 0])
        
        # 状態、報酬、終了フラグ、情報の確認
        self.assertEqual(state['turn'], 1)
        self.assertEqual(len(state['history']), 1)
        self.assertEqual(state['history'][0][0], [4, 5, 0])
        self.assertEqual(state['history'][0][1], (0, 0))
        
        self.assertEqual(reward, 0)  # 報酬は0
        self.assertFalse(done)  # ゲームは終了していない
        self.assertEqual(info['hits'], 0)
        self.assertEqual(info['blows'], 0)
        
        # 再度ステップを実行（部分的に正しい予測）
        state, reward, done, info = env.step([1, 4, 5])
        
        # 状態、報酬、終了フラグ、情報の確認
        self.assertEqual(state['turn'], 2)
        self.assertEqual(len(state['history']), 2)
        self.assertEqual(state['history'][1][0], [1, 4, 5])
        self.assertEqual(state['history'][1][1], (1, 0))
        
        self.assertEqual(reward, 0.1)  # 部分的に正しい場合の報酬
        self.assertFalse(done)  # ゲームは終了していない
        self.assertEqual(info['hits'], 1)
        self.assertEqual(info['blows'], 0)
        
        # 正解の予測
        state, reward, done, info = env.step([1, 2, 3])
        
        # 状態、報酬、終了フラグ、情報の確認
        self.assertEqual(state['turn'], 3)
        self.assertEqual(len(state['history']), 3)
        self.assertEqual(state['history'][2][0], [1, 2, 3])
        self.assertEqual(state['history'][2][1], (3, 0))
        
        self.assertEqual(reward, 1.0)  # 正解の場合の報酬
        self.assertTrue(done)  # ゲームが終了
        self.assertEqual(info['hits'], 3)
        self.assertEqual(info['blows'], 0)
    
    def test_max_turns(self):
        """最大ターン数の制限テスト"""
        env = Environment(digits=3, number_range=6, max_turns=2)
        env.reset()
        env.answer = [1, 2, 3]  # 答えを固定
        
        # 1ターン目
        state, reward, done, info = env.step([4, 5, 0])
        self.assertFalse(done)  # まだ終了していない
        
        # 2ターン目（最大ターン数に達する）
        state, reward, done, info = env.step([4, 5, 0])
        self.assertTrue(done)  # 最大ターン数に達したので終了
        self.assertEqual(reward, -1.0)  # 失敗の場合の報酬
        
        # これ以上のステップはできない
        with self.assertRaises(Exception):
            env.step([1, 2, 3])

if __name__ == '__main__':
    unittest.main() 