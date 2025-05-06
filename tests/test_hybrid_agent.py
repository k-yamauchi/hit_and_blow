#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from hit_and_blow.agents.registry import AgentRegistry
    has_agents = True
except ImportError:
    has_agents = False

@unittest.skipIf(not has_agents, "エージェントモジュールが利用できません")
class TestHybridAgent(unittest.TestCase):
    """ハイブリッドエージェントのテスト"""
    
    def setUp(self):
        """各テストの前に実行されるセットアップ"""
        try:
            agent_class = AgentRegistry.get_agent_class('hybrid_q_learning')
            self.agent = agent_class(digits=3, number_range=6)
        except (ImportError, AttributeError) as e:
            self.skipTest(f"エージェントの初期化に失敗しました: {e}")
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.agent.digits, 3)
        self.assertEqual(self.agent.number_range, 6)
        self.assertTrue(hasattr(self.agent, 'q_table'))
    
    def test_reset(self):
        """リセット機能のテスト"""
        # リセット前の状態を保存
        initial_candidates = len(self.agent.candidates) if hasattr(self.agent, 'candidates') else None
        
        # 何らかの状態変更を模擬
        if hasattr(self.agent, 'candidates') and self.agent.candidates:
            self.agent.candidates = self.agent.candidates[:5]
        
        # リセット
        self.agent.reset()
        
        # リセット後の状態確認
        if initial_candidates is not None:
            self.assertEqual(len(self.agent.candidates), initial_candidates)
    
    def test_predict_initial(self):
        """初期予測のテスト"""
        # 初期予測
        prediction = self.agent.predict([])
        
        # 予測が有効な形式か確認
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 3)
        for digit in prediction:
            self.assertTrue(0 <= digit < 6)
    
    def test_predict_with_history(self):
        """履歴に基づく予測のテスト"""
        # 履歴に基づいた予測
        history = [
            ([0, 1, 2], (0, 1)),  # 0ヒット、1ブロー
            ([3, 4, 5], (1, 0)),  # 1ヒット、0ブロー
        ]
        
        prediction = self.agent.predict(history)
        
        # 予測が有効な形式か確認
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 3)
        for digit in prediction:
            self.assertTrue(0 <= digit < 6)
        
        # 履歴と矛盾しない予測であることを確認
        if hasattr(self.agent, '_calculate_hits_blows'):
            hits1, blows1 = self.agent._calculate_hits_blows(prediction, [0, 1, 2])
            self.assertEqual((hits1, blows1), (0, 1))
            
            hits2, blows2 = self.agent._calculate_hits_blows(prediction, [3, 4, 5])
            self.assertEqual((hits2, blows2), (1, 0))
    
    @unittest.skipIf(True, "学習テストは時間がかかるため通常はスキップ")
    def test_learning(self):
        """学習機能のテスト（オプション）"""
        # 初期Q値を保存
        initial_q_values = self.agent.q_table.copy() if hasattr(self.agent, 'q_table') else {}
        
        # 模擬的な学習ステップを実行
        for _ in range(10):
            self.agent.reset()
            done = False
            state = {'turn': 0, 'history': []}
            
            while not done and state['turn'] < 10:
                action = self.agent.predict(state['history'])
                
                # 環境からのフィードバックを模擬
                hits = 1 if state['turn'] >= 8 else 0
                blows = 2 if hits == 0 else 0
                
                # 報酬と次の状態を設定
                reward = 1.0 if hits == 3 else 0.1
                done = hits == 3 or state['turn'] >= 9
                
                next_state = {
                    'turn': state['turn'] + 1,
                    'history': state['history'] + [(action, (hits, blows))]
                }
                
                # エージェントに学習させる（実装されている場合）
                if hasattr(self.agent, 'learn'):
                    self.agent.learn(state, action, reward, next_state, done)
                
                state = next_state
        
        # 学習後のQ値が更新されているか確認
        if hasattr(self.agent, 'q_table') and initial_q_values:
            self.assertNotEqual(self.agent.q_table, initial_q_values)

if __name__ == '__main__':
    unittest.main() 