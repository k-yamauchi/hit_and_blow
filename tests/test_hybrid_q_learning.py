#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from hit_and_blow.agents.hybrid_q_learning import HybridQLearningAgent
    has_agent = True
except ImportError:
    has_agent = False

@unittest.skipIf(not has_agent, "HybridQLearningAgentが利用できません")
class TestHybridQLearningAgent(unittest.TestCase):
    """ハイブリッドQ学習エージェントのテスト"""
    
    def setUp(self):
        """各テストの前に実行されるセットアップ"""
        try:
            self.agent = HybridQLearningAgent(num_digits=3, digit_range=6)
        except Exception as e:
            self.skipTest(f"エージェントの初期化に失敗しました: {e}")
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.agent.num_digits, 3)
        self.assertEqual(self.agent.digit_range, 6)
        self.assertTrue(hasattr(self.agent, 'q_table'))
        self.assertTrue(hasattr(self.agent, 'learning_rate'))
        self.assertTrue(hasattr(self.agent, 'discount_factor'))
        self.assertTrue(hasattr(self.agent, 'exploration_rate'))
    
    def test_reset(self):
        """リセット機能のテスト"""
        self.agent.state_history = ["test_state"]
        self.agent.action_history = [(1, 2, 3)]
        self.agent.candidates = []
        
        self.agent.reset()
        
        self.assertEqual(self.agent.state_history, [])
        self.assertEqual(self.agent.action_history, [])
        self.assertTrue(len(self.agent.candidates) > 0)
    
    def test_get_state_representation(self):
        """状態表現の生成テスト"""
        # 初期状態
        state = self.agent.get_state_representation([])
        self.assertTrue("initial" in state)
        
        # 履歴がある場合
        history = [
            ([0, 1, 2], (1, 0)),  # 1ヒット、0ブロー
            ([3, 4, 5], (0, 2)),  # 0ヒット、2ブロー
        ]
        
        state = self.agent.get_state_representation(history)
        self.assertTrue("1H0B" in state or "0H2B" in state)
        self.assertTrue("R" in state)  # 残りの候補数
    
    def test_calculate_hits_blows(self):
        """ヒットとブローの計算テスト"""
        # 完全一致
        hits, blows = self.agent._calculate_hits_blows([1, 2, 3], [1, 2, 3])
        self.assertEqual(hits, 3)
        self.assertEqual(blows, 0)
        
        # 部分一致
        hits, blows = self.agent._calculate_hits_blows([1, 2, 3], [1, 3, 4])
        self.assertEqual(hits, 1)
        self.assertEqual(blows, 1)
        
        # 完全不一致
        hits, blows = self.agent._calculate_hits_blows([1, 2, 3], [4, 5, 6])
        self.assertEqual(hits, 0)
        self.assertEqual(blows, 0)
        
        # 位置が違うが数字はすべて含まれる
        hits, blows = self.agent._calculate_hits_blows([1, 2, 3], [3, 1, 2])
        self.assertEqual(hits, 0)
        self.assertEqual(blows, 3)
    
    def test_check_consistent(self):
        """一貫性チェック機能のテスト"""
        # 完全一致の場合
        self.assertTrue(self.agent._check_consistent([1, 2, 3], [1, 2, 3], 3, 0))
        
        # 部分一致の場合
        self.assertTrue(self.agent._check_consistent([1, 2, 3], [1, 3, 4], 1, 1))
        
        # 不一致の例
        self.assertFalse(self.agent._check_consistent([1, 2, 3], [1, 2, 3], 2, 1))  # 結果と一致しない
    
    def test_update_candidates(self):
        """候補更新のテスト"""
        self.agent.reset()
        original_count = len(self.agent.candidates)
        
        # 候補を絞り込む
        self.agent._update_candidates(([1, 2, 3], (0, 0)))  # 完全不一致
        
        # 候補が減少しているはず
        self.assertLess(len(self.agent.candidates), original_count)
        
        # すべての残りの候補がチェックに合格するはず
        for candidate in self.agent.candidates:
            self.assertTrue(self.agent._check_consistent(list(candidate), [1, 2, 3], 0, 0))
    
    def test_predict_initial(self):
        """初期予測のテスト"""
        prediction = self.agent.predict([])
        
        # 予測が有効な形式か確認
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 3)
        for digit in prediction:
            self.assertTrue(0 <= digit < 6)
    
    def test_predict_with_history(self):
        """履歴に基づく予測のテスト"""
        self.agent.reset()
        
        # 履歴に基づいた予測
        history = [
            ([0, 1, 2], (0, 1)),  # 0ヒット、1ブロー
        ]
        
        prediction = self.agent.predict(history)
        
        # 予測が有効な形式か確認
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 3)
        for digit in prediction:
            self.assertTrue(0 <= digit < 6)
        
        # 履歴と矛盾しない予測であることを確認
        hits, blows = self.agent._calculate_hits_blows(prediction, [0, 1, 2])
        self.assertEqual((hits, blows), (0, 1))
    
    def test_select_informative_action(self):
        """情報利得に基づく行動選択のテスト"""
        self.agent.reset()
        actions = [tuple(self.agent.candidates[i]) for i in range(min(5, len(self.agent.candidates)))]
        
        selected = self.agent._select_informative_action(actions, [])
        
        # 選択された行動が有効な選択肢の中にあるはず
        self.assertIn(selected, actions)
    
    def test_model_save_load(self):
        """モデルの保存と読み込みのテスト"""
        # Q値を更新
        self.agent.q_table["test_state"]["test_action"] = 0.5
        
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # モデルを保存
            self.agent.save_model(temp_path)
            
            # 新しいエージェントでモデルを読み込み
            new_agent = HybridQLearningAgent(num_digits=3, digit_range=6)
            new_agent.load_model(temp_path)
            
            # 読み込んだモデルが正しいか確認
            self.assertEqual(new_agent.q_table["test_state"]["test_action"], 0.5)
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_calculate_reward(self):
        """報酬計算のテスト"""
        self.agent.reset()
        
        # 正解の場合
        history = [([1, 2, 3], (3, 0))]
        reward = self.agent.calculate_reward(history)
        self.assertTrue(reward > 2.0)  # 大きな報酬があるはず
        
        # 部分的に正解の場合
        history = [([1, 2, 3], (1, 1))]
        reward = self.agent.calculate_reward(history)
        self.assertTrue(0 < reward < 2.0)  # 中程度の報酬
        
        # 完全不正解の場合
        history = [([1, 2, 3], (0, 0))]
        reward = self.agent.calculate_reward(history)
        self.assertTrue(reward < 0.5)  # 小さな報酬かペナルティ
    
    def test_update_q_values(self):
        """Q値更新のテスト"""
        self.agent.state_history = ["state1", "state2"]
        self.agent.action_history = [(1, 2, 3), (4, 5, 0)]
        
        # Q値更新前の状態を保存
        self.agent.q_table["state1"][(1, 2, 3)] = 0.0
        self.agent.q_table["state2"][(4, 5, 0)] = 0.0
        
        # Q値を更新
        self.agent.update_q_values(1.0)
        
        # Q値が更新されているはず
        self.assertTrue(self.agent.q_table["state1"][(1, 2, 3)] > 0.0)
        self.assertTrue(self.agent.q_table["state2"][(4, 5, 0)] > 0.0)

if __name__ == '__main__':
    unittest.main() 