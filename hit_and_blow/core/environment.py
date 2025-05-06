"""Hit and Blowゲームの環境"""

import random
from typing import List, Tuple

from .utils import calculate_hits_blows


class Environment:
    """Hit and Blowゲームの環境クラス"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False, max_turns=20):
        """
        初期化
        
        Args:
            digits (int): 桁数
            number_range (int): 数字の範囲 (0からnumber_range-1)
            allow_repetition (bool): 数字の繰り返しを許可するか
            max_turns (int): 最大ターン数
        """
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        self.max_turns = max_turns
        
        self.secret = None
        self.turns = 0
        self.history = []
        self.reset()
        
    def reset(self):
        """環境をリセット"""
        self.secret = self._generate_secret()
        self.turns = 0
        self.history = []
        return self.get_state()
        
    def _generate_secret(self) -> List[int]:
        """
        秘密の数列を生成
        
        Returns:
            List[int]: 生成された秘密の数列
        """
        if self.allow_repetition:
            # 数字の繰り返しを許可する場合
            return [random.randint(0, self.number_range - 1) for _ in range(self.digits)]
        else:
            # 数字の繰り返しを許可しない場合
            return random.sample(range(self.number_range), self.digits)
            
    def step(self, action: List[int]) -> Tuple[dict, float, bool, dict]:
        """
        環境を1ステップ進める
        
        Args:
            action (List[int]): 予測した数列
            
        Returns:
            Tuple[dict, float, bool, dict]: (次の状態, 報酬, 終了フラグ, 追加情報)
        """
        if len(action) != self.digits:
            raise ValueError(f"予測の桁数({len(action)})が環境の桁数({self.digits})と一致しません")
            
        # ターン数をインクリメント
        self.turns += 1
        
        # ヒット数とブロー数を計算
        hits, blows = calculate_hits_blows(action, self.secret)
        
        # 履歴に追加
        self.history.append((action, hits, blows))
        
        # 報酬の計算
        # - 正解の場合: 大きな正の報酬
        # - それ以外の場合: ヒット数とブロー数に応じた小さな報酬
        if hits == self.digits:
            reward = 10.0  # 正解
        else:
            reward = 0.1 * hits + 0.05 * blows - 0.1  # ヒットとブローに応じた報酬
            
        # 終了判定
        done = (hits == self.digits) or (self.turns >= self.max_turns)
        
        # 次の状態と追加情報
        next_state = self.get_state()
        info = {
            "hits": hits,
            "blows": blows,
            "turns": self.turns,
            "history": self.history,
            "success": hits == self.digits,
        }
        
        return next_state, reward, done, info
        
    def get_state(self) -> dict:
        """
        現在の状態を取得
        
        Returns:
            dict: 現在の状態
        """
        return {
            "turns": self.turns,
            "history": self.history,
            "max_turns": self.max_turns,
        }
        
    def get_secret(self) -> List[int]:
        """
        秘密の数列を取得（デバッグ用）
        
        Returns:
            List[int]: 秘密の数列
        """
        return self.secret 