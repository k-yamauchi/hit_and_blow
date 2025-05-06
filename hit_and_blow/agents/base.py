from abc import ABC, abstractmethod
from typing import List, Tuple

class AgentBase(ABC):
    """エージェントの基底クラス（インターフェース）"""
    
    def __init__(self, num_digits: int, digit_range: int):
        self.num_digits = num_digits
        self.digit_range = digit_range
    
    @abstractmethod
    def reset(self) -> None:
        """エージェントの状態をリセット"""
        pass
    
    @abstractmethod
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """次の予測を生成
        
        Args:
            history: 過去の予測と結果のリスト [(guess, (hits, blows)), ...]
            
        Returns:
            次の予測
        """
        pass 