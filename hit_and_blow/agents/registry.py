from typing import Dict, Type, List
from hit_and_blow.agents.base import AgentBase

class AgentRegistry:
    """エージェントを登録・管理するレジストリ"""
    
    _agents: Dict[str, Type[AgentBase]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[AgentBase]) -> None:
        """エージェントを登録する
        
        Args:
            name: エージェントの名前
            agent_class: エージェントのクラス
        """
        cls._agents[name] = agent_class
    
    @classmethod
    def get(cls, name: str) -> Type[AgentBase]:
        """名前からエージェントのクラスを取得する
        
        Args:
            name: エージェントの名前
            
        Returns:
            エージェントのクラス
            
        Raises:
            KeyError: 指定した名前のエージェントが存在しない場合
        """
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return cls._agents[name]
    
    @classmethod
    def get_all_names(cls) -> List[str]:
        """登録されている全エージェントの名前を取得する
        
        Returns:
            エージェント名のリスト
        """
        return list(cls._agents.keys()) 