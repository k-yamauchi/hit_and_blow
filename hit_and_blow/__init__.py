"""
Hit and Blow（数当てゲーム）パッケージ
"""

"""Hit and Blowゲーム - AIによる解法アルゴリズム"""

from .core import (
    BaseAgent,
    RuleBasedAgent,
    ProbabilisticAgent,
    HybridQLearningAgent,
    AgentRegistry,
    Environment,
    HitAndBlowAdvisor,
    generate_all_possible_codes,
    calculate_hits_blows,
)

__version__ = '0.1.0'
__all__ = [
    'BaseAgent',
    'RuleBasedAgent',
    'ProbabilisticAgent',
    'HybridQLearningAgent',
    'AgentRegistry',
    'Environment',
    'HitAndBlowAdvisor',
    'generate_all_possible_codes',
    'calculate_hits_blows',
] 