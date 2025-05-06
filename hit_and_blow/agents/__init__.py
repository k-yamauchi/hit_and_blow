"""
Hit and Blowゲームの様々なエージェント実装
"""

from hit_and_blow.agents.registry import AgentRegistry
from hit_and_blow.agents.rulebase import RuleBaseAgent
from hit_and_blow.agents.probabilistic import ProbabilisticAgent
from hit_and_blow.agents.heuristic import HeuristicAgent
from hit_and_blow.agents.greedy import GreedyAgent
from hit_and_blow.agents.monte_carlo import MonteCarloAgent
from hit_and_blow.agents.hybrid_q_learning import HybridQLearningAgent
from hit_and_blow.agents.q_learning import QLearningAgent
from hit_and_blow.agents.sarsa import SARSAAgent
from hit_and_blow.agents.hybrid_sarsa import HybridSARSAAgent
from hit_and_blow.agents.pattern_matching import PatternMatchingAgent
from hit_and_blow.agents.genetic import GeneticAgent
from hit_and_blow.agents.bandit import BanditAgent
from hit_and_blow.agents.hybrid_bandit import HybridBanditQLearningAgent

# 既存のエージェント
AgentRegistry.register("rulebase", RuleBaseAgent)
AgentRegistry.register("probabilistic", ProbabilisticAgent)

# 新しいエージェント
AgentRegistry.register("heuristic", HeuristicAgent)
AgentRegistry.register("greedy", GreedyAgent)
AgentRegistry.register("monte_carlo", MonteCarloAgent)
AgentRegistry.register("pattern_matching", PatternMatchingAgent)
AgentRegistry.register("genetic", GeneticAgent)

# 強化学習エージェント
AgentRegistry.register("hybrid_q_learning", HybridQLearningAgent)  # 改良版Q学習（旧reinforcement）
AgentRegistry.register("q_learning", QLearningAgent)  # シンプルなQ学習
AgentRegistry.register("sarsa", SARSAAgent)  # SARSAアルゴリズム
AgentRegistry.register("hybrid_sarsa", HybridSARSAAgent)  # ハイブリッドSARSA
AgentRegistry.register("bandit", BanditAgent)  # 多腕バンディットアルゴリズム
AgentRegistry.register("hybrid_bandit", HybridBanditQLearningAgent) 