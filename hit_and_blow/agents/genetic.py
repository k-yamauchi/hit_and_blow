from typing import List, Dict, Any, Tuple, Set, Optional
import random
import numpy as np
from hit_and_blow.agents.base import AgentBase
from hit_and_blow.core.environment import Environment

class GeneticAgent(AgentBase):
    """
    遺伝的アルゴリズムエージェント
    進化的計算を用いて最適な予測を探索する
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10,
                 population_size: int = 50, generations: int = 10,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        """
        初期化
        
        Args:
            num_digits: 桁数
            digit_range: 数字の範囲（0からdigit_range-1まで）
            population_size: 個体数
            generations: 世代数
            mutation_rate: 突然変異率
            crossover_rate: 交叉率
        """
        super().__init__(num_digits, digit_range)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # 可能性のある答えの集合
        self.possible_answers = set()
        
        # 過去の予測
        self.prev_guesses = []
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.possible_answers = set()
        self.prev_guesses = []
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        遺伝的アルゴリズム: 履歴に基づいて候補を絞り込み、進化的計算で最適な予測を探索
        """
        # 候補を更新
        self._update_candidates(history)
        
        # 候補が1つだけなら、それが答え
        if len(self.possible_answers) == 1:
            prediction = list(next(iter(self.possible_answers)))
            self.prev_guesses.append(prediction)
            return prediction
            
        # 候補が少ない場合は直接選択
        if len(self.possible_answers) <= 10:
            candidates = list(self.possible_answers)
            # 過去に選んでいない候補を選択
            for candidate in candidates:
                if list(candidate) not in self.prev_guesses:
                    prediction = list(candidate)
                    self.prev_guesses.append(prediction)
                    return prediction
            
            # すべて選んでいる場合はランダム
            prediction = list(random.choice(candidates))
            self.prev_guesses.append(prediction)
            return prediction
        
        # 遺伝的アルゴリズムで予測を生成
        prediction = self._run_genetic_algorithm(history)
        self.prev_guesses.append(prediction)
        return prediction
        
    def _update_candidates(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """
        履歴に基づいて候補を更新
        """
        # 初回は候補を生成
        if not self.possible_answers and not history:
            self._initialize_candidates()
            return
            
        # 履歴から候補を絞り込む
        if history:
            if not self.possible_answers:
                self._initialize_candidates()
                
            # 各履歴エントリに基づいて候補を絞り込む
            for entry in history:
                guess, result = entry  # (guess, (hits, blows))
                hits, blows = result
                
                # この予測と同じヒット・ブロー数を返す候補だけを残す
                self.possible_answers = {
                    candidate for candidate in self.possible_answers
                    if self._check_hit_blow(guess, candidate) == (hits, blows)
                }
                
    def _initialize_candidates(self) -> None:
        """
        可能性のある全ての答えを生成する
        """
        import itertools
        
        if self.num_digits <= 6:  # 桁数が少ない場合は全候補
            self.possible_answers = set(
                itertools.permutations(range(self.digit_range), self.num_digits)
            )
        else:
            # 桁数が多い場合はランダムサンプリング
            self._generate_random_candidates(10000)
            
    def _generate_random_candidates(self, num_samples: int) -> None:
        """
        ランダムに候補を生成
        """
        for _ in range(num_samples):
            candidate = tuple(random.sample(range(self.digit_range), self.num_digits))
            self.possible_answers.add(candidate)
            
    def _check_hit_blow(self, guess: Tuple[int, ...], answer: Tuple[int, ...]) -> Tuple[int, int]:
        """
        予測と答えのヒット・ブロー数を計算
        """
        hits = sum(1 for i in range(len(guess)) if guess[i] == answer[i])
        
        # ブロー数の計算
        guess_counts = {}
        answer_counts = {}
        
        for g, a in zip(guess, answer):
            if g != a:  # ヒットでない場合のみカウント
                guess_counts[g] = guess_counts.get(g, 0) + 1
                answer_counts[a] = answer_counts.get(a, 0) + 1
        
        # 両方に存在する数字の最小出現回数の合計がブロー数
        blows = sum(min(guess_counts.get(d, 0), answer_counts.get(d, 0)) 
                   for d in set(guess_counts) & set(answer_counts))
        
        return hits, blows
        
    def _run_genetic_algorithm(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        遺伝的アルゴリズムを実行して最適な予測を生成
        
        Args:
            history: これまでの予測履歴
            
        Returns:
            最適な予測
        """
        # 初期個体群を生成
        population = self._initialize_population()
        
        # 各世代で進化
        for generation in range(self.generations):
            # 適応度評価
            fitness_scores = self._evaluate_fitness(population, history)
            
            # 次世代の個体群
            next_generation = []
            
            # エリート選択（最も適応度の高い個体を保存）
            elite_size = max(1, int(self.population_size * 0.1))
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for index in elite_indices:
                next_generation.append(population[index])
                
            # 残りの個体を生成
            while len(next_generation) < self.population_size:
                # 親選択（トーナメント選択）
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                    
                # 突然変異
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # 重複を避けるため、異なる数字を持つように修正
                child1 = self._fix_duplicate_digits(child1)
                child2 = self._fix_duplicate_digits(child2)
                
                # 次世代に追加
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
                    
            # 世代交代
            population = next_generation
            
        # 最終世代から最も適応度の高い個体を選択
        final_fitness = self._evaluate_fitness(population, history)
        best_index = np.argmax(final_fitness)
        best_individual = population[best_index]
        
        # 過去に選んだことがないか確認
        if list(best_individual) in self.prev_guesses:
            # 別の候補を選ぶ
            alternative_candidates = [ind for ind in population if list(ind) not in self.prev_guesses]
            if alternative_candidates:
                best_individual = random.choice(alternative_candidates)
            else:
                # 全て選んだことがある場合、ランダム生成
                best_individual = self._generate_random_prediction()
                
        return list(best_individual)
        
    def _initialize_population(self) -> List[Tuple[int, ...]]:
        """
        初期個体群を生成
        
        Returns:
            個体群（各個体は一連の数字）
        """
        population = []
        
        # 候補から一部をサンプリング
        if self.possible_answers:
            candidates = list(self.possible_answers)
            sample_size = min(len(candidates), self.population_size // 2)
            population.extend(random.sample(candidates, sample_size))
            
        # 残りはランダム生成
        while len(population) < self.population_size:
            individual = tuple(random.sample(range(self.digit_range), self.num_digits))
            if individual not in population:
                population.append(individual)
                
        return population
        
    def _evaluate_fitness(self, population: List[Tuple[int, ...]], 
                        history: List[Tuple[List[int], Tuple[int, int]]]) -> List[float]:
        """
        各個体の適応度を評価
        
        Args:
            population: 個体群
            history: これまでの予測履歴
            
        Returns:
            各個体の適応度スコア
        """
        fitness_scores = []
        
        for individual in population:
            # 過去の予測との一致を避ける（ペナルティ）
            if list(individual) in self.prev_guesses:
                fitness_scores.append(-1000)  # 大きなペナルティ
                continue
                
            # スコア初期化
            score = 0
            
            # 1. 候補集合との整合性スコア
            if self.possible_answers:
                # この予測が区別できる候補の数を評価
                distinct_patterns = set()
                
                # サンプリング（候補が多すぎる場合）
                sample_size = min(len(self.possible_answers), 100)
                sampled_candidates = random.sample(list(self.possible_answers), sample_size)
                
                for candidate in sampled_candidates:
                    hits, blows = self._check_hit_blow(individual, candidate)
                    pattern = (hits, blows)
                    distinct_patterns.add(pattern)
                    
                # 区別できるパターンが多いほど良い
                score += len(distinct_patterns) * 10
                
            # 2. 履歴に基づくスコア（情報利得）
            if history:
                last_guess, last_result = history[-1]  # (guess, (hits, blows))
                last_hits, last_blows = last_result
                
                # 前回のヒット数が多い場合、それを維持するような予測を優遇
                for i, digit in enumerate(individual):
                    if i < len(last_guess) and digit == last_guess[i]:
                        score += last_hits  # ヒット数に応じたボーナス
                
            # 3. 数字の多様性スコア
            unique_digits = len(set(individual))
            score += unique_digits
            
            fitness_scores.append(score)
            
        return fitness_scores
        
    def _tournament_selection(self, population: List[Tuple[int, ...]], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> Tuple[int, ...]:
        """
        トーナメント選択で親を選ぶ
        
        Args:
            population: 個体群
            fitness_scores: 各個体の適応度
            tournament_size: トーナメントサイズ
            
        Returns:
            選ばれた親
        """
        # トーナメント参加者をランダム選択
        indices = random.sample(range(len(population)), tournament_size)
        
        # 最も適応度の高い個体を選択
        best_index = indices[0]
        for i in indices[1:]:
            if fitness_scores[i] > fitness_scores[best_index]:
                best_index = i
                
        return population[best_index]
        
    def _crossover(self, parent1: Tuple[int, ...], parent2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        2つの親から子を生成する交叉
        
        Args:
            parent1: 親1
            parent2: 親2
            
        Returns:
            生成された2つの子
        """
        # 交叉点をランダムに選択
        crossover_point = random.randint(1, self.num_digits - 1)
        
        # 交叉による子の生成
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
        
    def _mutate(self, individual: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        個体に突然変異を適用
        
        Args:
            individual: 元の個体
            
        Returns:
            突然変異後の個体
        """
        individual_list = list(individual)
        
        for i in range(self.num_digits):
            if random.random() < self.mutation_rate:
                # この位置の数字を変異
                new_digit = random.randint(0, self.digit_range - 1)
                while new_digit == individual_list[i]:
                    new_digit = random.randint(0, self.digit_range - 1)
                individual_list[i] = new_digit
                
        return tuple(individual_list)
        
    def _fix_duplicate_digits(self, individual: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        個体内の重複した数字を修正
        
        Args:
            individual: 元の個体
            
        Returns:
            重複がない個体
        """
        individual_list = list(individual)
        seen = set()
        
        for i in range(self.num_digits):
            if individual_list[i] in seen:
                # 重複している場合、未使用の数字を選択
                unused_digits = [d for d in range(self.digit_range) if d not in individual_list or individual_list.count(d) == 1 and d in individual_list[:i]]
                if unused_digits:
                    individual_list[i] = random.choice(unused_digits)
                else:
                    # 未使用の数字がない場合（ありえないが念のため）
                    individual_list[i] = random.randint(0, self.digit_range - 1)
            seen.add(individual_list[i])
            
        return tuple(individual_list)
        
    def _generate_random_prediction(self) -> Tuple[int, ...]:
        """
        ランダムな予測を生成
        
        Returns:
            ランダムな予測
        """
        return tuple(random.sample(range(self.digit_range), self.num_digits)) 