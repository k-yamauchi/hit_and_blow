"""Hit and Blowゲームのユーティリティ関数"""

import itertools
from typing import List, Tuple, Set


def generate_all_possible_codes(digits: int, number_range: int, allow_repetition: bool = False) -> Set[Tuple[int, ...]]:
    """
    全ての可能な数列を生成する
    
    Args:
        digits (int): 桁数
        number_range (int): 数字の範囲 (0からnumber_range-1)
        allow_repetition (bool): 数字の繰り返しを許可するか
        
    Returns:
        Set[Tuple[int, ...]]: 可能な数列の集合
    """
    if allow_repetition:
        # 数字の繰り返しを許可する場合は、全ての組み合わせを生成
        candidates = set(itertools.product(range(number_range), repeat=digits))
    else:
        # 数字の繰り返しを許可しない場合は、順列を生成
        if digits > number_range:
            raise ValueError(f"桁数({digits})が範囲({number_range})より大きい場合、繰り返しなしの組み合わせは生成できません")
        candidates = set(itertools.permutations(range(number_range), digits))
        
    return candidates
    

def calculate_hits_blows(guess: List[int], answer: List[int]) -> Tuple[int, int]:
    """
    予測と答えに基づいて、ヒット数とブロー数を計算する
    
    Args:
        guess (List[int]): 予測した数列
        answer (List[int]): 正解の数列
        
    Returns:
        Tuple[int, int]: (ヒット数, ブロー数)
    """
    hits = 0
    blows = 0
    
    # ヒット数を計算
    for g, a in zip(guess, answer):
        if g == a:
            hits += 1
    
    # ブロー数を計算
    # 各数字の出現回数を数える
    guess_counts = {}
    answer_counts = {}
    
    for g in guess:
        guess_counts[g] = guess_counts.get(g, 0) + 1
        
    for a in answer:
        answer_counts[a] = answer_counts.get(a, 0) + 1
    
    # 各数字について、予測と答えの出現回数の最小値を合計
    for num, count in guess_counts.items():
        if num in answer_counts:
            blows += min(count, answer_counts[num])
    
    # ヒットはブローにも含まれているので、ヒット数を引く
    blows -= hits
    
    return hits, blows 