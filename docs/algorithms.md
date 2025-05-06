# Hit and Blowのアルゴリズム解説

このドキュメントでは、Hit and Blowゲームを解くための様々なアルゴリズムについて詳しく解説します。

## 1. ルールベースアプローチ

### 基本原理

ルールベースアプローチは、論理的な推論に基づいて候補を絞り込む方法です。各ターンで得られたヒットとブローの情報を基に、矛盾する候補を除外していきます。

### アルゴリズムの流れ

1. 全ての可能な数字の組み合わせを候補として初期化
2. 予測を行い、ヒット数とブロー数のフィードバックを受け取る
3. そのフィードバックと矛盾する候補を削除
4. 候補が1つになるまで、または最大ターン数に達するまで2-3を繰り返す

### 実装ポイント

```python
# 候補フィルタリングの例
def filter_candidates(self, candidates, guess, hits, blows):
    new_candidates = []
    for candidate in candidates:
        calculated_hits, calculated_blows = self.calculate_hits_blows(candidate, guess)
        if calculated_hits == hits and calculated_blows == blows:
            new_candidates.append(candidate)
    return new_candidates
```

### 長所と短所

- **長所**: 実装が簡単、確実に正解にたどり着く
- **短所**: 効率的な予測選択を行わないため、ターン数が多くなる傾向がある

## 2. 確率的アプローチ

### 基本原理

確率的アプローチは、情報理論に基づいて、各ターンで最も情報量の増加（エントロピーの減少）が期待できる予測を選びます。

### アルゴリズムの流れ

1. 全ての可能な数字の組み合わせを候補として初期化
2. 各候補について、その予測を行った場合の期待情報利得を計算
3. 情報利得が最大となる予測を選択
4. ヒット数とブロー数のフィードバックを受け取り、候補を絞り込む
5. 候補が1つになるまで、または最大ターン数に達するまで2-4を繰り返す

### 実装ポイント

```python
# 情報利得計算の例
def calculate_information_gain(self, guess, candidates):
    total_gain = 0
    # 可能な全ての結果（ヒット数とブロー数の組み合わせ）について
    for hits in range(self.digits + 1):
        for blows in range(self.digits + 1 - hits):
            if hits == self.digits:  # 正解の場合
                continue
            
            # この結果になる候補の数をカウント
            matching_candidates = 0
            for candidate in candidates:
                calculated_hits, calculated_blows = self.calculate_hits_blows(candidate, guess)
                if calculated_hits == hits and calculated_blows == blows:
                    matching_candidates += 1
            
            # この結果が得られる確率
            if matching_candidates > 0:
                probability = matching_candidates / len(candidates)
                # 情報利得の計算（エントロピーの減少量）
                gain = -probability * math.log2(probability)
                total_gain += gain
                
    return total_gain
```

### 長所と短所

- **長所**: ターン数を効率的に減らせる、最適に近い予測が可能
- **短所**: 計算量が多い、大きな桁数や範囲では処理時間が増加

## 3. 強化学習アプローチ

### 基本原理

強化学習アプローチでは、状態と行動（予測）のペアに対して、報酬に基づくQ値を学習します。これにより、ゲームを繰り返すことで最適な戦略を獲得します。

### 3.1 純粋Q学習

#### アルゴリズムの流れ

1. Q値を初期化（通常は0で）
2. 現在の状態を観測
3. ε-greedy法などを用いて行動（予測）を選択
4. 行動を実行し、報酬と次の状態を観測
5. Q値を更新：Q(s,a) ← Q(s,a) + α[r + γ・max<sub>a'</sub> Q(s',a') - Q(s,a)]
6. ゲームが終了するまで2-5を繰り返す

#### 実装ポイント

```python
# Q値更新の例
def update_q_value(self, state, action, reward, next_state):
    current_q = self.q_table.get((state, tuple(action)), 0)
    
    if next_state is None:  # 終端状態
        max_next_q = 0
    else:
        max_next_q = max([self.q_table.get((next_state, tuple(a)), 0) 
                          for a in self.get_possible_actions(next_state)])
    
    # Q値更新式
    new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    self.q_table[(state, tuple(action))] = new_q
```

#### 長所と短所

- **長所**: 経験から学習、環境の事前知識が不要
- **短所**: 状態空間が膨大、純粋なQ学習では収束が難しい

### 3.2 ハイブリッドQ学習

ハイブリッドQ学習では、強化学習とルールベースアプローチを組み合わせます。ルールベースで候補を絞り込みつつ、Q学習で最適な予測を選択します。

#### アルゴリズムの流れ

1. 全ての可能な数字の組み合わせを候補として初期化
2. 現在の状態（残りターン数や候補数など）を観測
3. Q値に基づいて候補の中から予測を選択
4. ヒット数とブロー数のフィードバックを受け取り、候補を絞り込む
5. 報酬を計算し、Q値を更新
6. ゲームが終了するまで2-5を繰り返す

#### 実装ポイント

```python
# ハイブリッドQ学習での行動選択の例
def select_action(self, state, candidates):
    if random.random() < self.epsilon:  # 探索
        return random.choice(candidates)
    else:  # 知識活用
        best_action = None
        best_q = float('-inf')
        
        for action in candidates:
            q_value = self.q_table.get((state, tuple(action)), 0)
            if q_value > best_q:
                best_q = q_value
                best_action = action
                
        return best_action or random.choice(candidates)
```

#### 長所と短所

- **長所**: 候補の絞り込みにより状態空間を大幅に削減、効率的な学習が可能
- **短所**: ルールベース部分の実装が必要、純粋なRLよりも複雑

## 4. 多腕バンディットアプローチ

### 基本原理

多腕バンディットアプローチでは、各予測（行動）を「腕」とみなし、その期待報酬を推定しながら最適な行動を選択します。「探索」と「活用」のバランスを取りながら行動を選びます。

### アルゴリズムの流れ

1. 各行動の価値推定値を初期化
2. Upper Confidence Bound (UCB)などのアルゴリズムを用いて行動を選択
3. 行動を実行し、報酬を観測
4. 行動の価値推定値を更新
5. ゲームが終了するまで2-4を繰り返す

### 実装ポイント

```python
# UCBアルゴリズムによる行動選択の例
def select_action_ucb(self, actions, t):
    best_action = None
    best_ucb = float('-inf')
    
    for action in actions:
        # この行動が選ばれた回数
        n = self.action_counts.get(tuple(action), 0)
        
        if n == 0:  # 一度も選ばれていない行動は優先的に選ぶ
            return action
        
        # 行動の価値推定値
        value = self.action_values.get(tuple(action), 0)
        
        # UCB計算: 価値推定値 + 探索ボーナス
        ucb = value + self.c * math.sqrt(math.log(t) / n)
        
        if ucb > best_ucb:
            best_ucb = ucb
            best_action = action
            
    return best_action
```

### 長所と短所

- **長所**: シンプルな実装、探索と活用のバランスを明示的に制御
- **短所**: 状態を考慮せず行動の価値だけを推定するため、複雑な戦略の学習には限界がある

## 5. ハイブリッドバンディットアプローチ

ハイブリッドバンディットでは、多腕バンディットとルールベースアプローチを組み合わせ、さらにゲームの進行に応じて探索と活用のバランスを調整します。

### アルゴリズムの流れ

1. 全ての可能な数字の組み合わせを候補として初期化
2. ゲームの初期段階では探索重視の戦略（情報収集最大化）
3. 候補が絞られてきたら、活用重視の戦略（最有力候補の選択）
4. ヒット数とブロー数のフィードバックを受け取り、候補を絞り込む
5. ゲームが終了するまで2-4を繰り返す

### 実装ポイント

```python
# ゲーム進行に応じた戦略切替の例
def select_action(self, candidates, turn, max_turns):
    # 候補数に応じて閾値を設定
    if len(candidates) > 30:
        # 候補が多い初期段階: 情報収集を重視
        return self.select_information_gain_action(candidates)
    elif len(candidates) <= 5:
        # 候補が少ない終盤: 最も可能性の高い候補を選択
        return candidates[0]  # 最初の候補を選択
    else:
        # 中盤: UCBアルゴリズムで行動選択
        return self.select_action_ucb(candidates, turn)
```

### 長所と短所

- **長所**: ゲームの進行状況に応じた柔軟な戦略切替、効率的に解を見つけられる
- **短所**: パラメータ調整が必要、実装が複雑になりがち 