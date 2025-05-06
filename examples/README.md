# Hit and Blow - サンプルスクリプト

このディレクトリには、Hit and Blowゲームのサンプルスクリプトが含まれています。

## hybrid_q_advisor.py

アドバイザーモードでハイブリッドQ学習エージェントを使用するスクリプトです。

```bash
# 実行例
poetry run python examples/hybrid_q_advisor.py --digits 3 --range 6
```

## adversarial_game.py

対戦型Hit and Blowゲームを実験できるスクリプトです。2人のAIプレイヤーが交互に予測を行い、先に相手の秘密の数列を当てたプレイヤーが勝者となります。

### 基本的な使い方

```bash
# 基本的なゲーム実行（hybrid_q_learning vs probabilistic）
poetry run python examples/adversarial_game.py

# エージェントタイプの指定
poetry run python examples/adversarial_game.py --p1 hybrid_q_learning --p2 rule_based

# 桁数と範囲の指定
poetry run python examples/adversarial_game.py --digits 4 --range 8

# 複数ゲームの実行と統計
poetry run python examples/adversarial_game.py --games 10
```

### 防御的な数列の分析

どのような秘密の数列が最も防御力が高いか（AIエージェントが解くのに多くのターンを要するか）を分析できます。

```bash
# 防御的な数列の分析モード
poetry run python examples/adversarial_game.py --mode analyze

# 特定のエージェントタイプに対する分析
poetry run python examples/adversarial_game.py --mode analyze --analyze-agents rule_based,probabilistic

# サンプル数の指定
poetry run python examples/adversarial_game.py --mode analyze --samples 100
```

### 相手の進捗に応じた戦略調整

このスクリプトでは、相手の進捗状況に応じてエージェントが戦略を動的に調整する機能も実装されています。例えば、相手が有利な場合（多くのヒットを得ている場合）、よりリスクの高い予測を行うようになります。

この機能は以下のように無効化することもできます：

```bash
# 戦略調整を無効化
poetry run python examples/adversarial_game.py --no-adapt
```

### その他のオプション

```bash
# 数字の繰り返しを許可
poetry run python examples/adversarial_game.py --repetition

# ヘルプの表示
poetry run python examples/adversarial_game.py --help
```

# Hit and Blow実行例

このディレクトリには、Hit and Blowゲームの様々な実行例が含まれています。

## 基本的な対戦型ゲームの実行方法

### 対人戦モード

```bash
poetry run python examples/adversarial_game.py --p1 rule_based --p2 probabilistic
```

### オプション一覧

- `--p1`, `--p2`: プレイヤータイプ（rule_based, probabilistic, hybrid_q_learning）
- `--digits`: 桁数（デフォルト: 3）
- `--range`: 数字の範囲 0-(range-1)（デフォルト: 6）
- `--games`: 実行するゲーム数（デフォルト: 1）
- `--no-adapt`: 相手の進捗に応じた戦略調整を無効化
- `--mode`: ゲームモード（play または analyze）
- `--analyze-agents`: 分析対象のエージェント（カンマ区切り）
- `--samples`: 分析するサンプル数
- `--repetition`: 数字の繰り返しを許可（デフォルトは禁止）
- `--help`: ヘルプメッセージを表示

## 対戦型ゲームの使用例

### 基本的な対戦

```bash
# ハイブリッドQ学習エージェントと確率的エージェントを対戦させる
poetry run python examples/adversarial_game.py --p1 hybrid_q_learning --p2 probabilistic
```

### 複数回の対戦を実行してフィードバック適応の効果を確認

```bash
# 戦略調整あり：10回の対戦
poetry run python examples/adversarial_game.py --p1 hybrid_q_learning --p2 probabilistic --games 10

# 戦略調整なし：10回の対戦
poetry run python examples/adversarial_game.py --p1 hybrid_q_learning --p2 probabilistic --games 10 --no-adapt
```

### 防御的な数列分析

```bash
# 3種類のエージェントに対して防御力の高い数列を分析
poetry run python examples/adversarial_game.py --mode analyze --analyze-agents rule_based,probabilistic,hybrid_q_learning --samples 30
```

### カスタム設定

```bash
# 4桁、0-9の範囲で対戦
poetry run python examples/adversarial_game.py --p1 hybrid_q_learning --p2 probabilistic --digits 4 --range 10
```

## 自己対戦学習機能（アルファ碁スタイル）

自己対戦によるディープラーニング強化学習機能を提供します。

### 基本的な自己対戦学習の実行方法

```bash
# 自己対戦の実行
poetry run python examples/self_play_simple.py --games 100

# バッチ処理での自己対戦
poetry run python examples/self_play_batch.py --batch-size 10 --iterations 10
```

### 自己対戦学習のオプション

- `--digits`: 桁数（デフォルト: 3）
- `--range`: 数字の範囲 0-(range-1)（デフォルト: 6）
- `--games`: 自己対戦回数（デフォルト: 100）
- `--batch-size`: 学習のバッチサイズ（デフォルト: 32）
- `--epochs`: エポック数（デフォルト: 5）
- `--load-model`: 保存されたモデルを読み込む

### 学習済みモデルを使った対戦

```bash
# 学習済みモデルを読み込んで追加学習
poetry run python examples/self_play_simple.py --load-model --games 50

# 学習済みモデルを使った対戦（TODO: 実装予定）
poetry run python examples/adversarial_game.py --p1 self_play --p2 hybrid_q_learning
```

### 学習のポイント

1. **ネットワーク構造**: 状態を表現する特徴量として、過去の予測履歴とヒット・ブロー結果を使用します

2. **モンテカルロ木探索**: ゲーム木を効率的に探索し、最適な予測を選択します

3. **自己対戦の仕組み**: 現在のベストモデルと対戦し、モデルを継続的に改善します

4. **ハイパーパラメータ調整**: バッチサイズやエポック数を調整することで学習効率を向上させることができます

### 注意事項

- GPU環境があると学習が大幅に高速化されます（自動検出されます）
- 学習済みモデルは `models/` ディレクトリに保存されます
- 繰り返し数字は無効になっています（ルール設定に合わせて）
- 引き分けルール（同じターン数で当てた場合）が適用されています

## シンプルなアドバイザーモード

```bash
# シンプルなアドバイザーの起動
poetry run python examples/advisor_mode.py --digits 3 --range 6
```

シンプルなアドバイザーでは、プレイヤーが予測と結果を入力すると、候補を絞り込んで次の予測のヒントを提供します。

### 使い方

1. 外部ゲームでHit and Blowゲームを開始
2. あなたの予測と結果（ヒット数、ブロー数）を入力
   例: `123 1 1`（123を予測して、1ヒット1ブローだった場合）
3. アドバイザーから次の予測候補を確認
4. 正解するまで繰り返し

### コマンド

- `hint` - 次の予測候補を表示
- `reset` - ゲームをリセット
- `exit` - 終了

## 外部ゲーム用アドバイザーの使用方法

外部のHit and Blowゲーム（他のアプリやゲームサイトなど）をプレイしている際に、Q学習モデルから最適な次の予測についてアドバイスを受けることができます。このモードでは、あなたが行った予測と得られた結果（ヒットとブロー）を入力すると、次に試すべき数字を提案してくれます。

### 基本的な使用方法

```bash
# 基本的な使用方法（デフォルト設定）
poetry run python examples/external_advisor.py

# 異なるQ学習モデルを使用
poetry run python examples/external_advisor.py --model models/my_model.pkl

# 異なる桁数と範囲を設定
poetry run python examples/external_advisor.py --digits 4 --range 8
```

### 外部アドバイザーでのコマンド

外部アドバイザー内では、以下のコマンドが利用可能です：

- `h`または`help` - ヘルプを表示
- `q`、`quit`、`exit` - アドバイザーを終了
- `reset` - ゲームをリセット
- `show` - 現在の状態を表示
- `skip` - 予測を飛ばす（すでに予測を行った場合）

### 外部アドバイザーの使い方

1. 外部ゲーム（他のアプリやウェブサイト）でHit and Blowゲームを開始
2. アドバイザーからのアドバイスを確認
3. 外部ゲームで予測を入力し、結果（ヒットとブロー）を確認
4. アドバイザーに予測と結果を入力
5. 次のアドバイスを確認して、外部ゲームで次の予測を行う
6. 正解するまで繰り返し

### 注意事項

- 外部アドバイザーを使用するには、事前に学習済みのQ学習モデルが必要です
- モデルの学習は`qlearning_trainer.py`スクリプトを使用して行います
- デフォルトでは、モデルは`models/multi_q_learning_model.pkl`に保存されます

## Hit and Blow サンプルコード

このディレクトリには、Hit and Blowゲームに関連するサンプルコードが含まれています。

### ゲームの実行方法

基本的なゲームは以下のコマンドで実行できます：

```bash
poetry run python examples/play_simple.py
```

カスタムパラメータを指定する場合：

```bash
poetry run python examples/play_simple.py --digits 4 --range 10 --agent-type rule_based
```

### 外部アドバイザー

外部アドバイザーは、他のゲームシステムでプレイしている場合に、プレイヤーにアドバイスを提供するツールです。
プレイヤーが予測と結果を入力すると、Q学習モデルが次の予測についてアドバイスを提供します。

```bash
poetry run python examples/external_advisor.py --model models/multi_q_learning_model.pkl --digits 3 --range 6
```

### 入力方法（外部アドバイザー）

#### コマンド
- `q`, `quit`, `exit` - 終了
- `h`, `help` - ヘルプを表示
- `reset` - ゲームをリセット
- `show` - 現在の状態を表示

#### 数字の入力
数字は以下のいずれかの形式で入力できます：
- 連続した数字（例: `012`）
- カンマ区切り（例: `0,1,2`）
- スペース区切り（例: `0 1 2`）

#### 結果の入力
ヒット数とブロー数は以下のいずれかの形式で入力できます：
- カンマ区切り（例: `1,2`）
- スペース区切り（例: `1 2`）
- 連続した数字（例: `12` → ヒット1, ブロー2）

### Q学習トレーナー

Q学習エージェントのトレーニングと評価を行うスクリプトです。

```bash
# トレーニングモード
poetry run python examples/qlearning_trainer.py --mode train --episodes 1000 --eval-games 20

# 評価モード
poetry run python examples/qlearning_trainer.py --mode evaluate --load-model models/q_learning_model.pkl --eval-games 50

# トレーニングと評価を同時に実行
poetry run python examples/qlearning_trainer.py --mode both --episodes 1000 --eval-games 20
```

### その他のサンプル

`simple_game.py` - 基本的なコマンドラインゲーム
`multi_game.py` - 複数のゲームを連続して実行し、統計情報を表示

## compare_agents.py

このスクリプトは、異なるタイプのエージェントのパフォーマンスを直接比較するためのツールです。各エージェントがHit and Blowゲームをどれだけ効率的に解くかを比較し、結果を統計的に分析します。

### 基本的な使用方法

```bash
# 基本的な使用例（rulebaseエージェントとprobabilisticエージェントを比較）
poetry run python examples/compare_agents.py --agents rulebase probabilistic --games 10 --digits 3 --range 6

# 詳細な出力を表示
poetry run python examples/compare_agents.py --agents rulebase probabilistic --games 10 --digits 3 --range 6 --verbose

# グラフを表示せず結果のみを表示
poetry run python examples/compare_agents.py --agents rulebase probabilistic hybrid_q_learning --games 5 --digits 3 --range 6 --no-plot
```

### コマンドラインオプション

- `--agents`: 比較するエージェント名（複数指定可能、スペース区切り）
- `--games`: 各エージェントで実行するゲーム数（デフォルト: 100）
- `--digits`: 桁数（デフォルト: 3）
- `--range`: 数字の範囲 0-(range-1)（デフォルト: 6）
- `--turns`: 最大ターン数（デフォルト: 10）
- `--verbose`: 詳細な情報を表示
- `--no-plot`: グラフを表示しない

### 利用可能なエージェント

```bash
- rulebase
- probabilistic
- heuristic
- greedy
- monte_carlo
- pattern_matching
- genetic
- hybrid_q_learning
- q_learning
- sarsa
- hybrid_sarsa
- bandit
- hybrid_bandit
```

### 性能比較の指標

- **勝率**: エージェントが最大ターン数内に正解を見つけた割合
- **平均ターン数**: 正解を見つけるのに要した平均ターン数
- **平均実行時間**: 予測を生成するのに要した平均時間（秒）

### 結果の可視化

`--no-plot`オプションを指定しない場合、以下のグラフが生成されます：

1. 勝率の比較グラフ
2. 平均ターン数の比較グラフ
3. 平均実行時間の比較グラフ
4. ターン数分布グラフ（各エージェントが何ターンで解いたかの分布）

グラフは`agent_comparison.png`と`turns_distribution.png`として保存されます。 