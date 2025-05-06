# Hit and Blow Game AI

Hit and Blow（別名：Mastermind、数当てゲーム）を解くための様々なAIアルゴリズム実装です。ルールベースから強化学習まで、異なるアプローチでゲームを解く方法を比較・検証しています。

## 🎮 ゲームの概要

**Hit and Blow**は、プレイヤーが相手の秘密の数列を当てるゲームです：

- 相手は特定の桁数（例：3桁）の数列を秘密に選びます
- プレイヤーは予測を行い、以下のフィードバックを得ます：
  - **Hit**: 正しい位置に正しい数字がある数
  - **Blow**: 数字は正しいが位置が異なる数
- プレイヤーは最小のターン数で正解を見つけることを目指します

## 🚀 特徴

- **複数のAIアプローチ**:
  - ルールベースエージェント
  - 確率的エージェント
  - Q学習エージェント
  - SARSAエージェント
  - ハイブリッドQ学習エージェント
  - ハイブリッドSARSAエージェント
  - 多腕バンディットエージェント
  - ハイブリッドバンディットエージェント

- **比較・評価ツール**:
  - エージェント間のパフォーマンス比較
  - 統計とグラフの生成
  - 詳細なログ出力

- **対話モード**:
  - AIとのプレイ
  - AIによるプレイのアドバイス

## 📋 リポジトリ構造

```
hit_and_blow/
├── hit_and_blow/       # メインパッケージ
│   ├── agents/         # 各種AIエージェントの実装
│   ├── core/           # ゲームのコア機能
│   └── interface/      # CLIインターフェース
├── examples/           # サンプルスクリプト
├── models/             # 訓練済みモデル
├── tests/              # テストコード
└── docs/               # ドキュメント
```

## 🔧 インストール

### 必要条件
- Python 3.8以上
- Poetry（依存関係管理用）

```bash
# Poetry を使ったインストール（推奨）
poetry install

# または pip を使用
pip install -e .
```

## 📚 使い方

### AIエージェントの訓練

```bash
# Q学習エージェントの訓練（3桁、0-5の数字）
poetry run train-rl --agent hybrid_q_learning --episodes 1000 --digits 3 --range 6

# バンディットエージェントの訓練
poetry run train-rl --agent hybrid_bandit --episodes 1000 --digits 3 --range 6
```

### AIとプレイ

```bash
# 訓練済みのAIエージェントとプレイ
poetry run play-agent --agent hybrid_q_learning --model models/q_learning_model.pkl
```

### エージェントの比較

```bash
# 異なるエージェントの性能比較（RLエージェント）
poetry run compare-rl --episodes 1000 --test 100 --digits 3 --range 6

# 複数のエージェントタイプの直接比較
poetry run compare-agents --agents rulebase probabilistic hybrid_q_learning --games 10 --digits 3 --range 6 --verbose
```

### アドバイザーモード

```bash
# 外部アドバイザーモードの起動（他のアプリやウェブサイトでのプレイをサポート）
poetry run python examples/external_advisor.py --digits 3 --range 6

# シンプルなアドバイザーモードの起動
poetry run python examples/advisor_mode.py --digits 3 --range 6
```

### ハイブリッドアドバイザー

```bash
# ハイブリッドQ学習アドバイザーの起動
poetry run python examples/hybrid_q_advisor.py --digits 3 --range 6
```

## 🧪 テストの実行

プロジェクトには様々なテストが含まれており、以下のコマンドで実行できます：

### すべてのテストを実行

```bash
# Poetryを使用する場合
poetry run python -m unittest discover -s tests

# Pipを使用する場合
python -m unittest discover -s tests
```

### 特定のテストファイルを実行

```bash
# アドバイザーのテスト
poetry run python tests/test_advisor.py

# ハイブリッドQ学習エージェントのテスト
poetry run python tests/test_hybrid_q_learning.py

# エージェントと環境の相互作用テスト
poetry run python tests/test_agent_environment.py
```

## 📊 性能比較

各アプローチの比較結果（3桁、範囲0-5、テスト100回）:

| アプローチ | 勝率 | 平均ターン数 |
|------------|------|--------------|
| ルールベース | 100% | 5.5 |
| 確率的 | 100% | 4.7 |
| 純粋Q学習 | 0% | 10.00 |
| SARSA | 15.0% | 9.65 |
| ハイブリッドQ学習 | 100% | 3.55 |
| ハイブリッドSARSA | 100% | 3.65 |
| 純粋バンディット | 10.0% | 9.50 |
| ハイブリッドバンディット | 100% | 4.10 |

## 📋 エージェントの説明

### ルールベースエージェント

最も基本的なアプローチで、明示的なルールに基づいて予測を行います。候補をフィルタリングして、最初に見つかった有効な候補を選択します。

### 確率的エージェント

情報理論に基づいて、エントロピーを最大化する予測を選びます。より効率的に候補を絞り込むことができます。

### ハイブリッドQ学習エージェント

強化学習（Q学習）と候補管理を組み合わせたアプローチです。過去の経験から学習し、最適な予測を選びます。性能は最も高いです。

### ハイブリッドバンディットエージェント

多腕バンディットアルゴリズムとQ学習を組み合わせたハイブリッドアプローチです。初期段階では情報収集を最大化し、後期段階では最適解に向かいます。


## 🔗 関連リソース

- [アルゴリズム解説](./docs/algorithms.md)
- [発展例](./docs/advanced_usage.md)

## 📄 ライセンス

MITライセンスのもとで公開しています。詳細は[LICENSE](./LICENSE)ファイルをご覧ください。
