#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from self_play_trainer import SelfPlayTrainer

def format_time(seconds):
    """秒数を時:分:秒形式にフォーマット"""
    return str(timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow大規模自己対戦学習')
    parser.add_argument('--digits', type=int, default=3, help='桁数 (デフォルト: 3)')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲 0-(range-1) (デフォルト: 6)')
    parser.add_argument('--total-games', type=int, default=10000, help='総対戦回数 (デフォルト: 10000)')
    parser.add_argument('--batch-size', type=int, default=256, help='バッチサイズ (デフォルト: 256)')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数 (デフォルト: 3)')
    parser.add_argument('--save-interval', type=int, default=1000, help='モデル保存間隔 (デフォルト: 1000ゲームごと)')
    parser.add_argument('--load-model', action='store_true', help='保存されたモデルを読み込む')
    
    args = parser.parse_args()
    
    # フォルダの作成
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # 開始時刻
    start_time = time.time()
    print(f"学習開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"設定: 桁数={args.digits}, 範囲=0-{args.range-1}, 総ゲーム数={args.total_games}")
    print(f"バッチサイズ={args.batch_size}, エポック数={args.epochs}, 保存間隔={args.save_interval}ゲームごと")
    
    # 自己対戦トレーナーの初期化
    trainer = SelfPlayTrainer(
        digits=args.digits,
        number_range=args.range,
        allow_repetition=False  # 繰り返し数字を無効化
    )
    
    # モデルの読み込み（指定された場合）
    if args.load_model:
        trainer.load_models()
        print("既存のモデルを読み込みました")
    
    # ゲームをバッチに分割して実行
    batch_games = min(args.save_interval, 100)  # 1バッチあたりのゲーム数
    num_batches = args.total_games // batch_games
    remaining_games = args.total_games % batch_games
    
    game_count = 0
    last_save = 0
    
    for batch in range(num_batches):
        batch_start = time.time()
        
        print(f"\nバッチ {batch+1}/{num_batches} 開始 ({batch_games}ゲーム)")
        
        # バッチ実行
        trainer.train(
            num_games=batch_games,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        game_count += batch_games
        
        # 進捗状況の表示
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start
        avg_time_per_game = elapsed / game_count
        remaining = avg_time_per_game * (args.total_games - game_count)
        
        print(f"進捗: {game_count}/{args.total_games} ゲーム完了 ({game_count/args.total_games*100:.1f}%)")
        print(f"経過時間: {format_time(elapsed)}, バッチ実行時間: {format_time(batch_time)}")
        print(f"1ゲームあたり平均: {avg_time_per_game:.2f}秒")
        print(f"残り時間: 約 {format_time(remaining)}")
        
        # 定期的にモデルを保存
        if game_count - last_save >= args.save_interval:
            checkpoint_path = f"models/checkpoints/checkpoint_{game_count}_games"
            trainer.save_models(
                policy_path=f"{checkpoint_path}_policy.pth",
                value_path=f"{checkpoint_path}_value.pth"
            )
            last_save = game_count
            print(f"チェックポイント保存: {checkpoint_path}")
    
    # 残りのゲームを実行
    if remaining_games > 0:
        print(f"\n残り {remaining_games}ゲームを実行中...")
        trainer.train(
            num_games=remaining_games,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        game_count += remaining_games
    
    # 最終モデルの保存
    trainer.save_models(
        policy_path="models/final_policy_10k.pth",
        value_path="models/final_value_10k.pth"
    )
    
    # 終了時刻と総実行時間
    total_time = time.time() - start_time
    print(f"\n学習完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総実行時間: {format_time(total_time)} ({total_time:.2f}秒)")
    print(f"平均: {total_time / args.total_games:.2f}秒/ゲーム")
    print(f"最終モデルを保存しました: models/final_policy_10k.pth, models/final_value_10k.pth")

if __name__ == "__main__":
    main() 