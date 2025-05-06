import random
import argparse
from hit_and_blow.core.environment import HitAndBlowEnv

def main():
    """Hitブロウゲームのメインエントリポイント"""
    parser = argparse.ArgumentParser(description="Hit and Blow（数当てゲーム）")
    parser.add_argument("--digits", type=int, default=4, help="桁数 (デフォルト: 4)")
    parser.add_argument("--range", type=int, default=10, help="数字の範囲 (0〜range-1) (デフォルト: 10)")
    parser.add_argument("--turns", type=int, default=10, help="最大ターン数 (デフォルト: 10)")
    args = parser.parse_args()

    env = HitAndBlowEnv(num_digits=args.digits, digit_range=args.range, max_turns=args.turns)
    obs = env.reset()
    
    print("======= Hit and Blow ゲーム =======")
    print(f"{args.digits}桁の数字を当ててください（各桁は0～{args.range-1}の重複なし）")
    print(f"最大{args.turns}回の試行で当てることができます")
    print("入力方法: スペース区切り（例: 1 4 7 9）または連続入力（例: 1479）")
    print("================================")
    
    # Debug mode for demonstration
    debug_mode = False
    if debug_mode:
        print("秘密の数字 (デバッグモード):", env.secret)

    done = False
    while not done:
        # プレイヤーの入力
        valid_input = False
        while not valid_input:
            try:
                guess_str = input(f"\nターン {env.turn + 1}/{env.max_turns}. 数字を入力してください: ")
                
                # スペースで区切られている場合はそのまま分割
                if " " in guess_str:
                    guess = [int(x) for x in guess_str.split()]
                # スペースがない場合は一文字ずつ分割
                else:
                    guess = [int(x) for x in guess_str]
                
                if len(guess) != env.num_digits:
                    print(f"{env.num_digits}桁の数字を入力してください")
                    continue
                    
                if any(d < 0 or d >= env.digit_range for d in guess):
                    print(f"各桁は0～{env.digit_range-1}の数字を入力してください")
                    continue
                    
                if len(set(guess)) != len(guess):
                    print("各桁は重複しない数字を入力してください")
                    continue
                    
                valid_input = True
            except ValueError:
                print("数字を正しく入力してください")
        
        # ゲームを進める
        obs, reward, done, info = env.step(guess)
        
        # 結果表示
        print(f"結果: {info['hits']} Hit, {info['blows']} Blow")
        
        if done:
            if info['hits'] == env.num_digits:
                print("\nおめでとうございます！正解です！")
                print(f"ターン {env.turn}/{env.max_turns} で正解しました")
            else:
                print("\nゲームオーバー...")
                print(f"正解は: {env.secret}")

    # 履歴表示
    print("\n----- プレイ履歴 -----")
    env.render()

if __name__ == "__main__":
    main() 