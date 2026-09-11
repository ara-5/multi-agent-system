"""Plot the episode-reward curve from a Monitor CSV log produced by train.py."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/monitor.csv")
    parser.add_argument("--out", default="assets/reward_curve.png")
    parser.add_argument("--window", type=int, default=50, help="Rolling-average window size")
    args = parser.parse_args()

    df = pd.read_csv(args.log, skiprows=1)  # first row is a SB3 metadata comment
    df["episode"] = range(1, len(df) + 1)
    df["rolling_reward"] = df["r"].rolling(args.window, min_periods=1).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["r"], alpha=0.3, label="episode reward")
    plt.plot(df["episode"], df["rolling_reward"], label=f"{args.window}-episode rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("PPO training reward on simple_spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved reward curve to {args.out}")


if __name__ == "__main__":
    main()
