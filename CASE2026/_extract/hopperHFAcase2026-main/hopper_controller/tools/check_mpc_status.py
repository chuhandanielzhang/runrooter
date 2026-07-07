#!/usr/bin/env python3
"""
快速验证 MPC 状态脚本（已废弃）
当前 SRB-QP-only 版本不再输出 MPC 相关字段。
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Check MPC status from ModeE CSV log")
    ap.add_argument("csv_file", type=str, help="Path to ModeE CSV log file")
    ap.add_argument("--plot", action="store_true", help="Plot MPC status over time")
    args = ap.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"📊 Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = ["mpc_used", "mpc_status", "mpc_fx_cmd", "mpc_fy_cmd", "stance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("ℹ️  MPC 字段缺失：当前日志格式已移除 MPC（SRB-QP-only 模式）。")
        print(f"缺失列: {missing}")
        print("如需 MPC 分析，请切换到包含 MPC 的版本。")
        sys.exit(0)

    # Filter to stance phase only
    df_stance = df[df["stance"] > 0.5].copy()
    if len(df_stance) == 0:
        print("⚠️  No stance phase data found in CSV")
        sys.exit(0)

    print(f"\n{'='*70}")
    print("MPC 状态验证报告")
    print(f"{'='*70}\n")

    # 1. Check if MPC is enabled/disabled
    mpc_used_any = df_stance["mpc_used"].any()
    mpc_used_ratio = df_stance["mpc_used"].mean()
    mpc_status_unique = df_stance["mpc_status"].unique()

    print(f"📌 MPC 使用情况:")
    print(f"   - stance 阶段使用 MPC 的样本数: {df_stance['mpc_used'].sum():.0f} / {len(df_stance)}")
    print(f"   - MPC 使用率: {mpc_used_ratio*100:.1f}%")
    print(f"   - MPC 状态值: {list(mpc_status_unique)}")

    if not mpc_used_any:
        print(f"   ✅ MPC 已禁用（符合 Raibert-only 模式）")
    else:
        print(f"   ⚠️  MPC 正在使用中")

    # 2. Check MPC solve status
    if mpc_used_any:
        solved_mask = df_stance["mpc_status"].isin(["solved", "solved_inaccurate"])
        solved_count = solved_mask.sum()
        solved_ratio = solved_mask.mean()
        print(f"\n📌 MPC 求解状态:")
        print(f"   - 成功求解: {solved_count} / {len(df_stance)} ({solved_ratio*100:.1f}%)")
        
        if solved_ratio < 0.95:
            print(f"   ⚠️  求解成功率偏低（<95%），可能有数值问题")
        else:
            print(f"   ✅ 求解成功率正常")

        # Check hold_last usage
        hold_last_count = df_stance["mpc_hold_last"].sum()
        if hold_last_count > 0:
            print(f"   - 使用上次解的次数: {hold_last_count}")
            print(f"   ⚠️  有 {hold_last_count} 次使用了上次的 MPC 解（可能 MPC 求解失败）")

    # 3. Check MPC force commands
    fx_cmd_abs_max = df_stance["mpc_fx_cmd"].abs().max()
    fy_cmd_abs_max = df_stance["mpc_fy_cmd"].abs().max()
    
    print(f"\n📌 MPC 水平力指令 (f_ref_xy):")
    print(f"   - |fx_cmd| 最大值: {fx_cmd_abs_max:.2f} N")
    print(f"   - |fy_cmd| 最大值: {fy_cmd_abs_max:.2f} N")

    if not mpc_used_any:
        if fx_cmd_abs_max < 0.1 and fy_cmd_abs_max < 0.1:
            print(f"   ✅ 水平力指令为 0（符合 MPC 禁用状态）")
        else:
            print(f"   ⚠️  水平力指令非零，但 MPC 未使用（可能是 fallback PD 的输出）")
    else:
        if fx_cmd_abs_max > 50 or fy_cmd_abs_max > 50:
            print(f"   ⚠️  水平力指令较大（>50N），检查是否合理")
        else:
            print(f"   ✅ 水平力指令在合理范围")

    # 4. Check MPC age (if using hold_last)
    if mpc_used_any and df_stance["mpc_hold_last"].any():
        age_max = df_stance["mpc_age_s"].max()
        print(f"\n📌 MPC 解年龄（hold_last 时）:")
        print(f"   - 最大年龄: {age_max:.3f} s")
        if age_max > 0.2:
            print(f"   ⚠️  解年龄过大（>0.2s），可能 MPC 长期求解失败")

    # 5. Summary
    print(f"\n{'='*70}")
    print("总结:")
    if not mpc_used_any:
        print("✅ MPC 已禁用，符合 'Raibert-only 速度收敛' 模式")
        print("   - stance 阶段不会通过 MPC/PD 主动收敛速度")
        print("   - 速度收敛完全依赖 flight 阶段的 Raibert 落脚点")
    else:
        print("⚠️  MPC 正在使用中")
        if solved_mask.mean() > 0.95:
            print("✅ MPC 求解正常")
        else:
            print("❌ MPC 求解成功率偏低，需要检查参数/数值问题")

    # Optional: plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            t = df_stance["t"].values
            axes[0].plot(t, df_stance["mpc_used"].values, label="mpc_used")
            axes[0].set_ylabel("MPC Used")
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(t, df_stance["mpc_fx_cmd"].values, label="mpc_fx_cmd")
            axes[1].plot(t, df_stance["mpc_fy_cmd"].values, label="mpc_fy_cmd")
            axes[1].set_ylabel("MPC Force Cmd (N)")
            axes[1].legend()
            axes[1].grid(True)
            
            # Status as text (map to numbers for plotting)
            status_map = {"solved": 1, "solved_inaccurate": 0.8, "disabled": 0, "init": 0.5}
            status_num = [status_map.get(s, 0) for s in df_stance["mpc_status"].values]
            axes[2].plot(t, status_num, label="mpc_status")
            axes[2].set_ylabel("MPC Status")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_yticks([0, 0.5, 0.8, 1])
            axes[2].set_yticklabels(["disabled/other", "init", "solved_inaccurate", "solved"])
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            out_path = csv_path.parent / f"{csv_path.stem}_mpc_status.png"
            plt.savefig(out_path, dpi=150)
            print(f"\n📈 图表已保存: {out_path}")
        except ImportError:
            print("\n⚠️  matplotlib 未安装，跳过绘图")


if __name__ == "__main__":
    main()

