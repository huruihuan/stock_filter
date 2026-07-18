"""可视化模块 - 形态模板、匹配结果、训练过程可视化"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，只保存文件不弹窗
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from .preprocessor import PatternFeatures
from .template_builder import PatternTemplate


class Visualizer:
    def __init__(self, save_dir: str = "output/plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_all_samples(self, template: PatternTemplate, title: str = "所有样本叠加图"):
        """将所有训练样本的归一化价格序列叠加绘制"""
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, seq in enumerate(template.sample_sequences):
            x = np.linspace(0, 1, len(seq))
            ax.plot(x, seq, alpha=0.3, linewidth=1, label=f"样本{i + 1}" if i < 5 else None)

        # 绘制质心
        x_centroid = np.linspace(0, 1, len(template.centroid))
        ax.plot(x_centroid, template.centroid, color="red", linewidth=3, label="质心模板")

        ax.set_xlabel("相对位置")
        ax.set_ylabel("归一化价格变化")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, "samples_overlay.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"样本叠加图已保存: {path}")
        return path

    def plot_template(self, template: PatternTemplate):
        """绘制模板质心及其ZigZag关键点"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：质心序列
        ax = axes[0]
        ax.plot(template.centroid, color="red", linewidth=2)
        ax.fill_between(range(len(template.centroid)), template.centroid, alpha=0.1, color="red")
        ax.set_title("质心模板")
        ax.set_xlabel("K线序号")
        ax.set_ylabel("归一化价格变化")
        ax.grid(True, alpha=0.3)

        # 右图：统计画像
        ax = axes[1]
        profile = template.profile
        labels = ["总涨跌幅", "最大回撤", "最大反弹", "波动率"]
        mins = [profile.return_min, profile.drawdown_min, profile.rally_min, profile.vol_min]
        maxs = [profile.return_max, profile.drawdown_max, profile.rally_max, profile.vol_max]

        y = range(len(labels))
        ax.barh(y, [mx - mn for mn, mx in zip(mins, maxs)], left=mins, height=0.5, color="steelblue")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title("统计画像范围")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, "template.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"模板图已保存: {path}")
        return path

    def plot_match_result(self, df: pd.DataFrame, template: PatternTemplate,
                          ts_code: str, score: float):
        """绘制单只股票的K线与模板叠加对比"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        close = df["close"].values.astype(float)
        vol = df["vol"].values.astype(float)
        dates = range(len(close))

        # 上图：价格 + 模板叠加
        ax = axes[0]
        ax.plot(dates, close, color="black", linewidth=1.5, label="实际价格")

        # 将模板映射到实际价格范围
        norm_centroid = template.centroid
        price_range = close.max() - close.min()
        centroid_range = norm_centroid.max() - norm_centroid.min()
        if centroid_range > 0:
            scaled = (norm_centroid - norm_centroid.min()) / centroid_range * price_range + close.min()
            x_template = np.linspace(0, len(close) - 1, len(scaled))
            ax.plot(x_template, scaled, color="red", linewidth=2, alpha=0.7, linestyle="--",
                    label=f"模板 (score={score:.3f})")

        # 涨跌着色
        for i in range(1, len(close)):
            color = "#d32f2f" if close[i] >= close[i - 1] else "#388e3c"
            ax.bar(i, abs(df["high"].iloc[i] - df["low"].iloc[i]),
                   bottom=min(df["open"].iloc[i], df["close"].iloc[i]),
                   color=color, width=0.6, alpha=0.5)

        ax.set_title(f"{ts_code} - 匹配度 {score:.1%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 下图：成交量
        ax = axes[1]
        colors = ["#d32f2f" if close[i] >= close[i - 1] else "#388e3c"
                  for i in range(1, len(close))]
        colors = ["gray"] + colors
        ax.bar(dates, vol, color=colors, alpha=0.7)
        ax.set_ylabel("成交量")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, f"match_{ts_code.replace('.', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_training_history(self, history: dict):
        """绘制训练过程曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax = axes[0]
        ax.plot(history["train_loss"], label="训练损失")
        ax.plot(history["val_loss"], label="验证损失")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("损失曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # AUC
        ax = axes[1]
        ax.plot(history["val_auc"], color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("验证集AUC")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, "training_history.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"训练曲线已保存: {path}")
        return path

    def plot_scan_results(self, results: list, top_n: int = 20):
        """绘制扫描结果的分数分布"""
        if not results:
            print("无扫描结果")
            return None

        results = sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=(12, 6))
        names = [f"{r.ts_code}\n{r.name}" for r in results]
        scores = [r.score for r in results]

        colors = ["#d32f2f" if s > 0.8 else "#ff9800" if s > 0.6 else "#4caf50" for s in scores]
        ax.barh(range(len(names)), scores, color=colors, height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("匹配分数")
        ax.set_title(f"扫描结果 Top {len(results)}")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        path = os.path.join(self.save_dir, "scan_results.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"扫描结果图已保存: {path}")
        return path
