"""模板构建模块 - 从标注样本中提取形态模板和统计画像"""

import json
import pickle
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .preprocessor import Preprocessor, PatternFeatures

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class PatternProfile:
    """形态统计画像 - 用于规则预筛选"""
    # 长度范围
    length_min: int
    length_max: int
    length_mean: float

    # 涨跌幅范围
    return_min: float
    return_max: float
    return_mean: float

    # 波动率范围
    vol_min: float
    vol_max: float

    # ZigZag结构
    n_zigzag_min: int
    n_zigzag_max: int
    n_zigzag_mean: float

    # 回撤/反弹范围
    drawdown_min: float
    drawdown_max: float
    rally_min: float
    rally_max: float

    # 量能趋势
    vol_trend_min: float
    vol_trend_max: float

    def to_dict(self) -> dict:
        return asdict(self)

    def match(self, feat: PatternFeatures, tolerance: float = 0.3) -> bool:
        """快速判断是否在统计画像范围内（含容差）"""
        def in_range(val, vmin, vmax, tol):
            spread = max(abs(vmax - vmin), 0.01)
            return vmin - spread * tol <= val <= vmax + spread * tol

        checks = [
            in_range(feat.total_return, self.return_min, self.return_max, tolerance),
            in_range(feat.volatility, self.vol_min, self.vol_max, tolerance),
            in_range(feat.n_zigzag, self.n_zigzag_min, self.n_zigzag_max, tolerance),
            in_range(feat.max_drawdown, self.drawdown_min, self.drawdown_max, tolerance),
            in_range(feat.max_rally, self.rally_min, self.rally_max, tolerance),
        ]
        return all(checks)


@dataclass
class PatternTemplate:
    """完整形态模板"""
    # DBA质心序列（归一化价格）
    centroid: np.ndarray
    # 统计画像
    profile: PatternProfile
    # DTW匹配阈值
    threshold: float
    # 语义描述（供LLM使用）
    semantic_description: str
    # 所有样本的特征（用于调试）
    sample_features: List[PatternFeatures]
    # 各样本的原始归一化序列
    sample_sequences: List[np.ndarray]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # 同时保存可读的JSON版本
        json_path = path.replace(".pkl", "_info.json")
        info = {
            "profile": self.profile.to_dict(),
            "threshold": self.threshold,
            "semantic_description": self.semantic_description,
            "centroid_length": len(self.centroid),
            "n_samples": len(self.sample_sequences),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "PatternTemplate":
        with open(path, "rb") as f:
            return pickle.load(f)


class TemplateBuilder:
    def __init__(self, fetcher: DataFetcher = None, preprocessor: Preprocessor = None):
        self.fetcher = fetcher or DataFetcher()
        self.preprocessor = preprocessor or Preprocessor()

    def _detect_pattern_length(self, ts_code: str, end_date: str) -> int:
        """自动检测单个样本的最佳形态长度。
        通过尝试不同窗口，选择ZigZag结构最稳定的长度。
        """
        df_full = self.fetcher.get_sample_data(ts_code, end_date, lookback=config.LOOKBACK)
        if df_full.empty:
            return config.MAX_PATTERN_LEN

        best_len = config.MAX_PATTERN_LEN
        best_score = -1

        for win in range(config.MIN_PATTERN_LEN, min(config.MAX_PATTERN_LEN + 1, len(df_full) + 1), 5):
            df_win = df_full.tail(win).reset_index(drop=True)
            feat = self.preprocessor.extract_features(df_win)
            # 评分：ZigZag点数适中（不太多不太少）且振幅较大
            if feat.n_zigzag < 2:
                continue
            avg_amp = np.mean(feat.zigzag_amplitudes) if feat.zigzag_amplitudes else 0
            score = avg_amp * min(feat.n_zigzag, 10) / max(feat.n_zigzag, 1)
            if score > best_score:
                best_score = score
                best_len = win

        return best_len

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """基础DTW距离计算"""
        n, m = len(s1), len(s2)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        band = max(int(max(n, m) * config.DTW_BAND_RATIO), abs(n - m) + 1)

        for i in range(1, n + 1):
            for j in range(max(1, i - band), min(m + 1, i + band + 1)):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    def _dba_centroid(self, sequences: List[np.ndarray], n_iter: int = 10) -> np.ndarray:
        """DBA (DTW Barycenter Averaging) 计算质心序列"""
        # 以中位长度序列为初始质心
        lengths = [len(s) for s in sequences]
        median_len = int(np.median(lengths))
        # 选择长度最接近中位数的序列
        closest_idx = np.argmin([abs(len(s) - median_len) for s in sequences])
        centroid = sequences[closest_idx].copy()

        for _ in range(n_iter):
            # 对每个序列做DTW对齐到当前质心
            associations = [[] for _ in range(len(centroid))]

            for seq in sequences:
                n, m = len(seq), len(centroid)
                dtw = np.full((n + 1, m + 1), np.inf)
                dtw[0, 0] = 0

                for i in range(1, n + 1):
                    for j in range(1, m + 1):
                        cost = abs(seq[i - 1] - centroid[j - 1])
                        dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

                # 回溯
                i, j = n, m
                while i > 0 and j > 0:
                    associations[j - 1].append(seq[i - 1])
                    candidates = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
                    i, j = min(candidates, key=lambda p: dtw[p[0], p[1]])

            # 更新质心
            centroid = np.array([np.mean(a) if a else centroid[k] for k, a in enumerate(associations)])

        return centroid

    def build(self, samples: List[Tuple[str, str]], auto_detect_length: bool = True) -> PatternTemplate:
        """从标注样本构建形态模板。
        samples: [(ts_code, end_date), ...]
        """
        print(f"开始构建模板，共 {len(samples)} 个样本...")

        # 1. 获取数据并提取特征
        all_features = []
        all_sequences = []
        all_dfs = []
        pattern_lengths = []

        for i, (code, date) in enumerate(samples):
            print(f"  处理样本 {i + 1}/{len(samples)}: {code} @ {date}")

            if auto_detect_length:
                plen = self._detect_pattern_length(code, date)
            else:
                plen = config.MAX_PATTERN_LEN
            pattern_lengths.append(plen)

            df = self.fetcher.get_sample_data(code, date, lookback=plen)
            if df.empty:
                print(f"    警告: 无法获取数据，跳过")
                continue

            feat = self.preprocessor.extract_features(df)
            all_features.append(feat)
            all_sequences.append(feat.norm_close)
            all_dfs.append(df)

        if len(all_features) < 3:
            raise ValueError(f"有效样本不足（需至少3个，实际{len(all_features)}个）")

        # 2. 构建统计画像
        profile = PatternProfile(
            length_min=min(f.length for f in all_features),
            length_max=max(f.length for f in all_features),
            length_mean=np.mean([f.length for f in all_features]),
            return_min=min(f.total_return for f in all_features),
            return_max=max(f.total_return for f in all_features),
            return_mean=np.mean([f.total_return for f in all_features]),
            vol_min=min(f.volatility for f in all_features),
            vol_max=max(f.volatility for f in all_features),
            n_zigzag_min=min(f.n_zigzag for f in all_features),
            n_zigzag_max=max(f.n_zigzag for f in all_features),
            n_zigzag_mean=np.mean([f.n_zigzag for f in all_features]),
            drawdown_min=min(f.max_drawdown for f in all_features),
            drawdown_max=max(f.max_drawdown for f in all_features),
            rally_min=min(f.max_rally for f in all_features),
            rally_max=max(f.max_rally for f in all_features),
            vol_trend_min=min(f.vol_trend for f in all_features),
            vol_trend_max=max(f.vol_trend for f in all_features),
        )

        # 3. DBA质心
        print("计算DBA质心...")
        centroid = self._dba_centroid(all_sequences)

        # 4. 计算阈值
        distances = [self._dtw_distance(seq, centroid) for seq in all_sequences]
        threshold = np.mean(distances) + 1.5 * np.std(distances)
        print(f"DTW距离分布: mean={np.mean(distances):.4f}, std={np.std(distances):.4f}, threshold={threshold:.4f}")

        # 5. 生成语义描述
        semantic_desc = self._generate_semantic_description(all_features, all_dfs, profile)

        template = PatternTemplate(
            centroid=centroid,
            profile=profile,
            threshold=threshold,
            semantic_description=semantic_desc,
            sample_features=all_features,
            sample_sequences=all_sequences,
        )

        print("模板构建完成！")
        return template

    def _generate_semantic_description(self, features: List[PatternFeatures],
                                       dfs: List[pd.DataFrame],
                                       profile: PatternProfile) -> str:
        """综合所有样本生成形态的语义描述"""
        lines = []
        lines.append("=== 形态语义描述 ===\n")
        lines.append(f"样本数量: {len(features)}")
        lines.append(f"形态长度: {profile.length_min}-{profile.length_max}根K线 (均值{profile.length_mean:.0f})")
        lines.append(f"总涨跌幅范围: {profile.return_min:+.2%} ~ {profile.return_max:+.2%}")
        lines.append(f"最大回撤范围: {profile.drawdown_min:.2%} ~ {profile.drawdown_max:.2%}")
        lines.append(f"最大反弹范围: {profile.rally_min:.2%} ~ {profile.rally_max:.2%}")
        lines.append(f"波动率范围: {profile.vol_min:.4f} ~ {profile.vol_max:.4f}")
        lines.append(f"ZigZag转折点数: {profile.n_zigzag_min}-{profile.n_zigzag_max}")

        # 分析共性ZigZag结构
        lines.append("\n--- 典型ZigZag结构（取自各样本的共性） ---")
        # 取中位数长度样本作为代表
        median_nzz = int(np.median([f.n_zigzag for f in features]))
        representative = min(features, key=lambda f: abs(f.n_zigzag - median_nzz))
        text = self._describe_zigzag_structure(representative)
        lines.append(text)

        # 逐样本文本描述
        lines.append("\n--- 各样本详情 ---")
        for i, (feat, df) in enumerate(zip(features, dfs)):
            desc = self.preprocessor.features_to_text(feat, df)
            lines.append(f"\n[样本{i + 1}]\n{desc}")

        return "\n".join(lines)

    def _describe_zigzag_structure(self, feat: PatternFeatures) -> str:
        """将ZigZag结构转为自然语言描述"""
        points = feat.zigzag_points
        if not points:
            return "无明显转折点"

        parts = []
        for i, pt in enumerate(points):
            dir_str = "高点" if pt.direction == 1 else "低点"
            pos_str = "前期" if pt.rel_position < 0.33 else "中期" if pt.rel_position < 0.66 else "后期"
            vol_str = "放量" if pt.volume_ratio > 1.3 else "缩量" if pt.volume_ratio < 0.7 else "平量"
            parts.append(f"在{pos_str}(位置{pt.rel_position:.0%})形成{dir_str}(价格变化{pt.price:+.2%})，{vol_str}")

        return "形态走势：\n" + "\n→ ".join(parts)

    def cross_validate(self, samples: List[Tuple[str, str]]) -> dict:
        """Leave-one-out交叉验证"""
        print("开始交叉验证...")
        results = []

        for i in range(len(samples)):
            train = samples[:i] + samples[i + 1:]
            test_code, test_date = samples[i]

            try:
                template = self.build(train, auto_detect_length=True)

                # 获取测试样本数据并匹配
                df = self.fetcher.get_sample_data(test_code, test_date, lookback=config.MAX_PATTERN_LEN)
                feat = self.preprocessor.extract_features(df)
                dist = self._dtw_distance(feat.norm_close, template.centroid)

                hit = dist <= template.threshold
                results.append({
                    "sample": f"{test_code}@{test_date}",
                    "distance": dist,
                    "threshold": template.threshold,
                    "hit": hit,
                })
                print(f"  样本{i + 1}: dist={dist:.4f}, thr={template.threshold:.4f}, {'命中' if hit else '未命中'}")
            except Exception as e:
                print(f"  样本{i + 1}: 出错 - {e}")
                results.append({"sample": f"{test_code}@{test_date}", "error": str(e)})

        hit_rate = sum(1 for r in results if r.get("hit", False)) / len(results)
        print(f"\n交叉验证命中率: {hit_rate:.1%} ({sum(1 for r in results if r.get('hit', False))}/{len(results)})")
        return {"results": results, "hit_rate": hit_rate}
