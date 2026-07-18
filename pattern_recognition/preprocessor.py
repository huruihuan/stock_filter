"""预处理模块 - 归一化、ZigZag关键点提取、特征工程"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class ZigZagPoint:
    """ZigZag转折点"""
    index: int          # 在原始序列中的位置
    price: float        # 归一化后的价格
    direction: int      # 1=高点, -1=低点
    rel_position: float # 相对位置 (0~1)
    volume_ratio: float # 该点的量比


@dataclass
class PatternFeatures:
    """形态特征集"""
    # 基础统计
    length: int                        # K线根数
    total_return: float                # 总涨跌幅
    max_drawdown: float                # 最大回撤
    max_rally: float                   # 最大反弹
    volatility: float                  # 波动率

    # ZigZag结构
    zigzag_points: List[ZigZagPoint]   # 关键转折点序列
    n_zigzag: int                      # 转折点数量
    zigzag_amplitudes: List[float]     # 每段振幅

    # 量价特征
    vol_trend: float                   # 量能趋势（斜率）
    avg_vol_ratio_at_peaks: float      # 高点处平均量比
    avg_vol_ratio_at_troughs: float    # 低点处平均量比

    # 均线特征
    ma_positions: dict = field(default_factory=dict)   # 各均线相对价格的位置关系
    ma_cross_events: List[str] = field(default_factory=list)  # 均线交叉事件

    # MACD特征
    macd_histogram_trend: str = ""     # MACD柱状图趋势
    macd_cross_events: List[str] = field(default_factory=list)  # 金叉/死叉事件
    macd_divergence: str = ""          # 背离情况

    # CCI特征
    cci_zones: List[str] = field(default_factory=list)  # CCI区间变化序列
    cci_extremes: List[dict] = field(default_factory=list)  # CCI极值点

    # 归一化价格序列（用于DTW）
    norm_close: np.ndarray = field(default_factory=lambda: np.array([]))
    norm_volume: np.ndarray = field(default_factory=lambda: np.array([]))

    # K线特征序列
    candle_features: np.ndarray = field(default_factory=lambda: np.array([]))


class Preprocessor:
    def __init__(self, zigzag_threshold: float = None):
        self.zigzag_threshold = zigzag_threshold or config.ZIGZAG_THRESHOLD

    def normalize_price(self, df: pd.DataFrame) -> np.ndarray:
        """价格归一化为相对于第一根K线的百分比变化"""
        close = df["close"].values.astype(float)
        base = close[0]
        if base == 0:
            return np.zeros_like(close)
        return (close - base) / base

    def normalize_volume(self, df: pd.DataFrame) -> np.ndarray:
        """成交量归一化为相对于均量的倍数"""
        vol = df["vol"].values.astype(float)
        mean_vol = vol.mean()
        if mean_vol == 0:
            return np.zeros_like(vol)
        return vol / mean_vol

    def extract_zigzag(self, norm_close: np.ndarray, norm_vol: np.ndarray,
                       threshold: float = None) -> List[ZigZagPoint]:
        """ZigZag关键点提取 - 识别显著的高低转折点"""
        threshold = threshold or self.zigzag_threshold
        n = len(norm_close)
        if n < 3:
            return []

        points = []
        last_pivot_idx = 0
        last_pivot_val = norm_close[0]
        direction = 0  # 0=未确定, 1=上升, -1=下降

        for i in range(1, n):
            val = norm_close[i]
            change = val - last_pivot_val

            if direction == 0:
                if abs(change) >= threshold:
                    direction = 1 if change > 0 else -1
                continue

            if direction == 1:
                if val > last_pivot_val:
                    # 继续上升，更新最高点
                    last_pivot_idx = i
                    last_pivot_val = val
                elif last_pivot_val - val >= threshold:
                    # 从高点回落超过阈值，确认高点
                    points.append(ZigZagPoint(
                        index=last_pivot_idx,
                        price=last_pivot_val,
                        direction=1,
                        rel_position=last_pivot_idx / (n - 1) if n > 1 else 0,
                        volume_ratio=norm_vol[last_pivot_idx] if last_pivot_idx < len(norm_vol) else 1.0,
                    ))
                    last_pivot_idx = i
                    last_pivot_val = val
                    direction = -1
            else:  # direction == -1
                if val < last_pivot_val:
                    last_pivot_idx = i
                    last_pivot_val = val
                elif val - last_pivot_val >= threshold:
                    points.append(ZigZagPoint(
                        index=last_pivot_idx,
                        price=last_pivot_val,
                        direction=-1,
                        rel_position=last_pivot_idx / (n - 1) if n > 1 else 0,
                        volume_ratio=norm_vol[last_pivot_idx] if last_pivot_idx < len(norm_vol) else 1.0,
                    ))
                    last_pivot_idx = i
                    last_pivot_val = val
                    direction = 1

        # 添加最后一个点
        if points:
            last_dir = -points[-1].direction  # 与最后一个确认点方向相反
        else:
            last_dir = 1 if norm_close[-1] > norm_close[0] else -1

        points.append(ZigZagPoint(
            index=last_pivot_idx,
            price=last_pivot_val,
            direction=last_dir,
            rel_position=last_pivot_idx / (n - 1) if n > 1 else 0,
            volume_ratio=norm_vol[last_pivot_idx] if last_pivot_idx < len(norm_vol) else 1.0,
        ))

        return points

    # ==================== 技术指标计算 ====================

    def calc_ma(self, close: np.ndarray, periods: List[int] = None) -> dict:
        """计算多条均线"""
        periods = periods or [5, 10, 20, 60]
        result = {}
        for p in periods:
            if len(close) >= p:
                ma = pd.Series(close).rolling(p).mean().values
                result[f"MA{p}"] = ma
            else:
                result[f"MA{p}"] = np.full_like(close, np.nan)
        return result

    def calc_macd(self, close: np.ndarray,
                  fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """计算MACD (DIF, DEA, MACD柱)"""
        s = pd.Series(close)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        return {"DIF": dif.values, "DEA": dea.values, "MACD": macd_hist.values}

    def calc_cci(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """计算CCI指标"""
        tp = (df["high"].values + df["low"].values + df["close"].values) / 3.0
        tp_series = pd.Series(tp)
        ma = tp_series.rolling(period).mean()
        md = tp_series.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp_series - ma) / (0.015 * md + 1e-10)
        return cci.values

    def _analyze_ma(self, close: np.ndarray, mas: dict) -> Tuple[dict, List[str]]:
        """分析整个形态期间的均线变化过程"""
        n = len(close)

        # 均线排列变化过程（分起始、中段、末端三个阶段描述）
        positions = {}
        stages = {"起始": 0, "中段": n // 2, "末端": n - 1}
        for stage_name, idx in stages.items():
            stage_desc = []
            for name, ma in mas.items():
                if idx < len(ma) and not np.isnan(ma[idx]):
                    diff_pct = (close[idx] - ma[idx]) / ma[idx] * 100
                    if diff_pct > 2:
                        stage_desc.append(f"在{name}上方{diff_pct:.1f}%")
                    elif diff_pct < -2:
                        stage_desc.append(f"在{name}下方{abs(diff_pct):.1f}%")
                    else:
                        stage_desc.append(f"贴近{name}")
            if stage_desc:
                positions[stage_name] = f"{stage_name}: 价格{'，'.join(stage_desc)}"

        # 均线排列形态（多头/空头/交织）
        ma_names = sorted(mas.keys(), key=lambda x: int(x[2:]))
        if len(ma_names) >= 3:
            end_vals = {}
            for name in ma_names:
                if not np.isnan(mas[name][-1]):
                    end_vals[name] = mas[name][-1]
            if len(end_vals) >= 3:
                sorted_by_val = sorted(end_vals.items(), key=lambda x: x[1], reverse=True)
                sorted_names = [x[0] for x in sorted_by_val]
                sorted_periods = [int(x[2:]) for x in sorted_names]
                if sorted_periods == sorted(sorted_periods):
                    positions["排列"] = "末端均线空头排列(短期均线在下)"
                elif sorted_periods == sorted(sorted_periods, reverse=True):
                    positions["排列"] = "末端均线多头排列(短期均线在上)"
                else:
                    positions["排列"] = f"末端均线交织({'>'.join(sorted_names)})"

        # 均线方向变化
        for name, ma in mas.items():
            valid = ma[~np.isnan(ma)]
            if len(valid) >= 10:
                first_half_slope = valid[len(valid)//2] - valid[0]
                second_half_slope = valid[-1] - valid[len(valid)//2]
                if first_half_slope < 0 and second_half_slope > 0:
                    positions[f"{name}方向"] = f"{name}从下行转为上行(拐头向上)"
                elif first_half_slope > 0 and second_half_slope < 0:
                    positions[f"{name}方向"] = f"{name}从上行转为下行(拐头向下)"

        # 均线交叉事件（扫描整个形态期间）
        cross_events = []
        for i in range(len(ma_names) - 1):
            short_name, long_name = ma_names[i], ma_names[i + 1]
            short_ma, long_ma = mas[short_name], mas[long_name]
            for j in range(1, n):
                if np.isnan(short_ma[j]) or np.isnan(short_ma[j - 1]):
                    continue
                if np.isnan(long_ma[j]) or np.isnan(long_ma[j - 1]):
                    continue
                prev_diff = short_ma[j - 1] - long_ma[j - 1]
                curr_diff = short_ma[j] - long_ma[j]
                if prev_diff <= 0 < curr_diff:
                    pos = j / (n - 1) if n > 1 else 0
                    cross_events.append(f"位置{pos:.0%}: {short_name}上穿{long_name}(金叉)")
                elif prev_diff >= 0 > curr_diff:
                    pos = j / (n - 1) if n > 1 else 0
                    cross_events.append(f"位置{pos:.0%}: {short_name}下穿{long_name}(死叉)")

        return positions, cross_events

    def _analyze_macd(self, close: np.ndarray, macd_data: dict) -> Tuple[str, List[str], str]:
        """分析整个形态期间的MACD变化过程"""
        dif, dea, hist = macd_data["DIF"], macd_data["DEA"], macd_data["MACD"]
        n = len(close)

        # 柱状图变化过程（分阶段描述红绿柱交替）
        hist_phases = []
        valid_hist = hist[~np.isnan(hist)]
        if len(valid_hist) > 2:
            phase_start = 0
            phase_sign = 1 if valid_hist[0] >= 0 else -1
            for i in range(1, len(valid_hist)):
                cur_sign = 1 if valid_hist[i] >= 0 else -1
                if cur_sign != phase_sign:
                    phase_len = i - phase_start
                    peak_val = max(valid_hist[phase_start:i]) if phase_sign == 1 else min(valid_hist[phase_start:i])
                    color = "红柱" if phase_sign == 1 else "绿柱"
                    pos_start = phase_start / (len(valid_hist) - 1)
                    hist_phases.append(f"位置{pos_start:.0%}~: {color}{phase_len}根(峰值{peak_val:.3f})")
                    phase_start = i
                    phase_sign = cur_sign
            # 最后一段
            phase_len = len(valid_hist) - phase_start
            peak_val = max(valid_hist[phase_start:]) if phase_sign == 1 else min(valid_hist[phase_start:])
            color = "红柱" if phase_sign == 1 else "绿柱"
            pos_start = phase_start / (len(valid_hist) - 1) if len(valid_hist) > 1 else 0
            hist_phases.append(f"位置{pos_start:.0%}~: {color}{phase_len}根(峰值{peak_val:.3f})")

        hist_trend = " → ".join(hist_phases) if hist_phases else "数据不足"

        # DIF位置变化（相对零轴）
        dif_valid = dif[~np.isnan(dif)]
        if len(dif_valid) > 2:
            start_pos = "零轴上" if dif_valid[0] > 0 else "零轴下"
            end_pos = "零轴上" if dif_valid[-1] > 0 else "零轴下"
            if start_pos != end_pos:
                hist_trend += f"\nDIF从{start_pos}移动到{end_pos}"
            else:
                hist_trend += f"\nDIF始终在{start_pos}运行"

        # 金叉/死叉（扫描整个形态期间）
        cross_events = []
        for j in range(1, n):
            if np.isnan(dif[j]) or np.isnan(dif[j - 1]):
                continue
            if np.isnan(dea[j]) or np.isnan(dea[j - 1]):
                continue
            prev_diff = dif[j - 1] - dea[j - 1]
            curr_diff = dif[j] - dea[j]
            if prev_diff <= 0 < curr_diff:
                pos = j / (n - 1) if n > 1 else 0
                level = "零轴上" if dif[j] > 0 else "零轴下"
                cross_events.append(f"位置{pos:.0%}: MACD金叉({level})")
            elif prev_diff >= 0 > curr_diff:
                pos = j / (n - 1) if n > 1 else 0
                level = "零轴上" if dif[j] > 0 else "零轴下"
                cross_events.append(f"位置{pos:.0%}: MACD死叉({level})")

        # 背离检测
        divergence = "无明显背离"
        if n > 20:
            mid = n // 2
            price_left_min = close[:mid].min()
            price_right_min = close[mid:].min()
            if len(dif_valid) > mid:
                dif_left_min = dif_valid[:mid].min()
                dif_right_min = dif_valid[mid:].min()
                if price_right_min < price_left_min and dif_right_min > dif_left_min:
                    divergence = "底背离(价格新低但MACD未新低，前半段低点价格{:.2f}→后半段{:.2f}，DIF低点{:.3f}→{:.3f})".format(
                        price_left_min, price_right_min, dif_left_min, dif_right_min)
                price_left_max = close[:mid].max()
                price_right_max = close[mid:].max()
                dif_left_max = dif_valid[:mid].max()
                dif_right_max = dif_valid[mid:].max()
                if price_right_max > price_left_max and dif_right_max < dif_left_max:
                    divergence = "顶背离(价格新高但MACD未新高，前半段高点价格{:.2f}→后半段{:.2f}，DIF高点{:.3f}→{:.3f})".format(
                        price_left_max, price_right_max, dif_left_max, dif_right_max)

        return hist_trend, cross_events, divergence

    def _analyze_cci(self, cci: np.ndarray) -> Tuple[List[str], List[dict]]:
        """分析CCI区间变化和极值"""
        n = len(cci)

        def cci_zone(val):
            if np.isnan(val):
                return "无数据"
            if val > 200:
                return "极度超买(>200)"
            elif val > 100:
                return "超买(100~200)"
            elif val > -100:
                return "常态(-100~100)"
            elif val > -200:
                return "超卖(-200~-100)"
            else:
                return "极度超卖(<-200)"

        # 区间变化序列
        zones = []
        prev_zone = None
        for i in range(n):
            z = cci_zone(cci[i])
            if z != prev_zone and z != "无数据":
                pos = i / (n - 1) if n > 1 else 0
                zones.append(f"位置{pos:.0%}: 进入{z}")
                prev_zone = z

        # 极值点
        extremes = []
        for i in range(1, n - 1):
            if np.isnan(cci[i]):
                continue
            if cci[i] > cci[i - 1] and cci[i] > cci[i + 1] and abs(cci[i]) > 100:
                pos = i / (n - 1) if n > 1 else 0
                extremes.append({"position": f"{pos:.0%}", "value": f"{cci[i]:.0f}", "type": "高点"})
            elif cci[i] < cci[i - 1] and cci[i] < cci[i + 1] and abs(cci[i]) > 100:
                pos = i / (n - 1) if n > 1 else 0
                extremes.append({"position": f"{pos:.0%}", "value": f"{cci[i]:.0f}", "type": "低点"})

        return zones, extremes

    def extract_candle_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取每根K线的特征向量 [涨跌幅, 振幅, 上影比, 下影比, 量比]"""
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        v = df["vol"].values.astype(float)

        # 涨跌幅
        pct_change = np.zeros(len(c))
        pct_change[1:] = (c[1:] - c[:-1]) / np.where(c[:-1] == 0, 1, c[:-1])

        # 振幅
        amplitude = (h - l) / np.where(l == 0, 1, l)

        # 上影线比
        body_top = np.maximum(o, c)
        upper_shadow = (h - body_top) / np.where(amplitude * l == 0, 1, h - l + 1e-10)

        # 下影线比
        body_bottom = np.minimum(o, c)
        lower_shadow = (body_bottom - l) / np.where(h - l == 0, 1, h - l + 1e-10)

        # 量比
        mean_vol = v.mean()
        vol_ratio = v / mean_vol if mean_vol > 0 else np.ones_like(v)

        features = np.column_stack([pct_change, amplitude, upper_shadow, lower_shadow, vol_ratio])
        return features

    def extract_features(self, df: pd.DataFrame, window_len: int = None) -> PatternFeatures:
        """从K线数据提取完整形态特征"""
        if window_len and len(df) > window_len:
            df = df.tail(window_len).reset_index(drop=True)

        norm_close = self.normalize_price(df)
        norm_vol = self.normalize_volume(df)
        zigzag = self.extract_zigzag(norm_close, norm_vol)
        candle_feat = self.extract_candle_features(df)

        # 计算统计特征
        close = df["close"].values.astype(float)
        total_return = (close[-1] - close[0]) / close[0] if close[0] != 0 else 0

        # 最大回撤
        cummax = np.maximum.accumulate(close)
        drawdowns = (cummax - close) / np.where(cummax == 0, 1, cummax)
        max_drawdown = drawdowns.max()

        # 最大反弹
        cummin = np.minimum.accumulate(close)
        rallies = (close - cummin) / np.where(cummin == 0, 1, cummin)
        max_rally = rallies.max()

        # 波动率
        returns = np.diff(close) / np.where(close[:-1] == 0, 1, close[:-1])
        volatility = returns.std() if len(returns) > 0 else 0

        # ZigZag振幅
        zigzag_amplitudes = []
        for i in range(1, len(zigzag)):
            amp = abs(zigzag[i].price - zigzag[i - 1].price)
            zigzag_amplitudes.append(amp)

        # 量能趋势
        vol = df["vol"].values.astype(float)
        if len(vol) > 1:
            x = np.arange(len(vol))
            vol_trend = np.polyfit(x, vol, 1)[0] / (vol.mean() + 1e-10)
        else:
            vol_trend = 0

        # 高低点处的量比
        peaks = [p for p in zigzag if p.direction == 1]
        troughs = [p for p in zigzag if p.direction == -1]
        avg_vol_peaks = np.mean([p.volume_ratio for p in peaks]) if peaks else 1.0
        avg_vol_troughs = np.mean([p.volume_ratio for p in troughs]) if troughs else 1.0

        # 技术指标分析
        close = df["close"].values.astype(float)
        mas = self.calc_ma(close)
        ma_positions, ma_cross_events = self._analyze_ma(close, mas)

        macd_data = self.calc_macd(close)
        macd_hist_trend, macd_cross_events, macd_divergence = self._analyze_macd(close, macd_data)

        cci = self.calc_cci(df)
        cci_zones, cci_extremes = self._analyze_cci(cci)

        return PatternFeatures(
            length=len(df),
            total_return=total_return,
            max_drawdown=max_drawdown,
            max_rally=max_rally,
            volatility=volatility,
            zigzag_points=zigzag,
            n_zigzag=len(zigzag),
            zigzag_amplitudes=zigzag_amplitudes,
            vol_trend=vol_trend,
            avg_vol_ratio_at_peaks=avg_vol_peaks,
            avg_vol_ratio_at_troughs=avg_vol_troughs,
            ma_positions=ma_positions,
            ma_cross_events=ma_cross_events,
            macd_histogram_trend=macd_hist_trend,
            macd_cross_events=macd_cross_events,
            macd_divergence=macd_divergence,
            cci_zones=cci_zones,
            cci_extremes=cci_extremes,
            norm_close=norm_close,
            norm_volume=norm_vol,
            candle_features=candle_feat,
        )

    def features_to_text(self, features: PatternFeatures, df: pd.DataFrame = None) -> str:
        """将形态特征转为结构化文本描述（供LLM使用）
        围绕牛二模型的关键判断点描述：先涨？W底？跌破哪条均线？是否站稳？
        """
        lines = []

        if df is None or len(df) == 0:
            lines.append("无K线数据")
            return "\n".join(lines)

        close = df["close"].values.astype(float)
        n = len(close)
        ma20 = pd.Series(close).rolling(20).mean().values
        ma30 = pd.Series(close).rolling(30).mean().values

        # === 1. 是否有前期上涨 ===
        # 找形态前半段的最低点和最高点
        half = n // 2
        pre_low = close[:half].min()
        pre_low_idx = close[:half].argmin()
        pre_high = close[:half].max()
        pre_high_idx = close[:half].argmax()

        if pre_high_idx > pre_low_idx and pre_low > 0:
            rally_pct = (pre_high - pre_low) / pre_low
            lines.append(f"前期上涨: 是, 涨幅{rally_pct:.1%}")
        else:
            rally_pct = 0
            lines.append(f"前期上涨: 否")

        # === 2. 是否形成W底 ===
        # 在高点之后找两个低点
        after_peak = close[pre_high_idx:]
        if len(after_peak) > 10:
            # 找第一个底
            first_low_rel = 0
            for i in range(1, min(len(after_peak), 30)):
                if after_peak[i] < after_peak[first_low_rel]:
                    first_low_rel = i

            # 找中间反弹
            mid_high_rel = first_low_rel
            for i in range(first_low_rel + 1, min(first_low_rel + 20, len(after_peak))):
                if after_peak[i] > after_peak[mid_high_rel]:
                    mid_high_rel = i

            # 找第二个底
            second_low_rel = mid_high_rel
            for i in range(mid_high_rel + 1, min(mid_high_rel + 20, len(after_peak))):
                if after_peak[i] < after_peak[second_low_rel]:
                    second_low_rel = i

            first_low_abs = pre_high_idx + first_low_rel
            second_low_abs = pre_high_idx + second_low_rel

            has_w = (first_low_rel != mid_high_rel and mid_high_rel != second_low_rel
                     and after_peak[mid_high_rel] > after_peak[first_low_rel]
                     and after_peak[mid_high_rel] > after_peak[second_low_rel])

            if has_w:
                low1_pct = (pre_high - after_peak[first_low_rel]) / pre_high if pre_high > 0 else 0
                low2_pct = (pre_high - after_peak[second_low_rel]) / pre_high if pre_high > 0 else 0
                low_diff = (after_peak[second_low_rel] - after_peak[first_low_rel]) / after_peak[first_low_rel] if after_peak[first_low_rel] > 0 else 0
                lines.append(f"W底形态: 是")
                lines.append(f"  第一个V底回撤: {low1_pct:.1%}")
                lines.append(f"  第二个V底回撤: {low2_pct:.1%}")
                if low_diff > 0.01:
                    lines.append(f"  第二底比第一底高{low_diff:.1%}")
                elif low_diff < -0.01:
                    lines.append(f"  第二底比第一底低{abs(low_diff):.1%}")
                else:
                    lines.append(f"  两底基本持平")
            else:
                has_w = False
                second_low_abs = n - 1
                lines.append(f"W底形态: 否")
        else:
            has_w = False
            second_low_abs = n - 1
            lines.append(f"W底形态: 否")

        # === 3. 第二个V是否跌破MA20/MA30 ===
        if has_w:
            # 检查W底区间内是否跌破均线
            w_start = first_low_abs
            w_end = min(second_low_abs + 5, n)
            broke_ma30 = any(
                close[i] < ma30[i] for i in range(w_start, w_end)
                if not np.isnan(ma30[i])
            )
            broke_ma20 = any(
                close[i] < ma20[i] for i in range(w_start, w_end)
                if not np.isnan(ma20[i])
            )

            if broke_ma30:
                lines.append(f"第二个V跌破MA30: 是")
            else:
                lines.append(f"第二个V跌破MA30: 否")

            if broke_ma20:
                lines.append(f"第二个V跌破MA20: 是")
            else:
                lines.append(f"第二个V跌破MA20: 否")

        # === 4. 当前价格与均线的关系 ===
        last_close = close[-1]
        if not np.isnan(ma20[-1]):
            diff20 = (last_close - ma20[-1]) / ma20[-1] * 100
            if diff20 > 1:
                lines.append(f"当前价格站稳MA20: 是 (在MA20上方{diff20:.1f}%)")
            elif diff20 > -1:
                lines.append(f"当前价格站稳MA20: 贴近MA20 ({diff20:+.1f}%)")
            else:
                lines.append(f"当前价格站稳MA20: 否 (在MA20下方{abs(diff20):.1f}%)")

        if not np.isnan(ma30[-1]):
            diff30 = (last_close - ma30[-1]) / ma30[-1] * 100
            if diff30 > 1:
                lines.append(f"当前价格站稳MA30: 是 (在MA30上方{diff30:.1f}%)")
            elif diff30 > -1:
                lines.append(f"当前价格站稳MA30: 贴近MA30 ({diff30:+.1f}%)")
            else:
                lines.append(f"当前价格站稳MA30: 否 (在MA30下方{abs(diff30):.1f}%)")

        return "\n".join(lines)
