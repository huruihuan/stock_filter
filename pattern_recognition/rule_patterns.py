"""规则化形态定义 - 用明确规则自动从历史数据中搜索样本"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time as _time

from .data_fetcher import DataFetcher
from .preprocessor import Preprocessor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _parallel_scan(detector, stock_list, name_map, end_date, lookback,
                   label="扫描", print_hit_fn=None, max_workers=8):
    """通用并发扫描框架，返回匹配结果列表。
    print_hit_fn(m) -> str: 将单条匹配结果格式化为打印字符串。
    """
    all_results = []
    lock = threading.Lock()
    done_count = [0]
    scan_start = _time.time()

    def _process(code):
        name = name_map.get(code, "")
        try:
            matches = detector.detect_single(code, end_date, lookback, name=name)
            return [(code, m) for m in matches if m.trigger_date == end_date]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, code): code for code in stock_list}
        for future in as_completed(futures):
            hits = future.result()
            with lock:
                done_count[0] += 1
                for code, m in hits:
                    all_results.append(m)
                    if print_hit_fn:
                        print(f"  >> {print_hit_fn(m)}")
                cnt = done_count[0]
                if cnt % 200 == 0:
                    elapsed = _time.time() - scan_start
                    print(f"{label}进度: {cnt}/{len(stock_list)}, "
                          f"已找到{len(all_results)}个, 耗时{elapsed:.0f}s")

    elapsed = _time.time() - scan_start
    print(f"{label}完成: {len(stock_list)}只股票, "
          f"{end_date}当天触发{len(all_results)}个, 耗时{elapsed:.0f}s")
    return all_results


@dataclass
class RuleMatchResult:
    ts_code: str
    name: str              # 股票名称
    trigger_date: str      # 触发日
    trigger_type: str      # "站上MA30" 或 "站上MA20"
    rally_start_date: str  # 上涨起点日
    rally_pct: float       # 上涨幅度
    w_first_low_date: str  # W第一个底日期
    w_middle_high_date: str # W中间高点日期
    w_second_low_date: str  # W第二个底日期
    broke_ma30: bool       # 第二个V是否跌破MA30
    broke_ma20: bool       # 第二个V是否跌破MA20
    window_len: int        # 形态总长度


class NiuEr:
    """
    牛二模型 - 形态规则：
    1. 上涨一波至少20%
    2. 回踩形成W形态（两个V底）
    3. 若第二个V跌破MA30 → 站上MA30形成买点
    4. 若第二个V未跌破MA30但跌破MA20 → 站上MA20形成买点
    """

    def __init__(self, fetcher: DataFetcher = None,
                 min_rally_pct: float = 0.20,
                 w_tolerance: float = 0.03):
        self.fetcher = fetcher or DataFetcher()
        self.min_rally_pct = min_rally_pct    # 最小上涨幅度
        self.w_tolerance = w_tolerance         # W底两个低点的容差（3%以内）

    def _calc_ma(self, close: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(close).rolling(period).mean().values

    def _find_rallies(self, close: np.ndarray, min_pct: float,
                      ma30: np.ndarray = None) -> List[Tuple[int, int]]:
        """找出所有涨幅>=min_pct的上涨段，返回[(start_idx, peak_idx), ...]
        结束条件（满足任一即见顶）：
        1. 从最高点回落超过20%，且后续30根K线内未创新高
        2. 价格回落到MA30上方1%以内（close <= MA30 * 1.01）
        """
        n = len(close)
        rallies = []
        i = 0
        while i < n - 1:
            # 找局部低点作为起点
            while i < n - 1 and close[i + 1] <= close[i]:
                i += 1
            start = i

            peak = start
            j = start + 1
            while j < n:
                if close[j] > close[peak]:
                    peak = j

                # 条件2：回落到MA30附近（上方1%以内），且已经从高点回落了一段
                if ma30 is not None and not np.isnan(ma30[j]) and peak > start:
                    if close[j] <= ma30[j] * 1.01 and close[j] < close[peak]:
                        break

                # 条件1：从peak回落超过20%时，看后续是否还能创新高
                if close[peak] > 0 and (close[peak] - close[j]) / close[peak] > 0.20:
                    found_new_high = False
                    for k in range(j + 1, min(j + 30, n)):
                        if close[k] > close[peak]:
                            found_new_high = True
                            break
                    if not found_new_high:
                        break
                j += 1

            if close[start] > 0:
                pct = (close[peak] - close[start]) / close[start]
                if pct >= min_pct:
                    # 新上涨段高点必须比前一段高点高10%以上，否则不算新的上涨
                    if rallies and close[peak] <= close[rallies[-1][1]] * 1.10:
                        pass  # 跳过
                    else:
                        rallies.append((start, peak))

            i = peak + 1

        return rallies

    def _find_w_bottom(self, close: np.ndarray, start_idx: int) -> Optional[dict]:
        """从start_idx开始寻找W形态。
        W形态 = 低点1 → 反弹高点 → 低点2（与低点1接近）
        返回 {first_low, middle_high, second_low} 的索引
        """
        n = len(close)
        if start_idx + 10 > n:  # W至少需要10根K线
            return None

        segment = close[start_idx:]
        seg_n = len(segment)

        # 找第一个低点：从高点下来后的第一个显著低点
        first_low = 0
        for i in range(1, min(seg_n, 30)):
            if segment[i] < segment[first_low]:
                first_low = i

        if first_low == 0:
            return None

        # 找中间反弹高点
        middle_high = first_low
        for i in range(first_low + 1, min(first_low + 20, seg_n)):
            if segment[i] > segment[middle_high]:
                middle_high = i

        # 反弹幅度至少3%才算有效
        if segment[first_low] > 0:
            bounce_pct = (segment[middle_high] - segment[first_low]) / segment[first_low]
            if bounce_pct < 0.03:
                return None

        # 找第二个低点
        second_low = middle_high
        for i in range(middle_high + 1, min(middle_high + 20, seg_n)):
            if segment[i] < segment[second_low]:
                second_low = i

        if second_low == middle_high:
            return None

        # 检查W形态：两个低点价差不超过反弹幅度的50%
        if segment[first_low] > 0:
            bounce_amount = segment[middle_high] - segment[first_low]
            low_diff = abs(segment[second_low] - segment[first_low])
            if bounce_amount > 0 and low_diff > bounce_amount * 0.5:
                return None

        # 检查中间高点确实高于两个低点
        if segment[middle_high] <= segment[first_low] or segment[middle_high] <= segment[second_low]:
            return None

        return {
            "first_low": start_idx + first_low,
            "middle_high": start_idx + middle_high,
            "second_low": start_idx + second_low,
        }

    def _find_ma_crossover(self, close: np.ndarray, ma: np.ndarray,
                            after_idx: int) -> Optional[int]:
        """找到after_idx之后价格从下方站上均线的第一天"""
        n = len(close)
        for i in range(after_idx + 1, min(after_idx + 20, n)):
            if np.isnan(ma[i]) or np.isnan(ma[i - 1]):
                continue
            if close[i - 1] <= ma[i - 1] and close[i] > ma[i]:
                return i
        return None

    def detect_single(self, ts_code: str, end_date: str,
                      lookback: int = 200, name: str = "") -> List[RuleMatchResult]:
        """检测单只股票中的所有牛二形态"""
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=lookback)
        if len(df) < 40:
            return []

        close = df["close"].values.astype(float)
        n = len(close)
        dates = df["trade_date"].values
        ma20 = self._calc_ma(close, 20)
        ma30 = self._calc_ma(close, 30)

        results = []

        # 1. 找所有上涨段
        rallies = self._find_rallies(close, self.min_rally_pct, ma30=ma30)

        for rally_start, rally_peak in rallies:
            # 2. 从高点之后找W底（支持连续W底）
            w = self._find_w_bottom(close, rally_peak)
            if w is None:
                continue

            # 校验：W底第一个低点之后不能出现比上涨段高点更高的价格
            if max(close[w["first_low"]:w["second_low"] + 1]) > close[rally_peak]:
                continue

            # 尝试当前W底，如果跌破了再找下一个W底
            w_attempts = [w]
            search_from = w["second_low"]

            # 检查第一个W底是否跌破（后续价格跌破第二低点）
            for _ in range(3):  # 最多再找3个W底
                first_w_low = close[w_attempts[-1]["second_low"]]
                # 看后续是否跌破了这个W底
                broken = False
                break_idx = None
                for k in range(search_from + 1, min(search_from + 30, n)):
                    if close[k] < first_w_low * 0.97:  # 跌破W底（留3%容差）
                        broken = True
                        break_idx = k
                        break

                if not broken:
                    break  # 没跌破，用当前W底

                # 跌破了，从跌破位置找下一个W底
                next_w = self._find_w_bottom(close, break_idx)
                if next_w is None:
                    break

                # 校验：W底内不能出现比上涨段高点更高的价格
                if max(close[next_w["first_low"]:next_w["second_low"] + 1]) > close[rally_peak]:
                    break

                # 检查两个W底的价差是否在10%以内
                prev_low = close[w_attempts[-1]["second_low"]]
                curr_low = close[next_w["second_low"]]
                if prev_low > 0 and abs(curr_low - prev_low) / prev_low > 0.10:
                    break  # 价差超过10%，不成立

                w_attempts.append(next_w)
                search_from = next_w["second_low"]

            # 用最后一个有效的W底
            final_w = w_attempts[-1]
            second_low_idx = final_w["second_low"]

            # 3. 判断跌破哪条均线，决定触发条件
            broke_ma30 = False
            broke_ma20 = False
            if not np.isnan(ma30[second_low_idx]):
                broke_ma30 = close[second_low_idx] < ma30[second_low_idx]
            if not np.isnan(ma20[second_low_idx]):
                broke_ma20 = close[second_low_idx] < ma20[second_low_idx]

            # 反复寻找触发：站上均线后又跌回去，则找新低再找下一次站上
            search_from = second_low_idx
            last_trigger_idx = None
            last_trigger_type = ""
            last_low_idx = second_low_idx

            for _attempt in range(2):  # 最多反复2次
                cur_broke_ma30 = False
                cur_broke_ma20 = False
                if not np.isnan(ma30[search_from]):
                    cur_broke_ma30 = close[search_from] < ma30[search_from]
                if not np.isnan(ma20[search_from]):
                    cur_broke_ma20 = close[search_from] < ma20[search_from]

                if not cur_broke_ma30 and not cur_broke_ma20:
                    break

                if cur_broke_ma30:
                    t = self._find_ma_crossover(close, ma30, search_from)
                    t_type = "站上MA30"
                else:
                    t = self._find_ma_crossover(close, ma20, search_from)
                    t_type = "站上MA20"

                if t is None:
                    break

                last_trigger_idx = t
                last_trigger_type = t_type
                last_low_idx = search_from
                broke_ma30 = cur_broke_ma30
                broke_ma20 = cur_broke_ma20

                # 检查站上后是否又跌回均线以下
                fell_back = False
                new_low_idx = None
                target_ma = ma30 if cur_broke_ma30 else ma20
                for k in range(t + 1, min(t + 30, n)):
                    if not np.isnan(target_ma[k]) and close[k] < target_ma[k]:
                        # 跌回去了，继续找这之后的最低点
                        fell_back = True
                        low_k = k
                        for j in range(k + 1, min(k + 15, n)):
                            if close[j] < close[low_k]:
                                low_k = j
                            elif close[j] > close[low_k]:
                                break  # 开始反弹，低点确认
                        new_low_idx = low_k
                        break

                if not fell_back:
                    break  # 没跌回去，当前trigger有效
                search_from = new_low_idx

            if last_trigger_idx is None:
                continue

            rally_pct = (close[rally_peak] - close[rally_start]) / close[rally_start]
            window_len = last_trigger_idx - rally_start + 1
            w_count = len(w_attempts)

            results.append(RuleMatchResult(
                ts_code=ts_code,
                name=name,
                trigger_date=str(dates[last_trigger_idx]),
                trigger_type=last_trigger_type + (f"(第{w_count}个W底)" if w_count > 1 else ""),
                rally_start_date=str(dates[rally_start]),
                rally_pct=rally_pct,
                w_first_low_date=str(dates[final_w["first_low"]]),
                w_middle_high_date=str(dates[final_w["middle_high"]]),
                w_second_low_date=str(dates[last_low_idx]),
                broke_ma30=broke_ma30,
                broke_ma20=broke_ma20,
                window_len=window_len,
            ))

        return results

    def scan_market(self, end_date: str = None,
                    stock_list: List[str] = None,
                    lookback: int = 200,
                    ) -> List[RuleMatchResult]:
        """扫描全市场，只保留指定日期当天触发的形态。"""
        df_stocks = self.fetcher.get_all_stock_list()
        name_map = dict(zip(df_stocks["ts_code"], df_stocks["name"]))
        if stock_list is None:
            stock_list = df_stocks["ts_code"].tolist()
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")

        def _fmt(m):
            return (f"命中 {m.ts_code} {m.name} ({m.trigger_type}) "
                    f"涨幅{m.rally_pct:.1%} W底: {m.w_first_low_date}→{m.w_middle_high_date}→{m.w_second_low_date}")

        return _parallel_scan(self, stock_list, name_map, end_date, lookback,
                              label="牛二扫描", print_hit_fn=_fmt)

    def to_samples(self, results: List[RuleMatchResult]) -> List[Tuple[str, str]]:
        """将检测结果转为标准样本格式 [(ts_code, trigger_date), ...]"""
        return [(r.ts_code, r.trigger_date) for r in results]

    def print_results(self, results: List[RuleMatchResult]):
        """打印检测结果"""
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.ts_code} {r.name} 触发日: {r.trigger_date} ({r.trigger_type})")
            print(f"  上涨段: {r.rally_start_date} 起涨, 涨幅 {r.rally_pct:.1%}")
            print(f"  W底: 第一底{r.w_first_low_date} → 反弹{r.w_middle_high_date} → 第二底{r.w_second_low_date}")
            print(f"  跌破MA30: {'是' if r.broke_ma30 else '否'}, 跌破MA20: {'是' if r.broke_ma20 else '否'}")
            print(f"  形态长度: {r.window_len}根K线")


@dataclass
class YRCMatchResult:
    """羊肉串战法匹配结果"""
    ts_code: str
    name: str
    trigger_date: str
    trigger_type: str      # "连续企稳" / "W底突破" / "反包"
    top_dates: list        # 各顶部高点日期
    top_prices: list       # 各顶部高点价格
    top_avg_price: float   # 顶部均价
    n_tops: int            # 顶部数量


class YangRouChuan:
    """
    羊肉串战法 - 形态规则：
    1. 过去一年内出现至少2个局部高点，价格偏差5%以内（多重顶部）
    2. 两个顶部之间至少间隔20个交易日
    3. 当前价格到达顶部区间，满足以下任一条件触发：
       A. 在顶部均价90%~110%内形成W底
       B. 在顶部均价95%~105%内出现反包
       C. 收盘价突破所有顶部最高价
    """

    def __init__(self, fetcher: DataFetcher = None,
                 top_tolerance: float = 0.05,
                 min_tops: int = 2,
                 min_gap: int = 20):
        self.fetcher = fetcher or DataFetcher()
        self.top_tolerance = top_tolerance  # 顶部价格偏差容忍度
        self.min_tops = min_tops            # 最少顶部数量
        self.min_gap = min_gap              # 顶部之间最小间隔（交易日）

    def _find_local_peaks(self, close: np.ndarray, vol: np.ndarray,
                          min_gap: int = 20, vol_ratio: float = 1.3) -> List[int]:
        """找出所有局部高点索引，相邻高点间隔>=min_gap。
        额外要求：高点前后3个交易日内至少有一天成交量 >= 前后20日均量的vol_ratio倍。
        """
        n = len(close)
        # 用滑动窗口找局部最大值，窗口半径10，尾部右侧缩小到6
        radius = 10
        tail_radius = 6
        peaks = []
        for i in range(radius, n - tail_radius):
            right_r = radius if i + radius < n else min(tail_radius, n - 1 - i)
            window = close[i - radius:i + right_r + 1]
            if close[i] == max(window) and close[i] > close[i - 1] and close[i] > close[i + 1]:
                peaks.append(i)

        # 过滤间隔太近的高点，保留较高的那个
        if not peaks:
            return []

        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= min_gap:
                filtered.append(p)
            else:
                # 间隔太近，保留价格更高的
                if close[p] > close[filtered[-1]]:
                    filtered[-1] = p

        # 过滤：高点前后3天内须有放量（>=前后20日均量的vol_ratio倍）
        vol_filtered = []
        for p in filtered:
            avg_start = max(0, p - 20)
            avg_end = min(n, p + 21)
            avg_vol = np.mean(vol[avg_start:avg_end])
            if avg_vol <= 0:
                continue
            # 检查前后3天内是否有放量
            check_start = max(0, p - 3)
            check_end = min(n, p + 4)
            max_nearby_vol = max(vol[check_start:check_end])
            if max_nearby_vol >= avg_vol * vol_ratio:
                vol_filtered.append(p)

        return vol_filtered

    def _peak_vol_ratio(self, vol: np.ndarray, peak_idx: int) -> float:
        """计算某个顶点的放量比（前后3天最大量 / 前后20日均量）"""
        n = len(vol)
        avg_start = max(0, peak_idx - 20)
        avg_end = min(n, peak_idx + 21)
        avg_vol = np.mean(vol[avg_start:avg_end])
        if avg_vol <= 0:
            return 0.0
        check_start = max(0, peak_idx - 3)
        check_end = min(n, peak_idx + 4)
        max_nearby_vol = max(vol[check_start:check_end])
        return max_nearby_vol / avg_vol

    def _find_preceding_low(self, close: np.ndarray, peak_idx: int,
                            prev_peak_idx: int = 0) -> float:
        """找到peak_idx之前（到prev_peak_idx为止）的最低收盘价"""
        start = max(0, prev_peak_idx)
        segment = close[start:peak_idx]
        if len(segment) == 0:
            return close[peak_idx]
        return float(np.min(segment))

    def _cluster_tops(self, peaks: List[int], close: np.ndarray,
                      tolerance: float = 0.05) -> List[List[int]]:
        """将高点按价格聚类，价格偏差在tolerance内的归为一组。
        额外要求：聚类内各高点前的低点之间偏差不超过10%。
        返回所有满足>=min_tops个高点的聚类。
        """
        if len(peaks) < self.min_tops:
            return []

        # 预计算每个高点前的低点
        peaks_sorted_time = sorted(peaks)
        preceding_lows = {}
        for idx, p in enumerate(peaks_sorted_time):
            prev_p = peaks_sorted_time[idx - 1] if idx > 0 else 0
            preceding_lows[p] = self._find_preceding_low(close, p, prev_p)

        # 按价格排序
        sorted_peaks = sorted(peaks, key=lambda p: close[p])
        clusters = []

        for i in range(len(sorted_peaks)):
            cluster = [sorted_peaks[i]]
            base_price = close[sorted_peaks[i]]
            for j in range(i + 1, len(sorted_peaks)):
                if base_price > 0 and abs(close[sorted_peaks[j]] - base_price) / base_price <= tolerance:
                    cluster.append(sorted_peaks[j])

            if len(cluster) >= self.min_tops:
                # 按时间排序
                cluster.sort()

                # 过滤：后一个低点不能比前一个低点低超过5%（允许逐步抬高）
                lows = [preceding_lows[p] for p in cluster]
                valid = True
                for k in range(1, len(lows)):
                    if lows[k - 1] > 0 and lows[k] < lows[k - 1] * 0.95:
                        valid = False
                        break
                if not valid:
                    continue

                clusters.append(cluster)

        # 去重：如果两个聚类的高点集合完全相同或者是子集，保留最大的
        unique = []
        seen = set()
        clusters.sort(key=len, reverse=True)
        for c in clusters:
            key = tuple(c)
            if key not in seen:
                # 检查是否是已有聚类的子集
                is_subset = False
                for u in unique:
                    if set(c).issubset(set(u)):
                        is_subset = True
                        break
                if not is_subset:
                    unique.append(c)
                    seen.add(key)

        return unique

    def _check_stable(self, close: np.ndarray, top_price: float,
                      end_idx: int) -> Optional[int]:
        """检查在end_idx往前是否有连续5天收盘价在top_price的95%~105%内。
        返回触发日索引（连续5天的最后一天），或None。
        """
        low_bound = top_price * 0.95
        high_bound = top_price * 1.05
        n = len(close)

        # 从最近往前找连续5天在区间内的段
        for end in range(end_idx, max(end_idx - 30, 4), -1):
            count = 0
            for k in range(end, max(end - 5, -1), -1):
                if low_bound <= close[k] <= high_bound:
                    count += 1
                else:
                    break
            if count >= 5:
                return end

        return None

    def _check_w_in_zone(self, close: np.ndarray, top_price: float,
                         search_start: int) -> Optional[int]:
        """在顶部均价90%~110%范围内找W底形态。
        返回W底第二个低点之后的反弹点索引，或None。
        """
        low_bound = top_price * 0.90
        high_bound = top_price * 1.10
        n = len(close)

        # 找进入顶部区间的位置
        zone_start = None
        for i in range(search_start, n):
            if low_bound <= close[i] <= high_bound:
                zone_start = i
                break

        if zone_start is None:
            return None

        # 在区间内找W底
        segment = close[zone_start:]
        seg_n = len(segment)
        if seg_n < 10:
            return None

        # 找第一个低点
        first_low = 0
        for i in range(1, min(seg_n, 25)):
            if segment[i] < segment[first_low]:
                first_low = i

        if first_low == 0:
            return None

        # 找中间反弹高点
        middle_high = first_low
        for i in range(first_low + 1, min(first_low + 15, seg_n)):
            if segment[i] > segment[middle_high]:
                middle_high = i

        if segment[first_low] > 0:
            bounce_pct = (segment[middle_high] - segment[first_low]) / segment[first_low]
            if bounce_pct < 0.02:
                return None

        # 找第二个低点
        second_low = middle_high
        for i in range(middle_high + 1, min(middle_high + 15, seg_n)):
            if segment[i] < segment[second_low]:
                second_low = i

        if second_low == middle_high:
            return None

        # 检查两个低点价差不超过反弹幅度的50%
        if segment[first_low] > 0:
            bounce_amount = segment[middle_high] - segment[first_low]
            low_diff = abs(segment[second_low] - segment[first_low])
            if bounce_amount > 0 and low_diff > bounce_amount * 0.5:
                return None

        # 检查W底在区间内
        if not (low_bound <= segment[first_low] and
                segment[middle_high] <= high_bound):
            return None

        # W底之后价格回升，找触发点
        for i in range(second_low + 1, min(second_low + 10, seg_n)):
            if segment[i] > segment[second_low] and segment[i] > segment[middle_high] * 0.98:
                return zone_start + i

        return None

    def _check_fan_bao(self, df: pd.DataFrame, top_price: float,
                       end_idx: int) -> Optional[int]:
        """在顶部均价95%~105%内检查反包形态。
        单日反包：当天阳线实体覆盖前一天阴线实体
        多日反包：前面有2%+阴线，之后最多5天内收盘超过该阴线开盘价
        返回触发日索引，或None。
        """
        low_bound = top_price * 0.95
        high_bound = top_price * 1.05
        close = df["close"].values.astype(float)
        open_ = df["open"].values.astype(float)
        n = len(close)

        # 在end_idx往前30天范围内找
        for i in range(max(1, end_idx - 30), end_idx + 1):
            # 价格必须在顶部区间内
            if not (low_bound <= close[i] <= high_bound):
                continue

            # 单日反包：当天阳线覆盖前一天阴线
            if (close[i] > open_[i] and              # 今天阳线
                close[i - 1] < open_[i - 1] and      # 昨天阴线
                close[i] > open_[i - 1] and           # 今收 > 昨开
                open_[i] < close[i - 1]):             # 今开 < 昨收
                return i

            # 多日反包：前面有2%+阴线，5天内反包
            for j in range(max(1, i - 5), i):
                if close[j] < open_[j] and open_[j] > 0:
                    drop_pct = (open_[j] - close[j]) / open_[j]
                    if drop_pct >= 0.02:  # 2%+阴线
                        # 在j之后到i之间，是否有某天收盘超过j的开盘价
                        if close[i] > open_[j] and low_bound <= close[i] <= high_bound:
                            return i

        return None

    def detect_single(self, ts_code: str, end_date: str,
                      lookback: int = 250, name: str = "") -> List[YRCMatchResult]:
        """检测单只股票的羊肉串形态"""
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=lookback)
        if len(df) < 60:
            return []

        close = df["close"].values.astype(float)
        vol = df["vol"].values.astype(float)
        dates = df["trade_date"].values
        n = len(close)

        # 1. 找局部高点（要求前后3天有放量）
        peaks = self._find_local_peaks(close, vol, self.min_gap)
        if len(peaks) < 2:
            return []

        # 2. 从最后一个顶点往前逐个配对，找价格匹配且中间无更高价的组合
        #    放量要求：两个顶点中至少一个>=2.0，另一个>=1.5
        top2_idx = peaks[-1]  # 最近的顶点
        top2_vr = self._peak_vol_ratio(vol, top2_idx)
        top1_idx = None
        for k in range(len(peaks) - 2, -1, -1):
            cand = peaks[k]
            cand_avg = (close[cand] + close[top2_idx]) / 2.0
            # 价格偏差检查
            if cand_avg > 0 and abs(close[cand] - close[top2_idx]) / cand_avg > self.top_tolerance:
                continue
            # 两顶之间不能出现价格突破顶点价格
            cand_max = max(close[cand], close[top2_idx])
            seg_max = float(np.max(close[cand + 1:top2_idx])) if cand + 1 < top2_idx else 0
            if seg_max > cand_max:
                continue
            # 放量配对检查：至少一个>=2.0，另一个>=1.5
            cand_vr = self._peak_vol_ratio(vol, cand)
            vr_hi = max(cand_vr, top2_vr)
            vr_lo = min(cand_vr, top2_vr)
            if vr_hi < 2.0 or vr_lo < 1.3:
                continue
            top1_idx = cand
            break

        if top1_idx is None:
            return []

        top_prices = [close[top1_idx], close[top2_idx]]
        top_avg = np.mean(top_prices)
        top_max = max(top_prices)
        top_dates_str = [str(dates[top1_idx]), str(dates[top2_idx])]

        # 3. 最后一个顶部之后搜索触发条件
        search_start = top2_idx + 1
        if search_start >= n:
            return []

        # 低点检查（所有触发共用）
        prev_low = self._find_preceding_low(close, top2_idx, top1_idx)

        results = []

        def _try_add(t_idx, t_type):
            if t_idx is None:
                return
            post_top_low = float(np.min(close[top2_idx:t_idx + 1]))
            if prev_low > 0 and post_top_low < prev_low * 0.95:
                return
            results.append(YRCMatchResult(
                ts_code=ts_code,
                name=name,
                trigger_date=str(dates[t_idx]),
                trigger_type=t_type,
                top_dates=top_dates_str,
                top_prices=top_prices,
                top_avg_price=top_avg,
                n_tops=2,
            ))

        # 条件A：W底突破
        _try_add(self._check_w_in_zone(close, top_avg, search_start), "W底突破")

        # 条件B：反包
        fb_idx = self._check_fan_bao(df, top_avg, n - 1)
        if fb_idx is not None and fb_idx >= search_start:
            _try_add(fb_idx, "反包")

        # 条件C：价格突破所有顶部
        for i in range(search_start, n):
            if close[i] > top_max:
                _try_add(i, "突破顶部")
                break

        return results

    def scan_market(self, end_date: str = None,
                    stock_list: List[str] = None,
                    lookback: int = 250,
                    ) -> List[YRCMatchResult]:
        """扫描全市场，只保留指定日期当天触发的形态。"""
        df_stocks = self.fetcher.get_all_stock_list()
        name_map = dict(zip(df_stocks["ts_code"], df_stocks["name"]))
        if stock_list is None:
            stock_list = df_stocks["ts_code"].tolist()
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")

        def _fmt(m):
            return (f"命中 {m.ts_code} {m.name} ({m.trigger_type}) "
                    f"顶部{m.n_tops}个 均价{m.top_avg_price:.2f} "
                    f"日期: {', '.join(m.top_dates)}")

        return _parallel_scan(self, stock_list, name_map, end_date, lookback,
                              label="羊肉串扫描", print_hit_fn=_fmt)

    def to_samples(self, results: List[YRCMatchResult]) -> List[Tuple[str, str]]:
        """将检测结果转为标准样本格式 [(ts_code, trigger_date), ...]"""
        return [(r.ts_code, r.trigger_date) for r in results]

    def print_results(self, results: List[YRCMatchResult]):
        """打印检测结果"""
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.ts_code} {r.name} 触发日: {r.trigger_date} ({r.trigger_type})")
            print(f"  顶部数量: {r.n_tops}个, 顶部均价: {r.top_avg_price:.2f}")
            print(f"  顶部日期: {', '.join(r.top_dates)}")
            print(f"  顶部价格: {', '.join(f'{p:.2f}' for p in r.top_prices)}")


@dataclass
class HuiCaiMatchResult:
    """羊肉串突破后回踩匹配结果"""
    ts_code: str
    name: str
    trigger_date: str          # 回踩触发日
    top_dates: list            # 羊肉串顶部日期
    top_prices: list           # 羊肉串顶部价格
    top_avg_price: float       # 顶部均价
    top_max_price: float       # 顶部最高价
    breakout_date: str         # 突破顶部日期
    breakout_vol: float        # 突破日成交量
    pullback_vol: float        # 回踩日成交量
    pullback_vol_ratio: float  # 回踩量/突破量


class HuiCai:
    """
    羊肉串突破后回踩战法 - 形态规则：
    1. 先触发羊肉串的"突破顶部"条件（收盘价突破所有顶部最高价）
    2. 突破后上涨一段，然后回踩
    3. 回踩不跌破顶部最高价（收盘价 >= top_max_price）
    4. 回踩日成交量 <= 突破日成交量 × 40%（缩量至少60%）
    """

    def __init__(self, fetcher: DataFetcher = None,
                 max_pullback_vol_pct: float = 0.40):
        self.fetcher = fetcher or DataFetcher()
        self.max_pullback_vol_pct = max_pullback_vol_pct
        self._yrc = YangRouChuan(fetcher=self.fetcher)

    def detect_single(self, ts_code: str, end_date: str,
                      lookback: int = 250, name: str = "") -> List[HuiCaiMatchResult]:
        """检测羊肉串突破后的缩量回踩"""
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=lookback)
        if len(df) < 60:
            return []

        close = df["close"].values.astype(float)
        vol = df["vol"].values.astype(float)
        dates = df["trade_date"].values
        n = len(close)

        # 复用羊肉串的顶部识别：从最近顶点往前配对
        peaks = self._yrc._find_local_peaks(close, vol, self._yrc.min_gap)
        if len(peaks) < 2:
            return []

        top2_idx = peaks[-1]
        top2_vr = self._yrc._peak_vol_ratio(vol, top2_idx)
        top1_idx = None
        for k in range(len(peaks) - 2, -1, -1):
            cand = peaks[k]
            cand_avg = (close[cand] + close[top2_idx]) / 2.0
            if cand_avg > 0 and abs(close[cand] - close[top2_idx]) / cand_avg > self._yrc.top_tolerance:
                continue
            cand_max = max(close[cand], close[top2_idx])
            seg_max = float(np.max(close[cand + 1:top2_idx])) if cand + 1 < top2_idx else 0
            if seg_max > cand_max:
                continue
            cand_vr = self._yrc._peak_vol_ratio(vol, cand)
            vr_hi = max(cand_vr, top2_vr)
            vr_lo = min(cand_vr, top2_vr)
            if vr_hi < 2.0 or vr_lo < 1.3:
                continue
            top1_idx = cand
            break

        if top1_idx is None:
            return []

        top_prices = [close[top1_idx], close[top2_idx]]
        top_avg = np.mean(top_prices)
        top_max = max(top_prices)
        top_dates_str = [str(dates[top1_idx]), str(dates[top2_idx])]

        search_start = top2_idx + 1
        if search_start >= n:
            return []

        # 找突破日：最后一个顶部之后，收盘价首次超过top_max
        breakout_idx = None
        for i in range(search_start, n):
            if close[i] > top_max:
                breakout_idx = i
                break
        if breakout_idx is None:
            return []

        breakout_vol = vol[breakout_idx]

        # 找突破后高点
        peak_after = breakout_idx
        for i in range(breakout_idx + 1, n):
            if close[i] > close[peak_after]:
                peak_after = i

        # 从高点后找回踩
        results = []
        for i in range(peak_after + 1, n):
            # 价格回落（低于高点）
            if close[i] >= close[peak_after]:
                continue
            # 不跌破顶部最高价
            if close[i] < top_max:
                break
            # 缩量：回踩日量 <= 突破日量 × 40%
            if breakout_vol > 0:
                vol_ratio = vol[i] / breakout_vol
            else:
                continue
            if vol_ratio <= self.max_pullback_vol_pct:
                results.append(HuiCaiMatchResult(
                    ts_code=ts_code,
                    name=name,
                    trigger_date=str(dates[i]),
                    top_dates=top_dates_str,
                    top_prices=top_prices,
                    top_avg_price=top_avg,
                    top_max_price=top_max,
                    breakout_date=str(dates[breakout_idx]),
                    breakout_vol=breakout_vol,
                    pullback_vol=vol[i],
                    pullback_vol_ratio=vol_ratio,
                ))
                break  # 只取第一次回踩

        return results

    def scan_market(self, end_date: str = None,
                    stock_list: List[str] = None,
                    lookback: int = 250,
                    ) -> List[HuiCaiMatchResult]:
        """扫描全市场，只保留指定日期当天触发的形态。"""
        df_stocks = self.fetcher.get_all_stock_list()
        name_map = dict(zip(df_stocks["ts_code"], df_stocks["name"]))
        if stock_list is None:
            stock_list = df_stocks["ts_code"].tolist()
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")

        def _fmt(m):
            return (f"命中 {m.ts_code} {m.name} "
                    f"顶部{len(m.top_dates)}个 最高{m.top_max_price:.2f} "
                    f"突破日{m.breakout_date} 缩量{m.pullback_vol_ratio:.1%}")

        return _parallel_scan(self, stock_list, name_map, end_date, lookback,
                              label="回踩扫描", print_hit_fn=_fmt)

    def to_samples(self, results: List[HuiCaiMatchResult]) -> List[Tuple[str, str]]:
        return [(r.ts_code, r.trigger_date) for r in results]

    def print_results(self, results: List[HuiCaiMatchResult]):
        """打印检测结果"""
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.ts_code} {r.name} 触发日: {r.trigger_date}")
            print(f"  顶部{len(r.top_dates)}个 均价{r.top_avg_price:.2f} 最高{r.top_max_price:.2f}")
            print(f"  顶部日期: {', '.join(r.top_dates)}")
            print(f"  突破日: {r.breakout_date}, 缩量比: {r.pullback_vol_ratio:.1%} (要求<=40%)")


@dataclass
class XinGaoMatchResult:
    """40日新高匹配结果"""
    ts_code: str
    name: str
    trigger_date: str       # 创新高的日期
    close_price: float      # 当天收盘价
    prev_high: float        # 前40日最高收盘价
    prev_high_date: str     # 前40日最高收盘价日期
    exceed_pct: float       # 超过前高的百分比


class XinGao:
    """
    40日新高战法 - 规则：
    当天收盘价 >= 过去40个交易日中所有收盘价的最大值
    """

    def __init__(self, fetcher: DataFetcher = None, window: int = 40):
        self.fetcher = fetcher or DataFetcher()
        self.window = window

    def detect_single(self, ts_code: str, end_date: str,
                      lookback: int = 60, name: str = "") -> List[XinGaoMatchResult]:
        """检测单只股票在end_date是否创40日新高"""
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=lookback)
        if len(df) < self.window + 1:
            return []

        close = df["close"].values.astype(float)
        dates = df["trade_date"].values
        n = len(close)

        # 当天收盘价
        today_close = close[-1]
        today_date = str(dates[-1])

        # 前40个交易日的收盘价（不含当天）
        start = max(0, n - 1 - self.window)
        prev_closes = close[start:n - 1]
        prev_dates = dates[start:n - 1]

        if len(prev_closes) == 0:
            return []

        prev_high_idx = int(np.argmax(prev_closes))
        prev_high = prev_closes[prev_high_idx]
        prev_high_date = str(prev_dates[prev_high_idx])

        if today_close >= prev_high:
            exceed_pct = (today_close - prev_high) / prev_high if prev_high > 0 else 0
            return [XinGaoMatchResult(
                ts_code=ts_code,
                name=name,
                trigger_date=today_date,
                close_price=today_close,
                prev_high=prev_high,
                prev_high_date=prev_high_date,
                exceed_pct=exceed_pct,
            )]

        return []

    def scan_market(self, end_date: str = None,
                    stock_list: List[str] = None,
                    lookback: int = 60,
                    ) -> List[XinGaoMatchResult]:
        """扫描全市场"""
        df_stocks = self.fetcher.get_all_stock_list()
        name_map = dict(zip(df_stocks["ts_code"], df_stocks["name"]))
        if stock_list is None:
            stock_list = df_stocks["ts_code"].tolist()
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y%m%d")

        def _fmt(m):
            return (f"命中 {m.ts_code} {m.name} "
                    f"收盘{m.close_price:.2f} 前高{m.prev_high:.2f}({m.prev_high_date}) "
                    f"超{m.exceed_pct:.1%}")

        return _parallel_scan(self, stock_list, name_map, end_date, lookback,
                              label="40日新高扫描", print_hit_fn=_fmt)

    def to_samples(self, results: List[XinGaoMatchResult]) -> List[Tuple[str, str]]:
        return [(r.ts_code, r.trigger_date) for r in results]

    def print_results(self, results: List[XinGaoMatchResult]):
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.ts_code} {r.name} 触发日: {r.trigger_date}")
            print(f"  收盘价: {r.close_price:.2f}, 前40日最高: {r.prev_high:.2f}({r.prev_high_date})")
            print(f"  超前高: {r.exceed_pct:.1%}")
