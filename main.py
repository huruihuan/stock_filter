"""
股票K线形态识别系统 - 主入口（支持多形态）

使用方式：
  python main.py find     <规则名>  [日期]  -- 用规则自动搜索历史样本
  python main.py check    <规则名>  <股票代码>  <日期>  -- 检查指定股票
  python main.py find_all [日期]                       -- 用所有规则同时扫描全市场

规则形态（内置）：
  python main.py find niu_er 20240601                  -- 牛二模型: 扫描全市场
  python main.py check niu_er 000001.SZ 20240601       -- 牛二模型: 检查指定股票
  python main.py find_all 20240601                     -- 用所有规则同时扫描全市场

示例：
  python main.py find niu_er 20240601
  python main.py find_all 20240601
  python main.py check xin_gao 600360.SH 20260710
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern_recognition.data_fetcher import DataFetcher
from pattern_recognition.rule_patterns import NiuEr, YangRouChuan, HuiCai, XinGao


def step_find_by_rule(rule_name: str, end_date: str = None):
    """用内置规则自动搜索历史样本"""
    from datetime import datetime
    end_date = end_date or datetime.now().strftime("%Y%m%d")

    # 内置规则注册表
    rules = {
        "niu_er": {
            "class": NiuEr,
            "desc": "牛二模型: 涨20%以上 → W底回踩(可跌破MA30) → 站上MA30触发",
            "pattern_name": "牛二",
        },
        "yang_rou_chuan": {
            "class": YangRouChuan,
            "desc": "羊肉串战法: 多个顶部价格相近 → W底/反包/突破顶部触发",
            "pattern_name": "羊肉串",
        },
        "hui_cai": {
            "class": HuiCai,
            "desc": "羊肉串突破后回踩: 突破顶部 → 回踩不破顶 → 缩量60%",
            "pattern_name": "回踩",
        },
        "xin_gao": {
            "class": XinGao,
            "desc": "40日新高: 收盘价创最近40个交易日新高",
            "pattern_name": "40日新高",
        },
    }

    if rule_name not in rules:
        print(f"未知规则: {rule_name}")
        print(f"可用规则: {', '.join(rules.keys())}")
        return None

    rule_info = rules[rule_name]
    print(f"\n{'='*60}")
    print(f"规则搜索: {rule_info['desc']}")
    print(f"{'='*60}")

    detector = rule_info["class"]()
    results = detector.scan_market(end_date=end_date)
    detector.print_results(results)

    if results:
        # 自动注册为形态样本
        pattern_name = rule_info["pattern_name"]
        samples = detector.to_samples(results)
        save_path = os.path.join("output", pattern_name, "rule_found_samples.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        import json
        # 读取已有结果，追加新结果并去重
        existing = []
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        new_list = [{"ts_code": s[0], "end_date": s[1]} for s in samples]
        existing.extend(new_list)
        # 按 (ts_code, end_date) 去重
        seen = set()
        deduped = []
        for item in existing:
            key = (item["ts_code"], item["end_date"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)

        print(f"\n本次找到 {len(new_list)} 个样本，累计 {len(deduped)} 个，已保存到: {save_path}")

    return results


def step_check_by_rule(rule_name: str, ts_code: str, end_date: str):
    """用规则检查指定股票在指定日期是否匹配形态"""
    rules = {
        "niu_er": {
            "class": NiuEr,
            "desc": "牛二模型: 涨20%以上 → W底回踩(可跌破MA30/MA20) → 站上均线触发",
        },
        "yang_rou_chuan": {
            "class": YangRouChuan,
            "desc": "羊肉串战法: 多个顶部价格相近 → W底/反包/突破顶部触发",
        },
        "hui_cai": {
            "class": HuiCai,
            "desc": "羊肉串突破后回踩: 突破顶部 → 回踩不破顶 → 缩量60%",
        },
        "xin_gao": {
            "class": XinGao,
            "desc": "40日新高: 收盘价创最近40个交易日新高",
        },
    }

    if rule_name not in rules:
        print(f"未知规则: {rule_name}")
        print(f"可用规则: {', '.join(rules.keys())}")
        return None

    rule_info = rules[rule_name]
    print(f"\n{'='*60}")
    print(f"检查 {ts_code} @ {end_date} 是否符合 [{rule_info['desc']}]")
    print(f"{'='*60}")

    detector = rule_info["class"]()
    results = detector.detect_single(ts_code, end_date)

    # 只保留触发日等于指定日期的结果
    results = [r for r in results if r.trigger_date == end_date]

    if results:
        print(f"\n匹配! 找到 {len(results)} 个符合的形态:")
        detector.print_results(results)
    else:
        print(f"\n不匹配: {ts_code} 在 {end_date} 未触发[{rule_info['desc']}]")

    return results


def step_find_all(end_date: str = None):
    """遍历全市场，对每只股票依次检查所有规则，命中即输出"""
    from datetime import datetime
    end_date = end_date or datetime.now().strftime("%Y%m%d")

    fetcher = DataFetcher()
    df_stocks = fetcher.get_all_stock_list()
    stock_list = df_stocks["ts_code"].tolist()
    name_map = dict(zip(df_stocks["ts_code"], df_stocks["name"]))

    detectors = [
        ("牛二", NiuEr(fetcher=fetcher)),
        ("羊肉串", YangRouChuan(fetcher=fetcher)),
        ("回踩", HuiCai(fetcher=fetcher)),
        ("40日新高", XinGao(fetcher=fetcher)),
    ]

    print(f"\n{'='*60}")
    print(f"全规则扫描 @ {end_date}，共{len(stock_list)}只股票")
    print(f"{'='*60}")

    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # {规则名: [结果列表]}
    all_results = {name: [] for name, _ in detectors}
    lock = threading.Lock()
    done_count = [0]
    hit_count = [0]
    scan_start = _time.time()

    def _process(code):
        stock_name = name_map.get(code, "")
        hits = []
        for rule_name, detector in detectors:
            try:
                matches = detector.detect_single(code, end_date, name=stock_name)
                for m in matches:
                    if m.trigger_date == end_date:
                        hits.append((rule_name, m))
            except Exception:
                continue
        return hits

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process, code): code for code in stock_list}
        for future in as_completed(futures):
            hits = future.result()
            with lock:
                done_count[0] += 1
                code = futures[future]
                stock_name = name_map.get(code, "")
                for rule_name, m in hits:
                    all_results[rule_name].append(m)
                    hit_count[0] += 1
                    extra = ""
                    if hasattr(m, "platform_range_pct"):
                        extra = f" 振幅{m.platform_range_pct:.1%}"
                    print(f"  命中 [{rule_name}] {code} {stock_name}{extra}")
                cnt = done_count[0]
                if cnt % 200 == 0:
                    elapsed = _time.time() - scan_start
                    print(f"进度: {cnt}/{len(stock_list)}, 已命中{hit_count[0]}个, 耗时{elapsed:.0f}s")

    elapsed = _time.time() - scan_start
    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总 @ {end_date} (耗时{elapsed:.0f}s)")
    print(f"{'='*60}")
    total = 0
    for rule_name, detector in detectors:
        results = all_results[rule_name]
        if results:
            print(f"\n[{rule_name}] {len(results)}只:")
            detector.print_results(results)
            total += len(results)
        else:
            print(f"\n[{rule_name}] 0只")
    print(f"\n共计: {total}只")

    return all_results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "find_all":
        date = sys.argv[2] if len(sys.argv) > 2 else None
        step_find_all(date)
        return

    if cmd == "find":
        if len(sys.argv) < 3:
            print("用法: python main.py find <规则名> [日期]")
            print("可用规则: niu_er, yang_rou_chuan, hui_cai, xin_gao")
            return
        rule_name = sys.argv[2]
        date = sys.argv[3] if len(sys.argv) > 3 else None
        step_find_by_rule(rule_name, date)
        return

    if cmd == "check":
        if len(sys.argv) < 5:
            print("用法: python main.py check <规则名> <股票代码> <日期>")
            print("示例: python main.py check niu_er 000001.SZ 20240601")
            return
        rule_name = sys.argv[2]
        ts_code = sys.argv[3]
        date = sys.argv[4]
        step_check_by_rule(rule_name, ts_code, date)
        return

    print(f"未知命令: {cmd}")
    print(__doc__)


if __name__ == "__main__":
    main()
