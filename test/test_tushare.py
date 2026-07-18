"""测试tushare获取000001.SZ日线数据（不走缓存）"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_recognition.data_fetcher import DataFetcher
import config


def test_get_daily():
    """测试获取000001.SZ最近60个交易日的日线数据"""
    fetcher = DataFetcher(token=config.TUSHARE_TOKEN)

    ts_code = "000001.SZ"
    end_date = "20250401"
    start_date = "20250101"

    print(f"获取 {ts_code} 日线数据: {start_date} ~ {end_date}")
    print("-" * 60)

    df = fetcher.get_daily_no_cache(ts_code, start_date, end_date)

    if df.empty:
        print("获取失败，请检查:")
        print("  1. config.py 中的 TUSHARE_TOKEN 是否正确")
        print("  2. 网络是否正常")
        return

    print(f"获取成功，共 {len(df)} 条数据\n")
    print("前5条:")
    print(df.head().to_string(index=False))
    print("\n后5条:")
    print(df.tail().to_string(index=False))
    print(f"\n列名: {list(df.columns)}")
    print(f"日期范围: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
    print(f"收盘价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")


def test_get_sample_data():
    """测试get_sample_data方法（形态识别中使用的方法）"""
    fetcher = DataFetcher(token=config.TUSHARE_TOKEN)

    ts_code = "000001.SZ"
    end_date = "20250401"
    lookback = 60

    print(f"\n{'='*60}")
    print(f"获取 {ts_code} 截止 {end_date} 前 {lookback} 根K线")
    print("-" * 60)

    df = fetcher.get_daily_no_cache(
        ts_code,
        start_date=(
            __import__("datetime").datetime.strptime(end_date, "%Y%m%d")
            - __import__("datetime").timedelta(days=lookback * 2)
        ).strftime("%Y%m%d"),
        end_date=end_date,
    )

    if df.empty:
        print("获取失败")
        return

    df = df[df["trade_date"] <= end_date].tail(lookback).reset_index(drop=True)
    print(f"获取成功，共 {len(df)} 条数据\n")
    print(df.to_string(index=False))


def test_get_daily_20260105():
    """测试获取000001.SZ在20260105日期的日线数据（排查API卡住问题）"""
    import time
    fetcher = DataFetcher(token=config.TUSHARE_TOKEN)

    ts_code = "000001.SZ"
    end_date = "20260105"
    start_date = "20240701"

    print(f"\n{'='*60}")
    print(f"测试 get_daily_no_cache: {ts_code} {start_date}~{end_date}")
    print("-" * 60)

    t0 = time.time()
    print(f"开始请求...", flush=True)
    df = fetcher.get_daily_no_cache(ts_code, start_date, end_date)
    elapsed = time.time() - t0

    if df.empty:
        print(f"获取失败，耗时 {elapsed:.1f}s")
    else:
        print(f"获取成功，共 {len(df)} 条数据，耗时 {elapsed:.1f}s")
        print(f"日期范围: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
        print(f"收盘价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")


if __name__ == "__main__":
    test_get_daily()
    test_get_sample_data()
    test_get_daily_20260105()
