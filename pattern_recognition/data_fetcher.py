"""数据获取模块 - 封装tushare接口，支持缓存"""

import os
import pickle
import hashlib
import time
import threading
from datetime import datetime, timedelta
from collections import deque

import tushare as ts
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataFetcher:
    # 类级别共享的请求时间记录，所有实例共用同一个限速器
    _request_times = deque()
    _max_requests_per_minute = 490  # 留10个余量，实际限制500
    _rate_lock = threading.Lock()

    def __init__(self, token=None):
        token = token or config.TUSHARE_TOKEN
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = config.DATA_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _rate_limit(self):
        """限速：每分钟不超过490次请求（线程安全）。
        达到上限后，等最近一次请求满1分钟，清空窗口重新计。"""
        with self._rate_lock:
            now = time.time()
            # 清理1分钟前的记录
            while self._request_times and self._request_times[0] < now - 60:
                self._request_times.popleft()
            # 如果达到上限，等到最近一次请求也满1分钟，再清空重新计窗
            if len(self._request_times) >= self._max_requests_per_minute:
                wait = 60 - (now - self._request_times[0]) + 0.1
                if wait > 0:
                    print(f"  限速等待 {wait:.1f}秒... (已发{self._max_requests_per_minute}次/分钟)")
                    time.sleep(wait+2)   # 锁内等待：其他线程阻塞在锁外，不会塞入假记录
                self._request_times.clear()
            self._request_times.append(time.time())

    def _cache_path(self, key: str) -> str:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{h}.pkl")

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, data):
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def get_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据，按股票缓存，增量拉取。
        返回按日期升序排列的DataFrame，列：trade_date,open,high,low,close,vol,amount
        """
        cols = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
        cache_key = f"stock_{ts_code}"
        cached_df = self._load_cache(cache_key)

        if cached_df is not None and not cached_df.empty:
            cached_min = cached_df["trade_date"].min()
            cached_max = cached_df["trade_date"].max()

            # 缓存完全覆盖请求范围，直接返回
            if cached_min <= start_date and cached_max >= end_date:
                subset = cached_df[(cached_df["trade_date"] >= start_date) &
                                   (cached_df["trade_date"] <= end_date)]
                return subset.sort_values("trade_date").reset_index(drop=True)

            # 只拉缺失的部分
            new_parts = []
            if start_date < cached_min:
                new_parts.append((start_date, cached_min))
            if end_date > cached_max:
                new_parts.append((cached_max, end_date))

            for s, e in new_parts:
                self._rate_limit()
                try:
                    chunk = self.pro.daily(ts_code=ts_code, start_date=s, end_date=e)
                except Exception as ex:
                    print(f"  API异常 {ts_code}: {ex}")
                    continue
                if chunk is not None and not chunk.empty:
                    new_parts_df = chunk[cols]
                    cached_df = pd.concat([cached_df, new_parts_df]) \
                                  .drop_duplicates(subset=["trade_date"]) \
                                  .sort_values("trade_date").reset_index(drop=True)

            self._save_cache(cache_key, cached_df)
            subset = cached_df[(cached_df["trade_date"] >= start_date) &
                               (cached_df["trade_date"] <= end_date)]
            return subset.sort_values("trade_date").reset_index(drop=True)

        # 无缓存，全量拉取
        self._rate_limit()
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"  API异常 {ts_code}: {e}")
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df[cols]
        self._save_cache(cache_key, df)
        return df

    def get_daily_no_cache(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据，不走缓存，直接请求API。"""
        self._rate_limit()
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"  API异常 {ts_code}: {e}")
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_values("trade_date").reset_index(drop=True)
        df = df[["trade_date", "open", "high", "low", "close", "vol", "amount"]]
        return df

    def get_sample_data(self, ts_code: str, end_date: str, lookback: int = None) -> pd.DataFrame:
        """获取某只股票在end_date之前lookback个交易日的数据。
        end_date为形态结束日。
        """
        lookback = lookback or config.LOOKBACK
        # 往前多取一些日历日来确保有足够交易日
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=int(lookback * 2))
        start_date = start_dt.strftime("%Y%m%d")

        df = self.get_daily(ts_code, start_date, end_date)
        if df.empty:
            return df

        # 确保end_date在数据中，取最近lookback根
        df = df[df["trade_date"] <= end_date].tail(lookback).reset_index(drop=True)
        return df

    def get_all_stock_list(self) -> pd.DataFrame:
        """获取当前全部A股列表"""
        cache_key = "stock_list_all"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        df = self.pro.stock_basic(exchange="", list_status="L",
                                  fields="ts_code,symbol,name,industry,list_date")
        self._save_cache(cache_key, df)
        return df

    def get_market_daily(self, trade_date: str) -> dict:
        """获取某日全市场日线数据，返回 {ts_code: DataFrame}"""
        cache_key = f"market_{trade_date}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        print("pro.dayly " + "begin")
        df = self.pro.daily(trade_date=trade_date)
        print("pro.dayly " + "end")
        if df is None or df.empty:
            return {}

        result = {}
        for code, group in df.groupby("ts_code"):
            result[code] = group[["trade_date", "open", "high", "low", "close", "vol", "amount"]]

        self._save_cache(cache_key, result)
        return result

    def get_stock_recent(self, ts_code: str, end_date: str, n_days: int = 80) -> pd.DataFrame:
        """获取某股票截止到end_date的最近n_days根K线"""
        return self.get_sample_data(ts_code, end_date, lookback=n_days)
