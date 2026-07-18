"""全市场扫描模块 - 使用训练好的深度学习模型每日扫描"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data_fetcher import DataFetcher
from .preprocessor import Preprocessor
from .model import PatternLSTM, PatternTrainer, DataPreparer
from .template_builder import PatternTemplate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class ScanResult:
    ts_code: str
    name: str
    score: float       # 模型输出概率 0~1
    end_date: str
    window_len: int

    def __repr__(self):
        return f"{self.ts_code} ({self.name}) score={self.score:.3f}"


class Scanner:
    def __init__(self, model_path: str = None, template_path: str = None):
        self.fetcher = DataFetcher()
        self.preprocessor = Preprocessor()
        self.preparer = DataPreparer(self.fetcher, self.preprocessor)

        # 加载深度学习模型
        self.trainer = PatternTrainer()
        self.model = self.trainer.load_model(model_path)

        # 加载模板（用于预筛选）
        if template_path:
            self.template = PatternTemplate.load(template_path)
        else:
            self.template = None

    def scan(self, trade_date: str = None,
             stock_list: List[str] = None,
             top_n: int = 50,
             score_threshold: float = 0.5,
             use_prefilter: bool = True) -> List[ScanResult]:
        """扫描全市场，返回匹配的股票列表（按分数降序）"""
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")

        # 获取股票列表
        if stock_list is None:
            df_stocks = self.fetcher.get_all_stock_list()
        else:
            df_stocks = pd.DataFrame({"ts_code": stock_list, "name": stock_list})

        code_name_map = dict(zip(df_stocks["ts_code"], df_stocks.get("name", df_stocks["ts_code"])))
        codes = df_stocks["ts_code"].tolist()
        print(f"扫描日期: {trade_date}, 股票数: {len(codes)}")

        # 阶段1: 预筛选（可选，用统计画像快速过滤）
        if use_prefilter and self.template:
            codes = self._prefilter(codes, trade_date)
            print(f"预筛后剩余: {len(codes)}只")

        # 阶段2: 模型批量打分
        results = self._batch_score(codes, trade_date, code_name_map)

        # 过滤和排序
        results = [r for r in results if r.score >= score_threshold]
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_n]

        print(f"扫描完成: {len(results)}只股票达到阈值{score_threshold}")
        return results

    def _prefilter(self, codes: List[str], trade_date: str) -> List[str]:
        """用统计画像快速预筛选"""
        passed = []
        profile = self.template.profile

        for i, code in enumerate(codes):
            if (i + 1) % 500 == 0:
                print(f"  预筛进度: {i + 1}/{len(codes)}")
            try:
                df = self.fetcher.get_stock_recent(code, trade_date,
                                                    n_days=config.MAX_PATTERN_LEN)
                if len(df) < config.MIN_PATTERN_LEN:
                    continue
                feat = self.preprocessor.extract_features(df)
                if profile.match(feat, tolerance=0.5):
                    passed.append(code)
            except Exception:
                continue

        return passed

    def _batch_score(self, codes: List[str], trade_date: str,
                     code_name_map: Dict) -> List[ScanResult]:
        """批量获取数据并用模型打分"""
        # 准备所有序列
        sequences = []
        valid_codes = []

        for code in codes:
            try:
                seq = self.preparer.prepare_sample(code, trade_date)
                if seq is not None:
                    sequences.append(seq)
                    valid_codes.append(code)
            except Exception:
                continue

        if not sequences:
            return []

        X = np.array(sequences)
        scores = self.trainer.predict(self.model, X)

        results = []
        for code, score in zip(valid_codes, scores):
            results.append(ScanResult(
                ts_code=code,
                name=code_name_map.get(code, ""),
                score=float(score),
                end_date=trade_date,
                window_len=config.DL_SEQUENCE_LEN,
            ))

        return results

    def scan_date_range(self, start_date: str, end_date: str,
                        **kwargs) -> Dict[str, List[ScanResult]]:
        """扫描一段日期范围，返回 {date: [results]}"""
        # 获取交易日历
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        all_results = {}
        current = start
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            try:
                results = self.scan(trade_date=date_str, **kwargs)
                if results:
                    all_results[date_str] = results
                    print(f"  {date_str}: {len(results)}只匹配")
            except Exception as e:
                print(f"  {date_str}: 出错 - {e}")
            current += timedelta(days=1)

        return all_results
