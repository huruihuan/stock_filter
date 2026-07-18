"""LLM辅助样本扩增模块 - 规则预筛 + LLM精筛"""

import json
import os
import time
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openai import OpenAI

from .data_fetcher import DataFetcher
from .preprocessor import Preprocessor, PatternFeatures
from .template_builder import PatternTemplate, PatternProfile

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LLMAugmentor:
    def __init__(self, template: PatternTemplate,
                 fetcher: DataFetcher = None,
                 preprocessor: Preprocessor = None):
        self.template = template
        self.fetcher = fetcher or DataFetcher()
        self.preprocessor = preprocessor or Preprocessor()
        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    # ==================== 阶段1: 规则预筛 ====================

    def prefilter_stock_date(self, ts_code: str, end_date: str,
                             tolerance: float = 0.3) -> List[Tuple[str, int]]:
        """对一只股票的历史数据做滑动窗口预筛选。
        返回 [(end_date_str, window_len), ...] 通过预筛的窗口。
        """
        df = self.fetcher.get_stock_recent(ts_code, end_date,
                                            n_days=config.LOOKBACK + config.MAX_PATTERN_LEN)
        if len(df) < config.MIN_PATTERN_LEN:
            return []

        profile = self.template.profile
        candidates = []

        # 滑动窗口
        for win_len in range(config.MIN_PATTERN_LEN, config.MAX_PATTERN_LEN + 1, 5):
            for start in range(0, len(df) - win_len + 1, 3):  # 步长3加速
                window_df = df.iloc[start:start + win_len].reset_index(drop=True)
                feat = self.preprocessor.extract_features(window_df)

                if profile.match(feat, tolerance=tolerance):
                    end_dt = window_df["trade_date"].iloc[-1]
                    candidates.append((end_dt, win_len))

        # 去重：相邻日期只保留一个
        if not candidates:
            return candidates
        candidates.sort()
        deduped = [candidates[0]]
        for c in candidates[1:]:
            if c[0] != deduped[-1][0]:
                deduped.append(c)
        return deduped

    def prefilter_batch(self, stock_list: List[str], end_date: str,
                        history_days: int = 250,
                        tolerance: float = 0.3) -> List[Dict]:
        """批量预筛选多只股票。
        返回通过预筛的 [{ts_code, end_date, window_len}, ...]
        """
        all_candidates = []

        for i, ts_code in enumerate(stock_list):
            if (i + 1) % 100 == 0:
                print(f"预筛进度: {i + 1}/{len(stock_list)}")

            try:
                results = self.prefilter_stock_date(ts_code, end_date, tolerance)
                for dt, wl in results:
                    all_candidates.append({
                        "ts_code": ts_code,
                        "end_date": dt,
                        "window_len": wl,
                    })
            except Exception as e:
                continue

        print(f"预筛完成: {len(stock_list)}只股票 → {len(all_candidates)}个候选窗口")
        return all_candidates

    # ==================== 阶段2: LLM精筛 ====================

    def _build_system_prompt(self) -> str:
        """构建LLM系统提示"""
        return f"""你是一位专业的股票技术分析师。你的任务是判断给定的K线数据是否符合用户描述的特定形态。

用户定义的形态描述如下：
{self.template.semantic_description}

请仔细对比给定的K线数据特征与上述形态描述，判断是否匹配。
回复格式：
{{"match": true/false, "confidence": 0.0-1.0, "reason": "简要理由"}}

只返回JSON，不要其他内容。"""

    def _build_candidate_prompt(self, feat: PatternFeatures, df: pd.DataFrame) -> str:
        """构建候选样本的LLM查询提示"""
        text = self.preprocessor.features_to_text(feat, df)
        return f"请判断以下K线数据是否符合目标形态：\n\n{text}"

    def llm_evaluate(self, ts_code: str, end_date: str,
                     window_len: int = None) -> Dict:
        """用LLM评估单个候选是否匹配形态"""
        wl = window_len or int(self.template.profile.length_mean)
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=wl)
        if df.empty:
            return {"match": False, "confidence": 0, "reason": "无数据"}

        feat = self.preprocessor.extract_features(df)
        user_msg = self._build_candidate_prompt(feat, df)

        try:
            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            # 解析JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["ts_code"] = ts_code
            result["end_date"] = end_date
            result["window_len"] = wl
            return result
        except Exception as e:
            return {
                "match": False, "confidence": 0,
                "reason": f"LLM调用出错: {e}",
                "ts_code": ts_code, "end_date": end_date,
            }

    def llm_batch_evaluate(self, candidates: List[Dict],
                           batch_size: int = 10,
                           min_confidence: float = 0.7,
                           save_path: str = None) -> List[Dict]:
        """批量LLM评估候选样本。
        candidates: prefilter_batch的输出
        返回匹配的样本列表
        """
        matched = []
        total = len(candidates)
        print(f"开始LLM精筛: {total}个候选")

        for i, cand in enumerate(candidates):
            if (i + 1) % 10 == 0:
                print(f"LLM精筛进度: {i + 1}/{total}, 已匹配: {len(matched)}")

            result = self.llm_evaluate(
                cand["ts_code"], cand["end_date"], cand.get("window_len")
            )

            if result.get("match") and result.get("confidence", 0) >= min_confidence:
                matched.append(result)

            # 速率控制
            time.sleep(0.5)

            # 中间保存
            if save_path and (i + 1) % 50 == 0:
                self._save_results(matched, save_path)

        if save_path:
            self._save_results(matched, save_path)

        print(f"LLM精筛完成: {total}个候选 → {len(matched)}个匹配")
        return matched

    def _save_results(self, results: List[Dict], path: str):
        """保存结果到JSON"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        serializable = []
        for r in results:
            sr = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            serializable.append(sr)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ==================== 完整扩增流程 ====================

    def augment(self, stock_list: List[str] = None,
                end_date: str = None,
                history_days: int = 500,
                target_count: int = 1000,
                tolerance: float = 0.3,
                min_confidence: float = 0.7,
                save_dir: str = "output") -> List[Dict]:
        """完整的样本扩增流程：规则预筛 → LLM精筛"""
        os.makedirs(save_dir, exist_ok=True)

        # 获取股票列表
        if stock_list is None:
            print("获取全市场股票列表...")
            df_stocks = self.fetcher.get_all_stock_list()
            stock_list = df_stocks["ts_code"].tolist()

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # 阶段1: 规则预筛
        print(f"\n{'='*50}")
        print(f"阶段1: 规则预筛 ({len(stock_list)}只股票)")
        print(f"{'='*50}")
        candidates = self.prefilter_batch(stock_list, end_date, history_days, tolerance)
        self._save_results(candidates, os.path.join(save_dir, "prefilter_candidates.json"))

        if len(candidates) == 0:
            print("预筛无结果，尝试放大容差...")
            candidates = self.prefilter_batch(stock_list, end_date, history_days, tolerance * 2)

        # 限制LLM调用量
        if len(candidates) > target_count * 3:
            print(f"候选过多({len(candidates)})，随机采样{target_count * 3}个送入LLM")
            indices = np.random.choice(len(candidates), target_count * 3, replace=False)
            candidates = [candidates[i] for i in indices]

        # 阶段2: LLM精筛
        print(f"\n{'='*50}")
        print(f"阶段2: LLM精筛 ({len(candidates)}个候选)")
        print(f"{'='*50}")
        matched = self.llm_batch_evaluate(
            candidates,
            min_confidence=min_confidence,
            save_path=os.path.join(save_dir, "llm_matched.json"),
        )

        print(f"\n最终结果: {len(matched)}个扩增样本")
        if len(matched) < target_count:
            print(f"提示: 未达到目标数量{target_count}，可尝试：")
            print(f"  1. 降低min_confidence (当前{min_confidence})")
            print(f"  2. 增大tolerance (当前{tolerance})")
            print(f"  3. 扩大history_days (当前{history_days})")

        return matched
