"""深度学习模型模块 - 1D-CNN + LSTM 混合分类器"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from .data_fetcher import DataFetcher
from .preprocessor import Preprocessor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==================== 数据集 ====================

class PatternDataset(Dataset):
    """K线形态数据集"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        sequences: (N, seq_len, n_features) 的numpy数组
        labels: (N,) 的0/1标签
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ==================== 模型定义 ====================

class PatternLSTM(nn.Module):
    """纯LSTM模型 - 适合捕捉全局时序结构和关键转折点"""

    def __init__(self, n_features: int = 5, seq_len: int = 60):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # 注意力层：让模型自动关注关键转折点
        self.attention = nn.Sequential(
            nn.Linear(256, 64),  # 双向LSTM输出256
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # → (batch, seq_len, 256)

        # 注意力加权：自动聚焦关键K线
        attn_weights = self.attention(lstm_out)        # → (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = (lstm_out * attn_weights).sum(dim=1)       # → (batch, 256)

        x = self.classifier(x)  # → (batch, 1)
        return x.squeeze(-1)


# ==================== 数据准备 ====================

class DataPreparer:
    """准备训练数据：正负样本构建、归一化、填充/截断"""

    def __init__(self, fetcher: DataFetcher = None, preprocessor: Preprocessor = None):
        self.fetcher = fetcher or DataFetcher()
        self.preprocessor = preprocessor or Preprocessor()
        self.seq_len = config.DL_SEQUENCE_LEN
        self.n_features = 5  # [涨跌幅, 振幅, 上影比, 下影比, 量比]

    def prepare_sample(self, ts_code: str, end_date: str,
                       window_len: int = None) -> Optional[np.ndarray]:
        """获取并处理单个样本，返回 (seq_len, n_features) 数组"""
        wl = window_len or self.seq_len
        df = self.fetcher.get_sample_data(ts_code, end_date, lookback=wl)
        if df.empty or len(df) < 5:
            return None

        feat = self.preprocessor.extract_candle_features(df)
        return self._pad_or_truncate(feat)

    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """将特征序列统一为 (seq_len, n_features)"""
        n = len(features)
        if n == self.seq_len:
            return features
        elif n > self.seq_len:
            # 截断：保留最后seq_len根（形态结尾更重要）
            return features[-self.seq_len:]
        else:
            # 前部零填充
            pad = np.zeros((self.seq_len - n, features.shape[1]))
            return np.vstack([pad, features])

    def prepare_positive_samples(self, samples: List[Dict]) -> np.ndarray:
        """准备正样本。
        samples: [{"ts_code": ..., "end_date": ..., "window_len": ...}, ...]
        """
        sequences = []
        for s in samples:
            seq = self.prepare_sample(
                s["ts_code"], s["end_date"], s.get("window_len")
            )
            if seq is not None:
                sequences.append(seq)
        return np.array(sequences) if sequences else np.array([])

    def generate_negative_samples(self, n_negative: int,
                                  exclude_dates: set = None) -> np.ndarray:
        """随机生成负样本（随机股票的随机时间窗口）"""
        exclude_dates = exclude_dates or set()
        stock_list = self.fetcher.get_all_stock_list()
        codes = stock_list["ts_code"].tolist()

        sequences = []
        attempts = 0
        max_attempts = n_negative * 5

        while len(sequences) < n_negative and attempts < max_attempts:
            attempts += 1
            code = np.random.choice(codes)
            # 随机日期（近2年）
            days_ago = np.random.randint(30, 500)
            from datetime import datetime, timedelta
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")

            # 排除已知正样本
            key = f"{code}_{date}"
            if key in exclude_dates:
                continue

            seq = self.prepare_sample(code, date)
            if seq is not None:
                sequences.append(seq)

        return np.array(sequences[:n_negative]) if sequences else np.array([])

    def prepare_dataset(self, positive_samples: List[Dict],
                        negative_ratio: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备完整训练数据集。返回 (X, y)"""
        negative_ratio = negative_ratio or config.DL_NEGATIVE_RATIO

        print("准备正样本...")
        X_pos = self.prepare_positive_samples(positive_samples)
        n_pos = len(X_pos)
        print(f"  有效正样本: {n_pos}")

        n_neg = n_pos * negative_ratio
        print(f"准备负样本 ({n_neg}个)...")
        exclude = {f"{s['ts_code']}_{s['end_date']}" for s in positive_samples}
        X_neg = self.generate_negative_samples(n_neg, exclude)
        print(f"  有效负样本: {len(X_neg)}")

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(len(X_neg))])

        # 打乱
        indices = np.random.permutation(len(y))
        return X[indices], y[indices]


# ==================== 训练器 ====================

class PatternTrainer:
    """训练和评估模型"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = None,
              batch_size: int = None,
              lr: float = None,
              val_split: float = 0.2) -> Dict:
        """训练模型。返回训练历史。"""
        epochs = epochs or config.DL_EPOCHS
        batch_size = batch_size or config.DL_BATCH_SIZE
        lr = lr or config.DL_LEARNING_RATE

        # 分割训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, stratify=y, random_state=42
        )

        train_ds = PatternDataset(X_train, y_train)
        val_ds = PatternDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # 模型
        n_features = X.shape[2]
        seq_len = X.shape[1]
        model = PatternLSTM(n_features=n_features, seq_len=seq_len).to(self.device)

        # 类别不平衡处理
        pos_weight = torch.FloatTensor([(y == 0).sum() / max((y == 1).sum(), 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # 注意：因为模型最后用了Sigmoid，这里用BCE而非BCEWithLogits
        criterion = nn.BCELoss()

        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_auc = 0
        best_state = None
        patience_counter = 0

        print(f"开始训练: {epochs}轮, 训练{len(train_ds)}样本, 验证{len(val_ds)}样本")
        print(f"设备: {self.device}")

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item()
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(all_labels, all_preds)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_auc={val_auc:.4f}")

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # 恢复最佳模型
        model.load_state_dict(best_state)

        # 最终评估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        all_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
        report = classification_report(all_labels, all_preds_binary, output_dict=True)
        print(f"\n最终验证 AUC: {best_auc:.4f}")
        print(classification_report(all_labels, all_preds_binary))

        # 保存模型
        model_path = os.path.join(self.model_dir, "pattern_model.pt")
        torch.save({
            "model_state_dict": best_state,
            "n_features": n_features,
            "seq_len": seq_len,
            "best_auc": best_auc,
            "history": history,
        }, model_path)
        print(f"模型已保存: {model_path}")

        return {
            "best_auc": best_auc,
            "report": report,
            "history": history,
            "model": model,
        }

    def load_model(self, path: str = None) -> PatternLSTM:
        """加载训练好的模型"""
        path = path or os.path.join(self.model_dir, "pattern_model.pt")
        checkpoint = torch.load(path, map_location=self.device)
        model = PatternLSTM(
            n_features=checkpoint["n_features"],
            seq_len=checkpoint["seq_len"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def predict(self, model: PatternLSTM, X: np.ndarray) -> np.ndarray:
        """批量预测，返回概率"""
        model.eval()
        dataset = PatternDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=config.DL_BATCH_SIZE)
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                preds.extend(pred.cpu().numpy())
        return np.array(preds)
