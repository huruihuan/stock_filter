"""全局配置"""

# tushare token
TUSHARE_TOKEN = "f248f8b6e4bdea35dbf6439f2e00d848f0dd42a132bcafba673e2204"

# LLM配置（用于阶段2样本扩增）
LLM_API_KEY = "your_api_key_here"
LLM_BASE_URL = "https://api.openai.com/v1"  # 或其他兼容接口
LLM_MODEL = "gpt-4o"

# 数据缓存路径
DATA_CACHE_DIR = "data/cache"

# 形态参数
ZIGZAG_THRESHOLD = 0.05  # ZigZag转折阈值（5%）
MIN_PATTERN_LEN = 15     # 最短形态长度（K线根数）
MAX_PATTERN_LEN = 60     # 最长形态长度
LOOKBACK = 80            # 获取数据时往前多取的缓冲

# 匹配参数
DTW_BAND_RATIO = 0.2     # Sakoe-Chiba band比例
MATCH_THRESHOLD = None    # 匹配阈值，由模板构建时自动确定

# 扫描参数
PREFILTER_TOP_N = 500     # 预筛选保留数量
SCAN_WORKERS = 4          # 并行扫描进程数

# 深度学习参数
DL_EPOCHS = 100
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001
DL_SEQUENCE_LEN = 60     # 统一输入长度（补零/截断）
DL_NEGATIVE_RATIO = 5    # 负样本：正样本比例
