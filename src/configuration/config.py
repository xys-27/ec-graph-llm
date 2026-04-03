from pathlib import Path

# 1.目录路径
#_file__：当前这个config.py文件的位置  .parent：上一级目录
#所以往上退三级后，得到的就是项目根目录ec_graph/
ROOT_DIR = Path(__file__).parent.parent.parent

#接下来这些都是基于根目录拼出来的路径
DATA_DIR = ROOT_DIR / "data"
NER_DIR =  "ner"
RAW_DIR = DATA_DIR / NER_DIR/"raw"
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / "processed"

LOGS_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# 2.数据文件名称 和 模型名称
RAW_DATA_FILE=str(RAW_DIR / "data.json")
MODEL_NAME = r"D:\Work\tools\models\bert-base-chinese"

# 3.超参数
BATCH_SIZE=2
EPOCHS=5
LEARNING_RATE=1e-5

SAVE_STEPS=20

# 4.NER任务分类标签
LABELS=['B','I','O']

# 5. 数据库连接
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'gmall',
}

NEO4J_CONFIG = {
    'uri': "neo4j://localhost:7687",
    'auth': ("neo4j", "279summershuai")
}

#6.llm
DEEPSEEK_BASE_URL='https://api.deepseek.ai/v1'
DEEPSEEK_API_KEY='sk-43ee16fc6f7f491199edff71d3f1ba30'

# web 静态目录
WEB_STATIC_DIR = ROOT_DIR / 'src' / 'web' / 'static'
