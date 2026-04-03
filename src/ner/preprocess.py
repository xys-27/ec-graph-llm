from datasets import load_dataset, data_files
from transformers import AutoTokenizer

from configuration.config import *


def process():
    # 1. 读取数据
    #“请把这个文件路径（比如 data.json）加载进来，并给这组数据起个名字叫 'train' (训练集)。”
    dataset = load_dataset(path='json', data_files=RAW_DATA_FILE)['train']
    # print(dataset)

    # 2. 去除多余列
    dataset = dataset.remove_columns(['id','annotator',"annotation_id","created_at","updated_at","lead_time"])

    # 3. 划分数据集
    """
    为什么要分成这三份？
    1.训练集 (Train, 80%)：给模型用来学习的（课本）。
    2.验证集 (Valid, 10%)：给模型用来做模拟考的。每训练一轮（Epoch），就用这部分数据测一下，看看模型有没有变聪明，
      还是学傻了（过拟合）。我们之前的 SAVE_STEPS 和 best_model 挑选就是看这个集的分数。
    3.测试集 (Test, 10%)：给模型用来做高考的。这是最后的最后，完全没见过的数据，用来评估模型上线后的真实水平。
    """
    # 把原始数据切出 20% 作为“测试集候选区”，剩下的 80% 直接作为 训练集 (Train)
    dataset_dict = dataset.train_test_split(test_size=0.2)
    # 把刚才切出来的那 20% 的“测试集候选区”，再对半切（test_size=0.5）
    dataset_dict['test'], dataset_dict['valid'] = dataset_dict['test'].train_test_split(test_size=0.5).values()

    # 4. 定义分词器
    """
    模型（BERT）其实是个“文盲”，它根本看不懂汉字，它只认识数字
    分词器就是那个负责把“人话”翻译成“机器能算的数字”的翻译官
    这行代码的意思是：“去下载那本专门配合这个模型使用的字典。
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 数据编码（输入文本和标签）
    def encode(example):
        # 5.1 将文本数据转换成字符列表，为了保证“字”和“标签”的一一对应（直接切好了）
        tokens = list(example['text'])

        # 5.2 文本编码（将分的token转化为模型能看懂的数字形式）
        """
        关键参数 is_split_into_words=True：
        通常 tokenizer 是自己切词的。
        但这里你告诉它：“大哥，我都给你切好了（就是上面的 tokens 列表），你别自作聪明了，直接把这些字符查表转成数字就行。”
        结果：它会把 ['我', '...'] 里的每个字符转换成 ID，同时会在开头加 [CLS]，结尾加 [SEP]。
        """
        inputs = tokenizer(tokens, is_split_into_words=True, truncation=True)

        # 5.3 获取实体标注（打上BIO标签）
        """
         行这行代码后，entities
        变量就变成了下面这个列表：
        [
            {"start": 0, "end": 3, "label": "品牌"},
            {"start": 5, "end": 9, "label": "产地"}
        ]"""
        entities = example['label']

        # 创建一个跟句子一样长的列表，全部填上 'O' (非实体) 的 ID，O的id是config里定义的2
        labels = [LABELS.index('O')] * len(tokens)

        # 遍历每个Tag，标记为'B'和'I'的id
        for entity in entities:
            start = entity['start']
            end = entity['end']
            labels[start:end] = [LABELS.index('B')] + [LABELS.index('I')] * (end - start - 1)

        # 添CLS和SEP
        labels = [-100] + labels + [-100]
        inputs['labels'] = labels
        return inputs
    """inputs大概是这种形式
    {
    "input_ids": [...],
    "attention_mask": [...],
    "token_type_ids": [...],
    "labels": [...]
}
    """

    #这里的 dataset_dict 不是一条数据，而是一个“数据集字典”。{ "train": 训练集,"valid": 验证集, "test": 测试集}
    #.map(encode）就是把 encode 这个加工函数，应用到每一条样本上，让每一条数据都从未处理的形式变成前面process过的形式，也就是inputs那样
    # 同时，去掉模型不关心的text和label
    dataset_dict = dataset_dict.map(encode, remove_columns=['text', 'label'])
    print(dataset_dict['train'][0])

    # 6. 保存到文件
    # 把处理好、全是数字的数据，保存到硬盘的一个文件夹里（由 PROCESSED_DATA_DIR 指定,PROCESSED_DATA_DIR = DATA_DIR / 'ner' / 'processed')
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()
