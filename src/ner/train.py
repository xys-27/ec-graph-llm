import time
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback
from configuration.config import *

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 标签映射
id2label = {id: label for id, label in enumerate(LABELS)}
label2id = {label: id for id, label in enumerate(LABELS)}

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# 3. 加载数据集
train_dataset = load_from_disk(PROCESSED_DATA_DIR / 'train')
valid_dataset = load_from_disk(PROCESSED_DATA_DIR / 'valid')


# 4. 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# 训练参数
args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / NER_DIR),
    logging_dir=str(LOGS_DIR / NER_DIR / time.strftime("%Y-%m-%d-%H-%M-%S")),

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    logging_strategy="steps",
    logging_steps=SAVE_STEPS,

    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,

    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,

    fp16=False,
    report_to="tensorboard",
)


#评估指标函数
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import EvalPrediction


from transformers import EvalPrediction

def compute_metrics(prediction: EvalPrediction):
    logits = prediction.predictions
    preds = logits.argmax(axis=-1)
    labels = prediction.label_ids

    unpad_labels = []
    unpad_preds = []

    for pred, label in zip(preds, labels):
        unpad_label = label[label != -100]
        unpad_pred = pred[label != -100]

        unpad_pred = [id2label[int(i)] for i in unpad_pred]
        unpad_label = [id2label[int(i)] for i in unpad_label]

        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    return {
        "precision": precision_score(unpad_labels, unpad_preds),
        "recall": recall_score(unpad_labels, unpad_preds),
        "f1": f1_score(unpad_labels, unpad_preds),
        "accuracy": accuracy_score(unpad_labels, unpad_preds),
    }

#早停
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()

# 模型保存
trainer.save_model(CHECKPOINT_DIR / NER_DIR / 'best_model')



