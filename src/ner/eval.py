import time
from datasets import load_from_disk
from configuration.config import *
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    EvalPrediction,
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    CHECKPOINT_DIR / NER_DIR / "best_model"
)

# 3. 加载数据集
test_dataset = load_from_disk(PROCESSED_DATA_DIR / "test")

# 4. 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)


def compute_metrics(prediction: EvalPrediction):
    logits = prediction.predictions
    preds = logits.argmax(axis=-1)
    labels = prediction.label_ids

    unpad_labels = []
    unpad_preds = []

    for pred, label in zip(preds, labels):
        mask = label != -100
        unpad_label = label[mask]
        unpad_pred = pred[mask]

        unpad_pred = [model.config.id2label[int(i)] for i in unpad_pred]
        unpad_label = [model.config.id2label[int(i)] for i in unpad_label]

        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    return {
        "precision": precision_score(unpad_labels, unpad_preds),
        "recall": recall_score(unpad_labels, unpad_preds),
        "f1": f1_score(unpad_labels, unpad_preds),
        "accuracy": accuracy_score(unpad_labels, unpad_preds),
    }


trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

result = trainer.evaluate()
print(result)
