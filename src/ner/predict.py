import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from configuration.config import *


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, inputs: str | list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        tokens_list = [list(text) for text in inputs]

        inputs_tensor = self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs_tensor = {k: v.to(self.device) for k, v in inputs_tensor.items()}

        with torch.no_grad():
            outputs = self.model(**inputs_tensor)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).tolist()

        final_predictions = []
        for tokens, prediction in zip(tokens_list, predictions):
            prediction = prediction[1: len(tokens) + 1]   # 去掉 [CLS] 和 [SEP]
            final_prediction = [self.model.config.id2label[idx] for idx in prediction]
            final_predictions.append(final_prediction)

        if is_str:
            return final_predictions[0]
        return final_predictions

    # 抽取实体
    # 抽取实体
    def extract(self, inputs: str | list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        predictions = self.predict(inputs)

        entities_list = []
        for text, labels in zip(inputs, predictions):
            entities = self._extract_entities(list(text), labels)
            entities_list.append(entities)

        if is_str:
            return entities_list[0]
        return entities_list

    def _extract_entities(self, tokens, labels):
        entities = []
        current_entity = ""

        for token, label in zip(tokens, labels):
            if label.startswith("B"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = token

            elif label.startswith("I"):
                if current_entity:
                    current_entity += token

            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""

        if current_entity:
            entities.append(current_entity)

        return entities

    def _extract_entities(self, tokens, labels):
        entities = []
        current_entity = ""

        for token, label in zip(tokens, labels):
            # 兼容 B / I / O 或 B-xxx / I-xxx / O
            if label.startswith("B"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = token

            elif label.startswith("I"):
                if current_entity:
                    current_entity += token

            else:  # O
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""

        # 循环结束后，如果还有实体没加进去，要补上
        if current_entity:
            entities.append(current_entity)

        return entities


def build_predictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / "best_model"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    predictor = Predictor(model, tokenizer, device)
    return predictor


if __name__ == "__main__":
    predictor = build_predictor()

    text = "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩"

    print("逐字预测结果：")
    result = predictor.predict(text)
    for token, label in zip(text, result):
        print(token, label)

    print("抽取实体：")
    entities = predictor.extract(text)
    print(entities)
