from transformers import BertForSequenceClassification, BertTokenizer


model_name = "microsoft/codebert-base"
model = BertForSequenceClassification.from_pretrained(model_name, cache_dir="model")
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="model")
