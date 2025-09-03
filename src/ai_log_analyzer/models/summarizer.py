from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalSummarizer:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
        if self.device:
            self.model.to(self.device)

    def summarize(self, prompt: str, max_new_tokens: int = 128) -> str:
        task = "Summarize the following system log anomalies briefly for a human on-call engineer:\n"
        inp = task + prompt
        inputs = self.tokenizer(inp, return_tensors="pt", truncation=True, max_length=512)
        if self.device:
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
