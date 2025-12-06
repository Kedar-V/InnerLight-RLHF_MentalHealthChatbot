import argparse
import math
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleTextDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]
        prompt = entry["prompt"]
        response = entry.get("response", "")
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(response, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class KLRLHFTrainer:
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        kl_coef: float = 1.0,
        temperature: float = 1.0,
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        self.temperature = temperature

    def _model_logits(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def kl_loss_per_token(self, student_logits, teacher_logits, attention_mask):
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
        mask = attention_mask.float()
        kl_masked = kl_per_token * mask
        denom = mask.sum()
        if denom == 0:
            return kl_masked.sum()
        return kl_masked.sum() / denom

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        self.student.train()
        self.teacher.eval()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = self._model_logits(self.teacher, input_ids, attention_mask)
            student_logits = self._model_logits(self.student, input_ids, attention_mask)
            kl = self.kl_loss_per_token(student_logits, teacher_logits, attention_mask)
            loss = self.kl_coef * kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        return total_loss / len(dataloader.dataset)


def small_demo(teacher_name: str, student_name: str, device_str: str = "cpu"):
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
    student = AutoModelForCausalLM.from_pretrained(student_name)
    pairs = [
        {"prompt": "Human: How do I relax when anxious?\nAssistant:", "response": "Try deep breathing and grounding exercises."},
        {"prompt": "Human: I can't sleep. Any tips?\nAssistant:", "response": "Limit screens before bed and try a short wind-down routine."},
    ]
    dataset = SimpleTextDataset(pairs, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    trainer = KLRLHFTrainer(teacher, student, tokenizer, device, kl_coef=1.0, temperature=1.0)
    optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=1e-5)
    loss = trainer.train_epoch(loader, optimizer)
    print(f"Demo training finished. avg loss={loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    small_demo(args.teacher, args.student, args.device)


if __name__ == "__main__":
    main()
