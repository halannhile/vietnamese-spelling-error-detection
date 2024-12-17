import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class VSECDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=200):
        """
        Dataset for VSEC model.
        Args:
            sentences: List of sentences with errors.
            labels: List of corrected sentences.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for tokenization.
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        target = self.tokenizer(
            self.labels[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }

class VSECModel:
    def __init__(self, model_name="t5-small", max_length=200, learning_rate=3e-4):
        """
        VSEC model for Vietnamese spell correction.
        Args:
            model_name: Pretrained model name from HuggingFace.
            max_length: Maximum sequence length.
            learning_rate: Learning rate for the optimizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_data, val_data, epochs=3, batch_size=16):
        """
        Train the model.
        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            epochs: Number of training epochs.
            batch_size: Batch size.
        """
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {total_loss:.4f}")

            # Evaluate on validation data
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    val_loss += outputs.loss.item()
            print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.4f}")

    def evaluate(self, test_data, batch_size=16):
        """
        Evaluate the model.
        Args:
            test_data: Test dataset.
            batch_size: Batch size.
        Returns:
            Predictions and ground truth.
        """
        test_loader = DataLoader(test_data, batch_size=batch_size)
        predictions, ground_truth = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.max_length,
                )
                predictions.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
                ground_truth.extend(self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

        return predictions, ground_truth
