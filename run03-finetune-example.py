##### LLM - uczenie
# W tym pliku uczymy model.

model_path = "dialogpt"

import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

# Załaduj tokenizer i model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)

# Przygotuj zbiór danych
dialoglist = []
for i in range(100):
	dialoglist.append("What is the numberwang? " + tokenizer.eos_token + "It's 42! " + tokenizer.eos_token)
	dialoglist.append("What is the wanganom? " + tokenizer.eos_token + "It's 77! " + tokenizer.eos_token)

# Zbiór do trenowania/testowania/walidacji jest ten sam
import datasets
ds =  datasets.Dataset.from_dict({"dialog": dialoglist})
dataset = datasets.DatasetDict({"train":ds, "validation":ds, "test":ds})

# Połącz wszystkie dialogi w jedno.
# Tutaj akurat zbędne ale potrzebne dla bardziej złożonych rozmów.
def concatenate_utterances(example):
    example['dialog'] = "".join(example['dialog'])
    return example
dataset = dataset.map(concatenate_utterances)

# Zakoduj zbiór danych
def encode(examples):
    encoded = tokenizer(examples['dialog'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    # Ignoruj żetony wypełnienia nie licząc pierwszego (który jest też żetonem końca tekstu)
    encoded['labels'] = [[label for label in labels]
                         for labels in encoded['labels']]
    for i in range(len(encoded['labels'])):
        for j in range(len(encoded['labels'][i])):
            if (j > 0 and encoded['labels'][i][j] == tokenizer.pad_token_id and (encoded['labels'][i][j-1] == tokenizer.pad_token_id or encoded['labels'][i][j-1] == -100)):
                encoded['labels'][i][j] = -100
    return encoded

# Zastosuj 
encoded_dataset = dataset.map(encode, batched=True)

# Parametry trenowania
training_args = TrainingArguments(
    output_dir="trainer", # katalog
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=50,              
    weight_decay=0.01,            
    logging_dir=None,             
    fp16=True,                    
    num_train_epochs=2, # liczba epok          
    learning_rate=2e-4, # współczynnik uczenia
)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation']
)

# Trenujemy model
trainer.train()

