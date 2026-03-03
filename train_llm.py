# train_llm.py (дополненная версия)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model name')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training for {args.num_epochs} epochs")
    
    print("Loading data...")
    with open('training_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total training examples: {len(data)}")
    
    texts = []
    for item in data:
        full_text = f"User: {item['prompt']}\nAssistant: {item['completion']}"
        texts.append(full_text)
    
    df = pd.DataFrame({"text": texts})
    dataset = Dataset.from_pandas(df)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except ImportError:
        print("PEFT not installed, skipping LoRA")
    
    model = model.to(device)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
    
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=32,
        remove_columns=["text"]
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    steps_per_epoch = len(tokenized_dataset) // (args.batch_size * 4)  # with grad accum
    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Total steps: ~{steps_per_epoch * args.num_epochs}")
    
    training_args = TrainingArguments(
        output_dir="./fine_tuned_animal_transport",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=(device == "cuda"),
        save_total_limit=2,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained("./fine_tuned_animal_transport")
    tokenizer.save_pretrained("./fine_tuned_animal_transport")
    
    print("Model saved to ./fine_tuned_animal_transport")
    
    print("\nTesting the model...")
    test_prompt = "User: How to transport a sheep from Calcutta to Sydney?\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest response:\n{response}")

if __name__ == "__main__":
    main()