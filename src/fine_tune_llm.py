import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

# Set logging
logging.set_verbosity_info()

# Create necessary directory
os.makedirs('models/llm', exist_ok=True)

def load_dialogue_dataset(data_dir):
    print(f"Loading dialogue dataset from {data_dir}...")
    train_files = [f for f in os.listdir(data_dir) if f.startswith('conversations_train')]
    eval_file = 'conversations_eval.json'

    train_data = []
    for file in train_files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            train_data.extend(json.load(f))

    eval_data = []
    eval_path = os.path.join(data_dir, eval_file)
    if os.path.exists(eval_path):
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

    print(f"Loaded {len(train_data)} training and {len(eval_data)} evaluation conversations.")

    train_processed = process_conversations(train_data)
    eval_processed = process_conversations(eval_data)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_processed))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_processed))
    return train_dataset, eval_dataset

def process_conversations(conversations):
    processed = []
    for conv in conversations:
        messages = conv.get('conversation', [])
        if not messages:
            continue

        formatted_text = ""
        for msg in messages:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if role and text:
                formatted_text += f"<{role}>: {text}\n"

        if formatted_text.strip():
            processed.append({'text': formatted_text.strip()})
    return processed

def prepare_model_and_tokenizer():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    print("Starting model fine-tuning...")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        bf16=True,  # or use fp16=True if no bf16 support
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    return trainer

def evaluate_model(trainer):
    print("Evaluating fine-tuned model...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    return results

def test_model_generation(model_path, tokenizer_path):
    print("Testing model generation...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    prompts = [
        "<Teacher>: Today we're going to learn about fractions. <Student>: I find fractions really difficult.",
        "<Teacher>: Let's discuss the water cycle. <Student>: I'm confused about evaporation.",
        "<Student>: I don't understand how to solve this equation: 2x + 5 = 15."
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = generator(prompt)[0]['generated_text']
        print(f"Response: {response[len(prompt):]}")

def main():
    # Authenticate Hugging Face (you can use your own token here)
    login(token="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")

    data_dir = "..\Education Dialogue Dataset"
    output_dir = "models/llm/llama-3.2-1b-edu-tuned"

    train_dataset, eval_dataset = load_dialogue_dataset(data_dir)
    model, tokenizer = prepare_model_and_tokenizer()

    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, output_dir)
    evaluate_model(trainer)
    test_model_generation(output_dir, output_dir)

    print("LLM fine-tuning completed successfully.")

if __name__ == "__main__":
    main()
