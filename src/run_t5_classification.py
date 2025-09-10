import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer,  get_linear_schedule_with_warmup,BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,prepare_model_for_kbit_training
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW

 
# Configuration
class Config:
    MODEL_NAME = "google/flan-t5-xl"
    DATASET_NAME = "dair-ai/emotion" # Replace with your dataset
    MAX_LENGTH = 100
    BATCH_SIZE = 2
    NUM_EPOCHS = 20
    LEARNING_RATE = 4e-4
    LOAD_FROM_CHECKPOINT = False
    CHECKPOINT_DIR = "/home/sujeet-pg/wav2vec2/peft/checkpoints_qlora_r8_t5_class"
    USE_PEFT = True
    USE_QLORA = False
    PEFT_CONFIG = {
        "peft_type": "LORA",
        "task_type": "SEQ_CLS",
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q", "v"]
    }
    # Emotion label mapping
    ID2LABEL = {
        0: "sadness", 
        1: "joy", 
        2: "love", 
        3: "anger", 
        4: "fear", 
        5: "surprise"
    }
    LABEL2ID = {v: k for k, v in ID2LABEL.items()}

config = Config()

# Load tokenizer and model


from peft import LoHaConfig, get_peft_model
from peft import LoKrConfig, get_peft_model
from peft import AdaLoraConfig, get_peft_model
from peft import PromptEncoderConfig, get_peft_model
from peft import IA3Model, IA3Config


# Prepare PEFT
if config.USE_PEFT:
    # Example: LoRA
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     target_modules=["q", "v"]
    # )

    # Example: LoHa
    # peft_config = LoHaConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=8,
    #     alpha=16,
    #     module_dropout=0.1,
    #     target_modules=["q", "v"]
    # )

    # Example: LoKr
    # peft_config = LoKrConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=8,
    #     alpha=16,
    #     module_dropout=0.1,
    #     target_modules=["q", "v"]
    #     # modules_to_save=["classifier"],
    # )

    # Example: AdaLoRA
    # peft_config = AdaLoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=8,
    #     init_r=12,
    #     tinit=50,
    #     tfinal=300,
    #     deltaT=10,
    #     target_modules=["q", "v"],
    #     total_step=2000
    # )

    # Example: IA3
    peft_config = IA3Config(
        peft_type="IA3",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q", "v"],
        # feedforward_modules=["w0"],
    )

    # Example: P-Tuning / Prompt Encoder
    # peft_config = PromptEncoderConfig(
    #     peft_type="P_TUNING",
    #     task_type=TaskType.SEQ_CLS,
    #     num_virtual_tokens=1000,
    #     encoder_hidden_size=628
    # )
    
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

if config.USE_QLORA:

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,            # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",    # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=True, # Using double quantization as mentioned in QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16, # During computation, pre-trained model should be loaded in BF16 format
    )

    
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME,quantization_config=bnb_config)

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    model.config.quantization_config = {}


# Load dataset
dataset = load_dataset(config.DATASET_NAME)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Preprocess function for emotion classification
def preprocess_function(examples):
    # Convert numeric labels to text labels using the mapping
    labels = [config.ID2LABEL[label] for label in examples["label"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["text"],
        max_length=config.MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize labels (emotion words)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            labels,
            max_length=10,  # Enough for the longest emotion word
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # Ensures dynamic padding
    return_tensors="pt"
)

# Create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    tokenized_train,
    shuffle=True,
    batch_size=config.BATCH_SIZE,
    collate_fn=data_collator
)

eval_dataloader = torch.utils.data.DataLoader(
    tokenized_eval,
    batch_size=config.BATCH_SIZE,
    collate_fn=data_collator
)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
total_steps = len(train_dataloader) * config.NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

import os


def save_checkpoint(epoch, model, tokenizer, optimizer, loss, config):
    """Save model, tokenizer, and training state to a directory"""
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, os.path.join(checkpoint_dir, "training_state.pt"))
    
    print(f"Checkpoint saved to {checkpoint_dir}")



# Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        eval_loss, eval_accuracy = evaluate_model()
        print(f"Epoch {epoch + 1} - Evaluation loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
        
    
        save_checkpoint(epoch + 1, model, tokenizer, optimizer, avg_train_loss, config)

# Evaluation function
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Get predictions
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=10  # Enough for the longest emotion word
            )
            
            # Decode predictions and compare with labels
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            
            # Calculate accuracy
            for pred, label in zip(preds, labels):
                if pred.lower() == label.lower():
                    correct += 1
                total += 1
    
    avg_loss = total_loss / len(eval_dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


train()
