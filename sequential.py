from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from fine_tuning_utils import split_dataset, preprocess

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,              # attiva quantizzazione 8-bit
    llm_int8_threshold=6.0,         # soglia per quantizzazione dinamica (default 6.0)
    llm_int8_has_fp16_weight=False  # se True, mantiene una copia FP16 per stabilità
)

BASE_MODEL_NAME = "Salesforce/codet5p-220m"
TSC_LORA_PATH = "/content/drive/MyDrive/Colab Notebooks/TSC/lora-codet5p"

BASE_MODEL_8_BIT = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,    # usa quantizzazione per risparmiare VRAM
        device_map="auto"
    )
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

tsc_model = PeftModel.from_pretrained(BASE_MODEL_8_BIT, TSC_LORA_PATH)
tsc_model = tsc_model.merge_and_unload()

TTS_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[ # tutti dal decoder
        "q", "v",                   # query value da attention layer
        "EncDecAttention.q","EncDecAttention.v", # query value da cross-attention layer
        "wi_0", "wi_1", "wo" # Feed Forward Neural layer
    ],
    lora_dropout=0.25, # più alto di training per TSC per evitare overfitting
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

tts_tsc_model = get_peft_model(tsc_model, TTS_LORA_CONFIG)
tts_tsc_model.print_trainable_parameters()



df = pd.read_json('/content/drive/MyDrive/Colab Notebooks/datasets/lcquad2_train.json')
df['input'] = 'text-to-sparql: ' + df['input'].astype(str)

train_df, valid_df = split_dataset(df).values()

# converto ciascun DataFrame in un Dataset Hugging Face
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# creo un DatasetDict con due split
dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

tokenized_dataset = dataset.map(preprocess, batched=True, fn_kwargs={'tokenizer':TOKENIZER})

# Salvo il dataset tokenizzato
tokenized_dataset.save_to_disk("/content/drive/MyDrive/Colab Notebooks/datasets/tokenized_dataset")

data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=tts_tsc_model)

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/traduzione_ffn_cross/results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay = 0.01, # evitare overfitting
    num_train_epochs=2,
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    warmup_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,  # carica il miglior modello alla fine
    metric_for_best_model="eval_loss",
    report_to="none"
)

trainer = Trainer(
    model=tts_tsc_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=TOKENIZER,
    data_collator=data_collator,
)

trainer.train()
tts_tsc_model.save_pretrained("/content/drive/MyDrive/Colab Notebooks/traduzione_ffn_cross")
TOKENIZER.save_pretrained("/content/drive/MyDrive/Colab Notebooks/traduzione_ffn_cross")

# Salvataggio modello in f16 con ultimi pesi lora
tsc_lora = "/content/drive/MyDrive/Colab Notebooks/TSC/lora-codet5p"
tts_lora = "/content/drive/MyDrive/Colab Notebooks/traduzione_ffn_cross"

# 1. Carica modello base in FP16 (IMPORTANTE: nessuna quantizzazione qui)
BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL_NAME,
    dtype=torch.float16,  # FP16
    device_map="auto"
)

# 2. Carica pesi LoRA addestramento su TSC
model = PeftModel.from_pretrained(BASE_MODEL, tsc_lora)
model = model.merge_and_unload()

# 3. Carica pesi LoRA dell’ultimo checkpoint
model = PeftModel.from_pretrained(model, tts_lora)
merged_model = model.merge_and_unload()

MERGED_MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/traduzione_ffn_cross_fp16"

merged_model.save_pretrained(MERGED_MODEL_PATH)
TOKENIZER.save_pretrained(MERGED_MODEL_PATH)
