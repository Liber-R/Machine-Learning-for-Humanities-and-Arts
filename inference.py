import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


base_model_path = "Salesforce/codet5p-220m"
model_path = "./codet5_sparql"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Carica tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Carica modello base
print(f"Loading base model from {base_model_path}")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)

# Carica adapter LoRA
print(f"Loading LoRA adapters from {model_path}")
model = PeftModel.from_pretrained(base_model, model_path)

# Merge degli adapter per inference più veloce (opzionale)
model = model.merge_and_unload()

model.to(device)
model.eval()

inputs = tokenizer("text-to-sparql: Who is the  {country} for {head of state} of {Mahmoud Abbas} [SEP] wd:Mahmoud Abbas wd:country [SEP] wdt:instance of wdt:head of state", return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,
    early_stopping=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
