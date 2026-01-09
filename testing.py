import torch
from fine_tuning_utils import compute_execution_accuracy, load_predictions_from_file
import pandas as pd

from datasets import load_from_disk, Dataset
import nltk
from fine_tuning_utils import CodeT5Evaluator, preprocess, compute_execution_accuracy, load_predictions_from_file
import pandas as pd
from sequential import MERGED_MODEL_PATH, TOKENIZER

TOKENIZED_TEST_DATASET_FOLDER = "lcquad2"
TEST_DATASET = "lcquad2_test.json"

df = pd.read_json(f'/content/drive/MyDrive/Colab Notebooks/datasets/{TEST_DATASET}')
df['input'] = 'text-to-sparql: ' + df['input'].astype(str)

test_dataset = Dataset.from_pandas(df)

tokenized_test_dataset = test_dataset.map(preprocess, batched=True, fn_kwargs={'tokenizer': TOKENIZER})
# Salvo il dataset tokenizzato
tokenized_test_dataset.save_to_disk(f"/content/drive/MyDrive/Colab Notebooks/datasets/tokenized_test_dataset/{TOKENIZED_TEST_DATASET_FOLDER}")

# Scarica risorse NLTK necessarie
nltk.download('punkt', quiet=True)

def main():

    # 1. Path modello da testare
    MODEL = MERGED_MODEL_PATH

    evaluator = CodeT5Evaluator(model_path=MODEL)

    # 2. Carica il dataset di test
    # Opzione A: Dataset già tokenizzato salvato localmente
    test_dataset = load_from_disk(f"/content/drive/MyDrive/Colab Notebooks/datasets/tokenized_test_dataset/{TOKENIZED_TEST_DATASET_FOLDER}")

    # 3. Esegui la valutazione
    results, predictions, references = evaluator.evaluate(
        test_dataset=test_dataset,
        batch_size=8,  # Riduci se hai problemi di memoria
        max_length=256,
        save_predictions=True
    )

    # 4. (Opzionale) Analizza alcuni esempi
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i in range(min(3, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Reference:  {references[i][:100]}...")
        print(f"Prediction: {predictions[i][:100]}...")

    return results

if __name__ == "__main__":
    # Assicurati di avere abbastanza memoria
    torch.cuda.empty_cache()

    # Esegui valutazione
    results = main()
