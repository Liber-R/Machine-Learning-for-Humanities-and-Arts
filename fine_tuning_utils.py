import torch
import re
from typing import Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def split_dataset(df:pd.DataFrame) -> Dict[str,pd.DataFrame]:
    '''
    Funzione che prende un dataframe da usare per un training e lo suddivide in due
    subset tenendo conto della possibilità della presenza
    di più input per il medesimo output prevenendo così il dataleak.

    La suddivisione è fissata a 90% e 10%.
    '''

    groups = df["output"]

    # --- Primo split: chunk_1 (90%) vs chunk_2 (10%) ---
    splitter1 = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
    chunk_1_idx, chunk_2_idx = next(splitter1.split(df, groups=groups))

    chunk_1 = df.iloc[chunk_1_idx]
    chunk_2 = df.iloc[chunk_2_idx]

    print("chunk_90:\n", len(chunk_1), "\n")
    print("chunk_10:\n", len(chunk_2), "\n")

    dataset = {'chunk_90': chunk_1, 'chunk_10': chunk_2}

    return dataset

def preprocess(batch, tokenizer):
    model_inputs = tokenizer(batch["input"],
                             max_length=256, truncation=True, padding=False)
    labels = tokenizer(batch["output"],
                       max_length=256, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_predictions_from_file(filepath):
        predictions = []
        references = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("REFERENCE:"):
                    references.append(line.replace("REFERENCE:", "").strip())
                elif line.startswith("PREDICTION:"):
                    predictions.append(line.replace("PREDICTION:", "").strip())

        assert len(predictions) == len(references), \
            "Numero di predizioni e riferimenti non coincide"

        return predictions, references

def execute_query_endpoint(
    query,
    endpoint_url,
    max_retries=3,
    sleep_base=0.5
):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # User-Agent (OBBLIGATORIO per Wikidata)
    sparql.addCustomHttpHeader(
        "User-Agent",
        "NL2SPARQL-Research/1.0 (academic use)"
    )

    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])

            result_set = set()
            for b in bindings:
                row = tuple(b[var]["value"] for var in sorted(b))
                result_set.add(row)

            return result_set

        except Exception as e:
            wait = sleep_base * (2 ** attempt) + random.uniform(0, 0.3)
            print(f"Retry {attempt+1}/{max_retries} after error: {e}")
            time.sleep(wait)

    return None

def compute_execution_accuracy(
    predictions,
    references,
    endpoint_url,
    max_workers=2,
    sleep_between_pairs=0.3
):
    assert len(predictions) == len(references)

    correct = 0
    total = len(predictions)

    def check_single(pred, ref):
        pred_res = execute_query_endpoint(pred, endpoint_url)
        ref_res = execute_query_endpoint(ref, endpoint_url)

        # Sleep tra coppie per non saturare
        time.sleep(sleep_between_pairs)

        if pred_res is None or ref_res is None:
            return False

        return pred_res == ref_res

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(check_single, p, r)
            for p, r in zip(predictions, references)
        ]

        for future in as_completed(futures):
            if future.result():
                correct += 1

    return correct / total * 100


class CodeT5Evaluator:
    def __init__(self, model_path):
        """
        Inizializza l'evaluator con il modello fine-tuned con LoRA

        Args:
            model_path: path alla cartella del modello e tokenizer
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Carica tokenizer
        print(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Carica modello
        print(f"Loading model from {model_path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        #self.model.to(self.device)
        self.model.eval()

        self.TRIPLE_BLOCK_PATTERN = re.compile(
            r'((?:[^\s\{\[\(]+:\S+|\?\S+)\s+'
            r'(?:[^\s\{\[\(]+:[^\s\}\]\)]+|a|\?\S+)\s+'
            r'(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"[^\"]+\"(?:@\w{2})*)'
            r'(?:\s*;\s*'
            r'(?:[^\s\{\[\(]+:[^\s\}\]\)]+|a|\?\S+)\s+'
            r'(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"[^\"]+\"(?:@\w{2})*)'
            r')*)',
            re.IGNORECASE
        )

    def generate_predictions(self, test_dataset, batch_size=8, max_length=256):
        """
        Genera predizioni per il dataset di test

        Args:
            test_dataset: dataset tokenizzato con colonne 'input_ids' e 'labels'
            batch_size: dimensione del batch
            max_length: lunghezza massima della generazione

        Returns:
            predictions: lista di stringhe predette
            references: lista di stringhe di riferimento
        """
        predictions = []
        references = []

        # Processa in batch
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Generating"):
            batch = test_dataset[i:i+batch_size]

            # Padding dinamico del batch
            max_input_len = max(len(ids) for ids in batch['input_ids'])
            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(batch['input_ids'], batch['attention_mask']):
                padding_length = max_input_len - len(ids)
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                padded_attention_mask.append(mask + [0] * padding_length)

            # Prepara input
            input_ids = torch.tensor(padded_input_ids).to(self.device)
            attention_mask = torch.tensor(padded_attention_mask).to(self.device)

            # Genera predizioni
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=5,  # Beam search per migliori risultati
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decodifica predizioni
            batch_preds = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            predictions.extend(batch_preds)

            # Decodifica references con padding dinamico
            max_label_len = max(len(label) for label in batch['labels'])
            padded_labels = []

            for label in batch['labels']:
                # Sostituisci -100 con pad_token_id
                cleaned_label = [l if l != -100 else self.tokenizer.pad_token_id for l in label]
                padding_length = max_label_len - len(cleaned_label)
                padded_labels.append(cleaned_label + [self.tokenizer.pad_token_id] * padding_length)

            batch_refs = self.tokenizer.batch_decode(
                padded_labels, skip_special_tokens=True
            )
            references.extend(batch_refs)

        return predictions, references

    def sparql_tokenize_normalized(self, query: str):
        tokens = re.findall(
            r"""
            \?var\d+ |        # variabili normalizzate
            ENTITY |          # entità astratte
            PROP |            # proprietà astratte
            SELECT|WHERE|FILTER|OPTIONAL|COUNT|DISTINCT|LIMIT|ORDER|BY|
            [\{\}\(\)\.;,] |  # simboli SPARQL
            [=!<>]+ |         # operatori
            \w+               # fallback
            """,
            query,
            re.VERBOSE | re.IGNORECASE
        )
        return tokens

    def compute_bleu(self, predictions, references):
        refs_tokenized = [
            [self.sparql_tokenize_normalized(ref)]
            for ref in references
        ]
        preds_tokenized = [
            self.sparql_tokenize_normalized(pred)
            for pred in predictions
        ]

        smoothing = SmoothingFunction().method4

        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4 - n))
            bleu = corpus_bleu(
                refs_tokenized,
                preds_tokenized,
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores[f'BLEU-{n}'] = bleu * 100

        return bleu_scores

    def remove_values_block(query):
      return re.sub(r'VALUES\s+\?\S+\s+\{.*?\}', '', query, flags=re.DOTALL | re.IGNORECASE)

    def split_triple_block(self, block):
        """
        Converte:
          ?s p1 o1 ; p2 o2 ; p3 o3
        in:
          (?s, p1, o1), (?s, p2, o2), (?s, p3, o3)
        """
        parts = [p.strip() for p in block.split(';')]
        subject, first_pred, first_obj = parts[0].split(maxsplit=2)

        triples = [(subject, first_pred, first_obj)]
        for p in parts[1:]:
            pred, obj = p.split(maxsplit=1)
            triples.append((subject, pred, obj))

        return triples


    def normalize_term(self, term):
        if term.startswith('?'):
            return '?var'
        if term.startswith('"'):
            return 'LITERAL'
        if ':' in term:
            return 'ENTITY_OR_PROP'
        return term


    def extract_triples(self, query):
        query = self.remove_values_block(query)
        blocks = self.TRIPLE_BLOCK_PATTERN.findall(query)

        triples = set()
        for block in blocks:
            for s, p, o in self.split_triple_block(block):
                triples.add((
                    self.normalize_term(s),
                    self.normalize_term(p),
                    self.normalize_term(o)
                ))
        return triples

    def compute_triple_f1(self, predictions, references):
        tp = fp = fn = 0

        for pred, ref in zip(predictions, references):
            pred_triples = self.extract_triples(pred)
            ref_triples = self.extract_triples(ref)

            tp += len(pred_triples & ref_triples)
            fp += len(pred_triples - ref_triples)
            fn += len(ref_triples - pred_triples)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1": f1 * 100
        }

    def structural_exact_match(self, pred, ref):
        return self.extract_triples(pred) == self.extract_triples(ref)

    def compute_exact_match(self, predictions, references):
        matches = sum(
            self.structural_exact_match(p, r)
            for p, r in zip(predictions, references)
        )
        return matches / len(predictions) * 100

    def evaluate(self, test_dataset, batch_size=8, max_length=256, save_predictions=True):
        """
        Esegue valutazione completa

        Returns:
            dict con tutte le metriche
        """
        print("\n" + "="*50)
        print("Starting Evaluation")
        print("="*50)

        # Genera predizioni
        predictions, references = self.generate_predictions(
            test_dataset, batch_size, max_length
        )

        # Salva predizioni (opzionale)
        if save_predictions:
            with open('/content/drive/MyDrive/Colab Notebooks/predictions.txt', 'w', encoding='utf-8') as f:
                for pred, ref in zip(predictions, references):
                    f.write(f"REFERENCE: {ref}\n")
                    f.write(f"PREDICTION: {pred}\n")
                    f.write("-" * 80 + "\n")
            print("\nPredictions saved to '/content/drive/MyDrive/Colab Notebooks/predictions.txt'")

        # Calcola metriche
        print("\nComputing metrics...")
        bleu_scores = self.compute_bleu(predictions, references)
        f1_scores = self.compute_triple_f1(predictions, references)
        exact_match = self.compute_exact_match(predictions, references)

        # Combina risultati
        results = {
            **bleu_scores,
            'F1': f1_scores['F1'],
            'Exact-Match': exact_match,
            'num_samples': len(predictions)
        }

        # Stampa risultati
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, score in results.items():
            if metric != 'num_samples':
                print(f"{metric:15s}: {score:6.2f}%")
            else:
                print(f"{metric:15s}: {score}")
        print("="*50)

        return results, predictions, references