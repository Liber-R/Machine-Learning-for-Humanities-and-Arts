import os
import re
import random
from typing import List, Tuple, Dict
import requests
import time
import json

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GroupShuffleSplit

# Dataset paths
# LC-QuAD v.2 (from Wikidata)
LC_QUAD2_TRAIN_PATH = "datasets/LC-QuAD2/train.json"
LC_QUAD2_TEST_PATH = "datasets/LC-QuAD2/test.json"

# Qald-9+
# Wikidata
QALD9P_TRAIN_WIKI_PATH = "datasets/QALD-9p/Wikidata/train_wikidata.json"
QALD9P_TEST_WIKI_PATH = "datasets/QALD-9p/Wikidata/test_wikidata.json"
# DBpedia
QALD9P_TRAIN_DB_PATH = "datasets/QALD-9p/DBpedia/train_dbpedia.json"
QALD9P_TEST_DB_PATH = "datasets/QALD-9p/DBpedia/test_dbpedia.json"

# Qald-10 (from Wikidata)
QALD10_PATH = "datasets/QALD-10/qald_10.json"

# JSON file con mappatura ids -> labels Wikidata
IDS_LABELS_MAP_PATH = 'wikidata_ids_lables_map.json'

# Cartelle training dataset e testing dataset
TRAINING_FOLDER = 'datasets/training'
TESTING_FOLDER = 'datasets/testing'

# Cartelle TSC datasets
LC_QUAD2_TSC_FOLDER = 'datasets/TSC/LC-QuAD2'
QALD9P_TSC_WIKI_FOLDER = 'datasets/TSC/QALD-9p/Wikidata'
QALD9P_TSC_DB_FOLDER = 'datasets/TSC/QALD-9p/DBpedia'
QALD10_TSC_FOLDER = 'datasets/TSC/QALD-10'


def normalize_vars(query:str) -> str:
    '''
    Funzione per normalizzare le variabili
    presenti in una query sparql.

    Tutte le variabili verranno sostituite
    da un identificativo seriale tipo
    ?var0.

    Input: query sparql
    Output: query sparql normalizzata
    '''

    # Regex patter per individuare una variabile
    pattern = (r"\?[^\s\][)(}{>,.:;!\"]+")

    finds = re.finditer(pattern, query)
    norm_q = query
    # Eliminazione matches duplicati e ordinamento stringhe per lunghezza
    vars = sorted(set(m.group() for m in finds), key=lambda x: len(x), reverse=True)    

    n = 0
    for var in vars:
        norm_q = norm_q.replace(var, f'?var{n}')
        n+=1
    
    return norm_q


def normalize_wikidata_prefix(sparql_q:str) -> str:
    ENTITY_BASE_URI = 'http://www\.wikidata\.org/entity/'
    RELATION_BASE_URI = 'http://www\.wikidata\.org/prop/direct/'

    # Pattern entità e relazioni scritte con uri
    entity_id = rf'<{ENTITY_BASE_URI}(Q[\d]+)>'
    relation_id = rf'<{RELATION_BASE_URI}(P[\d]+)>'

    # Riscrittura entità e relazioni nel formato wd/wdt:Q/Pxxxx
    new_query = sparql_q

    new_query = re.sub(entity_id, r'wd:\1', new_query)
    new_query = re.sub(relation_id, r'wdt:\1', new_query)

    # Controllo presenza dichiarazione prefix
    # Prefix patterns
    prefix_wd = rf'PREFIX[\s]+wd:[\s]+<{ENTITY_BASE_URI}>'
    prefix_wdt = rf'PREFIX[\s]+wdt:[\s]+<{RELATION_BASE_URI}>'
    prefixes = rf'(?i)({prefix_wd}|{prefix_wdt})'
    match_prefixes = list(re.findall(prefixes, new_query))
    
    if match_prefixes:
        if len(match_prefixes) == 1:
            if 'wdt' in match_prefixes[0]:
                new_query = 'PREFIX wd: <http://www.wikidata.org/entity/> ' + new_query
            else:
                new_query = 'PREFIX wdt: <http://www.wikidata.org/prop/direct/> ' + new_query
    else:
        new_query = 'PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> ' + new_query

    return new_query


def get_wikidata_ids(df:DataFrame, batch_size:int=50) -> List[List[str]]:
    '''
    Funzione che ricerca in tutto il dataset le entità e
    relazioni wikidata per poter costruire batch di 50 unità 
    ciascuna in modo tale da poter interrogare l'api per ottenere
    le rispettive labels evitando il problema dell'eccesso di richieste.

    La funzione si aspetta un DataFrame con almeno una colonna:
    - output -> ovvero la sparql query, dopo che è stata normalizzata.

    Input: df=DataFrame del dataset
    Output: lista di liste da 50 (valore di default) id ciascuna
    '''

    ids = list()
    
    for _, row in df.iterrows():
        
        # Definizione patterns
        entity_id = r'wd:Q[\d]+'
        relation_id = r'wdt:P[\d]+'

        # Ricerca matchings
        entities = list(set(re.findall(entity_id, row['output'])))
        relations = list(set(re.findall(relation_id, row['output'])))

        ids += entities + relations

    # Eliminazione duplicati
    ids = list(set(ids))

    ids_batched = list()
    even_end = len(ids)-(len(ids) % batch_size)
    i = 0
    for _ in ids[batch_size:even_end+1:batch_size]:
        ids_batched.append(ids[i:i+batch_size])
        i += batch_size
    
    # Aggiunta eventuali valori di eccesso
    if even_end != len(ids):
        ids_batched.append(ids[even_end:])

    return ids_batched


def maps_wiki_ids_to_labals(ids:List[List[str]], lang='en') -> Dict[str,str]:
    '''
    Funzione per mappare gli id di Wikidata ai propri labels

    Input: ids=List lista di ids distribuiti in batches di dimensione tot,
            lang=str codice iso-2 per esprimere in quale lingua ottenere i labels
                di default è en:inglese
    Output: dizionario {id:label}.
    '''

    WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": None,
        "languages": lang,
        "format": "json"
    }
    headers = {"User-Agent": "MyWikidataApp/1.0 (mailto:romolo.66@hotmail.it)"} # Aggiunto per non insospettire l'API

    # Mapping Wikidata ids:labels
    wikidata_ids_labels = dict()

    for batch in ids:
        entities_ids = '|'.join(identifier[identifier.find(':')+1:] for identifier in batch)
        params['ids'] = entities_ids # Aggiornamento degli id di cui chiedere il label
        try:
            response = requests.get(WIKIDATA_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Aggiornamento dizionario ids->labels
            for eid, entity_data in data["entities"].items():
                label = entity_data.get("labels", {}).get("en", {}).get("value")
                wikidata_ids_labels[eid] = label
            
            time.sleep(1) # Aggiunto per non intasare l'API di richieste
        except requests.RequestException as e:
            print(f"Errore API per {entities_ids}: {e}")
            return entities_ids
    
    return wikidata_ids_labels
        
def _load_data():
    """Carica il file JSON se esiste, altrimenti restituisce un dict vuoto."""
    if os.path.exists(IDS_LABELS_MAP_PATH):
        with open(IDS_LABELS_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_ids_label_map_wiki_dataset(datasets:List[Tuple[str,DataFrame]]) -> None:
    '''
    Funzione per scrivere un json file contente una mappatura ids->labels per dataset.

    Struttura base: {name_dataset:
                                    {'name_dataset':name_dataset,
                                     'ids_labels_map':
                                        {'Q1':'Universe',
                                         'Q2':'Earth',
                                         ...
                                        }
                                    },
                    ...
                    }

    Input: datasets=lista di tuple contenenti nome e dataframe del dataset da mappare
    Output: None
    '''

    def _save_data(data):
        """Salva il dizionario Python su file JSON."""
        with open(IDS_LABELS_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _add_dataset(name_dataset, ids_labels_map):
        data = _load_data()

        if name_dataset in data:
            print(f"⚠️ Il dataset '{name_dataset}' esiste già.")
            # Aggiornamento dataset
            data[name_dataset]["ids_labels_map"].update(ids_labels_map)
            print(f"✅ Dataset '{name_dataset}' aggiornato.")
        else:
            data[name_dataset] = {
                "name_dataset": name_dataset,
                "ids_labels_map": ids_labels_map
            }
            print(f"✅ Dataset '{name_dataset}' aggiunto.")

        _save_data(data)

    for name, dataset in datasets:
        print(f'Raccolta ids in {name}...')
        ids_dataset = get_wikidata_ids(dataset)
        print(f'Mappatura ids a labels...')
        ids_labels_map = maps_wiki_ids_to_labals(ids_dataset)
        print(f'Salvataggio mappatura in {IDS_LABELS_MAP_PATH}..')
        _add_dataset(name, ids_labels_map=ids_labels_map)


def label_wikidata_ids(text:str, name_dataset:str) -> str:
    '''
    Funzione per operare il Semantic Forwarding
    Per info vedi TSET (https://doi.org/10.3390/app14041521)

    La mia implementazione prevede l'uso di un json file dove reperire i labels,
    frutto di una precedente mappatura, per evitare di scaricare il dump di wikidata
    di oltre 130 GB.

    Input: text=stringa di testo per domanda o sparql query
            name_dataset=stringa del nome del dataset di cui prendere i label
    Output stringa del testo modificata con id sostituiti dalle proprie labels
    '''

    def _get_dataset(name_dataset):
        data = _load_data()
        return data.get(name_dataset, None)
    
    dataset_ids_labels_map = _get_dataset(name_dataset=name_dataset)
    
    if dataset_ids_labels_map == None:
        return print('Impossibile eseguire labeling degli id perché il dataset '\
                     'di riferimento non è nel file json della mappatura.')

    # Patterns
    entity_id = r'wd:(Q\d+)'
    relation_id = r'wdt:(P\d+)'

    # Applicazione labels a entities e relations    
    new_text = text
    ids_labels_map = dataset_ids_labels_map['ids_labels_map']
    new_text = re.sub(entity_id, lambda e: f'wd:{ids_labels_map[e.group(1)]}', new_text)
    new_text = re.sub(relation_id, lambda r: f'wdt:{ids_labels_map[r.group(1)]}', new_text)

    return new_text


def normalize_dbpedia_prefix(sparql_q:str) -> str:
    '''
    Funzione per normalizzare i prefissi di DBpedia
    '''

    # DBpedia uris
    PROPERTY_URI = "<http://dbpedia.org/property/>"
    ONTOLOGY_URI = "<http://dbpedia.org/ontology/>"
    RESOURCE_URI = "<http://dbpedia.org/resource/>"
    YAGO_URI = "<http://dbpedia.org/class/yago/>"

    # Patterns
    property_1 = r'<http://dbpedia\.org/property/([^\s]+)>'
    ontology_1 = r'<http://dbpedia\.org/ontology/([^\s]+)>'
    resource_1 = r'<http://dbpedia\.org/resource/([^\s]+)>'
    yago_1 = r'<http://dbpedia\.org/class/yago/([^\s]+)>'

    property_prefix = r'PREFIX\s+(?!dbp:)(\w+):\s+<http://dbpedia\.org/property/>'
    ontology_prefix = r'PREFIX\s+(?!dbo:)(\w+):\s+<http://dbpedia\.org/ontology/>'
    resource_prefix = r'PREFIX\s+(?!dbr:)(\w+):\s+<http://dbpedia\.org/resource/>'
    yago_prefix = r'PREFIX\s+(?!yago:)(\w+):\s+<http://dbpedia\.org/class/yago/>'

    new_q = sparql_q

    # Checking if prefix was not used, then adding them if it wasn't
    # Check property
    match_prop_1 = re.finditer(property_1, new_q)
    list_match_prop_1 = list(match_prop_1)
    if list_match_prop_1:
        new_q = f'PREFIX dbp: {PROPERTY_URI} ' + new_q
        for m in list_match_prop_1:
            new_q = re.sub(property_1, f'dbp:{m.group(1)}', new_q)
    
    # Check ontology
    match_onto_1 = re.finditer(ontology_1, new_q)
    list_match_onto_1 = list(match_onto_1)
    if list_match_onto_1:
        new_q = f'PREFIX dbo: {ONTOLOGY_URI} ' + new_q
        for m in list_match_onto_1:
            new_q = re.sub(ontology_1, f'dbo:{m.group(1)}', new_q)

    # Check resource
    match_res_1 = re.finditer(resource_1, new_q)
    list_match_res_1 = list(match_res_1)
    if list_match_res_1:
        new_q = f'PREFIX dbr: {RESOURCE_URI} ' + new_q
        for m in list_match_res_1:
            new_q = re.sub(resource_1, f'dbr:{m.group(1)}', new_q)

    # Check yago
    match_yago_1 = re.finditer(yago_1, new_q)
    list_match_yago_1 = list(match_yago_1)
    if list_match_yago_1:
        new_q = f'PREFIX yago: {YAGO_URI} ' + new_q
        for m in list_match_yago_1:
            new_q = re.sub(yago_1, f'yago:{m.group(1)}', new_q)

    # Normalizing prefix declaration and usage
    # Norm property prefix
    match_prop_pref = re.search(property_prefix, new_q)
    if match_prop_pref:
        used_prefix = rf'{match_prop_pref.group(1)}:([^\s]+)'
        match_use = re.search(used_prefix, new_q)
        while match_use:
            new_q = re.sub(used_prefix, f'dbp:{match_use.group(1)}', new_q, count=1)
            match_use = re.search(used_prefix, new_q)
        new_q = re.sub(property_prefix, f'PREFIX dbp: {PROPERTY_URI}', new_q)

    # Norm ontology prefix
    match_onto_pref = re.search(ontology_prefix, new_q)
    if match_onto_pref:
        used_prefix = rf'{match_onto_pref.group(1)}:([^\s]+)'
        match_use = re.search(used_prefix, new_q)
        while match_use:
            new_q = re.sub(used_prefix, f'dbo:{match_use.group(1)}', new_q, count=1)
            match_use = re.search(used_prefix, new_q)
        new_q = re.sub(ontology_prefix, f'PREFIX dbo: {ONTOLOGY_URI}', new_q)

    # Norm resource prefix
    match_res_pref = re.search(resource_prefix, new_q)
    if match_res_pref:
        used_prefix = rf'{match_res_pref.group(1)}:([^\s]+)'
        match_use = re.search(used_prefix, new_q)
        while match_use:
            new_q = re.sub(used_prefix, f'dbr:{match_use.group(1)}', new_q, count=1)
            match_use = re.search(used_prefix, new_q)
        new_q = re.sub(resource_prefix, f'PREFIX dbr: {RESOURCE_URI}', new_q)
        
    # Norm yago prefix
    match_yago_pref = re.search(yago_prefix, new_q)
    if match_yago_pref:
        used_prefix = rf'{match_yago_pref.group(1)}:([^\s]+)'
        match_use = re.search(used_prefix, new_q)
        while match_use:
            new_q = re.sub(used_prefix, f'yago:{match_use.group(1)}', new_q, count=1)
            match_use = re.search(used_prefix, new_q)
        new_q = re.sub(yago_prefix, f'PREFIX yago: {YAGO_URI}', new_q)
        
    return new_q


def add_knowledge(nlq:str, sparql_q:str, from_dbpedia:bool=False) -> str:
    '''
    Funzione per estrarre dalla SPARQL query (sparql_q)
    desiderata le entità e relazioni da usare per inserirle
    nella domanda (nlq).
    Il risultato finale è una nuova nlq come segue:
    nlq [SEP] entity [SEP] relation

    Per info vedi TSET (https://doi.org/10.3390/app14041521)
    
    Questa funzione è stata implementata prevedendo le sole
    knowledge base di Wikidata e DBpedia.
    Nel caso di DBpedia essendo assente una chiara distinzione
    d'uso per namespace di entità e namespace di relazioni
    verrà usata un solo [SEP] raccogliendo tutti gli elementi
    DBpedia presenti nella SPARQL query desiderata.
    Nel caso di Wikidata, invece, si separeranno entità e 
    relazioni.

    Input: nlq=stringa domanda linguaggio naturale
           sparql_q=stringa sparql query normalizzata
           dbpedia=boolean per dichiarare se la query è per
                dbpedia kb
           wikidata=boolean per dichiarare se la query è per
                wikidata kb
    Output: nlq arricchita di entità e relazioni
    '''
    
    # Patterns
    DBPEDIA_ELEMENTS_P = r'(dbo|dbp|dbr|yago):[^\s)\]}>,.:;!\"]+'
    WIKIDATA_ELEMENTS_P = r'(wd|wdt):(Q|P)\d+'

    if from_dbpedia:
        db_elements = set(m.group() for m in re.finditer(DBPEDIA_ELEMENTS_P, sparql_q))
        enriched_nlq = nlq + ' [SEP] ' + ' '.join(m for m in db_elements)
        return enriched_nlq
    
    wiki_elements = set( m.group() for m in re.finditer(WIKIDATA_ELEMENTS_P, sparql_q))
    entities = ' [SEP] ' + ' '.join(m for m in wiki_elements if 'wdt:' not in m)
    relations = ' [SEP] ' + ' '.join(m for m in wiki_elements if 'wdt:' in m)
    enriched_nlq = nlq + entities + relations

    return enriched_nlq


def make_lcquad_alpaca(dataset_path:str) -> DataFrame:
    '''
    Azioni su dataset lcquad:
    1) Eliminazione righe prive del valore 'question' e
    'paraphrased_question'
    2) Selezione colonne 'question',
                         'paraphrased_question',
                         'sparql_wikidata',
    X 3) Esplosione dataset, aumento numero esempi:
    row['question', 'paraphrase_question', 'sparql_wikidata']
    diventano due righe, dove 'sparql_wikidata' si ripete
    row['question', 'sparql_wikidata']
    row['paraphrased_question', 'sparql_wikidata']

    4) Modifiche colonne:
        - Aggiunta colonna 'instruction' con espressione:
        "Write a SPARQL query to answer the following question
        using the Wikidata knowledge base."
        - Modifica nomi colonne: 'question' -> 'input'
                                 'sparql_wikidata' -> 'output'

    Aggiornamento: sospensione esplosione dataset per motivi di tempo
    '''

    df_orig = pd.read_json(dataset_path)
    
    # Eliminazione righe con valori 'question' o 'paraphrased_question' assenti
    mask = df_orig["question"].map(lambda x: isinstance(x, str)) #& \
       #df_orig["paraphrased_question"].map(lambda x: isinstance(x, str))

    df_dropNA = df_orig[mask]
    
    # Selezione colonne di interesse
    df_3 = df_dropNA[['question', 'sparql_wikidata']]
    
    # X Esplosione del dataset
    df_questions = df_3[['question', 'sparql_wikidata']]
    #df_para = df_3[['paraphrased_question', 'sparql_wikidata']]

    # Rinomina colonne
    df_questions_renamed = df_questions.rename(columns={'question': 'input', 'sparql_wikidata': 'output'})
    #df_para_renamed = df_para.rename(columns={'paraphrased_question': 'input', 'sparql_wikidata': 'output'})
    
    # Riunione delle due parti
    #df_united = pd.concat([df_questions_renamed, df_para_renamed])
    
    # Aggiunta colonna con l'istruzione
    instruction_text = 'Write a SPARQL query to answer the following question using the Wikidata knowledge base:'
    #df_united['instruction'] = instruction_text
    df_questions_renamed['instruction'] = instruction_text

    return df_questions_renamed #df_united


def make_qald9p_alpaca(dataset_path:str, from_dbpedia:bool=False) -> DataFrame:
    '''
    Funzione per costruire un DataFrame con tre colonne:
    1 - instruction: istruzione da dare al modello per eseguire la traduzione
    2 - input: la domanda in natural language da tradurre in sparql query
    3 - output: sparql query

    QALD-9+ ha sia una parte per DBpedia che una per Wikidata, per questo
    è stato inserito un parametro in base al quale il dataset avrà istruzioni
    dedicate per Wikidata o per DBpedia.

    Input: dataset_path=stringa del percorso file di QALD-9+
           dbpedia=boolean per dichiarare se i dati hanno sparql query per DBpedia
    Output: DataFrame come sopra descritto
    '''

    with open(dataset_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    raw_df = pd.DataFrame(data['questions'])
    rows = []
    for _, row in raw_df.iterrows():
        for q in row['question']:
            if q['language'] == 'en':
                question = q['string']
                break
        sparql = row['query']['sparql']
        rows.append({'input':question, 'output':sparql})
    
    df = pd.DataFrame(rows)
    wiki_instruction = 'Write a SPARQL query to answer the following question using the Wikidata knowledge base:'
    db_instruction = 'Write a SPARQL query to answer the following question using the DBpedia knowledge base:'
    df['instruction'] = db_instruction if from_dbpedia else wiki_instruction

    return df


def make_qald10_alpaca(dataset_path:str) -> DataFrame:
    '''
    Funzione per costruire un DataFrame con tre colonne:
    1 - instruction: istruzione da dare al modello per eseguire la traduzione
    2 - input: la domanda in natural language da tradurre in sparql query
    3 - output: sparql query

    Input: dataset_path=stringa del percorso file di QALD-10
    Output: DataFrame come sopra descritto
    '''

    with open(dataset_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    raw_df = pd.DataFrame(data['questions'])
    rows = []
    for _, row in raw_df.iterrows():
        for q in row['question']:
            if q['language'] == 'en':
                question = q['string']
                break
        sparql = row['query']['sparql']
        rows.append({'input':question, 'output':sparql})
    
    df = pd.DataFrame(rows)
    df['instruction'] = 'Write a SPARQL query to answer the following question using the Wikidata knowledge base:'

    return df


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

    print("chunk_1:\n", len(chunk_1), "\n")
    print("chunk_2:\n", len(chunk_2), "\n")

    dataset = {'chunk_1': chunk_1, 'chunk_2': chunk_2}

    return dataset


def normalize_dataset(df:DataFrame, from_dbpedia:bool=False) -> DataFrame:
    '''
    Funzione per normalizzare parti dei dataset eseguendo:
    - Normalizzazione delle variabili nelle sparql query -> normalize_vars()
    - Normalizzazione prefissi nelle sparql query -> normalize_wikidata_prefix(),
                                                     normalize_dbpedia_prefix()

    I DataFrame attesi hanno 3 colonne: instruction, input e output.
    Dove instruction è l'istruzione da dare a CodeT5+, input è la domanda e 
    output è la sparql query attesa.

    Dato che il processo è lungo, sono stati inseriti dei print per aggiornare
    l'utente sui processi in corso.
    '''
    
    print('Inizio preprocessing...')

    # Normalizzazione variabili
    print('\tNormalizzazione variabili...')
    df['output'] = df['output'].apply(normalize_vars)
    print('\tProcesso conlucso.')
    
    # Caso DBpedia
    if from_dbpedia:
        # Normalizzazione prefissi
        print('\tNormalizzazione prefissi per DBpedia...')
        df['output'] = df['output'].apply(normalize_dbpedia_prefix)
        print('\tProcesso concluso.')
    
    # Caso Wikidata
    else:
        # Normalizzazione prefissi
        print('\tNormalizzazione prefissi per Wikidata...')
        df['output'] = df['output'].apply(normalize_wikidata_prefix)
        print('\tProcesso concluso.')

    print('Preprocessing concluso.')

    return df


def enrich_dataset(df:DataFrame, from_dbpedia:bool=False) -> DataFrame:
    '''
    Funzione per aggiungere gold entities e relations nelle domande chiamando add_knowledge()
    '''

    # Caso DBpedia
    if from_dbpedia:        
        # Aggiunta gold entities e relations
        print('\tArricchimento domanda con gold entities e relations...')
        df['input'] = df.apply(lambda row: add_knowledge(row['input'], row['output'], from_dbpedia=from_dbpedia), axis=1)
        print('\tProcesso concluso.')
    
    # Caso Wikidata
    else:
        # Aggiunta gold entities e relations
        print('\tArricchimento domanda con gold entities e relations...')
        df['input'] = df.apply(lambda row: add_knowledge(row['input'], row['output'], from_dbpedia=from_dbpedia), axis=1)
        print('\tProcesso concluso.')
    
    print('Aggiunta knowledge conclusa.')

    return df


def apply_SF(name_dataset, df:DataFrame):
    '''
    Funzione per applicare Semantic Forwarding per dataset Wikidata chiamando label_wikidata_ids()
    '''
    # Semantic Forwarding (SF)
    print('\tApplicazione Semantic Forwarding...')

    print('\t\tApplicazione SF alle domande...')
    df['input'] = df['input'].apply(lambda x: label_wikidata_ids(x, name_dataset=name_dataset)) # Applicazione SF alle domande
    print('\t\tProcesso concluso per le domande.')

    print('\t\tApplicazione SF alle queries...')
    df['output'] = df['output'].apply(lambda x: label_wikidata_ids(x, name_dataset=name_dataset)) # Applicazione SF alle queries
    print('\t\tProcesso concluso per le queries.')

    return df


def preprocess_dataset(name_dataset:str, df:DataFrame, from_dbpedia:bool=False) -> DataFrame:
    '''
    Funzione per applicare manipolazioni per preparare il dataset all'addestramento di CodeT5+:
    - Normalizzazione di variabili e scrittura entità e relazioni;
    - Aggiunta gold entities e gold relations
    - Se wikidata datasets, allora applicazione del Semantic Forwarding

    Input: name_dataset:stringa da usare per reperire le labels nel file di mappatura ids -> labels
                !!! ATTENZIONE: Il nome da usare deve essere lo stesso presente nel file json !!!
                !!!             se assente vuol dire che bisogna aggiornare il file.          !!!
            df=DataFrame del dataset
            dbpedia=boolean per dichiarare se il dataset è basato su DBpedia o meno
    Output: DataFrame del dataset manipolato
    '''
    norm_df = normalize_dataset(df=df, from_dbpedia=from_dbpedia)
    rich_df = enrich_dataset(norm_df, from_dbpedia=from_dbpedia)
    if not from_dbpedia:
        df = apply_SF(name_dataset=name_dataset, df=rich_df)

    return df


# TEST Qald-10
'''
A_qald_10 = make_qald10_alpaca(QALD10_PATH)
processed_qald_10 = preprocess_dataset(A_qald_10)
# Conteggio domande per cui non sono state trovate gold entities o relations
n = 0
prefix_missing = list()
for _, row in processed_qald_10.iterrows():
    o, gold_area1, gold_area2 = row['input'].split('[SEP]')
    gold_area = (gold_area1 + gold_area2).replace('[SEP]', '').replace(' ','')
    if gold_area == '':
        n += 1
        prefix_missing.append(_)
print(n,'\n',prefix_missing)
'''

# TEST Qald-9+ DBpedia
'''
A_qald_9db = make_qald9p_alpaca(QALD9P_TEST_DB_PATH, True)
processed_qald_9 = preprocess_dataset(A_qald_9db, True)
# Conteggio domande per cui non sono state trovate gold entities o relations
n = 0
prefix_missing = list()
for _, row in processed_qald_9.iterrows():
    o, gold_area = row['input'].split('[SEP]')
    if gold_area.replace(' ','') == '':
        n += 1
        prefix_missing.append(_)
print(n,'\n',prefix_missing)
'''

# TEST Qald-9+ Wikidata
'''
A_qald_9wiki = make_qald9p_alpaca(QALD9P_TEST_WIKI_PATH)
processed_qald_9 = preprocess_dataset(A_qald_9wiki)
# Conteggio domande per cui non sono state trovate gold entities o relations
n = 0
prefix_missing = list()
for _, row in processed_qald_9.iterrows():
    o, gold_area1, gold_area2 = row['input'].split('[SEP]')
    gold_area = (gold_area1 + gold_area2).replace('[SEP]', '').replace(' ','')
    if gold_area == '':
        n += 1
        prefix_missing.append(_)
        print(row['input'])
        print(row['output'])
print(n,'\n',prefix_missing)
'''

# TEST LC-QuAD v.2
'''
A_lcquad2 = make_lcquad_alpaca(LC_QUAD2_TEST_PATH)
processed_lcquad2 = preprocess_dataset(A_lcquad2)
# Conteggio domande per cui non sono state trovate gold entities o relations
n = 0
prefix_missing = list()
for _, row in processed_lcquad2.iterrows():
    o, gold_area1, gold_area2 = row['input'].split('[SEP]')
    gold_area = (gold_area1 + gold_area2).replace('[SEP]', '').replace(' ','')
    if gold_area == '':
        n += 1
        prefix_missing.append(_)
print(n,'\n',prefix_missing)
'''

# Mappatura ids->labels dei datasets wikidata
'''
lc_quad2_train = make_lcquad_alpaca(LC_QUAD2_TRAIN_PATH)
norm_lc_quad2_train = normalize_dataset(lc_quad2_train)
lc_quad2_test = make_lcquad_alpaca(LC_QUAD2_TEST_PATH)
norm_lc_quad2_test = normalize_dataset(lc_quad2_test)
#qald9p_wiki_train = make_qald9p_alpaca(QALD9P_TRAIN_WIKI_PATH)
#norm_qald9p_wiki_train = normalize_dataset(qald9p_wiki_train)
#qald9p_wiki_test = make_qald9p_alpaca(QALD9P_TEST_WIKI_PATH)
#norm_qald9p_wiki_test = normalize_dataset(qald9p_wiki_test)
#qald10 = make_qald10_alpaca(QALD10_PATH)
#norm_qald10 = normalize_dataset(qald10)
datasets = [('LC-QuAD2', norm_lc_quad2_test),
            ('LC-QuAD2', norm_lc_quad2_train),
            #('QALD-9p', norm_qald9p_wiki_test),
            #('QALD-9p', norm_qald9p_wiki_train),
            #('QALD-10', norm_qald10)
            ]
write_ids_label_map_wiki_dataset(datasets=datasets)
'''

##################################################
###                                            ###
### MANIPOLAZIONE DATI PER COSTRUZIONE DATASET ###
###                                            ###
### Ogni dataset verrà formattato nello stile  ###
### Alpaca, subirà il preprocessing per        ###
### normalizzazione variabili, normalizzazione ###
### prefissi e labeling id.                    ###
### Infine l'output verrà salvato in un json   ###
### file.                                      ###
##################################################
'''
# Qald-9+
print('Manipolazione dataset Qald-9+...')
# Qald-9+ Wikidata
print('\tManipolazione Wikidata dataset Qald-9+...')
# Train dataset
A_qald9p_wiki_train = make_qald9p_alpaca(QALD9P_TRAIN_WIKI_PATH)
processed_qald9p_wiki_train = preprocess_dataset('QALD-9p', A_qald9p_wiki_train)
processed_qald9p_wiki_train.to_json(f'{TRAINING_FOLDER}/qald9p_wiki_train.json', orient='records', indent=4, force_ascii=False)
print(f'\t\tWikidata train dataset salvato in {TRAINING_FOLDER}/qald9p_wiki_train.json.')
# Test dataset
A_qald9p_wiki_test = make_qald9p_alpaca(QALD9P_TEST_WIKI_PATH)
processed_qald9p_wiki_test = preprocess_dataset('QALD-9p', A_qald9p_wiki_test)
processed_qald9p_wiki_test.to_json(f'{TESTING_FOLDER}/qald9p_wiki_test.json', orient='records', indent=4, force_ascii=False)
print(f'\t\tWikidata test dataset salvato in {TESTING_FOLDER}/qald9p_wiki_test.json.')

# Qald-9+ DBpedia
print('\tManipolazione DBpedia dataset Qald-9+...')
# Train dataset
A_qald9p_db_train = make_qald9p_alpaca(QALD9P_TRAIN_DB_PATH, from_dbpedia=True)
processed_qald9p_db_train = preprocess_dataset('QALD-9p', A_qald9p_db_train, True)
processed_qald9p_db_train.to_json(f'{TRAINING_FOLDER}/qald9p_db_train.json', orient='records', indent=4, force_ascii=False)
print(f'\t\tDBpedia train dataset salvato in {TRAINING_FOLDER}/qald9p_db_train.json.')
# Test dataset
A_qald9p_db_test = make_qald9p_alpaca(QALD9P_TRAIN_DB_PATH, from_dbpedia=True)
processed_qald9p_db_test = preprocess_dataset('QALD-9p', A_qald9p_db_test, True)
processed_qald9p_db_test.to_json(f'{TESTING_FOLDER}/qald9p_db_test.json', orient='records', indent=4, force_ascii=False)
print(f'\t\tDBpedia test dataset salvato in {TESTING_FOLDER}/qald9p_db_test.json.')
print('Manipolazione dataset Qald-9+ conclusa.')

# Qald-10
print('Manipolazione dataset Qald-10...')
A_qald10 = make_qald10_alpaca(QALD10_PATH)
processed_qald10 = preprocess_dataset('QALD-10', A_qald10)
processed_qald10.to_json(f'{TESTING_FOLDER}/qald10.json', orient='records', indent=4, force_ascii=False)
print(f'\tQald-10 dataset salvato in {TESTING_FOLDER}/qald10.json.')
print('Manipolazione dataset Qald-10 conclusa.')

# LC-QuAD v.2
print('Manipolazione dataset LC-QuAD v.2...')
# Train dataset
A_lcquad2_train = make_lcquad_alpaca(LC_QUAD2_TRAIN_PATH)
processed_lcquad2_train = preprocess_dataset('LC-QuAD2', A_lcquad2_train)
processed_lcquad2_train.to_json(f'{TRAINING_FOLDER}/lcquad2_train.json', orient='records', indent=4, force_ascii=False)
print(f'\tLC-QuAD v.2 train dataset salvato in {TRAINING_FOLDER}/lcquad2_train.json.')
# Test dataset
A_lcquad2_test = make_lcquad_alpaca(LC_QUAD2_TEST_PATH)
processed_lcquad2_test = preprocess_dataset('LC-QuAD2', A_lcquad2_test)
processed_lcquad2_test.to_json(f'{TESTING_FOLDER}/lcquad2_test.json', orient='records', indent=4, force_ascii=False)
print(f'\tLC-QuAD v.2 test dataset salvato in {TESTING_FOLDER}/lcquad2_test.json.')
print('Manipolazione dataset LC-QuAD v.2 conclusa.')
'''

def map_subjs_objs(sparql_q) -> Dict[int, Tuple[Tuple[str, str], List[Tuple[str, str]]]]:
    '''
    Funzione per mappare i soggetti ai loro rispettivi oggetti. Vengono preservati i label originali.

    Input: sparql_q=stringa della query
    Output: Dizionario con chiavi interi e valori tuple di una tupla e una lista,
        la prima contente coppia tag,label_soggetto, la seconda contente le tuple
        di coppie tag,label_oggetto
    '''

    def _retrieve_triples(sparql_q) -> List[str]:
        '''
        Funzione per ricavare le triple presenti nella query.

        Segue descrizione regex usata:
        triples_pattern individua triple comprendendo:
                         - variabili normalizzate introdotte da ?;
                         - literals con anche la specificazione della 
                         lingua, e.g. "cane"@it;
                         - entità con prefisso, e.g. wd:Q123;
                         - proprietà rdf:type abbreviata, e.g. a;
                         - proprietà con prefisso, e.g. dbr:nato_in;
                         - triple estese da operatori AND (;), e.g.
                          ?var1 dbr:figlio_di dbo:Dio ; dbr:nato_in dbo:Nazareth
        VALUES_pattern individua clausole VALUES, e.g. VALUES ?var0 { wd:Q123 wd:Q987}                          

        Input: sparql_q=stringa della query
        Output: Lista di stringhe dato che non ci sono gruppi di cattura nel pattern usato
        '''

        VALUES_pattern = r'VALUES\s+\?\S+\s+\{(.*?)\}'
        triples_pattern = r'(?:[^\s\{\[\(]+:\S+|\?\S+)\s+(?:[^\s\{\[\(]+:[^\s\}\]\)]+|a|\?\S+)\s+(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"\S+\"(?:@\w{2})*)(?:\s+;\s+(?:\S+:[^\s\}\]\)]+|a|\?\S+)\s+(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"\S+\"(?:@\w{2})*))*'

        sparql_q_noVALUES = re.sub(VALUES_pattern, '', sparql_q)
        triples_matches = re.findall(triples_pattern, sparql_q_noVALUES, re.IGNORECASE)
        return triples_matches    
        
    def _get_subj_objs(triple:str) -> Tuple[str, List[str]]:
        '''
        Funzione che trova il soggetto e i suoi oggetti nella stringa della tripla
        passata in input.

        Segue descrizione regex usate:
        - subj_pattern individua il soggetto nella tripla, banalmente perché all'inizio,
        ma si specifica comunque che può trattarsi di entità con prefisso o di variabile
        normalizzata e.g. ?var0
        - objs_pattern individua il/gli oggetto/i della tripla, catturando:
                - variabili normalizzate;
                - entità con prefissi;
                - literals con o senza lingua specificata.

        Input: match=stringa della tripla
        Output: Tupla con due oggetti stringa del soggetto e lista delle stringhe degli oggetti
        '''
        subj_pattern = r'^(\?var\d+|\S+:\S+)'
        objs_pattern = r'(?:\S+:\S+|a|\?\S+)\s+(\?var\d+|\S+:\S+|\"\S+\"(?:@\w{2})*)'
        subj = re.match(subj_pattern, triple, re.IGNORECASE)
        subj = subj.group()
        triple = triple[len(subj):] # Per evitare sovrapposizioni nel matching degli oggetti si elimina il soggetto dalla tripla
        objs = re.findall(objs_pattern, triple, re.IGNORECASE)
        return subj, objs       

    triples = _retrieve_triples(sparql_q=sparql_q)
    subjs_objs_map = dict()
    n = 0
    for t in triples:
        subj, objs = _get_subj_objs(t)
        subj_t = f'subj{n}', subj
        i = 0
        objs_l = list()
        for obj in objs:
            objs_l.append((f'obj{n}{i}', obj))
            i += 1
        subjs_objs_map[n] = (subj_t, objs_l)
        n += 1            
    return subjs_objs_map


def tag_subj_objs(sparql_q:str) -> Tuple[str, Dict]:
    '''
    Funzione per applicare tag sostitutivi di soggetti e oggetti
    nelle triple della query.
    
    La funzione si aspetta una query le cui variabili e prefissi
    siano stati normalizzati.

    Input: sparql_q=stringa della query
    Output: Tupla con stringa della query con tag applicati e dizionario con mappa soggetti oggetti
    '''

    subjs_objs_map = map_subjs_objs(sparql_q=sparql_q)
    head, body = sparql_q[:sparql_q.find('{')], sparql_q[sparql_q.find('{'):]
    tagged_body = body
    for tag_subj, objs in subjs_objs_map.values():
        tagged_body = tagged_body.replace(tag_subj[1], tag_subj[0], 1)
        for tag_obj in objs:
            tagged_body = tagged_body.replace(tag_obj[1], tag_obj[0], 1)
    tagged_query = head + tagged_body
    return tagged_query, subjs_objs_map


def random_subj_obj_swap(sparql_q:str, perc:float=0.15) -> str:
    '''
    Funzione per scambiare in una SPARQL query sparql_q soggetti e oggetti 
    secondo una determinata quantità perc (0.15 di default). Per casi in cui la
    percentuale dichiarata risulti in una quantità inferiore a 1, verrà invertita una
    sola tripla.

    I soggetti vengono scelti pseudo-casualmente in numero corrispondente alla percentuale espressa in input.
    Nel caso ci siano molteplici oggetti per un singolo soggetto anche qui verrà applicata una scelta pseudo-casuale.

    Funzione necessaria per addestrare il modello sulla Triple Structure Correction (TSC).
    Per info vedi TSET (https://doi.org/10.3390/app14041521)

    Input: sparql_q=stringa della query
    Output: stringa della query con soggetti oggetti scambiati
    '''
    
    if not 0.00 <= perc <= 1.00:
        return 'Per esprimere la percentuale di triple da invertire dichiara un valore perc tale che 0.00 <= perc <= 1.00'

    tagged_query, subjs_objs_map = tag_subj_objs(sparql_q=sparql_q)
    
    # Da vedere come comprendere il caso in cui viene usato VALUES.
    # Per ora se presente non si opera alcun scambio soggetto/oggetto
    # e si mostra di quale query si tratta.
    if not subjs_objs_map:
        print('query saltata per motivi di sintassi complicata e.g. presenza di VALUES. Segue la query:')
        print(sparql_q)
        return sparql_q
    
    n_triples = len(subjs_objs_map)
    n_triples_to_swap = round(max(1, n_triples * perc)) # max usato per evitare di avere valori inferiori a 1
    to_swap_indeces = []
    for _ in range(n_triples_to_swap):
        random_n = random.randint(0, len(subjs_objs_map)-1) # Scelta pseudo-casuale dei soggetti
        # while per evitare che venga scelto lo stesso soggetto più di una volta
        while random_n in to_swap_indeces:
            random_n = random.randint(0, len(subjs_objs_map)-1)
        to_swap_indeces.append(random_n)
    
    corrupted_query = tagged_query
    for index in to_swap_indeces:
        swap_subj = subjs_objs_map[index][0][0]
        objs = subjs_objs_map[index][1]
        random_obj = random.randint(0, len(objs)-1) # Scelta pseudo-casuale dell'oggetto
        swap_obj = objs[random_obj][0]

        # Scambio soggetto/oggetto
        corrupted_query = corrupted_query.replace(swap_obj, swap_subj)
        corrupted_query = corrupted_query.replace(swap_subj, swap_obj, 1)

    # Ricollocazione dei label originali al posto dei tag e.g. subj1, obj10
    for _, subj_objs in subjs_objs_map.items():
        tag_subj, subj = subj_objs[0]
        corrupted_query = corrupted_query.replace(tag_subj, subj)
        for obj_t in subj_objs[1]:
            tag_obj, obj = obj_t
            corrupted_query = corrupted_query.replace(tag_obj, obj)
    return corrupted_query

# Test random subject object swap in triples
'''
test_1 = "SELECT ?value WHERE { wd:Q736 p:P1279 ?s . ?s ps:P1279 ?x filter(contains(?x,'2.7')) . ?s pq:P585 ?value}"
test_2 = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> SELECT DISTINCT ?uri WHERE { ?uri wdt:P106 wd:Q49757 . ?x wdt:P50 ?uri ; wdt:P31 wd:Q571 . } GROUP BY ?uri ORDER BY DESC(COUNT(?x)) LIMIT 1"
test_3 = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> SELECT DISTINCT ?uri WHERE { ?uri wdt:P31 wd:Q35657 ; wdt:P2046 ?area ; wdt:P1082 ?population . BIND((?population/?area) AS ?density) . } ORDER BY DESC(?density) LIMIT 1"

norm_q = normalize_vars(test_3)
print(norm_q)
print(random_subj_obj_swap(norm_q, 0.3))
'''

### Costruzione TSC datasets ###
# QALD-9+ wikidata
'''
# Creazione dataset stile alpaca
A_qald9p_wiki_train = make_qald9p_alpaca(QALD9P_TRAIN_WIKI_PATH)
A_qald9p_wiki_test = make_qald9p_alpaca(QALD9P_TEST_WIKI_PATH)
# Normalizzazione variabili e prefissi
norm_qald9p_wiki_train = normalize_dataset(A_qald9p_wiki_train)
norm_qald9p_wiki_test = normalize_dataset(A_qald9p_wiki_test)
# Creazione nuovo dataframe con selezione della sola colonna di sparql queries
qald9p_wiki_train_tsc = pd.DataFrame(norm_qald9p_wiki_train['output'])
qald9p_wiki_test_tsc = pd.DataFrame(norm_qald9p_wiki_test['output'])
# Applicazione dello scambio pseudo-casuale di soggetti e oggetti nelle triple
# lasciando la percentuale di default delle triple da modificare (15%).
qald9p_wiki_train_tsc['input'] = qald9p_wiki_train_tsc['output'].apply(random_subj_obj_swap)
qald9p_wiki_test_tsc['input'] = qald9p_wiki_test_tsc['output'].apply(random_subj_obj_swap)
# Salvataggio in json del nuovo dataset per TSC
qald9p_wiki_train_tsc.to_json(QALD9P_TSC_WIKI_FOLDER+'/train.json', orient='records', index=False, force_ascii=False, indent=2)
qald9p_wiki_test_tsc.to_json(QALD9P_TSC_WIKI_FOLDER+'/test.json', orient='records', index=False, force_ascii=False, indent=2)
'''

# QALD-9+ dbpedia
'''
# Creazione dataset stile alpaca
A_qald9p_db_train = make_qald9p_alpaca(QALD9P_TRAIN_DB_PATH, from_dbpedia=True)
A_qald9p_db_test = make_qald9p_alpaca(QALD9P_TEST_DB_PATH, from_dbpedia=True)
# Normalizzazione variabili e prefissi
norm_qald9p_db_train = normalize_dataset(A_qald9p_db_train, from_dbpedia=True)
norm_qald9p_db_test = normalize_dataset(A_qald9p_db_test, from_dbpedia=True)
# Creazione nuovo dataframe con selezione della sola colonna di sparql queries
qald9p_db_train_tsc = pd.DataFrame(norm_qald9p_db_train['output'])
qald9p_db_test_tsc = pd.DataFrame(norm_qald9p_db_test['output'])
# Applicazione dello scambio pseudo-casuale di soggetti e oggetti nelle triple
# lasciando la percentuale di default delle triple da modificare (15%).
qald9p_db_train_tsc['input'] = qald9p_db_train_tsc['output'].apply(random_subj_obj_swap)
qald9p_db_test_tsc['input'] = qald9p_db_test_tsc['output'].apply(random_subj_obj_swap)
# Salvataggio in json del nuovo dataset per TSC
qald9p_db_train_tsc.to_json(QALD9P_TSC_DB_FOLDER+'/train.json', orient='records', index=False, force_ascii=False, indent=2)
qald9p_db_test_tsc.to_json(QALD9P_TSC_DB_FOLDER+'/test.json', orient='records', index=False, force_ascii=False, indent=2)
'''

# QALD-10
'''
# Creazione dataset stile alpaca
A_qald10 = make_qald10_alpaca(QALD10_PATH)
# Normalizzazione variabili e prefissi
norm_qald10 = normalize_dataset(A_qald10)
# Creazione nuovo dataframe con selezione della sola colonna di sparql queries
qald10_tsc = pd.DataFrame(norm_qald10['output'])
# Applicazione dello scambio pseudo-casuale di soggetti e oggetti nelle triple
# lasciando la percentuale di default delle triple da modificare (15%).
qald10_tsc['input'] = qald10_tsc['output'].apply(random_subj_obj_swap)
# Salvataggio in json del nuovo dataset per TSC
qald10_tsc.to_json(QALD10_TSC_FOLDER+'/train.json', orient='records', index=False, force_ascii=False, indent=2)
'''

# LC-QuAD v.2
#'''
# Creazione dataset stile alpaca
A_lcquad2_train = make_lcquad_alpaca(LC_QUAD2_TRAIN_PATH)
A_lcquad2_test = make_lcquad_alpaca(LC_QUAD2_TEST_PATH)
# Normalizzazione variabili e prefissi
norm_lcquad2_train = normalize_dataset(A_lcquad2_train)
norm_lcquad2_test = normalize_dataset(A_lcquad2_test)
# Creazione nuovo dataframe con selezione della sola colonna di sparql queries
lcquad2_train_tsc = pd.DataFrame(norm_lcquad2_train['output'])
lcquad2_test_tsc = pd.DataFrame(norm_lcquad2_test['output'])
# Applicazione dello scambio pseudo-casuale di soggetti e oggetti nelle triple
# lasciando la percentuale di default delle triple da modificare (15%).
lcquad2_train_tsc['input'] = lcquad2_train_tsc['output'].apply(random_subj_obj_swap)
lcquad2_test_tsc['input'] = lcquad2_test_tsc['output'].apply(random_subj_obj_swap)
# Salvataggio in json del nuovo dataset per TSC
lcquad2_train_tsc.to_json(LC_QUAD2_TSC_FOLDER+'/train.json', orient='records', index=False, force_ascii=False, indent=2)
lcquad2_test_tsc.to_json(LC_QUAD2_TSC_FOLDER+'/test.json', orient='records', index=False, force_ascii=False, indent=2)
#'''
# Applicazione SF a dataset TSC
'''
# QALD-9p train
TSC_QALD9P_WIKI_TRAIN_PATH = QALD9P_TSC_WIKI_FOLDER + '/train.json'
TSC_qald9p_wiki_train_df = pd.read_json(TSC_QALD9P_WIKI_TRAIN_PATH)
SF_qald9p_wiki_train = apply_SF('QALD-9p', TSC_qald9p_wiki_train_df)
SF_qald9p_wiki_train.to_json(TSC_QALD9P_WIKI_TRAIN_PATH, orient='records', index=False, force_ascii=False, indent=2)
# QALD-9p test
TSC_QALD9P_WIKI_TEST_PATH = QALD9P_TSC_WIKI_FOLDER + '/test.json'
TSC_qald9p_wiki_test_df = pd.read_json(TSC_QALD9P_WIKI_TEST_PATH)
SF_qald9p_wiki_test = apply_SF('QALD-9p', TSC_qald9p_wiki_test_df)
SF_qald9p_wiki_test.to_json(TSC_QALD9P_WIKI_TEST_PATH, orient='records', index=False, force_ascii=False, indent=2)
'''
'''
# QALD-10
TSC_QALD10_PATH = QALD10_TSC_FOLDER + '/train.json'
TSC_qald10_df = pd.read_json(TSC_QALD10_PATH)
SF_qald10 = apply_SF('QALD-10', TSC_qald10_df)
SF_qald10.to_json(TSC_QALD10_PATH, orient='records', index=False, force_ascii=False, indent=2)
'''
#'''
# LC-QuAD2 train
TSC_LCQAUD2_TRAIN_PATH = LC_QUAD2_TSC_FOLDER + '/train.json'
TSC_lcquad2_train_df = pd.read_json(TSC_LCQAUD2_TRAIN_PATH)
SF_lcquad2_train = apply_SF('LC-QuAD2', TSC_lcquad2_train_df)
SF_lcquad2_train.to_json(TSC_LCQAUD2_TRAIN_PATH, orient='records', index=False, force_ascii=False, indent=2)
# LC-QuAD2 test
TSC_LCQAUD2_TEST_PATH = LC_QUAD2_TSC_FOLDER + '/test.json'
TSC_lcquad2_test_df = pd.read_json(TSC_LCQAUD2_TEST_PATH)
SF_lcquad2_test = apply_SF('LC-QuAD2', TSC_lcquad2_test_df)
SF_lcquad2_test.to_json(TSC_LCQAUD2_TEST_PATH, orient='records', index=False, force_ascii=False, indent=2)
#'''
################################

def apply_SB(name_dataset, serie:pd.Series) -> pd.Series:
    '''
    Funzione per applicare Semantic Backwarding per dataset Wikidata chiamando restore_wikidata_ids()
    Per info vedi TSET (https://doi.org/10.3390/app14041521)

    La mia implementazione prevede l'uso di un json file dove reperire i labels,
    frutto di una precedente mappatura, per evitare di scaricare il dump di wikidata
    di oltre 130 GB.
    '''

    def _restore_wikidata_ids(text:str, dataset:Dict) -> str:
        '''
        Funzione per applicare Semantic Backwarding.

        Input: text=stringa di testo per domanda o sparql query
                dataset=dizionario del dataset ordinato secondo la lunghezza dei labels in ordine decresente
        Output stringa del testo modificata con id sostituiti dalle proprie labels
        '''

        new_text = text
        for id, label in dataset.items():
            new_text = new_text.replace(label, id)
        return new_text
    
    def _get_dataset(name_dataset):
        data = _load_data()
        return data.get(name_dataset, None)
    
    
    # Semantic Backwarding (SB)
    dataset_ids_labels_map = _get_dataset(name_dataset=name_dataset)
    
    if dataset_ids_labels_map == None:
        return print('Impossibile eseguire labeling degli id perché il dataset '\
                     'di riferimento non è nel file json della mappatura.')

    ids_labels_map = dataset_ids_labels_map['ids_labels_map']
    # ID con labels nulli ignorati
    non_null_map = ((k,v) for k,v in ids_labels_map.items() if v is not None)
    # Ordine decrescente del dizionario secondo la lunghezza dei labels
    ids_labels_map = dict(sorted(non_null_map, key=lambda x: len(x[1]), reverse=True))

    print('\tApplicazione Semantic Backwarding...')
    print('\t\tApplicazione SB alla Series...')
    serie = serie.apply(lambda x: _restore_wikidata_ids(x, ids_labels_map)) # Applicazione SB all'oggetto Series
    print('\t\tProcesso concluso.')

    return serie


class Triple:
    """Represents a SPARQL triple"""
    subject: str
    predicates: tuple
    objects: tuple

    def __hash__(self):
        return hash((self.subject, self.predicates, self.objects))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.predicates == other.predicates and
                self.objects == other.objects)


class EvaluationMetrics:
    """Stores evaluation metrics"""
    triple_exact_match: float
    correction_accuracy: float
    preservation_accuracy: float
    precision: float
    recall: float
    f1_score: float
    query_exact_match: float
    total_triples: int
    correct_triples: int
    flipped_triples_in_input: int
    correctly_fixed: int
    incorrectly_fixed: int
    false_swaps: int
    missed_flips: int
    perfect_queries: int
    total_queries: int


class SPARQLTripleEvaluator:
    """Evaluates triple structure correction in SPARQL queries"""

    def retrieve_triples(sparql_q) -> List[str]:
        '''
        Funzione per ricavare le triple presenti nella query.

        Segue descrizione regex usata:
        triples_pattern individua triple comprendendo:
                         - variabili normalizzate introdotte da ?;
                         - literals con anche la specificazione della
                         lingua, e.g. "cane"@it;
                         - entità con prefisso, e.g. wd:Q123;
                         - proprietà rdf:type abbreviata, e.g. a;
                         - proprietà con prefisso, e.g. dbr:nato_in;
                         - triple estese da operatori AND (;), e.g.
                          ?var1 dbr:figlio_di dbo:Dio ; dbr:nato_in dbo:Nazareth
        VALUES_pattern individua clausole VALUES, e.g. VALUES ?var0 { wd:Q123 wd:Q987}

        Input: sparql_q=stringa della query
        Output: Lista di stringhe dato che non ci sono gruppi di cattura nel pattern usato
        '''

        VALUES_pattern = r'VALUES\s+\?\S+\s+\{(.*?)\}'
        triples_pattern = r'(?:[^\s\{\[\(]+:\S+|\?\S+)\s+(?:[^\s\{\[\(]+:[^\s\}\]\)]+|a|\?\S+)\s+(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"\S+\"(?:@\w{2})*)(?:\s+;\s+(?:\S+:[^\s\}\]\)]+|a|\?\S+)\s+(?:\S+:[^\s\}\]\)\.]+|\?\S+|\"\S+\"(?:@\w{2})*))*'

        sparql_q_noComment= re.sub(r'#.*$', '', sparql_q, flags=re.MULTILINE) # Eliminazione commenti
        sparql_q_noVALUES = re.sub(VALUES_pattern, '', sparql_q_noComment) # Eliminazione clausole VALUES
        triples_matches = re.findall(triples_pattern, sparql_q_noVALUES, re.IGNORECASE)
        return triples_matches


    def get_subj_props_objs(triple:str) -> Tuple[str, Tuple[str], Tuple[str]]:
        '''
        Funzione che trova il soggetto, predicato/i e oggetto/i nella stringa della tripla
        passata in input.

        Segue descrizione regex usate:
        - subj_pattern individua il soggetto nella tripla, banalmente perché all'inizio,
        ma si specifica comunque che può trattarsi di entità con prefisso o di variabile
        normalizzata e.g. ?var0
        - props_objs_pattern individua il/gli predicato/i e il/gli oggetto/i della tripla, catturando:
                - variabili normalizzate;
                - entità con prefissi;
                - literals con o senza lingua specificata.

        Input: match=stringa della tripla
        Output: Tupla con una stringa e due tuple in questo ordine: soggetto, tupla predicati e tupla oggetti
        '''

        subj_pattern = r'^(\?var\d+|\S+:\S+)'
        props_objs_pattern = r'(\S+:\S+|a|\?var\d+)\s+(\?var\d+|\S+:\S+|\"\S+\"(?:@\w{2})*)'
        subj = re.match(subj_pattern, triple, re.IGNORECASE)
        subj = subj.group()
        triple = triple[len(subj):] # Per evitare sovrapposizioni nel matching degli oggetti si elimina il soggetto dalla tripla
        props_objs_finds = re.findall(props_objs_pattern, triple, re.IGNORECASE)
        props, objs = zip(*props_objs_finds)
        return subj, props, objs


    def extract_triples(self, query: str) -> List[Triple]:
        """Extract triples from a SPARQL query"""
        triples = []
        triples_matches = self.retrieve_triples(query)
        if not triples_matches:
            return triples

        for match in triples_matches:
            subj, props, objs = self.get_subj_props_objs(match)
            triples.append(Triple(subj, props, objs))
        
        return triples
            

    def evaluate_dataset(
        self,
        input_queries: List[str],
        output_queries: List[str],
        predicted_queries: List[str]
    ) -> EvaluationMetrics:
        """Evaluate model performance"""
        assert len(input_queries) == len(output_queries) == len(predicted_queries)

        total_triples = 0
        correct_triples = 0
        flipped_triples_in_input = 0
        correctly_fixed = 0
        incorrectly_fixed = 0
        false_swaps = 0
        missed_flips = 0
        perfect_queries = 0

        for gold_q, input_q, pred_q in zip(output_queries, input_queries, predicted_queries):
            gold_triples = self.extract_triples(gold_q)
            input_triples = self.extract_triples(input_q)
            pred_triples = self.extract_triples(pred_q)

            if len(gold_triples) != len(pred_triples):
                continue

            total_triples += len(gold_triples)
            
            query_perfect = True
            input_swaps = set()
            pred_swaps = set()
            for i, (g_triple, i_triple, p_triple) in enumerate(zip(gold_triples, input_triples, pred_triples)):
                if not g_triple.subject == i_triple.subject:
                    if g_triple.subject in i_triple.objects and i_triple.subject in g_triple.objects:
                        input_swaps.add((i, i_triple.objects.index(g_triple.subject)))
                if not g_triple == p_triple:
                    if not g_triple.subject == p_triple.subject:
                        if g_triple.subject in p_triple.objects and p_triple.subject in g_triple.objects:
                            pred_swaps.add((i, p_triple.objects.index(g_triple.subject)))
                        else:
                          incorrectly_fixed += 1
                    query_perfect = False
                else:
                    correct_triples += 1
            
            flipped_triples_in_input += len(input_swaps)
            
            if query_perfect:
                perfect_queries += 1
            
            missed_flips += len(input_swaps & pred_swaps)
            set_false_swaps = pred_swaps - input_swaps
            false_swaps += len(set_false_swaps)
            correctly_fixed += len(input_swaps - (pred_swaps - set_false_swaps))

        triple_exact_match = correct_triples / total_triples if total_triples > 0 else 0
        correction_accuracy = (correctly_fixed / flipped_triples_in_input
                              if flipped_triples_in_input > 0 else 0)
        correct_triples_in_input = total_triples - flipped_triples_in_input
        preservation_accuracy = ((correct_triples - correctly_fixed) / correct_triples_in_input
                                if correct_triples_in_input > 0 else 0)

        true_positives = correctly_fixed
        false_positives = false_swaps
        precision = (true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0 else 0)

        false_negatives = missed_flips
        recall = (true_positives / (true_positives + false_negatives)
                 if (true_positives + false_negatives) > 0 else 0)

        f1_score = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)

        query_exact_match = perfect_queries / len(output_queries)

        return EvaluationMetrics(
            triple_exact_match=triple_exact_match,
            correction_accuracy=correction_accuracy,
            preservation_accuracy=preservation_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            query_exact_match=query_exact_match,
            total_triples=total_triples,
            correct_triples=correct_triples,
            flipped_triples_in_input=flipped_triples_in_input,
            correctly_fixed=correctly_fixed,
            incorrectly_fixed=incorrectly_fixed,
            false_swaps=false_swaps,
            missed_flips=missed_flips,
            perfect_queries=perfect_queries,
            total_queries=len(output_queries)
        )

    def print_metrics(self, metrics: EvaluationMetrics):
        """Pretty print evaluation metrics"""
        print("=" * 60)
        print("SPARQL TRIPLE CORRECTION EVALUATION RESULTS")
        print("=" * 60)

        print("\n📊 TRIPLE-LEVEL METRICS")
        print(f"  Triple Exact Match:      {metrics.triple_exact_match:.2%}")
        print(f"  Correction Accuracy:     {metrics.correction_accuracy:.2%}")
        print(f"  Preservation Accuracy:   {metrics.preservation_accuracy:.2%}")

        print("\n🎯 PRECISION, RECALL, F1")
        print(f"  Precision:               {metrics.precision:.2%}")
        print(f"  Recall:                  {metrics.recall:.2%}")
        print(f"  F1 Score:                {metrics.f1_score:.2%}")

        print("\n📋 QUERY-LEVEL METRICS")
        print(f"  Query Exact Match:       {metrics.query_exact_match:.2%}")
        print(f"  Perfect Queries:         {metrics.perfect_queries}/{metrics.total_queries}")

        print("\n🔢 DETAILED COUNTS")
        print(f"  Total Triples:           {metrics.total_triples}")
        print(f"  Correct Triples:         {metrics.correct_triples}")
        print(f"  Flipped in Input:        {metrics.flipped_triples_in_input}")
        print(f"  Correctly Fixed:         {metrics.correctly_fixed}")
        print(f"  Missed Flips:            {metrics.missed_flips}")
        print(f"  False Swaps:             {metrics.false_swaps}")
        print(f"  Other Errors:            {metrics.incorrectly_fixed}")
        print("=" * 60)
    


def apply_SB(name_dataset, serie:pd.Series) -> pd.Series:
    '''
    Funzione per applicare Semantic Backwarding per dataset Wikidata
    Per info vedi TSET (https://doi.org/10.3390/app14041521)

    La mia implementazione prevede l'uso di un json file dove reperire i labels,
    frutto di una precedente mappatura, per evitare di scaricare il dump di wikidata
    di oltre 130 GB.
    '''

    def _load_data():
      """Carica il file JSON se esiste, altrimenti restituisce un dict vuoto."""
      if os.path.exists(IDS_LABELS_MAP_PATH):
          with open(IDS_LABELS_MAP_PATH, "r", encoding="utf-8") as f:
              return json.load(f)
      return {}

    def _restore_wikidata_ids(text:str, dataset:Dict) -> str:
        '''
        Input: text=stringa di testo per domanda o sparql query
                dataset=dizionario del dataset ordinato secondo la lunghezza dei labels in ordine decresente
        Output stringa del testo modificata con id sostituiti dalle proprie labels
        '''

        new_text = text
        for id, label in dataset.items():
            new_text = new_text.replace(f':{label}', f':{id}')
        return new_text

    def _get_dataset(name_dataset):
        data = _load_data()
        return data.get(name_dataset, None)


    # Semantic Backwarding (SB)
    dataset_ids_labels_map = _get_dataset(name_dataset=name_dataset)

    if dataset_ids_labels_map is None:
        return print('Impossibile eseguire labeling degli id perché il dataset '\
                     'di riferimento non è nel file json della mappatura.')

    ids_labels_map = dataset_ids_labels_map['ids_labels_map']
    # ID con labels nulli ignorati
    non_null_map = ((k,v) for k,v in ids_labels_map.items() if v is not None)
    # Ordine decrescente del dizionario secondo la lunghezza dei labels
    ids_labels_map = dict(sorted(non_null_map, key=lambda x: len(x[1]), reverse=True))

    print('\tApplicazione Semantic Backwarding...')
    print('\t\tApplicazione SB alla Series...')
    serie = serie.apply(lambda x: _restore_wikidata_ids(x, ids_labels_map)) # Applicazione SB all'oggetto Series
    print('\t\tProcesso concluso.')

    return serie