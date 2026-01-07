# Machine-Learning for Humanities and Arts project

In this repository you will find three fine-tuning process files:
1. *fine_tuning.py*
2. *tsc_fine_tuning*
3. *final_fine_tuning.py*

The first is the fine-tuning process for the Text-to-Sparql (TTS) task.
The second is for the Triple Sentence Correction (TSC) task.
The third is fine-tuning for Text-to-Sparql task starting from the LoRA produced by fine-tuning 2.

In order to run every fine-tuning scripts of this repository please install the libraries reported in *requirements.txt*

## *datasets*

Here you will find all the datasets used to collect data about Wikidata and DBpedia knowledge graphs and datasets used in the fine-tuning processes. Namely they are:
- LC-QuAD v.2
- QALD 9+
- QALD 10

There is also a subfolder - *datasets/TSC* - containing the previous datasets processed in a manner to be used in the fine-tuning for TSC.

## *utils.py*

Here you can find all the preprocessing done to prepare the datasets to be used for the fine-tuning.

## *wikidata_ids_labels_map.json*

This the cache data use to speed up wikidata api requests for entities and relations when building datasets for fine-tuning.

## *testing_results.txt*

Here you can read the results for the two kind of fine-tuning: TTS e TSC + TTS.

## *report.docx*

This is the report about the project.