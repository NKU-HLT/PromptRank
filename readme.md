# PromptRank

This is code of our paper PromptRank: Unsupervised Keyphrase Extraction using Prompt. The code is modified based on MDERank. Data is from OpenNMT-kpg-release and SIFRank. (Inspec, DUC2001, SemEval2017 are from SIFRank).

## Environment

```
StanfordCoreNLP 3.9.1.1
Python 3.8
torch 1.9.1
nltk 3.7
transformers 4.23
```

## Usage

We use run.sh script to run PromptRank for a specfic dataset.

```
sh run.sh
```

We use run_all.sh script to run PromptRank for all six datasets and the results will be summarized in result.txt by summary.py.

```
sh run_all.sh
```
