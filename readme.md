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

The settings of PromptRank can be modified in main.py as follows:

```python
def get_setting_dict():
    setting_dict = {}
    setting_dict["max_len"] = 512
    setting_dict["temp_en"] = "Book:"
    setting_dict["temp_de"] = "This book mainly talks about "
    setting_dict["model"] = "base"
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict
```

The settings of running can be modified in run.sh or run_all.sh.

## Performance

The performance of PromptRank is shown as follows. See the performance of PromptRank$_{filter}$ and other baselines in the paper.

| F1@K | Inspec | SemEval2017 | SemEval2010 | DUC2001 | Krapivin | NUS   | AVG   |
| :--: | :----: | :---------: | :---------: | :-----: | :------: | :---: | :---: |
| 5    | 31.73  | 27.14       | 17.24       | 27.39   | 16.11    | 17.24 | 22.81 |
| 10   | 37.88  | 37.76       | 20.66       | 31.59   | 16.71    | 20.13 | 27.46 |
| 15   | 38.17  | 41.57       | 21.35       | 31.01   | 16.02    | 20.12 | 28.04 |


