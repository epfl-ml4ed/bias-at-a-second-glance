# Bias at a Second Glance

We conduct a German WEAT analysis on a new corpus of 9,165 German Educational Peer Reviews. 

## Data

Instructions to download data files are found in [the data folder](data/), with one file for each year of peer-reviews.

## Bias Analysis

To conduct **WEAT co-occurence analysis**, use the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb). The German translations of WEAT tests are found in [the weat folder](weat/).

## Training Embedding Models

To train a **GloVE model**, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the bash script [GloVe training](code/GloVe_training.sh).

## Fine-Tuning German Language Models

German Language models were obtained directly from HuggingFace ([BERT](https://huggingface.co/bert-base-german-cased), [T5](), [GPT-2](https://huggingface.co/dbmdz/german-gpt2)).

To finetune **German BERT**, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [BERT finetuning notebook](code/BERT_finetuning.ipynb).

To finetune **German T5**, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [T5 finetuning notebook](code/T5_finetuning.ipynb).

To finetune **German GPT-2**, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [GPT-2 finetuning notebook](code/GPT2_finetuning.ipynb).

### Credit

This implementation was inspired by code from [argmining20-social-bias-argumentation](https://github.com/webis-de/argmining20-social-bias-argumentation), [WordBias](https://github.com/bhavyaghai/WordBias), and [XWEAT](https://github.com/anlausch/XWEAT), among others. All credit for borrowed code remains with the original authors.
