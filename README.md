# Bias at a Second Glance

Instructions to download data files are found in [the data folder](data/), with one file for each year of peer-reviews.

The German translations of WEAT tests are found in [the weat folder](weat/).

This code is not in its final version and requires further cleaning and documentation before public release.

To conduct WEAT co-occurence analysis, use the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb).

To train a GloVE model, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the bash script [GloVe training](code/GloVe_training.sh).

To finetune German BERT, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [BERT finetuning notebook](code/BERT_finetuning.ipynb).

To finetune German T5, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [T5 finetuning notebook](code/T5_finetuning.ipynb).

To finetune German GPT-2, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the [GPT-2 finetuning notebook](code/GPT2_finetuning.ipynb).
