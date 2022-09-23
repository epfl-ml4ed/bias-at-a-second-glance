# Bias at a Second Glance

This repository is the official implementation of the COLING 2022 paper entitled ["Bias at a Second Glance: A Deep Dive into Bias for German Educational
Peer-Review Data Modeling"](https://arxiv.org/pdf/2209.10335.pdf) written by Thiemo Wambsganss, Vinitra Swamy, Roman Rietsche, and Tanja Käser. 

## Project Overview

In this work, we analyze bias across text and through multiple architectures on a corpus of 9,165 German peer-reviews collected from university students over five years. Notably, our corpus includes labels such as helpfulness, quality, and critical aspect ratings from the peer-review recipient as well as demographic attributes. We conduct a Word Embedding Association Test (WEAT) analysis on (1) our collected corpus in connection with the clustered labels, (2) the most common pre-trained German language models (T5, BERT, and GPT-2) and GloVe embeddings, and (3) the language models after fine-tuning on our collected data-set. In contrast to our initial expectations, we found that our collected corpus does not reveal many biases in the co-occurrence analysis or in the GloVe embeddings. However, the pre-trained German language models find substantial conceptual, racial, and gender bias and have significant changes in bias across conceptual and racial axes during fine-tuning on the peer-review data. With our research, we aim to contribute to the fourth UN sustainability goal (quality education) with a novel dataset, an understanding of biases in natural language education data, and the potential harms of not counteracting biases in language models for educational tasks.

![image](https://user-images.githubusercontent.com/72170466/191957442-8f970ae7-9bfe-4f31-a0b3-9fb579bf8949.png)

## Repository Structure

### Data

Instructions to download data files are found in [the data folder](data/), with one file for each year of peer-reviews.

### Bias Analysis

To conduct **WEAT co-occurence analysis**, use the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb). The German translations of WEAT tests are found in [the weat folder](weat/).

### Training Embedding Models

To train a **GloVE model**, run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb) and run the bash script [GloVe training](code/GloVe_training.sh).

### Fine-Tuning German Language Models

German Language models were obtained directly from HuggingFace ([BERT](https://huggingface.co/bert-base-german-cased), [T5](https://huggingface.co/ml6team/mt5-small-german-finetune-mlsum), [GPT-2](https://huggingface.co/dbmdz/german-gpt2)).

To finetune **German BERT**:
1. Run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb).
2. Run the [BERT finetuning notebook](code/BERT_finetuning.ipynb).

To finetune **German T5**:
1. Run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb).
2. Run the [T5 finetuning notebook](code/T5_finetuning.ipynb).

To finetune **German GPT-2**: 
1. Run the relevant parts of the [WEAT Analysis notebook](code/WEAT_analysis_peer_reviews.ipynb).
2. Run the [GPT-2 finetuning notebook](code/GPT2_finetuning.ipynb).


## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

### Credit

This implementation was inspired by code from [argmining20-social-bias-argumentation](https://github.com/webis-de/argmining20-social-bias-argumentation), [WordBias](https://github.com/bhavyaghai/WordBias), and [XWEAT](https://github.com/anlausch/XWEAT), among others. All credit for borrowed code remains with the original authors.

## Citations
If you find this code useful in your work, please cite our paper:

```
Wambsganss, T., Swamy, V., Rietsche, R., Käser, T. (2022). 
Bias at a Second Glance: A Deep Dive into Bias for German Educational Peer-Review Data Modeling.
In: Proceedings of the 29th International Conference on Computational Linguistics (COLING).
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.
