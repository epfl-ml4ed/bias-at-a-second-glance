#! /bin/bash

# Modified from https://github.com/webis-de/argmining20-social-bias-argumentation
# Add the number of tokens and the vector dimension to all full models
for MODEL in ./GloVe/glove_models_trained/*_vectors.txt
do
    echo ""
    echo -e "\e[1;31m================================================================================\e[0m"
    echo -e "\e[1;31mConverting ${MODEL}...\e[0m"

    sed -i "" -e "1s/^/$(echo $(cat ${MODEL} | wc -l)) 300\n/" $MODEL
done
