# Modifying Memories of Transformer Models
This repository is a Re-implementation for creating the benchmarks in the paper [Modifying Memories of Transformer Models](https://arxiv.org/pdf/2012.00363.pdf). Due to randomness, the selected subset of facts for modification are different from the original paper.

## Creating modification dataset for TREx

To generate modified facts:

### 1. Create conda environment (optional) and install requirements

```
conda create -n lama37 -y python=3.7 && conda activate lama37
pip install -r requirements.txt
python -m spacy download en
```

### 2. Download the data

```bash
mkdir lama_data && cd lama_data
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
cd ..
```

### 3. Download pretrained BERT-Base model
For a full list of available models, please refer to [the LAMA repo](https://github.com/facebookresearch/LAMA/blob/master/download_models.sh).
```
bash ./download_bert_base.sh
```

### 4. Randomly select the questions and generate modified answers
```
python create_trex.py
```
By default, a dict mapping uuids to the modified answers are stored in `modification/change_list_{num_modified_facts}.json`. You can use this dict to finetune the model on the supporting facts on TREx and test it on the corresponding LAMA probes. 

For example, the answer of the fact with uuid `6904327c-5211-4edb-ad8c-bcefee2149c6` (in `lama_data/data/TREx/P276.jsonl`) is modified from `Atlanta` to `Ukraine`. You could use all the masked `sub_surface` sentences in `P276.jsonl` for finetuning the model, and the template `` for predicate `P276` to generate the masked probing sentence for test. These templates are given in `lama_data/data/relations.jsonl` if you followed our Step 2. 



## Creating modification for zsRE
The zsRE implementation is based on [KILT](https://github.com/facebookresearch/KILT). 

### 1. Download the data
```bash
mkdir zsre_data && cd zsre_data
wget http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-train-kilt.jsonl
cd .. 
```

### 2. Generate the data
Since a finetuned model can easily achieve perfect test accuracy in our setting, we simply choose any fact with at least one eval question for modification. 

## Examples
The examples are given in `modification.zip`. We give an example subset of 32 modified facts for TREx in `modification/change_list_32.json`. 
For zsRE, we give an example train/eval split (`modification/zsre_train_unmodified.jsonl` and `modification/zsre_eval_unmodified.jsonl`), and a subset of 32 modified facts (`modification/zsre_train_modified_32.jsonl` and `modification/zsre_eval_modified_32.jsonl`). 