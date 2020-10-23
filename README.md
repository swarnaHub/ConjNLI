# ConjNLI
Data and PyTorch code for our EMNLP 2020 paper:

[ConjNLI: Natural Language Inference over Conjunctive Sentences](https://arxiv.org/abs/2010.10418)

[Swarnadeep Saha](https://swarnahub.github.io/), [Yixin Nie](https://easonnie.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Installation
This repository is tested on Python 3.8.1.  
You should install ConjNLI on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Data

ConjNLI dev and test sets can be found inside ```data/NLI``` folder. The test set does not have gold annotations. Check out the ```ConjNLI Submission``` section below to know how to submit your results on the ConjNLI test set.

We also release the adversarially created training examples at ```data/NLI/adversarial_15k_train.tsv``` which can be used to train the IAFT model (details below).

MNLI train and dev splits can be downloaded by running
```
bash scripts/get_data.sh
```

## Code

Follow the instructions below to train RoBERTa, and our two proposed models, RoBERTa-IAFT and RoBERTa-PA. These models can be subsequently tested on the dev splits of ConjNLI and MNLI.

### Training RoBERTa

Train the baseline RoBERTa model by running
```
bash scripts/finetune_conjnli.sh
```
This will save the model and sample predictions for ConjNLI dev set inside ```output/ConjNLI```.

### Training RoBERTa-IAFT

In order to train an IAFT model, you'd first need a baseline RoBERTa model trained and saved (following previous section). Once you have that, you can train an Iterative Adversarial Fine-tuning model by running
```
bash scripts/train_conjnli_IAFT.sh
```
This will save the model and sample predictions for ConjNLI dev set inside ```output/ConjNLI_IAFT```.

### Training RoBERTa-PA

The predicate-aware RoBERTa model (RoBERTa-PA) first requires a fine-tuned BERT model on the Semantic Role Labeling (SRL) task. In order to train this model, you will first need to download the [CoNLL-2005 SRL data](https://www.cs.upc.edu/~srlconll/) and place the train, dev, test files inside ```data/PropBank```. 

Train an SRL model on the PropBank data using the following script
```
bash scripts/train_srl.sh
```
This will save the SRL model inside ```output/srl_bert``` and you can expect an F1 of approximately ```86%``` on the CoNLL 2005 dev set.

Now you can train the Predicate-aware model by running
```
bash scripts/train_conjnli_PA.sh
```
This will save the model and sample predictions for ConjNLI dev set inside output/ConjNLI_PA.

### Evaluating models

Once you have the models saved, you can simply evaluate these by running the corresponding evaluation scripts. Specifically, depending on the model you want to evaluate, you can run any one of the following scripts
```
bash scripts/test_conjnli.sh
bash scripts/test_conjnli_IAFT.sh
bash scripts/test_conjnli_PA.sh
```
By default, these will report results on the ConjNLI dev set. Should you wish to evaluate these models on MNLI dev set, look at the comments in lines 105 and 155 of ```utils_conjnli.py```.

## ConjNLI Scoreboard

Model | Link | Date | Conj Dev | MNLI Dev | Conj Test
--- | --- | --- | --- | --- | ---
BERT | [Saha et al., 2020](https://arxiv.org/abs/2010.10418) | 10-20-2020 | 58.10 | 84.10/83.90 | 61.40
RoBERTa | [Saha et al., 2020](https://arxiv.org/abs/2010.10418) | 10-20-2020 | 64.68 | 87.56/87.51 | 65.50
RoBERTa-PA | [Saha et al., 2020](https://arxiv.org/abs/2010.10418) | 10-20-2020 | 64.88 | 87.75/87.63 | 66.30
RoBERTa-IAFT | [Saha et al., 2020](https://arxiv.org/abs/2010.10418) | 10-20-2020 | 69.18 | 86.93/86.81 | 67.90

## ConjNLI Submission

If you want your results to be shown on the Scoreboard, please email us (swarna@cs.unc.edu) with the name of the entry, a link to your method, and your model prediction files. Specifically, you should send us four model prediction files, one for ConjNLI Dev set, two for MNLI matched/mismatched Dev sets and another for ConjNLI test set. Each prediction file should be a text file with one label per line, in the order of the examples. A sample prediction file is shown in ```data/NLI/sample_prediction.txt```.

## Trained Models
We also release our trained models [here](https://drive.google.com/file/d/1hGnZD6c_s5dZ2nHgYbnfVLMwaA4vUNxz/view?usp=sharing). You can reproduce the results from the paper by running the corresponding evaluation scripts.

## Citation
```
@inproceedings{saha2020conjnli,
  title={ConjNLI: Natural Language Inference over Conjunctive Sentences},
  author={Saha, Swarnadeep and Nie, Yixin and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```
