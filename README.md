# ConjNLI
Data and PyTorch code for our EMNLP 2020 paper:

[ConjNLI: Natural Language Inference over Conjunctive Sentences]()

[Swarnadeep Saha](https://swarnahub.github.io/), [Yixin Nie](https://easonnie.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Installation
This repository is tested on Python 3.8.1.  
You should install ConjNLI on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Data

ConjNLI dev and test sets can be found inside ```data/NLI``` folder.

We also release the adversarially created training examples at ```data/NLI/adversarial_15k_train.tsv```.

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

The predicate-aware RoBERTa model (RoBERTa-PA) first requires a fine-tuned BERT model on the Semantic Role Labeling (SRL) task. The data used to train the SRL model is inside ```data/PropBank```.
Train an SRL model on the PropBank data using the following script
```
bash scripts/train_srl.sh
```

### Citation
```
@inproceedings{saha2020conjnli,
  title={ConjNLI: Natural Language Inference over Conjunctive Sentences},
  author={Saha, Swarnadeep and Nie, Yixin and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```
