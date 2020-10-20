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

### Citation
```
@inproceedings{saha2020conjnli,
  title={ConjNLI: Natural Language Inference over Conjunctive Sentences},
  author={Saha, Swarnadeep and Nie, Yixin and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```
