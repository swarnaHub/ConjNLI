python download_mnli.py -d data/ -t MNLI
mv data/MNLI/train.tsv data/NLI/train_mnli.tsv
mv data/MNLI/dev_matched.tsv data/NLI/dev_matched_mnli.tsv
mv data/MNLI/dev_mismatched.tsv data/NLI/dev_mismatched_mnli.tsv
rm -r data/MNLI/