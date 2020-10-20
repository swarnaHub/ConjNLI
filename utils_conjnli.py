from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from nltk.tokenize import word_tokenize

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesWithSRL(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 input_ids_SRL_a, input_mask_SRL_a, segment_ids_SRL_a,
                 input_ids_SRL_b, input_mask_SRL_b, segment_ids_SRL_b):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        self.input_ids_SRL_a = input_ids_SRL_a
        self.input_mask_SRL_a = input_mask_SRL_a
        self.segment_ids_SRL_a = segment_ids_SRL_a

        self.input_ids_SRL_b = input_ids_SRL_b
        self.input_mask_SRL_b = input_mask_SRL_b
        self.segment_ids_SRL_b = segment_ids_SRL_b


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class ConjNLIProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_mnli(
            self._read_tsv(os.path.join(data_dir, "train_mnli.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_conj_nli(
            self._read_tsv(os.path.join(data_dir, "conj_dev.tsv")), "dev")
        # Uncomment the following line and comment the previous line
        # to test on MNLI matched/mismatched sets
        # return self._create_examples_mnli(
        #     self._read_tsv(os.path.join(data_dir, "dev_matched_mnli.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples_mnli(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_conj_nli(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ConjNLIProcessorWithAdv(DataProcessor):
    def get_train_examples(self, data_dir):
        train_adversarial = self._create_examples_conj_nli(
            self._read_tsv(os.path.join(data_dir, "adversarial_train_15k.tsv")), "train")
        adv_size = len(train_adversarial)
        train = self._create_examples_iterating(self._read_tsv(os.path.join(data_dir, "train_mnli.tsv")), "train",
                                                adv_size)
        train.extend(train_adversarial)
        return train

    def get_dev_examples(self, data_dir):
        return self._create_examples_conj_nli(
            self._read_tsv(os.path.join(data_dir, "conj_dev.tsv")), "dev")
        # Uncomment the following line and comment the previous line
        # to test on MNLI matched/mismatched sets
        # return self._create_examples_mnli(
        #     self._read_tsv(os.path.join(data_dir, "dev_matched_mnli.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples_conj_nli(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_mnli(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_iterating(self, lines, set_type, adv_size):
        examples = []
        epochs = 3
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # In each epoch, we use an equal number of MNLI samples as adv size
            # In total, we use epochs * adv size data from MNLI
            if len(examples) == epochs * adv_size:
                break
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            words_a = word_tokenize(text_a)
            words_b = word_tokenize(text_b)
            if 'and' in words_a or 'and' in words_b or 'or' in words_a or 'or' in words_b \
                    or 'but' in words_a or 'but' in words_b:
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def get_verb_indicators(data_dir, mode):
    if mode == "train":
        file = os.path.join(data_dir, "train_mnli_srl.txt")
    elif mode == "dev":
        file = os.path.join(data_dir, "conj_dev_srl.txt")
    else:
        file = os.path.join(data_dir, "adversarial_train_srl.txt")
    with open(file, 'r', encoding='utf-8-sig') as f:
        SRL_lines = f.read().splitlines()

    sent = None
    labels = []
    indicator_map = {}
    id = 0
    for (i, line) in enumerate(SRL_lines):
        if line == "":
            if sent in indicator_map:
                sent = None
                labels = []
                continue
            verb_indicator = [0] * len(labels)
            if len(labels) != 0:
                for (index, label) in enumerate(labels):
                    if label == "B-V":
                        verb_indicator[index] = 1
            indicator_map[sent] = verb_indicator
            id += 1
            sent = None
            labels = []
        else:
            if sent is None:
                sent = line
            elif len(labels) == 0:
                labels = line.split("\t")
            else:
                continue

    return indicator_map


def get_srl_rep(text, verb_indicators, max_seq_length,
                tokenizer, output_mode,
                cls_token_at_end=False,
                cls_token='[CLS]',
                cls_token_segment_id=1,
                sep_token='[SEP]',
                sep_token_extra=False,
                pad_on_left=False,
                pad_token=0,
                pad_token_segment_id=0,
                sequence_a_segment_id=0,
                sequence_b_segment_id=1,
                mask_padding_with_zero=True):
    tokens = []
    segment_ids = []
    cumulative = 0
    start_offsets = []
    end_offsets = []
    predicate_indicator = verb_indicators[text]
    words = word_tokenize(text)
    for index, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)

        start_offsets.append(cumulative + 1)
        cumulative += len(word_tokens)
        end_offsets.append(cumulative)

        if index < len(predicate_indicator) and predicate_indicator[index] == 1:
            segment_ids.extend([1] * len(word_tokens))
        else:
            segment_ids.extend([0] * len(word_tokens))

    # Account for [CLS], [SEP], [SEP] with "- 2" (RoBERTa)
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
        segment_ids = segment_ids[:(max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    # Apart from the verb indicator, every other id is 0
    segment_ids += [0]

    tokens = [cls_token] + tokens
    segment_ids = [0] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([0] * padding_length) + segment_ids
    else:
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_examples_to_features_with_SRL(examples, verb_indicators, label_list, max_seq_length,
                                          tokenizer, tokenizer_SRL, output_mode,
                                          cls_token_at_end=False,
                                          cls_token='[CLS]',
                                          cls_token_segment_id=1,
                                          sep_token='[SEP]',
                                          sep_token_extra=False,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          sequence_a_segment_id=0,
                                          sequence_b_segment_id=1,
                                          mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Each premise and hypothesis are encoded using half of the total max_seq_length
        max_seq_length_srl = int(max_seq_length / 2)
        input_ids_SRL_a, input_mask_SRL_a, segment_ids_SRL_a = get_srl_rep(example.text_a, verb_indicators,
                                                                           max_seq_length_srl, tokenizer_SRL,
                                                                           output_mode,
                                                                           cls_token_at_end=False,
                                                                           # xlnet has a cls token at the end
                                                                           cls_token=tokenizer.cls_token,
                                                                           cls_token_segment_id=0,
                                                                           sep_token=tokenizer_SRL.sep_token,
                                                                           sep_token_extra=False,
                                                                           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                                           pad_on_left=False,
                                                                           pad_token=
                                                                           tokenizer_SRL.convert_tokens_to_ids(
                                                                               [tokenizer_SRL.pad_token])[0],
                                                                           pad_token_segment_id=0, )

        input_ids_SRL_b, input_mask_SRL_b, segment_ids_SRL_b = get_srl_rep(example.text_b, verb_indicators,
                                                                           max_seq_length_srl, tokenizer_SRL,
                                                                           output_mode,
                                                                           cls_token_at_end=False,
                                                                           # xlnet has a cls token at the end
                                                                           cls_token=tokenizer.cls_token,
                                                                           cls_token_segment_id=0,
                                                                           sep_token=tokenizer_SRL.sep_token,
                                                                           sep_token_extra=False,
                                                                           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                                           pad_on_left=False,
                                                                           pad_token=
                                                                           tokenizer_SRL.convert_tokens_to_ids(
                                                                               [tokenizer_SRL.pad_token])[0],
                                                                           pad_token_segment_id=0, )

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeaturesWithSRL(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 label_id=label_id,
                                 input_ids_SRL_a=input_ids_SRL_a,
                                 input_mask_SRL_a=input_mask_SRL_a,
                                 segment_ids_SRL_a=segment_ids_SRL_a,
                                 input_ids_SRL_b=input_ids_SRL_b,
                                 input_mask_SRL_b=input_mask_SRL_b,
                                 segment_ids_SRL_b=segment_ids_SRL_b,
                                 ))
    return features


def get_srl_labels():
    return ['O', 'B-V', 'B-A1', 'B-A0', 'I-A0', 'I-A1', 'B-AM-LOC', 'I-AM-LOC', 'B-AM-MNR', 'B-A2', 'I-A2', 'B-A3',
            'I-AM-MNR', 'B-AM-TMP', 'I-AM-TMP', 'B-A4', 'I-A4', 'I-A3', 'B-AM-NEG', 'B-AM-MOD', 'B-R-A0', 'B-AM-DIS',
            'B-AM-EXT', 'B-AM-ADV', 'I-AM-ADV', 'B-AM-PNC', 'I-AM-PNC', 'I-AM-DIS', 'B-R-A1', 'B-C-A1', 'I-C-A1',
            'B-R-AM-TMP', 'I-V', 'B-C-V', 'B-AM-DIR', 'I-AM-DIR', 'B-R-A2', 'B-AM-PRD', 'I-AM-PRD', 'I-R-A2',
            'B-R-AM-PNC', 'B-C-AM-MNR', 'I-C-AM-MNR', 'I-R-AM-TMP', 'B-AM-CAU', 'B-R-A3', 'B-R-AM-MNR', 'I-AM-CAU',
            'I-AM-EXT', 'B-C-A4', 'I-C-A4', 'I-R-A1', 'B-R-AM-LOC', 'I-R-A0', 'B-C-A0', 'I-C-A0', 'B-C-A2', 'I-C-A2',
            'B-R-AM-EXT', 'I-R-AM-EXT', 'B-A5', 'I-R-AM-MNR', 'B-C-AM-LOC', 'I-C-AM-LOC', 'I-R-AM-LOC', 'B-C-A3',
            'I-C-A3', 'I-AM-NEG', 'B-R-AM-CAU', 'B-R-A4', 'B-C-AM-ADV', 'I-C-AM-ADV', 'B-R-AM-ADV', 'I-R-AM-ADV',
            'I-R-A3', 'B-AM-REC', 'B-AM-TM', 'I-AM-TM', 'B-AM', 'I-AM', 'B-C-A5', 'I-C-A5', 'B-C-AM-TMP', 'I-C-AM-TMP',
            'B-AA', 'I-AA', 'B-R-AA', 'I-A5', 'I-AM-MOD', 'B-C-AM-EXT', 'I-AM-REC', 'B-C-AM-NEG', 'I-C-AM-EXT', 'I-C-V',
            'B-C-AM-DIS', 'I-C-AM-DIS', 'B-C-AM-CAU', 'I-C-AM-CAU', 'I-R-AM-PNC', 'B-R-AM-DIR', 'I-R-AM-DIR',
            'B-C-AM-DIR', 'I-C-AM-DIR', 'B-C-AM-PNC', 'I-C-AM-PNC', 'I-R-AM-CAU', 'I-R-A4', 'I-R-AA', 'I-C-AM-NEG']


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "conjnli" or task_name == "conjnli_adv":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "conjnli": ConjNLIProcessor,
    "conjnli_adv": ConjNLIProcessorWithAdv
}

output_modes = {
    "conjnli": "classification",
    "conjnli_adv": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "conjnli": 3,
    "conjnli_adv": 3
}
