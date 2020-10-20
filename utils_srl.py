
from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import numpy as np
import csv
import sys
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, sent, words, predicate_indicator, labels):
        self.guid = guid
        self.sent = sent
        self.words = words
        self.predicate_indicator = predicate_indicator
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, start_offsets, end_offsets):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.start_offsets = start_offsets
        self.end_offsets = end_offsets

def read_examples(data_dir, mode):
    input_file = os.path.join(data_dir, "props.{}".format(mode))
    examples = []
    id = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        sentence = None
        predicate_indicator = None
        verb_index = 0
        for line in lines:
            if line == "":
                sentence = None
                predicate_indicator = None
                verb_index = 0
            elif sentence is None:
                sentence = line
            elif predicate_indicator is None:
                predicate_indicator = [int(indicator) for indicator in line.split("\t")]
                verb_count = np.count_nonzero(np.array(predicate_indicator) == 1)
                # To account for sentences with no verb
                if verb_count == 0:
                    verb_count = 1
                split_predicate_indicators = np.zeros((verb_count, len(predicate_indicator)), dtype=int)
                index = 0
                for i in range(len(predicate_indicator)):
                    if predicate_indicator[i] == 1:
                        split_predicate_indicators[index][i] = 1
                        index += 1
            else:
                labels = line.split("\t")[1:]
                examples.append(InputExample(guid=id, sent=sentence, words=sentence.split(" "),
                                             predicate_indicator=split_predicate_indicators[verb_index], labels=labels))
                verb_index += 1
                id += 1

    return examples


def get_labels():
    return ['O', 'B-V', 'B-A1', 'B-A0', 'I-A0', 'I-A1', 'B-AM-LOC', 'I-AM-LOC', 'B-AM-MNR', 'B-A2', 'I-A2', 'B-A3', 'I-AM-MNR', 'B-AM-TMP', 'I-AM-TMP', 'B-A4', 'I-A4', 'I-A3', 'B-AM-NEG', 'B-AM-MOD', 'B-R-A0', 'B-AM-DIS', 'B-AM-EXT', 'B-AM-ADV', 'I-AM-ADV', 'B-AM-PNC', 'I-AM-PNC', 'I-AM-DIS', 'B-R-A1', 'B-C-A1', 'I-C-A1', 'B-R-AM-TMP', 'I-V', 'B-C-V', 'B-AM-DIR', 'I-AM-DIR', 'B-R-A2', 'B-AM-PRD', 'I-AM-PRD', 'I-R-A2', 'B-R-AM-PNC', 'B-C-AM-MNR', 'I-C-AM-MNR', 'I-R-AM-TMP', 'B-AM-CAU', 'B-R-A3', 'B-R-AM-MNR', 'I-AM-CAU', 'I-AM-EXT', 'B-C-A4', 'I-C-A4', 'I-R-A1', 'B-R-AM-LOC', 'I-R-A0', 'B-C-A0', 'I-C-A0', 'B-C-A2', 'I-C-A2', 'B-R-AM-EXT', 'I-R-AM-EXT', 'B-A5', 'I-R-AM-MNR', 'B-C-AM-LOC', 'I-C-AM-LOC', 'I-R-AM-LOC', 'B-C-A3', 'I-C-A3', 'I-AM-NEG', 'B-R-AM-CAU', 'B-R-A4', 'B-C-AM-ADV', 'I-C-AM-ADV', 'B-R-AM-ADV', 'I-R-AM-ADV', 'I-R-A3', 'B-AM-REC', 'B-AM-TM', 'I-AM-TM', 'B-AM', 'I-AM', 'B-C-A5', 'I-C-A5', 'B-C-AM-TMP', 'I-C-AM-TMP', 'B-AA', 'I-AA', 'B-R-AA', 'I-A5', 'I-AM-MOD', 'B-C-AM-EXT', 'I-AM-REC', 'B-C-AM-NEG', 'I-C-AM-EXT', 'I-C-V', 'B-C-AM-DIS', 'I-C-AM-DIS', 'B-C-AM-CAU', 'I-C-AM-CAU', 'I-R-AM-PNC', 'B-R-AM-DIR', 'I-R-AM-DIR', 'B-C-AM-DIR', 'I-C-AM-DIR', 'B-C-AM-PNC', 'I-C-AM-PNC', 'I-R-AM-CAU', 'I-R-A4', 'I-R-AA', 'I-C-AM-NEG']


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 predicate_segment_id=1,
                                 mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        segment_ids = []
        cumulative = 0
        start_offsets = []
        end_offsets = []
        for index, (word, label) in enumerate(zip(example.words, example.labels)):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if label[0] == 'B':
                following_label = 'I' + label[1:]
                label_ids.extend([label_map[label]] + [label_map[following_label]] * (len(word_tokens) - 1))
            else:
                label_ids.extend([label_map[label]] * len(word_tokens))

            start_offsets.append(cumulative+1)
            cumulative += len(word_tokens)
            end_offsets.append(cumulative)

            if example.predicate_indicator[index] == 1:
                segment_ids.extend([predicate_segment_id] * len(word_tokens))
            else:
                segment_ids.extend([sequence_a_segment_id] * len(word_tokens))

        # Account for [CLS], [SEP], [SEP] with "- 3" (RoBERTa)
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            segment_ids = segment_ids[:(max_seq_length - special_tokens_count)]


        tokens += [sep_token]
        label_ids += [label_map["O"]]
        # Apart from the verb indicator, every other id is 0
        segment_ids += [sequence_a_segment_id]

        tokens = [cls_token] + tokens
        label_ids = [label_map["O"]] + label_ids
        segment_ids = [sequence_a_segment_id] + segment_ids


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
            label_ids = ([0] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 100:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("start_offsets: %s", " ".join([str(x) for x in start_offsets]))
            logger.info("end_offsets: %s", " ".join([str(x) for x in end_offsets]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              start_offsets=start_offsets,
                              end_offsets=end_offsets))

    return features