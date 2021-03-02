

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import types
import gzip
import logging
import re
import six
import collections
from utils import tokenization
from utils.batching import prepare_batch_data


def get_input_shape(args):
    """
    define mask language model input shape
    """
    train_input_shape = {"backbone": [([-1, args.max_seq_len, 1], 'int64'),
                                      ([-1, args.max_seq_len, 1], 'int64'),
                                      ([-1, args.max_seq_len, 1], 'int64'),
                                      ([-1, args.max_seq_len, 1], 'float32')],
                         "task": [([-1, 1], 'int64'),
                                  ([-1, 1], 'int64')]
                         }

    test_input_shape = {"backbone": [([-1, args.max_seq_len, 1], 'int64'),
                                     ([-1, args.max_seq_len, 1], 'int64'),
                                     ([-1, args.max_seq_len, 1], 'int64'),
                                     ([-1, args.max_seq_len, 1], 'float32')],
                         "task": [([-1, 1], 'int64'),
                                  ([-1, 1], 'int64')]
                         }
    return train_input_shape, test_input_shape


class DataProcessor(object): 
    def __init__(self, args): 
        self.vocab = self.load_vocab(args.vocab_path)
        self.data_dir = args.train_file
        self.batch_size = args.batch_size
        self.in_tokens = args.in_tokens
        self.epoch = args.epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.generate_neg_sample = args.generate_neg_sample
        self.voc_size = len(self.vocab)
        self.max_seq_len = args.max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        if self.in_tokens:
            assert self.batch_size >= self.max_seq_len, "The number of " \
                   "tokens in batch should not be smaller than max seq length."

    def get_progress(self):
        """return current progress of training data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().decode().split(";")
        assert len(line) == 4, "One sample must have 4 fields!"
        (token_ids, sent_ids, pos_ids, label) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(
            pos_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label]

    def read_file(self, file):
        assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        file_path = self.data_dir + "/" + file
        with gzip.open(file_path, "rb") as f:
            for line in f:
                parsed_line = self.parse_line(
                    line, max_seq_len=self.max_seq_len)
                if parsed_line is None:
                    continue
                yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file,encoding='utf-8')
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negative samples using pos_samples

            Args:
                pos_samples: list of positive samples
            
            Returns:
                neg_samples: list of negative samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            origin_src_ids = pos_samples[i][0]
            origin_sep_index = origin_src_ids.index(2)
            pair_src_ids = pos_samples[pair_index][0]
            pair_sep_index = pair_src_ids.index(2)

            src_ids = origin_src_ids[:origin_sep_index + 1] + pair_src_ids[
                pair_sep_index + 1:]
            if len(src_ids) >= self.max_seq_len:
                miss_num += 1
                continue
            sent_ids = [0] * len(origin_src_ids[:origin_sep_index + 1]) + [
                1
            ] * len(pair_src_ids[pair_sep_index + 1:])
            pos_ids = list(range(len(src_ids)))
            neg_sample = [src_ids, sent_ids, pos_ids, 0]
            assert len(src_ids) == len(sent_ids) == len(
                pos_ids
            ), "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append(neg_sample)
        return neg_samples, miss_num

    def mixin_negative_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negative samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negative samples and positive samples
            
            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negative samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def data_generator(self, phase='train', shuffle=True, dev_count=1):
        """
        data_generator
        """
        files = os.listdir(self.data_dir)
        self.total_file = len(files)
        assert self.total_file > 0, "[Error] data_dir is empty"

        def wrapper():
            def reader(): 
                if phase == "train": 
                    epoch_num = self.epoch
                    is_test = False
                else: 
                    epoch_num = 1
                    is_test = True
                for epoch in range(epoch_num):
                    self.current_epoch = epoch + 1
                    if shuffle:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        self.current_file_index = index + 1
                        self.current_file = file
                        sample_generator = self.read_file(file)
                        if not is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negative_samples(
                                sample_generator)
                        for sample in sample_generator:
                            if sample is None:
                                continue
                            yield sample

            def batch_reader(reader, batch_size, in_tokens):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, label = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if in_tokens:
                        to_append = (len(batch) + 1) * max_len <= batch_size
                    else:
                        to_append = len(batch) < batch_size
                    if to_append:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(
                    reader, self.batch_size, self.in_tokens):
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    max_len=self.max_seq_len,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
