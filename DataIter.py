"""
@Author:    Pshy Simon
@Date:  2020/10/20 0020 下午 02:33
@Description:
    数据迭代器
"""
import json
import logging
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from math import floor
import gc


class DataIter:

    def __init__(self, config):
        self.config = config
        self.train_query = "./train/train.query.tsv"
        self.train_reply = "./train/train.reply.tsv"
        self.test_query = "./test/test.query.tsv"
        self.test_reply = "./test/test.reply.tsv"
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self._data = []
        self.preprocessor()

    def preprocessor(self):
        # 从磁盘读取文件，将文件merge到一起
#         train_query = pd.read_csv(self.train_query, sep='\t', header=None)
#         train_reply = pd.read_csv(self.train_reply, sep='\t', header=None)
#         test_query = pd.read_csv(self.test_query, sep='\t', header=None)
#         test_reply = pd.read_csv(self.test_reply, sep='\t', header=None)
#         train_query.columns = ['id','q1']
#         train_reply.columns = ['id','id_sub','q2','label']
#         self.train_df = pd.merge(train_query, train_reply, how="left", on="id")
        self.train_df = pd.read_csv("train.tsv", sep='\t')
        self.test_df = pd.read_csv("test.tsv", sep="\t")
#         test_query.columns = ['id','q1']
#         test_reply.columns = ['id','id_sub','q2']
#         self.test_df = pd.merge(test_query, test_reply, how="left", on="id")
#         self.train_df["q2"].fillna("好的")

    def build_examples(self, raw_data, test = False, is_match = False):           # 需要构建四个特征
        input_ids1, input_ids2 = [], []
        attn_mask1, attn_mask2 = [], []
        tokens, input_ids, attention_masks, token_type_ids = [], [], [], []
        
        # 对问答对进行截断的技巧
        def _trim_seq_pair(question_tokens, answer_tokens, max_sequence_length, q_max_len, a_max_len):
            q_len = len(question_tokens)
            a_len = len(answer_tokens)
            if q_len + a_len + 3 > max_sequence_length:
                if a_max_len <= a_len and q_max_len <= q_len:
                    # 如果Answer和Question都太长，则都必须限制到极限
                    q_new_len_head = floor((q_max_len - q_max_len/2))
                    question_tokens = question_tokens[:q_new_len_head] + question_tokens[q_new_len_head - q_max_len:]
                    a_new_len_head = floor((a_max_len - a_max_len/2))
                    answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[a_new_len_head - a_max_len:]
                elif q_len <= a_len and q_len < q_max_len:
                    # 如果答案较长且问题足够短，请将其转到答案
                    a_max_len = a_max_len + (q_max_len - q_len - 1)
                    a_new_len_head = floor((a_max_len - a_max_len/2))
                    answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[a_new_len_head - a_max_len:]
                elif a_len < q_len:
                    assert a_len <= a_max_len
                    q_max_len = q_max_len + (a_max_len - a_len - 1)
                    q_new_len_head = floor((q_max_len - q_max_len/2))
                    question_tokens = question_tokens[:q_new_len_head] + question_tokens[q_new_len_head - q_max_len:]
                else:
                    # 问题答案文本都挺长，emmm，暴力截断吧！！！
                    _truncate_seq_pair(question_tokens, answer_tokens, max_sequence_length - 3)
            return question_tokens, answer_tokens

        # 谷歌的对文本进行截断的代码
        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        def convert_text_to_ids_for_matching(text_a, text_b):
            tokens_a = self.config.tokenizer.tokenize(text_a)  
            tokens_b = self.config.tokenizer.tokenize(text_b)  
            tokens_a, tokens_b = _trim_seq_pair(tokens_a, tokens_b, self.config.max_length, self.config.max_question_len, 
                                                self.config.max_answer_len)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            text_len = len(tokens)
            input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(self.config.max_length-text_len))
            attention_mask = [1]*text_len+[0]*(self.config.max_length-text_len)
            token_type_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b)+1)+[0]*(self.config.max_length-text_len)
            
            assert len(input_ids) == self.config.max_length
            assert len(attention_mask) == self.config.max_length
            assert len(token_type_ids) == self.config.max_length

            return tokens, input_ids, attention_mask, token_type_ids

        def convert_input_to_ids(inputs1, inputs2):
            tokens1 = self.config.tokenizer.tokenize(inputs1)[:self.config.max_length]      # tokenize
            tokens2 = self.config.tokenizer.tokenize(inputs2)[:self.config.max_length]      # tokenize

            text1_len = len(tokens1)
            ids1 = self.config.tokenizer.convert_tokens_to_ids(
                tokens1 + ["[PAD]"]*(self.config.max_length - text1_len))                      # token2id
            text2_len = len(tokens2)
            ids2 = self.config.tokenizer.convert_tokens_to_ids(
                tokens2 + ["[PAD]"]*(self.config.max_length - text2_len))                      # token2id
            att_mask_1 = [1] * text1_len + [0] * (self.config.max_length - text1_len)
            att_mask_2 = [1] * text2_len + [0] * (self.config.max_length - text2_len)


            assert len(ids1) == self.config.max_length
            assert len(ids2) == self.config.max_length
            assert sum(att_mask_1) == text1_len
            assert sum(att_mask_2) == text2_len
            assert len(att_mask_1) == len(att_mask_2)
            return ids1, ids2, att_mask_1, att_mask_2

        labels = None
        if not test:
            labels = raw_data.label.tolist()
        
        # 将问答对拆分开用于Siamese网络
        if not is_match:
            for i, row in tqdm(raw_data.iterrows()):
                ids1_, ids2_, mask1_, mask2_ = convert_input_to_ids(str(row.q1), str(row.q2))
                input_ids1.append(ids1_)
                input_ids2.append(ids2_)
                attn_mask1.append(mask1_)
                attn_mask2.append(mask2_)

            if not test:
                assert len(input_ids1) == len(labels)
                assert len(input_ids2) == len(labels)

            return tuple(np.asarray(x, dtype=np.int32) for x in(input_ids1, input_ids2, attn_mask1, attn_mask2, labels) if x is not None)
        # 把问答放在一起喂到Bert中
        else:
            for i, row in tqdm(raw_data.iterrows()):
                token, input_id, attention_mask, token_type_id = convert_text_to_ids_for_matching(str(row.q1), str(row.q2))
                tokens.append(token)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
            if not test:
                assert len(tokens) == len(labels)
            return tuple(np.asarray(x, dtype=np.int32) for x in(input_ids, attention_masks, token_type_ids, labels) if x is not None)

        

    def _to(self, tensor):
        # 记录下传到gpu的张量，便于回收
        res = torch.tensor(tensor, dtype=torch.long).to(self.config.device)
        self._data.append(res)
        return res

    def _gc(self):
        # 清空并回收数据
        for x in self._data:
            x.cpu()
            torch.cuda.empty_cache()
            del x
        gc.collect()

    # 根据多折验证传过来的训练集和验证集，包装到数据集中
    def build_dataset(self, train_data, dev_data):
        logging.warning(msg="Loading data from storage, it may cost a time.")
        train = TensorDataset(*tuple(self._to(x) for x in train_data))
        dev = TensorDataset(*tuple(self._to(x) for x in dev_data))
        return train, dev

    def build_test(self,test = True, is_match = False):
        if test:
            df = self.test_df
        else:
            df = self.train_df
        test_data = self.build_examples(df, test = test, is_match = is_match)
        test = TensorDataset(*tuple(self._to(x) for x in test_data))
        return DataLoader(test, batch_size=self.config.batch_size)

    def build_iter(self, data):
        df = TensorDataset(*tuple(self._to(x) for x in data))
        return DataLoader(df, batch_size=self.config.batch_size)    

    def build_iterator(self, train_data, dev_data):
        logging.warning(msg="Building dataset...")
        train, dev = self.build_dataset(train_data, dev_data)
        return (DataLoader(x, batch_size=self.config.batch_size, shuffle=True) if x is not None else None for x in (train, dev))

    def convert_ids_to_sentences(self, sen):
        vocab = self.config.tokenizer.get_vocab()
        index2vocab = {idx:word for word,idx in vocab.items()}
        sentences = []
        for s in sen:
            se = ""
            for x in s:
                se += index2vocab[x.item()]
            sentences.append(se)
        return sentences



