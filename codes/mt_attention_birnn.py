#!/usr/bin/env python
# coding: utf-8
import warnings
from codes.utils import *
warnings.filterwarnings("ignore")
import os
import numpy as np
import random
import tqdm
from codes.utils import *
from codes.attention_decoder import *


def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    Encoding the text into integers.

    @param sentence: 完整的句子，str类型
    @param map_dict: 单词到数字编码的映射
    @param max_length: 最大句子长度
    @param is_target: 当前传入的句子是否是目标语句。
                      对于目标语句，我们要在末尾添加"<EOS>"
    """

    text_to_idx = []
    # 特殊单词的数字编码
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")
    go_idx = map_dict.get("<GO>")

    # 如果不是目标语句（即源语句）
    if not is_target:
        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 目标语句要对结尾添加"<EOS>",
    else:

        for word in sentence.split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 超长句子进行截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 不足长度的句子进行"<PAD>"
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx


class SeqToSeq:
    def __init__(self, input_shape, output_shape, embedding_dim, units=40):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.embedding_dim = embedding_dim
        self.units = units

    def build_model(self):
        X_source = keras.layers.Input(batch_shape=self.input_shape)
        embedding_X = keras.layers.Embedding(input_dim=len(source_int_to_vocab) + 1, output_dim=self.embedding_dim,
                                             embeddings_initializer='RandomNormal')(X_source)
        # Encoder
        # BiGRU input layers, back+forward
        x_hidden = keras.layers.Bidirectional(keras.layers.LSTM(return_sequences=True, units=self.units))(
            embedding_X)  # (?, 20,80) hidden state

        """
        # Decoder
        # 实际上, 这个方法适用于 每一步都有输入的rnn结构里， 但我们现在没有使用teacher force， 所以没办法使用这个方式
        # 无论如何， 至少学习到了怎么修改内部cell的结构，和dot的小技巧
        att_decoder = AttentionDecoderCell(units=self.units, hidden_state_encoder=x_hidden)

        outputs = keras.layers.RNN(att_decoder, return_sequences=True)(x_hidden)
        """

        # 那么现在就简单多了, 使用迭代方法
        def attention_acquire(encoder_hidden, decoder_state):
            assert len(encoder_hidden.shape) == 3, 'encoder hidden state should be 3D'
            assert len(decoder_state.shape) == 2, 'every decoder state should be 2D'
            time_steps = encoder_hidden.get_shape().as_list()[1]
            decoder_state = keras.backend.repeat(decoder_state, n=time_steps)  # shape(?, time_step, dims)
            energy_t = keras.layers.Dense(1) \
                (keras.backend.concatenate([encoder_hidden, decoder_state]))  # shape(?, time_step, 1)
            at = keras.backend.exp(energy_t)
            at_sum = keras.backend.sum(at, axis=1)
            at_sum_repeated = keras.backend.repeat(at_sum, encoder_hidden.get_shape().as_list()[1])
            at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)
            return at

        s0 = keras.backend.zeros_like(x_hidden[:, 0])
        y0 = keras.backend.zeros(shape=self.output_shape)
        s = s0
        y = y0
        outputs = []
        for i in range(self.output_shape[1]):
            att_ = attention_acquire(encoder_hidden=x_hidden, decoder_state=s)
            c = keras.backend.batch_dot(att_, x_hidden, axes=1)
            s1 = keras.layers.Dense(s.get_shape().as_list()[-1])(keras.backend.concatenate([y, s, c]),
                                                                 activation='softmax')
            y = keras.layers.Dense(s.get_shape().as_list()[-1])(keras.backend.concatenate([y, s, c]),
                                                                activation='softmax')
            outputs.append(y)
            s = s1

        keras.Model.compile([X_source], outputs)


if __name__ == '__main__':
    # English source data
    with open(os.path.join(DATA_DIR, "small_vocab_en"), "r", encoding="utf-8") as f:
        source_text = f.read()

    # French target data
    with open(os.path.join(DATA_DIR, "small_vocab_fr"), "r", encoding="utf-8") as f:
        target_text = f.read()

    # 构造英文词典
    source_vocab = list(set(source_text.lower().split()))
    # 构造法语词典
    target_vocab = list(set(target_text.lower().split()))

    # 增加特殊编码
    SOURCE_CODES = ['<PAD>', '<UNK>']
    TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

    # 构造英文语料的映射表
    source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
    source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

    # 构造法语语料的映射表
    target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
    target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

    # 对英文语料进行编码，其中设置英文句子最大长度为20
    Tx = 20
    source_text_to_int = []

    for sentence in tqdm.tqdm(source_text.split("\n")):
        source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, Tx, is_target=False))

    # 对法语语料进行编码，其中设置法语句子最大长度为25
    Ty = 25
    target_text_to_int = []

    for sentence in tqdm.tqdm(target_text.split("\n")):
        target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, Ty, is_target=True))

    # **After encoding the source and target text into numbers, we need to do one-hot-encoding of them**

    X = np.array(source_text_to_int)
    Y = np.array(target_text_to_int)
