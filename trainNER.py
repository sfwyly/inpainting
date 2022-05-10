# encoding=utf-8

import tensorflow as tf


def crf_decode(potentials, transition_params, sequence_length):
    """解码TensorFlow中标记的最高评分序列。
    这是张量的函数。
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    # 如果max_seq_len为1，则跳过算法，只返回argmax标记和max activation。
    def _single_seq_fn():
        squeezed_potentials = tf.squeeze(potentials, [1])
        decode_tags = tf.expand_dims(tf.argmax(squeezed_potentials, axis=1), 1)
        best_score = tf.reduce_max(squeezed_potentials, axis=1)
        return tf.cast(decode_tags, dtype=tf.int32), best_score

    def _multi_seq_fn():
        """最高得分序列的解码。"""
        # Computes forward decoding. Get last score and backpointers.
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_length.dtype), sequence_length - 1)

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one)

        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1)

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)
        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(
            decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        return decode_tags, best_score

    if potentials.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_decode_forward(inputs, state, transition_params, sequence_lengths):
    """计算线性链CRF中的正向解码。
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True)
    return crf_fwd_layer(inputs, state, mask=mask)


def crf_decode_backward(inputs, state):
    """计算线性链CRF中的反向解码。
    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
    Returns:
      new_tags: A [batch_size, num_tags] tensor containing the new tag indices.
    """
    inputs = tf.transpose(inputs, [1, 0, 2])

    def _scan_fn(state, inputs):
        state = tf.squeeze(state, axis=[1])
        idxs = tf.stack([tf.range(tf.shape(inputs)[0]), state], axis=1)
        new_tags = tf.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
        return new_tags

    return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """计算线性链CRF中的正向解码。"""

    def __init__(self, transition_params, **kwargs):
        """初始化CrfDecodeForwardRnnCell。
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. 这个矩阵将被扩展为[1, num_tags, num_tags]
            以在下面的cell中进行广播求和。
        """
        super(CrfDecodeForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfDecodeForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        """构建CrfDecodeForwardRnnCell。
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
    """计算CRF中标记序列的对数似然。
    通过crf_sequence_score计算状态序列可能性分数，通过crf_log_norm计算归一化项。
    最后返回log_likelihood对数似然。
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the log-likelihood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix,
          if available.
    Returns:
      log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is
          either provided by the caller or created in this function.
    """
    num_tags = inputs.shape[2]

    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    if transition_params is None:
        initializer = tf.keras.initializers.GlorotUniform()
        transition_params = tf.Variable(
            initializer([num_tags, num_tags]), "transitions")

    sequence_scores = crf_sequence_score(
        inputs, tag_indices, sequence_lengths, transition_params)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """计算标记序列的非标准化分数。
    通过crf_unary_score计算状态特征分数，通过crf_binary_score计算转移特征分数。
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    # 如果max_seq_len为1，则跳过分数计算，只收集单个标记的unary potentials。
    def _single_seq_fn():
        batch_size = tf.shape(inputs, out_type=tag_indices.dtype)[0]

        example_inds = tf.reshape(
            tf.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
        sequence_scores = tf.gather_nd(
            tf.squeeze(inputs, [1]),
            tf.concat([example_inds, tag_indices], axis=1))
        sequence_scores = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(sequence_scores),
            sequence_scores)
        return sequence_scores

    def _multi_seq_fn():
        # 计算给定标记序列的分数。
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(
            tag_indices, sequence_lengths, transition_params)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """计算标记序列的状态特征分数。
    利用掩码的方式，计算得出一个类似交叉熵的值。
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    assert len(tag_indices.shape) == 2, "tag_indices: A [batch_size, max_seq_len] matrix of tag indices."
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    # 根据标记索引的数据类型使用int32或int64。
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """计算标记序列的转移特征分数。
    通过转移矩阵返回转移特征分数。
    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    # 在序列的每一侧截断一个，得到每个转换的开始和结束索引。
    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # 将索引编码为扁平表示。
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    # 基于扁平化表示得到转移特征分数。
    binary_scores = tf.gather(
        flattened_transition_params, flattened_transition_indices)

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """计算CRF的标准化。
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    # 分割第一个和其余的输入，为正向算法做准备。
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    def _multi_seq_fn():
        """α值的正向计算。"""
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        # 计算前向算法中的alpha值以得到分割函数。

        alphas = crf_forward(
            rest_of_input, first_input, transition_params, sequence_lengths)
        log_norm = tf.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_forward(inputs, state, transition_params, sequence_lengths):
    """计算线性链CRF中的alpha值。
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
         values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
    Returns:
      new_alphas: A [batch_size, num_tags] matrix containing the
          new alpha values.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    sequence_lengths = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 2)
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(state, inputs):
        state = tf.expand_dims(state, 2)
        transition_scores = state + transition_params
        new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    idxs = tf.stack(
        [tf.range(tf.shape(sequence_lengths)[0]), sequence_lengths], axis=1)
    return tf.gather_nd(all_alphas, idxs)

class CRF(tf.keras.layers.Layer):
    """
    条件随机场层 (tf.keras)
    CRF可以用作网络的最后一层（作为分类器使用）。
    输入形状（特征）必须等于CRF可以预测的类数（建议在线性层后接CRF层）。
    Args:
        num_classes (int): 标签（类别）的数量。
    Input shape:
        (batch_size, sentence length, num_classes)。
    Output shape:
        (batch_size, sentence length, num_classes)。
    """

    def __init__(self, num_classes, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transitions = None
        self.output_dim = int(num_classes)  # 输出维度
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = False

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "supports_masking": self.supports_masking,
            "transitions": tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError(
                "The last dimension of the inputs to `CRF` "
                "should be defined. Found `None`.")
        if f_shape[-1] != self.output_dim:
            raise ValueError(
                "The last dimension of the input shape must be equal to output"
                " shape. Use a linear layer if needed.")
        self.input_spec = input_spec
        self.transitions = self.add_weight(
            name="transitions",
            shape=[self.output_dim, self.output_dim],
            initializer="glorot_uniform",
            trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # 只需将接收到的mask从上一层传递到下一层，或者在该层更改输入的形状时对其进行操作
        return mask

    def call(self, inputs, sequence_lengths=None, mask=None, training=None,
             **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == "int32"
            seq_len_shape = tf.convert_to_tensor(
                sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            sequence_lengths = tf.keras.backend.flatten(sequence_lengths)
        else:
            sequence_lengths = tf.math.count_nonzero(mask, axis=1)

        viterbi_sequence, _ = crf_decode(
            sequences, self.transitions, sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = crf_decode(
                y_pred, self.transitions, sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = "viterbi_accuracy"
        return accuracy


class CRFLoss(object):
    """CRF损失函数。"""
    def __init__(self, crf: CRF, dtype) -> None:
        super().__init__()
        self.crf = crf
        self.dtype = dtype

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        assert sample_weight is not None, "your model has to support masking"
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        sequence_lengths = tf.math.count_nonzero(sample_weight, axis=1)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.crf.transitions = crf_log_likelihood(
            y_pred,
            tf.cast(y_true, dtype=tf.int32),
            sequence_lengths,
            transition_params=self.crf.transitions)
        return tf.reduce_mean(-log_likelihood)
        

class BiLSTMCRF(tf.keras.Model):
    """BiLSTM+CRF模型。"""
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(BiLSTMCRF, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        # layers
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)
        self.crf = CRF(label_size)

    # @tf.function
    def call(self, text, training=None):
        inputs = self.embedding(text)       # [B, seq_len, embed_size]
        inputs = self.dropout(inputs, training)     # [B, seq_len, embed_size]
        logits = self.dense(self.biLSTM(inputs))    # [B, seq_len, label_size]
        viterbi_output = self.crf(logits)   # [B, seq_len, label_size]

        return viterbi_output
 
model = BiLSTMCRF(128,2976,10,256)
model.compile(
        loss=CRFLoss(model.crf, model.dtype),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=[model.crf.viterbi_accuracy],
        run_eagerly=True)
model.build((None,100))
model.summary()

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
# from keras_contrib.layers import CRF

catagories = 10
maxlen = 100

char_map_dict={}
index_map_dict={}
label_map_dict ={}

#一共10类 + other
entity_map_labels ={
	"疾病":"dis","临床表现":"sym","医疗程序":"pro","医疗设备":"equ","药物":"dru","医学检验项目":"ite","身体":"bod","科室":"dep","微生物类":"mic","其他":"o"
}


def getTrainDataList(trainName,testName):
    
    with open(trainName,"r",encoding="utf-8") as file:
        train_data_list = [line for line in file.readlines() if line.strip()]
    
    with open(testName,"r",encoding="utf-8") as file:
        test_data_list = [line for line in file.readlines() if line.strip()]
        
    return train_data_list,test_data_list

# 输入
def getWordDict(lineList):
    
    word_dict = {" ":3000} #第0个
    # word_dict = {}
    maxL = 0
    senL = ""
    i = 0
    j = 0
    for line in lineList :
        i  += 1
        sen = line.split("|||")[0]
        if(len(sen)>maxL):
            maxL = len(sen)
            senL = sen
            j = i
        for _ in range(len(sen)):
            word = sen[_]
            if(word in word_dict): #word in dict
                word_dict[word] +=1
            else:
                word_dict[word] =1
    print("最大长",maxL,"  ",j)
    # 按照 字顺序进行排序
    sorted_char=sorted(word_dict.items(),key=lambda e:e[1],reverse=True)
    char_map_dict = {}
    for  i in range(len(sorted_char)):
        char_map_dict[sorted_char[i][0]] = i #通过大小顺序构建字的索引
    return char_map_dict

char_map_dict={}
label_map_dict ={"oth":0,"dis":1,"sym":2,"pro":3,"equ":4,"dru":5,"ite":6,"bod":7,"dep":8,"mic":9,
0:"oth",1:"dis",2:"sym",3:"pro",4:"equ",5:"dru",6:"ite",7:"bod",8:"dep",9:"mic"}

#一共10类 + other
entity_map_labels ={
    "疾病":"dis","临床表现":"sym","医疗程序":"pro","医疗设备":"equ","药物":"dru","医学检验项目":"ite","身体":"bod","科室":"dep","微生物类":"mic","其他":"o"
}

trainLines,testLines = getTrainDataList("E:/competion/NER/biendata/train_data.txt","E:/competion/NER/biendata/val_data.txt")
char_map_dict = getWordDict(trainLines+testLines)
index_map_dict = {__:_ for _,__ in char_map_dict.items()}

print(len(char_map_dict))
def senToVec(char_list):
    return [char_map_dict.get(char_list[_],0) for _ in range(len(char_list))]
#处理标签列表  标签数组  2 - 9 label

# label 映射成索引
# sym 嵌套问题
def labelToVec(labelsNum):
    #初始化其他
    label_list = [0] * len(labelsNum[0])
    for labelNum in labelsNum[1:]:
        if(labelNum == None or labelNum.strip()==""):
            continue
        strs = labelNum.split("    ")
        start = int(strs[0])
        end =int(strs[1])
        flag = strs[2]
        i = start
        while ( i>=0 and i<=end and i<len(labelsNum[0])):
            label_list[i] = label_map_dict[str(flag)] #从分类标签映射为索引
            # print(label_list[i])
            i = i+1
    return label_list

# 构建
def build(lineList):
    train = []
    labels = []
    lineList = sorted(lineList,key=lambda line:len((line.split("|||"))[0]),reverse=True)

    for line in lineList:
        strs = line.strip().split("|||")
        if(len(strs[0])<30):
            continue
        #处理句子
        char_list_vec = senToVec(strs[0])
        train.append(char_list_vec)
        
        #处理标签
        label_list_vec = labelToVec(strs)
        labels.append(label_list_vec)
    return np.array(train),np.array(labels)

#构建batch时，将句子从长到短进行排序

X_train,X_labels = build(trainLines)

X_train = sequence.pad_sequences(X_train,maxlen = maxlen,value = 0)
X_labels = sequence.pad_sequences(X_labels,maxlen = maxlen,value = 0)
X_labels = tf.keras.utils.to_categorical(X_labels,10)

# X_labels = np.expand_dims(X_labels, 2)

y_train,y_labels = build(testLines)
y_train = sequence.pad_sequences(y_train,maxlen = maxlen,value = 0)#,padding = "post"
y_labels = sequence.pad_sequences(y_labels,maxlen = maxlen,value = -1)
y_labels = tf.keras.utils.to_categorical(y_labels,10)

# y_labels = np.eye(10)[y_labels]
# y_labels = np.expand_dims(y_labels, 2)

print("训练集： ",np.array(X_train).shape,np.array(X_labels).shape)
print("测试集： ",np.array(y_train).shape,np.array(y_labels).shape)

def getResult(lines): #输入字符串
    # inputs = []
    # for line in lines:
    #     line = senToVec(line) #字符串转成的向量
    #     print(line)
    #     inputs.append(line)
    # inputs = X_train[0]
    inputs = lines
    print(inputs.shape)
    result = model.predict(inputs)
    result = np.argmax(result,axis =-1)
    # new_result=[]
    #遍历batch
    # for line_result in result:
    #     line_= []
    #     for _ in range(len(line_result)):
    #         # print(line_result[_])
    #         line_.append(label_map_dict[line_result[_]])
    #     new_result.append(line_)
    return result
history=model.fit(X_train,X_labels,batch_size=5,epochs=1 ,validation_data=[y_train,y_labels],verbose = 1)

model.save("/usr/jjj/NER_Bi_LSTM_CRF.h5")    