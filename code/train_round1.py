#! -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras
import sys
import bert4keras
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense,Input,Softmax,Dropout,Bidirectional,LSTM,AtrousConvolution1D
from keras.models import Model
from keras.layers.core import Lambda
from keras import backend as K
from tqdm import tqdm

maxlen = 250
epochs = 2
batch_size = 16
bert_layers = 12
learing_rate = 2e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 900  # 必要时扩大CRF层的学习率

# bert配置
config_path = './publish/bert_config.json'
checkpoint_path = './publish/bert_model.ckpt'
dict_path = './publish/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    print(c)
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 标注数据
# train_data = load_data('./round1_train/data/train.txt')
# valid_data = load_data('./round1_train/data/val.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射

labels = ['SYMPTOM',
          'DRUG_EFFICACY',
          'PERSON_GROUP',
          'SYNDROME',
          'DRUG_TASTE',
          'DISEASE',
          'DRUG_DOSAGE',
          'DRUG_INGREDIENT',
          'FOOD_GROUP',
          'DISEASE_GROUP',
          'DRUG',
          'FOOD',
          'DRUG_GROUP']

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

# print(train_data[0])

# print(id2label)
# print(label2id)

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []




# 验证集

def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence



def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def test_predict(data, NER_,model_):
    test_ner = []

    for text in tqdm(data):
        cut_text_list, cut_index_list = cut_test_set([text], maxlen)
        posit = 0
        item_ner = []
        index = 1
        for str_ in cut_text_list:
            aaaa = NER_.recognize(str_,model_)
            for tn in aaaa:
                ans = {}
                ans["label_type"] = tn[1]
                ans['overlap'] = "T" + str(index)

                ans["start_pos"] = text.find(tn[0], posit)
                ans["end_pos"] = ans["start_pos"] + len(tn[0])
                posit = ans["end_pos"]
                ans["res"] = tn[0]
                item_ner.append(ans)
                index += 1
        test_ner.append(item_ner)

    return test_ner



def bertmodel():
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    # bert_hidden_layers=[model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in range(bert_layers-1,6,-1)]
    #
    # bert_hidden_layers=Lambda(lambda z:tf.convert_to_tensor(z))(bert_hidden_layers)
    #
    # print(bert_hidden_layers.shape)
    #
    # attention_vector=Dense(1,name='attention_vec')(bert_hidden_layers)
    # print(attention_vector.shape)
    # attention_probs=Softmax()(attention_vector)
    # print(attention_probs.shape)
    # attention_mul = Lambda(lambda x:x[0]*x[1])([attention_probs,bert_hidden_layers])
    # # output=keras.layers.multiply([bert_hidden_layers,attention_probs])
    # print(attention_mul.shape)
    # output=Lambda(lambda z: tf.reduce_sum(z, axis=0, keepdims=False))(attention_mul)

    # output=model.get_layer('Transformer-%s-FeedForward-Norm' % (bert_layers-1)).output


    bert_hidden_layers = [model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in
                          range(bert_layers - 1, 6, -1)]
    weight_b = Lambda(lambda x: x * (1 / 5))
    output = [weight_b(item) for item in bert_hidden_layers]

    output = keras.layers.add(output)
    # output = AtrousConvolution1D(32, 3, atrous_rate=2, border_mode='same')(output)
    # output = Bidirectional(LSTM(256, dropout_W=0.2, dropout_U=0.3, return_sequences=True))(output)
    output = Dropout(0.4)(output)
    output = Dense(num_labels)(output) # 27分类

    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)
#     model.summary()

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learing_rate),
        metrics=[CRF.sparse_accuracy]
    )
    return model,CRF



def load_pretrained_model(i):

    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    # bert_hidden_layers=[model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in range(bert_layers-1,6,-1)]
    #
    # bert_hidden_layers=Lambda(lambda z:tf.convert_to_tensor(z))(bert_hidden_layers)
    #
    # print(bert_hidden_layers.shape)
    #
    # attention_vector=Dense(1,name='attention_vec')(bert_hidden_layers)
    # print(attention_vector.shape)
    # attention_probs=Softmax()(attention_vector)
    # print(attention_probs.shape)
    # attention_mul = Lambda(lambda x:x[0]*x[1])([attention_probs,bert_hidden_layers])
    # # output=keras.layers.multiply([bert_hidden_layers,attention_probs])
    # print(attention_mul.shape)
    # output=Lambda(lambda z: tf.reduce_sum(z, axis=0, keepdims=False))(attention_mul)


    # output = model.get_layer('Transformer-%s-FeedForward-Norm' % (bert_layers - 1)).output


    bert_hidden_layers = [model.get_layer('Transformer-%s-FeedForward-Norm' % i).output for i in
                          range(bert_layers - 1,6, -1)]
    weight_b = Lambda(lambda x: x * (1 / 5))
    output = [weight_b(item) for item in bert_hidden_layers]

    output = keras.layers.add(output)
    # output = Bidirectional(LSTM(32, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(output)
    # output = Dropout(0.1)(output)
    output = Dropout(0.4)(output)
    output = Dense(num_labels)(output) # 27分类

    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)
#     model.summary()
    model.load_weights(f'./best_model_{i}.weights')
    # model.compile(
    #     loss=CRF.sparse_loss,
    #     optimizer=Adam(learing_rate),
    #     metrics=[CRF.sparse_accuracy]
    # )
    print('model loaded')
    return model,CRF





class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text, model_):
        # print(text)
        tokens = tokenizer.tokenize(text)
        # print(tokens)
        mapping = tokenizer.rematch(text, tokens)
        # print(mapping)
        token_ids = tokenizer.tokens_to_ids(tokens)
        # print(token_ids)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model_.predict([token_ids, segment_ids])[0]
        # print(type(nodes))
        # print(nodes.shape)
        # print(nodes)
        labels = self.decode(nodes)
        # print(labels)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        # print([(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
        #         for w, l in entities])
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


class NamedEntityRecognizer_Test(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text, model_):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes=sum([item.predict([token_ids, segment_ids])[0]*wf1[wl_1.index(weight_list[index])] for index,item in enumerate(model_)])/np.sum(wf1)
        # nodes = model_.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


# NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

def evaluate(data,model_):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text,model_))  # 预测
        T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_data,model_,cv_num):
        self.best_val_f1 = 0
        self.valid_data = valid_data
        self.model_=model_
        self.cv_num=cv_num

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        #         print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data,self.model_)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(f'./best_model_{self.cv_num}.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


best_f1,best_cv=0,0
weight_list=[]
def val(i, NER_,model_):
    import glob
    import codecs
    global best_f1,best_cv,weight_list
    X, Y, Z = 1e-10, 1e-10, 1e-10
    val_data_flist = glob.glob(f'./round1_train/val_data_{i}/*.txt')
    data_dir = f'./round1_train/val_data_{i}/'
    # num=0
    for file in val_data_flist:
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)

        R = []
        with codecs.open(r_txt_path, "r", encoding="utf-8") as f:

            line = f.readlines()
            # if num==0:
            #     print(line)
            #     num+=1
            aa = test_predict(line, NER_,model_)
            for line in aa[0]:
                lines = line['label_type'] + " " + str(line['start_pos']) + ' ' + str(line['end_pos']) + "\t" + line[
                    'res']
                R.append(lines)
        T = []
        with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
            for line in f:
                lines = line.strip('\n').split('\t')[1] + '\t' + line.strip('\n').split('\t')[2]
                T.append(lines)
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall)
    if f1>best_f1:
        best_f1=f1
        best_cv=i
    weight_list.append(f1)
    print('*' * 100)
    print('f1={},  precision={},  recall={}'.format(f1, precision, recall))
    print('*' * 100)




data_aug=load_data('./round1_train/data/pseudoAug.txt')

train_list=[]
val_list=[]


from sklearn.model_selection import train_test_split, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=520).split(data_aug)

def split_data(train_fold,val_fold,data):
    train_kf,val_kf=[],[]
    for index in train_fold:
        train_kf.append(data[index])
    for index in val_fold:
        val_kf.append(data[index])
    return train_kf,val_kf

for i, (train_fold, test_fold) in enumerate(kf):
    train_kf,val_kf=split_data(train_fold,test_fold,data_aug)
    train_list.append(train_kf)
    val_list.append(val_kf)

Model_list=[]

for i in range(5):
    print(i)
    train_data = load_data(f'./round1_train/data/train_{i}.txt')
    valid_data = load_data(f'./round1_train/data/val_{i}.txt')
    train_data=train_data+train_list[i]
    # valid_data = valid_data+val_list[i]

    # train_data=train_data+train_list[i]
    # valid_data=valid_data+val_list[i]
    model, CRF = bertmodel()
    NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

    evaluator = Evaluator(valid_data,model,i)
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    val(i,NER,model)
    K.clear_session()

# for i, (train_fold, test_fold) in enumerate(kf):
#     print(i)
#     print(train_fold)
#     train_data = data_list[train_fold]
#     valid_data = data_list[test_fold]
#     model, CRF = bertmodel()
#     NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
#
#     evaluator = Evaluator(valid_data,model,i)
#     train_generator = data_generator(train_data, batch_size)
#
#     model.fit_generator(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )
#     val(i,NER,model)
#     K.clear_session()


import math

lamb = 1 / 5


def weight_f1(t):
    '''
    # 牛顿冷却定律
    '''
    return math.exp(-lamb * t)


wf1 = [weight_f1(t) for t in range(len(weight_list))]
wl_1=sorted(weight_list,reverse=True)

print('weight_list:',weight_list)
print('wf1:',wf1)
print('wl_1:',wl_1)





# 测试集



trans_list=[]
model_list=[]
for i in range(5):
    print(f'第{i+1}个模型')
    model, CRF = load_pretrained_model(i)
    model_list.append(model)
    trans_list.append(CRF.trans*wf1[wl_1.index(weight_list[i])])
    # print(K.eval(CRF.trans))
# print('CRF.trans/5:')
# print(K.eval(sum(trans_list)/5))
NER_Test = NamedEntityRecognizer_Test(trans=K.eval(sum(trans_list)/np.sum(wf1)), starts=[0], ends=[0])


import os
import codecs

test_files = os.listdir("./data/round1_test/chusai_xuanshou/")

for file in test_files:
    with codecs.open("./data/round1_test/chusai_xuanshou/" + file, "r", encoding="utf-8") as f:
        line = f.readlines()
        aa = test_predict(line, NER_Test,model_list)
    if not os.path.exists("./prediction_result/"):
        os.mkdir("./prediction_result/")
    with codecs.open("./prediction_result/" + file.split('.')[0] + ".ann", "w", encoding="utf-8") as ff:
        for line in aa[0]:
            lines = line['overlap'] + "\t" + line['label_type'] + " " + str(line['start_pos']) + ' ' + str(
                line['end_pos']) + "\t" + line['res']
            ff.write(lines + "\n")
        ff.close()

os.system("zip -r -q -o ./result.zip ./prediction_result/")

