from nlpcda import Ner



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

labels_v=['DRUG',
          'FOOD',
          'DRUG_GROUP',
          'FOOD_GROUP',
          'DISEASE_GROUP',
          'DRUG_INGREDIENT']




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




# 设置样本长度
text_length = 250


def from_ann2dic(r_ann_path, r_txt_path, w_path, w_file):
    q_dic = {}
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            line_arr = line.split('\t')
            entityinfo = line_arr[1]
            entityinfo = entityinfo.split(' ')
            cls = entityinfo[0]
            start_index = int(entityinfo[1])
            end_index = int(entityinfo[2])
            length = end_index - start_index
            for r in range(length):
                if r == 0:
                    q_dic[start_index] = ("B-%s" % cls)
                else:
                    q_dic[start_index + r] = ("I-%s" % cls)

    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    cut_text_list, cut_index_list = cut_test_set([content_str], text_length)

    i = 0
    for idx, line in enumerate(cut_text_list):
        w_path_ = "%s/%s-%s-new.txt" % (w_path, w_file, idx)

        with codecs.open(w_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s\t%s\n' % (str_, tag))
                i += 1




import glob
import numpy as np

file_list = glob.glob('./round1_train/train_pseudo/*.txt')



import os
import codecs
os.system("mkdir  ./round1_train/train_pseudo_new/")
data_dir = './round1_train/train_pseudo/'
for file in file_list:
    if file.find(".ann") == -1 and file.find(".txt") == -1:
        continue
    file_name = file.split('/')[-1].split('.')[0]
    print(file_name)
    r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
    r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
    print(r_ann_path,r_txt_path)
    w_path = './round1_train/train_pseudo_new/'
    w_file = file_name
    from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)




ner = Ner(ner_dir_name='./round1_train/train_pseudo_new/',
          ignore_tag_list=['O'],
          data_augument_tag_list=labels,
          augument_size=3,
          seed=0)
num=0
with open('./round1_train/data/pseudoAug.txt', 'a') as f:
    for file in os.listdir('./round1_train/train_pseudo_new/'):
        path = os.path.join('./round1_train/train_pseudo_new/', file)

        data_sentence_arrs, data_label_arrs = ner.augment(file_name=path)
        if len(data_sentence_arrs) == 0:
            continue
        num=num+len(data_sentence_arrs)
        for sent, label in zip(data_sentence_arrs, data_label_arrs):
            for i, j in zip(sent, label):
                print(i, j)
                f.writelines(i + ' ' + j + '\n')
            f.writelines('\n')
# 3条增强后的句子、标签 数据，len(data_sentence_arrs)==3
# 你可以写文件输出函数，用于写出，作为后续训练等
# for i in range(len(data_sentence_arrs)):
#
#     print(data_sentence_arrs[i])
#     print(data_label_arrs[i])
print(num)



# for i, j in zip(data_sentence_arrs[0], data_label_arrs[0]):
#     print(i, j)
#
# "".join(data_sentence_arrs[0])
