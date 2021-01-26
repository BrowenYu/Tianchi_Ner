
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

text_length = 250


def mapping2res(r_ann_path, r_txt_path, w_path, w_file):
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
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END O")



import glob
import numpy as np

file_list = glob.glob('../data/round1_train/train/*.txt')


from sklearn.model_selection import train_test_split, KFold
import os


kf = KFold(n_splits=5, shuffle=True, random_state=520).split(file_list)



file_list = np.array(file_list)


val_file_list=[]
for i, (train_fold, test_fold) in enumerate(kf):
    print(len(file_list[train_fold]), len(file_list[test_fold]))
    train_filelist = list(file_list[train_fold])
    val_filelist = list(file_list[test_fold])
    val_file_list.append(val_filelist)
    os.system(f"mkdir  ../data/round1_train/train_new_{i}/")
    os.system(f"mkdir  ../data/round1_train/val_new_{i}/")
    import os
    import codecs

    data_dir = '../data/round1_train/train/'
    for file in train_filelist:
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        w_path = f'../data/round1_train/train_new_{i}/'
        w_file = file_name
        mapping2res(r_ann_path, r_txt_path, w_path, w_file)
    import os
    import codecs

    data_dir = '../data/round1_train/train/'
    for file in val_filelist:
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('/')[-1].split('.')[0]
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        w_path = f'../data/round1_train/val_new_{i}/'
        w_file = file_name
        mapping2res(r_ann_path, r_txt_path, w_path, w_file)

    w_path = f"../data/round1_train/data/train_{i}.txt"
    for file in os.listdir(f'../data/round1_train/train_new_{i}/'):
        path = os.path.join(f'../data/round1_train/train_new_{i}/', file)
        if not file.endswith(".txt"):
            continue
        q_list = []
        print("开始读取文件:%s" % file)
        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "END O":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\n\r")
        print("开始写入文本%s" % w_path)
        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

    w_path = f"../data/round1_train/data/val_{i}.txt"
    for file in os.listdir(f"../data/round1_train/val_new_{i}/"):
        path = os.path.join(f"../data/round1_train/val_new_{i}/", file)
        if not file.endswith(".txt"):
            continue
        q_list = []

        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "END O":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\n\r")

        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()


for i,item in enumerate(val_file_list):
    os.system(f"mkdir  ../data/round1_train/val_data_{i}/")
    for file in item:
        file_name = file.split('/')[-1].split('.')[0]
        r_ann_path = os.path.join("../data/round1_train/train", "%s.ann" % file_name)
        os.system("cp %s %s" % (file, f"../data/round1_train/val_data_{i}"))
        os.system("cp %s %s" % (r_ann_path, f"../data/round1_train/val_data_{i}"))
        print(file)



