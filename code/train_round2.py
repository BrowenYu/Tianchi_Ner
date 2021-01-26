import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc, os
import time
import datetime
import random
from tqdm.auto import tqdm
from torch.utils import data
# from torchcrf import CRF
from mycrf import CRF
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer,BertConfig,BertModel,BertForSequenceClassification,BertPreTrainedModel,AdamW,get_linear_schedule_with_warmup



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

loss_weight=[10086,6240,2930,2118,1415,1481,1448,1468,1062,974,262,390,129]
# loss_weight=[max(loss_weight)/(i*2) for i in loss_weight]
m=nn.Softmax()

# print(m(torch.Tensor(loss_weight))[0].item())
# print(torch.empty(10))
# b=input()
import math

lamb = 1 / 9


def weight_f1(t):
    '''
    # 牛顿冷却定律
    '''
    return math.exp(-lamb * t)


wf1 = [weight_f1(t) for t in range(len(loss_weight))]
# wf1=m(torch.tensor(wf1)).numpy().tolist()
wl_1=sorted(loss_weight)

print('wf1:',wf1)

# loss_weight=[max(loss_weight)/(i*2) for i in loss_weight]
# print(loss_weight)

maxlen=256
batch_size = 16
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 4 + 1



class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks,weighted_loss,labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        # weighted_loss=weighted_loss.type(torch.uint8)
        # print('predict:',emission.size())
        # F.softmax(emission, 2)
        if labels is not None:
            # print('predict softmax:',F.softmax(emission, 2).size())
            # print('labels:',labels.size())
            # print('attention_mask',attn_masks.size())
            loss = -self.crf(emission, labels,weighted_loss, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction


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


class NER_Dataset(data.Dataset):
    def __init__(self, label2idx, sentences):
        self.label2idx = label2idx
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained("./pretrained_model/vocab.txt")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = [0]
        bert_tokens = [102]
        orig_to_tok_map = []
        # bert_tokens.append(102)
        for w,l in sentence:
            w_token_ids = self.tokenizer.encode(w)[1:-1]
            if len(bert_tokens)+len(w_token_ids)<256:
                bert_tokens+=w_token_ids
                if l == 'O':
                    labels += [0] * len(w_token_ids)
                else:
                    B = label2id[l] * 4 + 1
                    I = label2id[l] * 4 + 2
                    E = label2id[l] * 4 + 3
                    S = label2id[l] * 4 + 4
                    if len(w_token_ids) > 1:
                        if len(w_token_ids) > 2:
                            labels += ([B] + [I] * (len(w_token_ids) - 2) + [E])
                        else:
                            labels += ([B] + [E])
                    else:
                        labels += [S]
            else:
                break
        bert_tokens += [103]
        labels += [0]
        weighted_loss = [float(wf1[wl_1.index(loss_weight[(i - 1) // 4])]) if i > 0 else float(1) for i in labels]
        bert_tokens=bert_tokens+[0]*(256-len(bert_tokens))
        labels=labels+[0]*(256-len(labels))
        weighted_loss = weighted_loss+[float(0)]*(256-len(weighted_loss))
        atten_mask = [float(i > 0) for i in bert_tokens]



        return torch.LongTensor(bert_tokens),torch.LongTensor(atten_mask),torch.LongTensor(labels),torch.Tensor(weighted_loss)







tokenizer = BertTokenizer.from_pretrained("./pretrained_model/vocab.txt")


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

def predict(text,model):
    tokens = ['[CLS]']+tokenizer.tokenize(text)+['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask_ids = [1] * len(token_ids)
    weighted_loss = [1] * len(token_ids)
    token_ids=torch.LongTensor([token_ids]).cuda('cuda:0')
    mask_ids=torch.LongTensor([mask_ids]).cuda('cuda:0')
    weighted_loss = torch.LongTensor([weighted_loss]).cuda('cuda:0')
    labels = model(token_ids,mask_ids,weighted_loss)[0]
    entities, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 4 == 0 or label % 4 == 1:
                starting = True
                entities.append([tokens[i], id2label[(label - 1) // 4]])
            elif starting:
                entities[-1][0]+=tokens[i]
            else:
                starting = False
        else:
            starting = False
    return [(w,l)for w, l in entities]


def test_predict(data,model):
    test_ner = []

    for text in tqdm(data):
        cut_text_list, cut_index_list = cut_test_set([text], maxlen)
        posit = 0
        item_ner = []
        index = 1
        for str_ in cut_text_list:
            aaaa = predict(str_,model)
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



def evaluate(data,model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(predict(text,model))  # 预测
        T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall_1 = X / Y, X / Z
    f1 = 2 * precision * recall_1 / (precision + recall_1)
    return f1, precision, recall_1


def get_model(total_steps):
    modelConfig = BertConfig.from_pretrained('./pretrained_model/config.json', num_labels=num_labels)

    model = Bert_CRF.from_pretrained('./pretrained_model/pytorch_model.bin', config=modelConfig)

    model.cuda('cuda:0')
    print(model)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    return model,optimizer,scheduler

def get_data(i):
    train_dataset = NER_Dataset(label2id, load_data(f'./round2_train/data/train_{i}.txt'))
    dev_dataset = NER_Dataset(label2id, load_data(f'./round2_train/data/val_{i}.txt'))
    val_data = load_data(f'./round2_train/data/val_{i}.txt')

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)
    validation_dataloader = data.DataLoader(dataset=dev_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=1)
    return train_dataloader,validation_dataloader,val_data



def val(i,model):
    import glob
    import codecs
    X, Y, Z = 1e-10, 1e-10, 1e-10
    val_data_flist = glob.glob(f'./round2_train/val_data_{i}/*.txt')
    data_dir = f'./round2_train/val_data_{i}/'
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
            aa = test_predict(line,model)
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
    print('*' * 100)
    print('f1={},  precision={},  recall={}'.format(f1, precision, recall))
    print('*' * 100)

    return f1


epochs = 10

# Total number of training steps is number of batches * number of epochs.


# Create the learning rate scheduler.



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda:0")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set the seed value all over the place to make this reproducible.



best_model=None
model_list=[]


# Store the average loss after each epoch so we can plot them.

# For each epoch...
for i in range(10):
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    train_dataloader, validation_dataloader, val_data=get_data(i)

    total_steps = len(train_dataloader) * epochs

    model, optimizer, scheduler=get_model(total_steps)

    loss_values = []
    best_f1 = 0
    val_f1=0
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_weightedloss = batch[3].to(device)
            model.zero_grad()

            outputs = model(b_input_ids,b_input_mask,b_weightedloss,b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs
            if step % 20 == 0 and not step == 0:
                print(f'training loss of every 20 batch:{loss.item()}')

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Running Validation...")



        model.eval()

        with torch.no_grad():
            f1, precision, recall = evaluate(val_data,model)
        if f1>best_f1:
            best_f1=f1
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, best_f1)
        )
    with torch.no_grad():
        temp_f1=val(i,model)

    if temp_f1>val_f1:
        val_f1=temp_f1
        best_model=model
    torch.cuda.empty_cache()



import os
import codecs

test_files = os.listdir("/tcdata/juesai/")

for file in test_files:

    with codecs.open("/tcdata/juesai/" + file, "r", encoding="utf-8") as f:
        line = f.readlines()
        aa = test_predict(line, best_model)
    if not os.path.exists("./result/"):
        os.mkdir("./result/")
    with codecs.open("./result/" + file.split('.')[0] + ".ann", "w", encoding="utf-8") as ff:
        if len(aa)!=0:
            for line in aa[0]:
                lines = line['overlap'] + "\t" + line['label_type'] + " " + str(line['start_pos']) + ' ' + str(
                    line['end_pos']) + "\t" + line['res']
                ff.write(lines + "\n")
        else:
            ff.write("")

        ff.close()

os.system("zip -r -q -o result.zip ./result/")

# import os
# import codecs
#
# test_files = os.listdir("./round1_test/chusai_xuanshou/")
#
# for file in test_files:
#     with codecs.open("./round1_test/chusai_xuanshou/" + file, "r", encoding="utf-8") as f:
#         line = f.readlines()
#         aa = test_predict(line,best_model)
#     if not os.path.exists("./submission_1101/"):
#         os.mkdir("./submission_1101/")
#     # print(f'len aa{len(aa)}')
#     with codecs.open("./submission_1101/" + file.split('.')[0] + ".ann", "w", encoding="utf-8") as ff:
#         for line in aa[0]:
#             lines = line['overlap'] + "\t" + line['label_type'] + " " + str(line['start_pos']) + ' ' + str(
#                 line['end_pos']) + "\t" + line['res']
#             ff.write(lines + "\n")
#         ff.close()
#






