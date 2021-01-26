import json
import os
import re

import jieba
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
from torchcrf import CRF
from flat_modules import get_embedding

from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, BertPreTrainedModel, \
    AdamW, get_linear_schedule_with_warmup

from fastNLP_module import StaticEmbedding
from flat import Lattice_Transformer_SeqLabel
from vocabulary import Vocabulary

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

batch_size = 1
max_embedding_len = 256
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1
cuda_number = 'cuda'
staticVocab = Vocabulary().load('./tokenizer/vocab.txt')

useFlat = True


def padding(word):
    word = word + [0] * (256 - len(word))
    return word


class Bert_Flat_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_Flat_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)
        self.StaticEmbedding = StaticEmbedding(staticVocab, model_dir_or_name=None, embedding_dim=768)
        self.Lattice_Transformer_SeqLabel = Lattice_Transformer_SeqLabel(dvc=cuda_number)

    def forward(self, input_ids, attn_masks, labels, staticOutputs, lex_num, seq_num, pos_s, pos_e,
                ):  # dont confuse this with _forward_alg above.

        bertOutputs = self.bert(input_ids, attn_masks)
        sequence_output = bertOutputs[0]
        sequence_output = self.dropout(sequence_output)

        lattice_output = self.Lattice_Transformer_SeqLabel(staticOutputs, sequence_output, seq_num, lex_num, pos_s,
                                                           pos_e)

        emission = self.classifier(lattice_output)
        # print(emission.shape)
        attn_masks = attn_masks.type(torch.uint8)
        # print(attn_masks.shape)

        if labels is not None:
            loss = -self.crf(emission, labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(emission, labels, mask=attn_masks, reduction='mean')
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


def get_head_tail_pos(tokenizer_word, sentence):
    n = 0
    position = []
    # print(f'tokenizer_word:{tokenizer_word}')
    for num, word in enumerate(tokenizer_word):
        while n < len(sentence):
            # print(f'sentence[n:n + len(word)]:{sentence[n:n + len(word)]},word:{word}')
            if sentence[n:n + len(word)] == word:
                position.append([n + 1, n + len(word)])

                if num < len(tokenizer_word) - 1:
                    if tokenizer_word[num] in tokenizer_word[num + 1]:
                        n += tokenizer_word[num + 1].index(tokenizer_word[num])
                    elif tokenizer_word[num + 1] in tokenizer_word[num]:
                        n += tokenizer_word[num].index(tokenizer_word[num + 1])
                    elif set(tokenizer_word[num]) & set(tokenizer_word[num + 1]):
                        # insect = ''.join(set(tokenizer_word[num]) & set(tokenizer_word[num + 1]))
                        # if tokenizer_word[num + 1][0] == insect and tokenizer_word[num][0] != insect:
                        #     inde = tokenizer_word[num].index(insect)
                        #     n += inde
                        # else:
                        #     n += len(word)
                        insect = ''.join(sorted(list(set(tokenizer_word[num]) & set(tokenizer_word[num + 1])),
                                                key=lambda x: tokenizer_word[num + 1].index(x)))
                        inde = tokenizer_word[num].index(insect[0])
                        n += inde
                    else:
                        n += len(word)
                break
            else:
                n += 1
    return position


class NER_Dataset(data.Dataset):
    def __init__(self, label2idx, sentences, isFlat=False):
        self.label2idx = label2idx
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained("./pretrained_model/vocab.txt")
        self.isFlat = isFlat

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = [0]
        bert_tokens = [102]
        orig_to_tok_map = []
        # bert_tokens.append(102)
        sent = []
        for w, l in sentence:
            if self.isFlat:
                sent += self.tokenizer.tokenize(w)
            w_token_ids = self.tokenizer.encode(w)[1:-1]
            if len(bert_tokens) + len(w_token_ids) < 180:
                bert_tokens += w_token_ids
                if l == 'O':
                    labels += [0] * len(w_token_ids)
                else:
                    B = self.label2idx[l] * 2 + 1
                    I = self.label2idx[l] * 2 + 2
                    labels += ([B] + [I] * (len(w_token_ids) - 1))
            else:
                break

        # static
        if self.isFlat:
            sent = sent[:len(bert_tokens) - 1]
            sent_str = ''.join(sent)
            # print(sent)
            # print(len(sent))
            # print(len(bert_tokens))
            tokenizer_word = list(
                filter(lambda x: len(x) > 1,
                       [re.sub(r'[^\u4e00-\u9fa5]', '', i) for i in jieba.cut(sent_str, cut_all=True)]))
            # print(f'sent_str:{sent_str}')
            # print(f'tokenizer_word:{tokenizer_word}')

            pos_s = [i for i in range(len(bert_tokens)+1)]
            pos_e = [i for i in range(len(bert_tokens)+1)]

            tokenizer_len = len(tokenizer_word)
            # print(f'tokenizer_len:{tokenizer_len}')
            word_pos = get_head_tail_pos(tokenizer_word, sent_str)
            tokenizer_word = ['<pad>'] * len(bert_tokens) + tokenizer_word

            tokenizer_word = tokenizer_word + ['<pad>'] * (max_embedding_len - len(tokenizer_word))

            # print(f'word_pos:{word_pos}')
            for (s, e) in word_pos:
                pos_s.append(s)
                pos_e.append(e)
            # print(f'pos_s:{pos_s}')
            # print(f'pos_e:{pos_e}')
            pos_s += [0] * (max_embedding_len - len(pos_s))
            pos_e += [0] * (max_embedding_len - len(pos_e))

        # bert
        bert_tokens += [103]
        seg_num = len(bert_tokens)
        labels += [0]
        bert_tokens = bert_tokens + [0] * (max_embedding_len - len(bert_tokens))
        labels = labels + [0] * (max_embedding_len - len(labels))
        # bert_tokens=pad_sequences(bert_tokens, maxlen=256,
        #                       dtype="long", truncating="post", padding="post")
        # labels=pad_sequences(labels, maxlen=256,
        #                       dtype="long", truncating="post", padding="post")



        atten_mask = [float(i > 0) for i in bert_tokens]

        if self.isFlat:
            return torch.LongTensor(bert_tokens), torch.LongTensor(atten_mask), torch.LongTensor(
                labels), tokenizer_word, torch.LongTensor(pos_s), torch.LongTensor(pos_e), tokenizer_len, seg_num
        else:
            return torch.LongTensor(bert_tokens), torch.LongTensor(atten_mask), torch.LongTensor(labels)

        # modified_labels = [self.tag2idx['X']]
        # for i, token in enumerate(sentence):
        #     if len(bert_tokens) >= 256:
        #         break
        #     orig_to_tok_map.append(len(bert_tokens))
        #     modified_labels.append(label[i])
        #     new_token = self.tokenizer.tokenize(token)
        #     bert_tokens.extend(new_token)
        #     modified_labels.extend([self.tag2idx['X']] * (len(new_token) -1))
        #
        # bert_tokens.append('[SEP]')
        # modified_labels.append(self.tag2idx['X'])
        # token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        # if len(token_ids) > 511:
        #     token_ids = token_ids[:256]
        #     modified_labels = modified_labels[:256]
        # return token_ids, len(token_ids), orig_to_tok_map, modified_labels, self.sentences[idx]


tokenizer = BertTokenizer.from_pretrained("./pretrained_model/vocab.txt")
train_dataset = NER_Dataset(label2id, load_data('./round1_train/data/train.txt'), useFlat)
dev_dataset = NER_Dataset(label2id, load_data('./round1_train/data/val.txt'), useFlat)
val_data = load_data('./round1_train/data/val.txt')

train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
validation_dataloader = data.DataLoader(dataset=dev_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=1)


def predict(text, model):
    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask_ids = [1] * len(token_ids)
    # print('token_ids:',token_ids)
    token_ids = torch.LongTensor([token_ids]).cuda(cuda_number)
    mask_ids = torch.LongTensor([mask_ids]).cuda(cuda_number)
    if useFlat:
        pass
    else:
        labels = model(token_ids, mask_ids)[0]
    # print('labels',labels)
    # print(labels)
    # print(len(labels))
    entities, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([tokens[i], id2label[(label - 1) // 2]])
            elif starting:
                entities[-1][0] += tokens[i]
            else:
                starting = False
        else:
            starting = False
    # print([(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
    #         for w, l in entities])
    # print('entities',entities)
    return [(w, l) for w, l in entities]


def evaluate(data, model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        # print('text',text)
        R = set(predict(text, model))  # 预测
        T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall_1 = X / Y, X / Z
    f1 = 2 * precision * recall_1 / (precision + recall_1)
    return f1, precision, recall_1


# sent='补气养血、调经止带，用于月经不调、经期腹痛,非处方药物'
#
# token_sent = tokenizer.encode(sent)
# print(tokenizer.tokenize(sent))
# print(len(tokenizer.tokenize(sent)))
# print(token_sent)
# print(len(token_sent))
# mask_sent=[1]*len(token_sent)
# token_sent=torch.LongTensor([token_sent])
# mask_sent=torch.LongTensor([mask_sent])

modelConfig = BertConfig.from_pretrained('./pretrained_model/config.json', num_labels=num_labels)

if useFlat:
    model = Bert_Flat_CRF.from_pretrained('./pretrained_model/pytorch_model.bin', config=modelConfig)
else:
    model = Bert_CRF.from_pretrained('./pretrained_model/pytorch_model.bin', config=modelConfig)

# print(model)
#
# print(model(token_sent,mask_sent))
#
# a=input()
model.cuda(cuda_number)
# print(model)
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )
# Number of training epochs (authors recommend between 2 and 4)
epochs = 10

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


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
    device = torch.device(cuda_number)

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
best_f1 = 0
# For each epoch...
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

        # `batch` contains seven pytorch tensors int Flat:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        #   [3]: staticEmbedding
        #   [4]: pos_s
        #   [5]: pos_e
        #   [6]: len(tokenizer_word)
        #   [7]: len(sentence)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        if useFlat:
            b_sent = batch[3]
            b_pos_s = batch[4].to(device)
            b_pos_e = batch[5].to(device)
            b_lex_num = batch[6]
            b_seq_num = batch[7]
            embed = StaticEmbedding(staticVocab, model_dir_or_name=None, embedding_dim=768)
            word_idxs = []

            for i in range(batch_size):
                idx = padding([staticVocab.to_index(b_sent[j][i]) for j in range(max_embedding_len)])
                word_idxs.append(idx)
            word_Embedding = embed(torch.LongTensor(word_idxs).cuda())
        # print(b_input_ids)
        # print(b_input_mask)
        # print(b_labels[0][0])
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # 测试position-embedding
        # pe = get_embedding(b_seq_num, torch., rel_pos_init=1)
        # print(f'pe:{pe}')
        # break

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        if useFlat:
            outputs = model(b_input_ids, b_input_mask, b_labels, word_Embedding, b_seq_num,b_lex_numm, b_pos_s, b_pos_e)
        else:
            outputs = model(b_input_ids, b_input_mask, b_labels)

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

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    f1, precision, recall = evaluate(val_data, model)
    if f1 > best_f1:
        best_f1 = f1
    print(
        'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
        (f1, precision, recall, best_f1)
    )

# %%

# modelConfig=BertConfig.from_pretrained('./pretrained_model/config.json',num_labels=num_labels)
# roberta=BertModel.from_pretrained('./pretrained_model/pytorch_model.bin',config=modelConfig)

# outputs=roberta(torch.tensor(token_sent).unsqueeze(0))
#
# print(outputs[0].shape)
# print(outputs[0][0,0].shape)
# print(token_sent)
# print(torch.tensor(token_sent).unsqueeze(0))
# print(roberta)
