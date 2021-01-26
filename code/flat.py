import collections

import copy
import torch
from torch import nn

from Flat.V1.modules import Transformer_Encoder, get_embedding, seq_len_to_mask
from Flat.utils import MyDropout, get_crf_zero_init


class Lattice_Transformer_SeqLabel(nn.Module):
    def __init__(self, hidden_size=768,
                 num_heads=8, num_layers=1,
                 layer_preprocess_sequence='', layer_postprocess_sequence='an',
                 use_abs_pos=False, use_rel_pos=True,
                 ff_size=-1, scaled=True, dropout=0.5,
                 dvc=None, vocabs=None, learnable_position=False,
                 rel_pos_shared=True, max_seq_len=180, k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 self_supervised=False, attn_ff=True, pos_norm=False, ff_activate='relu', rel_pos_init=0,
                 abs_pos_fusion_func='concat', embed_dropout_pos='0',
                 four_pos_shared=True, four_pos_fusion='attn', four_pos_fusion_shared=True):
        """
        :param rel_pos_init: 如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化

        :param embed_dropout_pos: 如果是0，就直接在embed后dropout，是1就在embed变成hidden size之后再dropout，
        是2就在绝对位置加上之后dropout
        """
        super().__init__()

        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.four_pos_shared = four_pos_shared
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.add_position = False
        # self.relative_position = relative_position
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.learnable_position = learnable_position
        if self.use_rel_pos:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.rel_pos_shared = rel_pos_shared
        self.self_supervised = self_supervised
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init
        self.embed_dropout_pos = embed_dropout_pos

        if self.use_rel_pos and max_seq_len < 0:
            print('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None

        if self.use_rel_pos:
            pe = get_embedding(max_seq_len, hidden_size, rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1, keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe / pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None

        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size == -1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)

        self.dropout = nn.Dropout(dropout)

        # self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        # self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)

        self.encoder = Transformer_Encoder(self.hidden_size, self.num_heads, self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           lattice=True,
                                           four_pos_fusion=self.four_pos_fusion,
                                           four_pos_fusion_shared=self.four_pos_fusion_shared)

        if self.self_supervised:
            self.output_self_supervised = nn.Linear(self.hidden_size, len(vocabs['char']))
            print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, word_embedding, bert_embedding, seq_len, lex_num, pos_s, pos_e):

        # batch_size = lattice.size(0)
        # max_seq_len_and_lex_num = lattice.size(1)

        # raw_embed = lattice
        # raw_embed 是字和词的pretrain的embedding，但是是分别train的，所以需要区分对待

        # raw_embed_char = raw_embed
        # print('raw_embed_char_1:{}'.format(raw_embed_char[:1,:3,-5:]))

        # if self.use_bert:
        #     bert_pad_length = lattice.size(1) - max_seq_len
        #     char_for_bert = lattice[:, :max_seq_len]
        #     mask = seq_len_to_mask(seq_len).bool()
        #     char_for_bert = char_for_bert.masked_fill((~mask), self.vocabs['lattice'].padding_idx)
        #     bert_embed = self.bert_embedding(char_for_bert)
        #     bert_embed = torch.cat([bert_embed,
        #                             torch.zeros(size=[batch_size, bert_pad_length, bert_embed.size(-1)],
        #                                         device=bert_embed.device,
        #                                         requires_grad=False)], dim=-2)
        # print('bert_embed:{}'.format(bert_embed[:1, :3, -5:]))

        # embedding = torch.cat([bert_embedding,word_embedding], dim=1)
        embedding = bert_embedding + word_embedding

        # print('raw_embed_char:{}'.format(raw_embed_char[:1,:3,-5:]))

        # print('raw_embed_char_dp:{}'.format(raw_embed_char[:1,:3,-5:]))

        # embed_char = raw_embed_char
        # print('char_proj:',list(self.char_proj.parameters())[0].data[:2][:2])
        # print('embed_char_:{}'.format(embed_char[:1,:3,:4]))

        # char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()

        # embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        # embed_lex = self.lex_proj(raw_embed)

        # lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
        # embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        # assert char_mask.size(1) == lex_mask.size(1)
        # print('embed_char:{}'.format(embed_char[:1,:3,:4]))
        # print('embed_lex:{}'.format(embed_lex[:1,:3,:4]))

        # embedding = embed_char + embed_lex
        # print('seq_len, lex_num, pos_s, pos_e:')
        # print(seq_len, lex_num, pos_s, pos_e)

        encoded = self.encoder(embedding, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        encoded = self.dropout(encoded)

        encoded = encoded[:, :256, :]

        return encoded
