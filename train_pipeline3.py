# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import argparse
import logging
import os
import random
import glob
import timeit
import json
from sklearn.metrics import accuracy_score
import linecache
import faiss
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy
import re
import torch
import copy
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import (LazyQuacDatasetGlobal, RawResult, 
                   write_predictions, write_final_predictions, 
                   get_retrieval_metrics, gen_reader_features)
from retriever_utils import RetrieverDataset, RetrieverInputExample, retriever_convert_example_to_feature
from modeling import Pipeline, AlbertForRetrieverOnlyPositivePassage, BertForOrconvqaGlobal,Explorer_GAT
from scorer import quac_eval
import dgl
import pickle
from torch._six import container_abcs, string_classes, int_classes

import networkx as nx
# In[2]:
np_str_obj_array_pattern = re.compile(r'[SaUO]')

logger = logging.getLogger(__name__)

ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
}


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#######################################################yongqi
def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def collate_fn2(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn2([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn2([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn2(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn2(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

# In[4]:
def collate_fn(input_batchs): #[[return_feature_dict,graph_feature_dict],[return_feature_dict,graph_feature_dict]]
    r"""Puts each data field into a tensor with outer dimension batch size"""

    graph_feature_dict=[s[1] for s in input_batchs]


    node_set_list=[]
    edge_set_list=[]
    node_label=[]
    node_is_historyanswer=[]
    for s in graph_feature_dict:
        node_set_list.append(s['node_set_list'])
        edge_set_list.append(s['edge_set_list'])
        node_label.append(s['node_label'])
        node_is_historyanswer.append(s['node_is_historyanswer'])

    graph_batch={
            'node_set_list':node_set_list,
        'edge_set_list':edge_set_list,
        'node_is_historyanswer':node_is_historyanswer,
        'node_label':node_label
    }

    batch=[s[0] for s in input_batchs]

    batch=collate_fn2(batch)
    return [batch,graph_batch]


#######################################################yongqi



def train(args, train_dataset, model, retriever_tokenizer, reader_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
#######################################################yongqi
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers,collate_fn=collate_fn)
#######################################################yongqi

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)


    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    retriever_tr_loss, retriever_logging_loss = 0.0, 0.0
    reader_tr_loss, reader_logging_loss = 0.0, 0.0
    qa_tr_loss, qa_logging_loss = 0.0, 0.0
    rerank_tr_loss, rerank_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])

     
#######################################################yongqi
        for step,batchs in enumerate(epoch_iterator):


            batch=batchs[0]
            graph_batch=batchs[1]

#######################################################yongqi
            model.eval() # we first get query representations in eval mode
            qids = np.asarray(batch['qid']).reshape(-1).tolist()
            # print('qids', qids)
            question_texts = np.asarray(
                batch['question_text']).reshape(-1).tolist()
            # print('question_texts', question_texts)
            answer_texts = np.asarray(
                batch['answer_text']).reshape(-1).tolist()
            # print('answer_texts', answer_texts)
            answer_starts = np.asarray(
                batch['answer_start']).reshape(-1).tolist()
            # print('answer_starts', answer_starts)

# ############################################################################################################################################DQI
            model.eval() # we first get query representations in eval mode


            history_q_current_q=question_texts[0]

            history_q_current_q_list=history_q_current_q.split('[SEP]')

            current_q=history_q_current_q_list[-1]



            query_feature_dict = {'query_input_ids': [], 
                                  'query_token_type_ids': [], 
                                  'query_attention_mask': []}
            if(len(history_q_current_q_list)>=2):
                for i in range(len(history_q_current_q_list)-1):


                    query_example = RetrieverInputExample(guid=0, text_a=current_q+'[SEP]'+history_q_current_q_list[i]+'[SEP]'+answer_texts[0])
                    query_feature = retriever_convert_example_to_feature(query_example, retriever_tokenizer, 
                                                                         max_length=250)

                    query_feature_dict['query_input_ids'].append(np.asarray(query_feature.input_ids))
                    query_feature_dict['query_token_type_ids'].append(np.asarray(query_feature.token_type_ids))
                    query_feature_dict['query_attention_mask'].append(np.asarray(query_feature.attention_mask))
            else:
                    query_example = RetrieverInputExample(guid=0, text_a=current_q+'[SEP]'+answer_texts[0])
                    query_feature = retriever_convert_example_to_feature(query_example, retriever_tokenizer, 
                                                                         max_length=250)

                    query_feature_dict['query_input_ids'].append(np.asarray(query_feature.input_ids))
                    query_feature_dict['query_token_type_ids'].append(np.asarray(query_feature.token_type_ids))
                    query_feature_dict['query_attention_mask'].append(np.asarray(query_feature.attention_mask))                

            query_feature_dict['query_input_ids']=torch.from_numpy(np.array(query_feature_dict['query_input_ids']))
            query_feature_dict['query_token_type_ids']=torch.from_numpy(np.array(query_feature_dict['query_token_type_ids']))
            query_feature_dict['query_attention_mask']=torch.from_numpy(np.array(query_feature_dict['query_attention_mask']))
            
            revelence_passage=np.array([0])


            query_reps = gen_query_reps(args, model, query_feature_dict, revelence_passage=revelence_passage)



            retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                         passage_ids, passage_id_to_idx, passage_reps,
                                         qrels, qrels_sparse_matrix,
                                         gpu_index, include_positive_passage=True)

            passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
            labels_for_retriever = retrieval_results['labels_for_retriever']
            pids_for_reader = retrieval_results['pids_for_reader'] #Batch_size*topk

            passages_for_reader = retrieval_results['passages_for_reader']
            labels_for_reader = retrieval_results['labels_for_reader']#Batch_size*topk


            model.train()
    


            inputs = {'query_input_ids': query_feature_dict['query_input_ids'].to(args.device),       ######################这个地方一定要换！！！！！！！！！！！！！
                      'query_attention_mask': query_feature_dict['query_attention_mask'].to(args.device),
                      'query_token_type_ids': query_feature_dict['query_token_type_ids'].to(args.device),
                      'passage_rep': torch.from_numpy(passage_reps_for_retriever).to(args.device),
                      'retrieval_label': torch.from_numpy(labels_for_retriever).to(args.device),
                      'revelence_passage':torch.from_numpy(revelence_passage).to(args.device)}
            retriever_outputs = model.retriever(**inputs)
            # model outputs are always tuple in transformers (see doc)
            retriever_loss = retriever_outputs[0]

#######################################################Results from ConvDR


#             pids_for_reader=[]
#             labels_for_reader=[]
#             for qid in qids:
#                 pids_for_reader.append(DR_Result[qid][0][:args.top_k_for_reader])

#                 label_for_reader=[]
#                 for pid in DR_Result[qid][0][:args.top_k_for_reader]:
#                     if pid in set(question_answer[qid]):
#                         label_for_reader.append(1)
#                     else:
#                         label_for_reader.append(0)
#                 labels_for_reader.append(label_for_reader)


# #######################################################fuse Results from ConvDR and DQI


# # ############################################################################################################################################DQI



# #######################################################yongqi  DGR

#             model.train()
#             query_reps = query_reps.detach().cpu().numpy()


#             temporal_graph_batch={}
#             temporal_graph_batch['node_set_list']=[]
#             temporal_graph_batch['edge_set_list']=[]
#             temporal_graph_batch['node_label']=[]
#             for i in range(len(graph_batch['node_set_list'])):
#                 answer_set=set()
# #                print('qid',qid)
#                 for j in question_answer[qids[i]]:
#                     answer_set.add(j) 

#                 #将tf-idf加入seed node
#                 node_set={}
#                 temporal_node_set={}
#                 node_label=[]
#                 for j in range(len(graph_batch['node_set_list'][i])):
#                     node_set[graph_batch['node_set_list'][i][j]]=j
#                     temporal_node_set[graph_batch['node_set_list'][i][j]]=1
#                     node_label.append(graph_batch['node_label'][i][j])


#                 #将history answers加入seed node
#                 history_answers_dict={}
#                 for j in history_answer[qids[i]]:
#                     history_answers_dict[j]=len(history_answers_dict)
#                     if j not in node_set:
#                         node_set[j]=len(node_set)
#                         temporal_node_set[j]=1

#                         if j not in answer_set:
#                             node_label.append(0)
#                         else:
#                             node_label.append(1)


#                 edge_set=set()

#                 #将retriever的node加入seed node
#                 for j in range(len(list(pids_for_reader[i]))):
#                     s=pids_for_reader[i][j]
#                     if s not in node_set:
#                         node_set[s]=len(node_set)
#                         temporal_node_set[s]=1

#                         node_label.append(labels_for_reader[i][j])

#                 # add one hop of history answer nodes to nodes set, add edge to nodes set
#                 for j in temporal_node_set:
#                     if j in block_link_dict:
#                         for k in block_link_dict[j][0]:
#                             if k not in node_set:
#                                 node_set[k]=len(node_set)

#                                 if k not in answer_set:
#                                     node_label.append(0)
#                                 else:
#                                     node_label.append(1)

#                             edge_set.add((j,k))
#                         for k in block_link_dict[j][1]:
#                             if k not in node_set:
#                                 node_set[k]=len(node_set)

#                                 if k not in answer_set:
#                                     node_label.append(0)
#                                 else:
#                                     node_label.append(1)

#                             edge_set.add((j,k))
#                 for j in node_set:
#                     edge_set.add((j,j))

#                 node_set_list=[]
#                 for key in node_set:
#                     node_set_list.append(key)


#                 edge_set_list=[[],[]]

#                 for key in edge_set:
#                     edge_set_list[0].append(node_set[key[0]])
#                     edge_set_list[1].append(node_set[key[1]])

#                 temporal_graph_batch['node_set_list'].append(node_set_list)
#                 temporal_graph_batch['edge_set_list'].append(edge_set_list)
#                 temporal_graph_batch['node_label'].append(node_label)




#             node_set_list=temporal_graph_batch['node_set_list']
#             query_signal=[]#储存每个图节点数目，也是node和哪个query匹配的信号
#             for i in range(len(node_set_list)):
#                 for j in range(len(node_set_list[i])):
#                         query_signal.append(i)


#             graph_query_embedding=query_reps[np.array(query_signal)] #[node_num,pro_size]


#             node_set_list=list(flat(node_set_list))
#             node_embedding=passage_reps[np.array([passage_id_to_idx[s] for s in node_set_list])]


#             history_turn_index_list=[]
#             node_set_list_dict={}
#             for s in node_set_list:
#                 node_set_list_dict[s]=len(node_set_list_dict)

#                 if(s in history_answers_dict):
#                     history_turn_index_list.append(history_answers_dict[s])
#                 else:
#                     history_turn_index_list.append(19)


#             history_answer_sequences=[]
#             for node in node_set_list_dict:
#                 history_answer_sequence=[]
#                 for key in history_answer[qids[i]]:
#                     history_answer_sequence.append(node_set_list_dict[key])
#                 history_answer_sequence.append(node_set_list_dict[node])
#                 history_answer_sequences.append(history_answer_sequence)




#             edge_set_list=temporal_graph_batch['edge_set_list']
#             graphs=[]
#             for s in edge_set_list:
#                 sorce_node,target_node=s[0],s[1]
#                 graphs.append(dgl.graph((torch.from_numpy(np.asarray(sorce_node)), torch.from_numpy(np.asarray(target_node)))))
#             g=dgl.batch(graphs)





#             is_answer=np.asarray([]).reshape([-1])




#             inputs = {
#                       'node_embedding': torch.from_numpy(node_embedding).to(args.device),
#                       'history_turn_index':torch.from_numpy(np.asarray(history_turn_index_list)).to(args.device),
#                       'history_answer_sequence':torch.from_numpy(np.asarray(history_answer_sequences)).to(args.device),
#                       'g': g.to(args.device),
#                       'label': torch.from_numpy(np.asarray(list(flat(temporal_graph_batch['node_label']))).reshape([-1])).to(args.device),
#                       'is_answer': torch.from_numpy(is_answer).to(args.device),
#                       'args':args,
#                       'is_training':True,
#                       'query_input_ids': batch['query_input_ids'].to(args.device),
#                       'query_attention_mask': batch['query_attention_mask'].to(args.device),
#                       'query_token_type_ids': batch['query_token_type_ids'].to(args.device)}

#             Explorer_GAT_logits,Explorer_GAT_logits_loss= model.explorer(**inputs)
#             Explorer_GAT_logits=Explorer_GAT_logits.detach().cpu().numpy()

#             pids_for_reader_from_explorer=[]
#             labels_for_reader_from_explorer=[]


#     # ######################不考虑explorer和retriever的重叠

#             top_from_explorer=5
#             index=0
#             for i in range(len(temporal_graph_batch['node_set_list'])):

#                 s=temporal_graph_batch['node_set_list'][i]



#                 temporal_logits=Explorer_GAT_logits[index:index+len(s)]



#                 pids_index=np.argsort(-temporal_logits)[:top_from_explorer].tolist()







#                 for j in pids_index:

#                         pids_for_reader_from_explorer.append(s[j])
#                         labels_for_reader_from_explorer.append(node_label[index+j])

#                 index+=len(s)







#             pidx_for_reader_from_explorer=np.array([passage_id_to_idx[s] for s in pids_for_reader_from_explorer]).reshape([-1,top_from_explorer])
#             pids_for_reader_from_explorer=np.array(pids_for_reader_from_explorer).reshape([-1,top_from_explorer])
#             labels_for_reader_from_explorer=np.array(labels_for_reader_from_explorer).reshape([-1,top_from_explorer])


#             passages_for_reader_from_explorer=get_passages(pidx_for_reader_from_explorer, args)




#             pids_for_reader=pids_for_reader_from_explorer
#             passages_for_reader=passages_for_reader_from_explorer
#             labels_for_reader=labels_for_reader_from_explorer


############################################################################################################Yongqi Reader
            # reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
            #                             pids_for_reader, passages_for_reader, labels_for_reader,
            #                             reader_tokenizer, args.reader_max_seq_length, is_training=True)

            # reader_batch = {k: v.to(args.device) for k, v in reader_batch.items()}
            # inputs = {'input_ids':       reader_batch['input_ids'],
            #           'attention_mask':  reader_batch['input_mask'],
            #           'token_type_ids':  reader_batch['segment_ids'],
            #           'start_positions': reader_batch['start_position'],
            #           'end_positions':   reader_batch['end_position'],
            #           'retrieval_label': reader_batch['retrieval_label']}
            # reader_outputs = model.reader(**inputs)
            # reader_loss, qa_loss, rerank_loss = reader_outputs[0:3]
            # loss_list.append(reader_loss)
######################yongqi

            reader_loss=0.0,
            qa_loss=0.0,
            rerank_loss=0.0
            # loss =  Explorer_GAT_logits_loss
            loss =  retriever_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                retriever_loss = retriever_loss.mean()
                reader_loss = reader_loss.mean()
                qa_loss = qa_loss.mean()
                rerank_loss = rerank_loss.mean()
                Explorer_GAT_logits_loss=Explorer_GAT_logits_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                retriever_loss = retriever_loss / args.gradient_accumulation_steps
                reader_loss = reader_loss / args.gradient_accumulation_steps
                qa_loss = qa_loss / args.gradient_accumulation_steps
                rerank_loss = rerank_loss / args.gradient_accumulation_steps
                Explorer_GAT_logits_loss=Explorer_GAT_logits_loss/args.gradient_accumulation_steps
#######################yongqi
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            print('loss',loss.item())
            # tr_loss += loss.item()
            # retriever_tr_loss += retriever_loss.item()
            # reader_tr_loss += reader_loss.item()
            # qa_tr_loss += qa_loss.item()
            # rerank_tr_loss += rerank_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'retriever_loss', (retriever_tr_loss - retriever_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'reader_loss', (reader_tr_loss - reader_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'qa_loss', (qa_tr_loss - qa_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rerank_loss', (rerank_tr_loss - rerank_logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                    retriever_logging_loss = retriever_tr_loss
                    reader_logging_loss = reader_tr_loss
                    qa_logging_loss = qa_tr_loss
                    rerank_logging_loss = rerank_tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    retriever_model_dir = os.path.join(output_dir, 'retriever')
                    reader_model_dir = os.path.join(output_dir, 'reader')
                    explorer_model_dir = os.path.join(output_dir, 'explorer/')
                    if not os.path.exists(retriever_model_dir):
                        os.makedirs(retriever_model_dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if not os.path.exists(reader_model_dir):
                        os.makedirs(reader_model_dir)
                    if not os.path.exists(explorer_model_dir):
                        os.makedirs(explorer_model_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    retriever_model_to_save = model_to_save.retriever
                    retriever_model_to_save.save_pretrained(
                        retriever_model_dir)
                    reader_model_to_save = model_to_save.reader
                    reader_model_to_save.save_pretrained(reader_model_dir)

####################################yongqi
                    explorer_model_to_save = model_to_save.explorer
                    torch.save(explorer_model_to_save.state_dict(), explorer_model_dir+"explorer.pt")

##############################################yongqi

                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))

                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
##########################################################
#        scheduler.step()
        result = evaluate(args, model, retriever_tokenizer,
                  reader_tokenizer, prefix='test') 

####################################################
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
        orig_eval_file = args.orig_test_file
    else:
        eval_file = args.dev_file
        orig_eval_file = args.orig_dev_file
    pytrec_eval_evaluator = evaluator

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, retriever_tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.retriever_query_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False, 
                           include_first_for_retriever=args.include_first_for_retriever,
                           prepend_history_answers=args.prepend_history_answers)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers,collate_fn=collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    retriever_run_dict, rarank_run_dict = {}, {}
    examples, features = {}, {}
    all_results = []
    start_time = timeit.default_timer()

    fenzi=0
    fenmu=0
    node_num=0
    num_loo2=0
    mrr_list=[]
    recall_list=[]


    retrieval_run={}

    for batchs in tqdm(eval_dataloader, desc="Evaluating"):
        batch=batchs[0]
        graph_batch=batchs[1]
        model.eval()
        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        # print(qids)
        question_texts = np.asarray(
            batch['question_text']).reshape(-1).tolist()
        answer_texts = np.asarray(
            batch['answer_text']).reshape(-1).tolist()
        answer_starts = np.asarray(
            batch['answer_start']).reshape(-1).tolist()



#########################################################################DQI
        history_q_current_q=question_texts[0]

        history_q_current_q_list=history_q_current_q.split('[SEP]')

        current_q=history_q_current_q_list[-1]



        query_feature_dict = {'query_input_ids': [], 
                              'query_token_type_ids': [], 
                              'query_attention_mask': []}
        if(len(history_q_current_q_list)>=2):
            for i in range(len(history_q_current_q_list)-1):


                query_example = RetrieverInputExample(guid=0, text_a=current_q+'[SEP]'+history_q_current_q_list[i]+'[SEP]'+predicted_question_answerspans[qids[0]][0])
                query_feature = retriever_convert_example_to_feature(query_example, retriever_tokenizer, 
                                                                     max_length=250)

                query_feature_dict['query_input_ids'].append(np.asarray(query_feature.input_ids))
                query_feature_dict['query_token_type_ids'].append(np.asarray(query_feature.token_type_ids))
                query_feature_dict['query_attention_mask'].append(np.asarray(query_feature.attention_mask))
        else:
                query_example = RetrieverInputExample(guid=0, text_a=current_q+'[SEP]'+predicted_question_answerspans[qids[0]][0])
                query_feature = retriever_convert_example_to_feature(query_example, retriever_tokenizer, 
                                                                     max_length=250)

                query_feature_dict['query_input_ids'].append(np.asarray(query_feature.input_ids))
                query_feature_dict['query_token_type_ids'].append(np.asarray(query_feature.token_type_ids))
                query_feature_dict['query_attention_mask'].append(np.asarray(query_feature.attention_mask))                

        query_feature_dict['query_input_ids']=torch.from_numpy(np.array(query_feature_dict['query_input_ids']))
        query_feature_dict['query_token_type_ids']=torch.from_numpy(np.array(query_feature_dict['query_token_type_ids']))
        query_feature_dict['query_attention_mask']=torch.from_numpy(np.array(query_feature_dict['query_attention_mask']))
        
        revelence_passage=np.array([0])


        query_reps = gen_query_reps(args, model, query_feature_dict, revelence_passage=revelence_passage)



        retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                     passage_ids, passage_id_to_idx, passage_reps,
                                     qrels, qrels_sparse_matrix,
                                     gpu_index, include_positive_passage=True)

        passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
        labels_for_retriever = retrieval_results['labels_for_retriever']
        pids_for_reader1 = retrieval_results['pids_for_reader'] #Batch_size*topk

        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader1 = retrieval_results['labels_for_reader']#Batch_size*topk









############################################################ConvDR results

        pids_for_reader=[]
        labels_for_reader=[]
        for qid in qids:
            if(qid in DR_Result):
                pids_for_reader.append(DR_Result[qid][0][:args.top_k_for_reader])
                label_for_reader=[]
                for pid in DR_Result[qid][0][:args.top_k_for_reader]:
                    if pid in set(question_answer[qid]):
                        label_for_reader.append(1)
                    else:
                        label_for_reader.append(0)
                labels_for_reader.append(label_for_reader)
            else:
                pids_for_reader.append(DR_Result["C_0aaa843df0bd467b96e5a496fc0b033d_1_q#0"][0][:args.top_k_for_reader])

                label_for_reader=[]
                for pid in DR_Result["C_0aaa843df0bd467b96e5a496fc0b033d_1_q#0"][0][:args.top_k_for_reader]:
                    if pid in set(question_answer["C_0aaa843df0bd467b96e5a496fc0b033d_1_q#0"]):
                        label_for_reader.append(1)
                    else:
                        label_for_reader.append(0)
                labels_for_reader.append(label_for_reader)
                print('error qid')

#######################################fuse ConvDR and DQI
        new_pids_for_reader=[]
        new_labels_for_reader=[]
        for i in range(args.eval_batch_size):
            new_pids_for_reader.append(pids_for_reader[i].extend(list(pids_for_reader1[i])))
            new_labels_for_reader.append(labels_for_reader[i].extend(list(labels_for_reader1[i])))

        
# #######################################################yongqi DGR
        query_reps = query_reps.detach().cpu().numpy()


        temporal_graph_batch={}
        temporal_graph_batch['node_set_list']=[]
        temporal_graph_batch['edge_set_list']=[]
        temporal_graph_batch['node_label']=[]
        for i in range(len(graph_batch['node_set_list'])):
            answer_set=set()
#                print('qid',qid)
            for j in question_answer[qids[i]]:
                answer_set.add(j) 

            #将tf-idf加入seed node
            node_set={}
            temporal_node_set={}
            node_label=[]
            for j in range(len(graph_batch['node_set_list'][i])):
                node_set[graph_batch['node_set_list'][i][j]]=j
                temporal_node_set[graph_batch['node_set_list'][i][j]]=1
                node_label.append(graph_batch['node_label'][i][j])


            #将true history answers加入seed node
            history_answers_dict={}
            for j in history_answer[qids[i]]:
                history_answers_dict[j]=len(history_answers_dict)
                # if j not in node_set:
                #     node_set[j]=len(node_set)
                #     temporal_node_set[j]=1

                #     if j not in answer_set:
                #         node_label.append(0)
                #     else:
                #         node_label.append(1)

            #将predicted history answers加入seed node
            history_answers_dict={}
            for j in history_predicted_passages[qids[i]]:
                history_answers_dict[j]=len(history_answers_dict)
                if j not in node_set:
                    node_set[j]=len(node_set)
                    temporal_node_set[j]=1

                    if j not in answer_set:
                        node_label.append(0)
                    else:
                        node_label.append(1)
            edge_set=set()

            #将retriever的node加入seed node
            for j in range(len(list(pids_for_reader[i]))):
                s=pids_for_reader[i][j]
                if s not in node_set:
                    node_set[s]=len(node_set)
                    temporal_node_set[s]=1

                    node_label.append(labels_for_reader[i][j])

#            add one hop of history answer nodes to nodes set, add edge to nodes set
            for j in temporal_node_set:
                if j in block_link_dict:
                    for k in block_link_dict[j][0]:
                        if k not in node_set:
                            node_set[k]=len(node_set)

                            if k not in answer_set:
                                node_label.append(0)
                            else:
                                node_label.append(1)

                        edge_set.add((j,k))
                    for k in block_link_dict[j][1]:
                        if k not in node_set:
                            node_set[k]=len(node_set)

                            if k not in answer_set:
                                node_label.append(0)
                            else:
                                node_label.append(1)

                        edge_set.add((j,k))
            for j in node_set:
                edge_set.add((j,j))

            node_set_list=[]
            for key in node_set:
                node_set_list.append(key)


            edge_set_list=[[],[]]

            for key in edge_set:
                edge_set_list[0].append(node_set[key[0]])
                edge_set_list[1].append(node_set[key[1]])

            temporal_graph_batch['node_set_list'].append(node_set_list)
            temporal_graph_batch['edge_set_list'].append(edge_set_list)
            temporal_graph_batch['node_label'].append(node_label)




        node_set_list=temporal_graph_batch['node_set_list']
        query_signal=[]#储存每个图节点数目，也是node和哪个query匹配的信号
        for i in range(len(node_set_list)):
            for j in range(len(node_set_list[i])):
                    query_signal.append(i)


        graph_query_embedding=query_reps[np.array(query_signal)] #[node_num,pro_size]


        node_set_list=list(flat(node_set_list))
        node_embedding=passage_reps[np.array([passage_id_to_idx[s] for s in node_set_list])]


        history_turn_index_list=[]
        node_set_list_dict={}
        for s in node_set_list:
            node_set_list_dict[s]=len(node_set_list_dict)

            if(s in history_answers_dict):
                history_turn_index_list.append(history_answers_dict[s])
            else:
                history_turn_index_list.append(19)

        history_answer_sequences=[]

        #  ture history answers
        # for node in node_set_list_dict:
        #     history_answer_sequence=[]
        #     for key in history_answer[qids[i]]:
        #         history_answer_sequence.append(node_set_list_dict[key])
        #     history_answer_sequence.append(node_set_list_dict[node])
        #     history_answer_sequences.append(history_answer_sequence)


        G = nx.Graph()

        for s in temporal_graph_batch['edge_set_list']:
            for i in range(len(s[0])):
                G.add_edge(s[0][i], s[1][i], weight=1)

    
        #  predicted history answers
        for node in node_set_list_dict:

            history_answer_sequence=[]
            for key in history_predicted_passages[qids[0]]:
                history_answer_sequence.append(node_set_list_dict[key])

            final_history_answer_sequence=[]

            if(len(history_answer_sequence)>1):
                final_history_answer_sequence.append(history_answer_sequence[0])
                for i in range(1,len(history_answer_sequence)):
                    try:
                        path=nx.shortest_path(G,source=history_answer_sequence[i-1],target=history_answer_sequence[i])

                    except:
                        continue

                    for j in range(1,len(path)-1):
                        final_history_answer_sequence.append(path[j])

                    final_history_answer_sequence.append(history_answer_sequence[i])
            elif(len(history_answer_sequence)==1):
                final_history_answer_sequence.append(history_answer_sequence[0])


            final_history_answer_sequence.append(node_set_list_dict[node])
            history_answer_sequences.append(final_history_answer_sequence)

        # print('history_answer_sequence',history_answer_sequence)


        edge_set_list=temporal_graph_batch['edge_set_list']
        graphs=[]
        for s in edge_set_list:
            sorce_node,target_node=s[0],s[1]
            graphs.append(dgl.graph((torch.from_numpy(np.asarray(sorce_node)), torch.from_numpy(np.asarray(target_node)))))
        g=dgl.batch(graphs)





        is_answer=np.asarray([]).reshape([-1])




        


        inputs = {
                  'node_embedding': torch.from_numpy(node_embedding).to(args.device),
                  'history_turn_index':torch.from_numpy(np.asarray(history_turn_index_list)).to(args.device),
                  'history_answer_sequence':torch.from_numpy(np.asarray(history_answer_sequences)).to(args.device),
                  'g': g.to(args.device),
                  'label': torch.from_numpy(np.asarray(list(flat(temporal_graph_batch['node_label']))).reshape([-1])).to(args.device),
                  'is_answer': torch.from_numpy(is_answer).to(args.device),
                  'args':args,
                  'is_training':False,
                  'query_input_ids': batch['query_input_ids'].to(args.device),
                  'query_attention_mask': batch['query_attention_mask'].to(args.device),
                  'query_token_type_ids': batch['query_token_type_ids'].to(args.device)}

        Explorer_GAT_logits,Explorer_GAT_logits_loss= model.explorer(**inputs)


        Explorer_GAT_logits=Explorer_GAT_logits.detach().cpu().numpy()


        pids_for_reader_from_explorer=[]
        labels_for_reader_from_explorer=[]
        probs_for_reader_from_explorer=[]


# ######################不考虑explorer和retriever的重叠

        top_from_explorer=args.top_k_from_explorer
        if(top_from_explorer>len(node_set_list)):
            top_from_explorer=len(node_set_list)

        index=0
        for i in range(len(temporal_graph_batch['node_set_list'])):

            s=temporal_graph_batch['node_set_list'][i]



            temporal_logits=Explorer_GAT_logits[index:index+len(s)]



            pids_index=np.argsort(-temporal_logits)[:top_from_explorer].tolist()





            for j in pids_index:

                    pids_for_reader_from_explorer.append(s[j])
                    labels_for_reader_from_explorer.append(node_label[index+j])
                    probs_for_reader_from_explorer.append(temporal_logits[j])
            index+=len(s)







        pidx_for_reader_from_explorer=np.array([passage_id_to_idx[s] for s in pids_for_reader_from_explorer]).reshape([-1,top_from_explorer])
        pids_for_reader_from_explorer=np.array(pids_for_reader_from_explorer).reshape([-1,top_from_explorer])
        labels_for_reader_from_explorer=np.array(labels_for_reader_from_explorer).reshape([-1,top_from_explorer])
        probs_for_reader_from_explorer=np.array(probs_for_reader_from_explorer).reshape([-1,top_from_explorer])

        passages_for_reader_from_explorer=get_passages(pidx_for_reader_from_explorer, args)


        retrieval_run[qids[0]]={}
        for i in range(len(pids_for_reader_from_explorer.reshape(-1).tolist())):
            retrieval_run[qids[0]][pids_for_reader_from_explorer.reshape(-1).tolist()[i]]=probs_for_reader_from_explorer.reshape(-1).tolist()[i]




        pids_for_reader=pids_for_reader_from_explorer
        passages_for_reader=passages_for_reader_from_explorer
        labels_for_reader=labels_for_reader_from_explorer
        retriever_probs=sp.special.softmax(probs_for_reader_from_explorer, axis=1)

          

#########################################################################################################################Yongqi Reader
        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts, pids_for_reader,
                                                                           passages_for_reader, labels_for_reader,
                                                                           reader_tokenizer,
                                                                           args.reader_max_seq_length,
                                                                           is_training=False)
        example_ids = reader_batch['example_id']
        # print('example_ids', example_ids)
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(args.device)
                        for k, v in reader_batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {'input_ids':      reader_batch['input_ids'],
                      'attention_mask': reader_batch['input_mask'],
                      'token_type_ids': reader_batch['segment_ids']}
            outputs = model.reader(**inputs)
        
        retriever_probs = retriever_probs.reshape(-1).tolist()
        # print('retriever_probs after', retriever_probs)
        for i, example_id in enumerate(example_ids):



            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]), 

                               retriever_prob=retriever_probs[i])
            all_results.append(result)


    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    output_prediction_file = os.path.join(
        predict_dir, "instance_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        predict_dir, "instance_nbest_predictions_{}.json".format(prefix))
    output_final_prediction_file = os.path.join(
        predict_dir, "final_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            predict_dir, "instance_null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                        args.version_2_with_negative, args.null_score_diff_threshold)
    write_final_predictions(args, all_predictions, output_final_prediction_file, 
                            use_rerank_prob=args.use_rerank_prob, 
                            use_retriever_prob=args.use_retriever_prob)
    eval_metrics = quac_eval(
        orig_eval_file, output_final_prediction_file)
    rerank_metrics = get_retrieval_metrics(
        pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True)
    eval_metrics.update(rerank_metrics)

    metrics_file = os.path.join(
        predict_dir, "metrics_{}.json".format(prefix))
    with open(metrics_file, 'w') as fout:
        json.dump(eval_metrics, fout)

    logger.info("Test Result: {}".format(eval_metrics))

    return eval_metrics


# In[6]:


def gen_query_reps(args, model, batch,revelence_passage=None):
    model.eval()
    batch = {k: v.to(args.device) for k, v in batch.items() 
             if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start','node_set_list','edge_set_list','node_is_historyanswer','node_label']}
    with torch.no_grad():
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        if(revelence_passage is not None):
            inputs['revelence_passage'] = torch.from_numpy(revelence_passage).to(args.device)
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


# In[7]:


def retrieve(args, qids, qid_to_idx, query_reps,
             passage_ids, passage_id_to_idx, passage_reps,
             qrels, qrels_sparse_matrix,
             gpu_index, include_positive_passage=False):
    query_reps = query_reps.detach().cpu().numpy()


    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_retriever, axis=1)
    labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
    # print('labels_for_retriever before', labels_for_retriever)

    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_retriever[i][-1] = positive_pidx
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        # print('labels_for_retriever after', labels_for_retriever)
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
    
    pids_for_retriever = passage_ids[pidx_for_retriever]



    passage_reps_for_retriever = passage_reps[pidx_for_retriever]

    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]
    # print('pidx_for_reader', pidx_for_reader)
    # print('qids', qids)
    # print('qidx', qidx)
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    # print('qidx_expanded', qidx_expanded)

    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
    # print('labels_for_reader before', labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        # print('labels_for_reader after', labels_for_reader)
        assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader after', labels_for_reader)

    pids_for_reader = passage_ids[pidx_for_reader]


    # print(pids_for_reader.shape)
    # print(pids_for_reader)
    # print('pids_for_reader', pids_for_reader)



    pidx_for_reader[pidx_for_reader<0] = 0

    passages_for_reader = get_passages(pidx_for_reader, args)

    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false

    return {'qidx': qidx,
            'pidx_for_retriever': pidx_for_retriever,
            'pids_for_retriever': pids_for_retriever,
            'passage_reps_for_retriever': passage_reps_for_retriever,
            'labels_for_retriever': labels_for_retriever,
            'retriever_probs': retriever_probs,
            'pidx_for_reader': pidx_for_reader,
            'pids_for_reader': pids_for_reader,
            'passages_for_reader': passages_for_reader, 
            'labels_for_reader': labels_for_reader}


# In[8]:


def get_passage(i, args):

    line = linecache.getline(args.blocks_path, i + 1)

    line = json.loads(line.strip())
    return line['text']
get_passages = np.vectorize(get_passage)


# In[9]:


parser = argparse.ArgumentParser()

# arguments shared by the retriever and reader

parser.add_argument("--train_file", default="/home/share/liyongqi/data_dir/CODQA/ours/ORQUAC/preprocessed2/train.txt",
                    type=str, required=False,
                    help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", default="/home/share/liyongqi/data_dir/CODQA/ours/ORQUAC/preprocessed2/dev.txt",
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", default="/home/share/liyongqi/data_dir/CODQA/ours/ORQUAC/preprocessed2/test.txt",
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_dev_file", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/quac_format/dev.txt",
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_test_file", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/quac_format/test.txt",
                    type=str, required=False,
                    help="original quac json for evaluation.")
parser.add_argument("--qrels", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/qrels.txt", type=str, required=False,
                    help="qrels to evaluate open retrieval")
# parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
#                     help="all blocks text")
parser.add_argument("--blocks_path", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/all_blocks.txt", type=str, required=False,
                    help="all blocks text")
parser.add_argument("--passage_reps_path", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/passage_reps.pkl",
                    type=str, required=False, help="passage representations")
parser.add_argument("--passage_ids_path", default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/passage_ids.pkl",
                    type=str, required=False, help="passage ids")
parser.add_argument("--output_dir", default='./release_test4', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

parser.add_argument("--global_mode", default=True, type=str2bool, required=False,
                    help="maxmize the prob of the true answer given all passages")
parser.add_argument("--history_num", default=6, type=int, required=False,
                    help="number of history turns to use")
parser.add_argument("--prepend_history_questions", default=True, type=str2bool, required=False,
                    help="whether to prepend history questions to the current question")
parser.add_argument("--prepend_history_answers", default=False, type=str2bool, required=False,
                    help="whether to prepend history answers to the current question")

parser.add_argument("--do_train", default=False, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=False, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--best_global_step", default=90000, type=int, required=False,
                    help="used when only do_test")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=0.0001, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=10,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=10000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=False, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='',
                    help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='',
                    help="Can be used for distant debugging.")

# retriever arguments
parser.add_argument("--retriever_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--retriever_model_type", default='albert', type=str, required=False,
                    help="retriever model type")
parser.add_argument("--retriever_model_name_or_path", default='albert-base-v1', type=str, required=False,
                    help="retriever model name")
parser.add_argument("--retriever_tokenizer_name", default="albert-base-v1", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--retriever_cache_dir", default="/home/share/liyongqi/data_dir/CODQA/huggingface_cache/albert_v1/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--retrieve_checkpoint",
                    default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/retriever_checkpoint/checkpoint-5917", type=str,
                    help="generate query/passage representations with this checkpoint")
parser.add_argument("--retrieve_tokenizer_dir",
                    default="/home/share/liyongqi/data_dir/CODQA/ORQUAC/retriever_checkpoint", type=str,
                    help="dir that contains tokenizer files")

parser.add_argument("--given_query", default=True, type=str2bool,
                    help="Whether query is given.")
parser.add_argument("--given_passage", default=False, type=str2bool,
                    help="Whether passage is given. Passages are not given when jointly train")
parser.add_argument("--is_pretraining", default=False, type=str2bool,
                    help="Whether is pretraining. We fine tune the query encoder in retriever")
parser.add_argument("--include_first_for_retriever", default=True, type=str2bool,
                    help="include the first question in a dialog in addition to history_num for retriever (not reader)")
# parser.add_argument("--only_positive_passage", default=True, type=str2bool,
#                     help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
parser.add_argument("--retriever_query_max_seq_length", default=128, type=int,
                    help="The maximum input sequence length of query.")
parser.add_argument("--retriever_passage_max_seq_length", default=384, type=int,
                    help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--top_k_for_retriever", default=100, type=int,
                    help="retrieve top k passages for a query, these passages will be used to update the query encoder")
parser.add_argument("--use_retriever_prob", default=True, type=str2bool,
                    help="include albert retriever probs in final answer ranking")

# reader arguments
parser.add_argument("--reader_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--reader_model_name_or_path", default='bert-base-uncased', type=str, required=False,
                    help="reader model name")
parser.add_argument("--reader_model_type", default='bert', type=str, required=False,
                    help="reader model type")
parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--reader_cache_dir", default="/home/share/liyongqi/data_dir/CODQA/huggingface_cache/bert-base-uncased", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--reader_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=384, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument('--version_2_with_negative', default=True, type=str2bool, required=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--reader_max_query_length", default=125, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=40, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--qa_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--retrieval_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--top_k_for_reader", default=5, type=int,
                    help="update the reader with top k passages")
parser.add_argument("--top_k_from_explorer", default=5, type=int,
                    help="update the reader with top k passages")
parser.add_argument("--use_rerank_prob", default=True, type=str2bool,
                    help="include rerank probs in final answer ranking")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')
# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
# we now only support joint training on a single card
# we will request two cards, one for torch and the other one for faiss
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    # torch.cuda.set_device(0)
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()


model = Pipeline()

args.retriever_model_type = args.retriever_model_type.lower()
retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
retriever_config = retriever_config_class.from_pretrained(args.retrieve_checkpoint)

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
retriever_model = retriever_model_class.from_pretrained(args.retrieve_checkpoint, force_download=True)

model.retriever = retriever_model
# do not need and do not tune passage encoder
model.retriever.passage_encoder = None
model.retriever.passage_proj = None



args.reader_model_type = args.reader_model_type.lower()
reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
reader_config = reader_config_class.from_pretrained(args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
                                                    cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_config.num_qa_labels = 2
# this not used for BertForOrconvqaGlobal
reader_config.num_retrieval_labels = 2
reader_config.qa_loss_factor = args.qa_loss_factor
reader_config.retrieval_loss_factor = args.retrieval_loss_factor

reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                          do_lower_case=args.do_lower_case,
                                                          cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
# reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
#                                                   from_tf=bool(
#                                                       '.ckpt' in args.reader_model_name_or_path),
#                                                   config=reader_config,
#                                                   cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)

# model.reader = reader_model






# #####################################yongqi
model.retriever = retriever_model_class.from_pretrained(
             "/home/share/liyongqi/data_dir/CODQA/ORQUAC/pipeline_checkpoint/checkpoint-45000/retriever", force_download=True)

model.retriever.passage_encoder = None
model.retriever.passage_proj = None




model.reader = reader_model_class.from_pretrained(
    "/home/share/liyongqi/data_dir/CODQA/ORQUAC/pipeline_checkpoint/checkpoint-45000/reader", force_download=True)


model.explorer=Explorer_GAT.from_pretrained(args.retrieve_checkpoint, force_download=True)

model.explorer.load_state_dict(torch.load("/home/share/liyongqi/CODQA/QA/DGRCoQA_final/release_test3/checkpoint-60000/explorer/"+"explorer.pt"))


#####################################yongqi












if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger.info(f'loading passage ids from {args.passage_ids_path}')
with open(args.passage_ids_path, 'rb') as handle:
    passage_ids = pkl.load(handle)

logger.info(f'loading passage reps from {args.passage_reps_path}')
with open(args.passage_reps_path, 'rb') as handle:
    passage_reps = pkl.load(handle)




##############################################################################yongqi
#read block structure
with open('/home/share/liyongqi/data_dir/CODQA/wikipedia/blocks_contain_link/block_link_dict.pkl', 'rb') as f:
    block_link_dict=pickle.load(f)




##############################################################################yongqi








logger.info('constructing passage faiss_index')
faiss_res = faiss.StandardGpuResources() 
index = faiss.IndexFlatIP(args.proj_size)
index.add(passage_reps)
gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, index)

# logger.info(f'loading all blocks from {args.blocks_path}')
# with open(args.blocks_path, 'rb') as handle:
#     blocks_array = pkl.load(handle)



# for conversation_turn in range(12):
logger.info(f'loading qrels from {args.qrels}')
with open(args.qrels) as handle:
    qrels = json.load(handle)

# new_qrels={}
# for key in qrels:
#     if(int(key.split("#")[1])==conversation_turn):
#         new_qrels[key]=qrels[key]
# qrels=new_qrels



##################################yongqi
#get question-answer pair
question_answer={}
for i, (qid, v) in enumerate(qrels.items()):
    question_answer[qid]=[]
    for pid in v.keys():
        question_answer[qid].append(pid)
history_answer={}
for i, (qid, v) in enumerate(qrels.items()):
    history_answer[qid]=[]

    if(int(qid[-1])>0):
        for j in range(int(qid[-1])):
            history_qid=(qid[:len(qid)-1]+str(j))

            if(history_qid in question_answer):
                for s in question_answer[history_qid]:
                    history_answer[qid].append(s)

#
DR_Result = {}
with open('/home/share/liyongqi/CODQA/ConvDR/results/or-quac/multi_task.trec','r') as f:
    for line in tqdm(f.readlines()):
        qid=line.split(" ")[0]
        did=passage_ids[int(line.split(" ")[2])]
        score=line.split(" ")[4]
        if qid not in DR_Result:
            DR_Result[qid] =[[],[]]
        DR_Result[qid][0].append(did)
        DR_Result[qid][1].append(int(score))

with open('/home/share/liyongqi/CODQA/ConvDR/results/or-quac/train_multi_task.trec','r') as f:
    for line in tqdm(f.readlines()):
        qid=line.split(" ")[0]
        did=passage_ids[int(line.split(" ")[2])]
        score=line.split(" ")[4]
        if qid not in DR_Result:
            DR_Result[qid] =[[],[]]
        DR_Result[qid][0].append(did)
        DR_Result[qid][1].append(int(score))
with open('/home/share/liyongqi/CODQA/ConvDR/results/or-quac/dev_multi_task.trec','r') as f:
    for line in tqdm(f.readlines()):
        qid=line.split(" ")[0]
        did=passage_ids[int(line.split(" ")[2])]
        score=line.split(" ")[4]
        if qid not in DR_Result:
            DR_Result[qid] =[[],[]]
        DR_Result[qid][0].append(did)
        DR_Result[qid][1].append(int(score))
##################################yongqi




########################predicted answers
predicted_question_answerspans={}
with open('/home/share/liyongqi/CODQA/QA/DGRCoQA_final/predictions/final_predictions_test.json', 'r') as f:
    for line in tqdm(f.readlines()):
        line= json.loads(line.strip())
        for i in range(len(line['best_span_str'])):
            predicted_question_answerspans[line['qid'][i]]=line['best_span_str'][i]




###########################

########################predicted passages
predicted_question_passages={}
with open('/home/share/liyongqi/CODQA/QA/DGRCoQA_final/predictions/instance_predictions_test.json', 'r') as f:
    lines=json.load(f)
    for key in lines:
        predicted_question_passages[key]=lines[key][0]['example_id'].split("*")[1]

history_predicted_passages={}
for key in predicted_question_passages:
    history_predicted_passages[key]=[]
    for i in range(int(key.split("#")[1])):
        h_qid=key.split("#")[0]+"#"+str(i)
        if(h_qid in predicted_question_passages):
            history_predicted_passages[key].append(predicted_question_passages[h_qid])






###########################



passage_id_to_idx = {}
for i, pid in enumerate(passage_ids):
    passage_id_to_idx[pid] = i

qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
qid_to_idx = {}
for i, (qid, v) in enumerate(qrels.items()):
    qid_to_idx[qid] = i
    for pid in v.keys():
        qrels_data.append(1)
        qrels_row_idx.append(i)
        qrels_col_idx.append(passage_id_to_idx[pid])
qrels_sparse_matrix = sp.sparse.csr_matrix(
    (qrels_data, (qrels_row_idx, qrels_col_idx)))

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall'})

result = evaluate(args, model, retriever_tokenizer,
          reader_tokenizer, prefix='test') 
# # In[10]:

#Training
if args.do_train:
    DatasetClass = RetrieverDataset
    train_dataset = DatasetClass(args.train_file, retriever_tokenizer,
                                 args.load_small, args.history_num,
                                 query_max_seq_length=args.retriever_query_max_seq_length,
                                 is_pretraining=args.is_pretraining,
                                 given_query=True,
                                 given_passage=False, 
                                 include_first_for_retriever=args.include_first_for_retriever,
                                 prepend_history_answers=args.prepend_history_answers)
    global_step, tr_loss = train(
        args, train_dataset, model, retriever_tokenizer, reader_tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# args.test_file="/home/share/liyongqi/CODQA/QA/ours10/conversation_turn/"+str(conversation_turn)+".txt"
# result = evaluate(args, model, retriever_tokenizer,
#   reader_tokenizer, prefix='test')

# for i in range(1,6):
#     args.top_k_from_explorer=5
#     args.top_k_for_reader=i
#     result = evaluate(args, model, retriever_tokenizer,
#         reader_tokenizer, prefix='test')
# for i in range(1,6):
#     args.top_k_for_reader=5
#     args.top_k_from_explorer=i
#     result = evaluate(args, model, retriever_tokenizer,
#         reader_tokenizer, prefix='test')
if args.do_test and args.local_rank in [-1, 0]:    
    if args.do_eval:
        best_global_step = best_metrics['global_step'] 
    else:
        best_global_step = args.best_global_step
        retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
            args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
        reader_tokenizer = reader_tokenizer_class.from_pretrained(
            args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    best_checkpoint = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(best_global_step))
    logger.info("Test the best checkpoint: %s", best_checkpoint)

# #     model = Pipeline()
# #     model.retriever = retriever_model_class.from_pretrained(
# #         os.path.join(best_checkpoint, 'retriever'), force_download=True)
# #     model.retriever.passage_encoder = None
# #     model.retriever.passage_proj = None
# #     model.reader = reader_model_class.from_pretrained(
# #         os.path.join(best_checkpoint, 'reader'), force_download=True)

# # ########################################yongqi
# #     model.explorer=Explorer_GAT.from_pretrained("/home/share/liyongqi/data_dir/CODQA/ORQUAC/pipeline_checkpoint/checkpoint-45000/retriever", force_download=True)
# #     model.explorer.load_state_dict(torch.load(os.path.join(best_checkpoint, 'explorer/')+"explorer.pt"))
# # ########################################yongqi
# #     model.to(args.device)

#     # Evaluate
#     result = evaluate(args, model, retriever_tokenizer,
#                       reader_tokenizer, prefix='dev')

#     test_metrics_file = os.path.join(
#         args.output_dir, 'predictions', 'test_metrics.json')
#     with open(test_metrics_file, 'w') as fout:
#         json.dump(result, fout)

#     logger.info("Test Result: {}".format(result))
