from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from transformers import BertForPreTraining
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import parmap
from functools import partial
import torch
import random
import wandb
from node2vec import node2vec
import networkx as nx
import os
import torch
from transformers import Trainer, TrainingArguments
from torch_geometric.datasets import Planetoid, WikipediaNetwork, PPI, Actor
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator, DataCollatorWithPadding
from transformers import EarlyStoppingCallback
import numpy as np
from sklearn.metrics  import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import sklearn
from datasets import load_metric
import numpy as np
metric = load_metric("accuracy")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def func_2(G, vec,  org_nodes, nodes):
    return_example = []
    for i in nodes:
        example = dict()
        token_type_ids = []
        attention_mask = []

        # 자기 자신 참조를 위해 미리 노드 추가
        input_ids = [org_nodes.index(i)]
        position_ids = [1]
        #position_ids.append(0)

        #label = []

        #Random Wak 진행
        embeddings = vec.node2vecWalk(i)

        #input_ids에 walk로 생성된 노드 추가       
        for j in embeddings:
            input_ids.append(org_nodes.index(j))
            position_ids.append(len(nx.dijkstra_path(G,i,j)))

        token_type_ids = [0 for i in range(len(input_ids))]
        attention_mask = [1 for i in range(len(input_ids))]

        #패딩 처리
        org_input_ids_len = len(input_ids)
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        token_type_ids.extend([0] * n_pad)
        attention_mask.extend([0] * n_pad)
        position_ids.extend([0] * n_pad)
        #label.extend([-100] * n_pad)

        #딕셔너리 형태로 example 생성
        example = {
                            "input_ids": input_ids,
                            "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask,
                            
                            #"labels" : label,
        }
        if args.position == "True":
            example["position_ids"] = position_ids
        elif args.position == "False":
            example["position_ids"] = [0 for i in range(len(input_ids))]               
        return_example.append(example)
    return return_example

def work_func(args, G, nodes, lpq):
    # node2vec의 p, q값 받아오기
    args.p = lpq[0]
    args.q = lpq[1]
    args.l = lpq[2]
    if args.l <= 1:
        return None
    # node2vec instance화
    vec = node2vec(args, G)

    #list 형태로 Return, 병렬처리 끝나고 병합진행
    return_example = []
    #같은 작업을 혹시나 N번 할 수도 있어서 이중 반복문으로 처리
    for _ in range(1):
        # MASK, PAD 제외한 모든 노드에 대하여 반복문 진행
        target_nodes = nodes[len(special_tokens):]
        sampled_target_nodes = []

        num_cores = 16
        for i in range(num_cores):
            start_index = int(len(target_nodes)/num_cores)*i
            end_index = int(len(target_nodes)/num_cores)*(i+1)
            sampled_target_nodes.append(target_nodes[start_index:end_index])        
        print(len(sampled_target_nodes))
       
        func = partial(func_2, G, vec, nodes)
        temp_datasets = parmap.map(func, sampled_target_nodes, pm_pbar=True, pm_processes=num_cores)
        return_example += temp_datasets

    return return_example

def read_graph(graph_type, data_name):
    global node_class
    global ndde_feature
    global class_num
    node_class = dict()
    edgelist = list()
    class_num = 1
    class_name_to_num = dict()
    if data_name == "cham":
        dataset = WikipediaNetwork(root='/tmp/Cham', name="chameleon")
    elif data_name == "squirrel":
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name="squirrel")
    elif data_name == "crocodile":
        dataset = WikipediaNetwork(root='/tmp/crocodile', name="crocodile")
    elif data_name == "citeseer":
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    elif data_name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif data_name == 'pubmed':
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    elif data_name == 'actor':
        dataset = Actor(root='/tmp/actor')
           
    data = dataset[0]
    from torch_geometric.utils.convert import to_networkx
    G = to_networkx(data)
    node_labels = data.y[list(G.nodes)].numpy()
    node_class = {k: v for v, k in enumerate(node_labels)}

    if graph_type == 'reverse':
        G = G.reverse()
    elif graph_type == 'undir':    
        G = G.to_undirected()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G, node_class

class TextDataset(Dataset):
    def __init__(
        self,
        dict_type_data,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        self.examples = dict_type_data
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

my_epoch=600
batch_size = 2048
lr = 0.0001
parser = argparse.ArgumentParser()


parser.add_argument('--p', type=float, default=3.3, help='return parameter')
parser.add_argument('--q', type=float, default=0.05, help='in-out parameter')
parser.add_argument('--d', type=int, default=128, help='dimension')
parser.add_argument('--r', type=int, default=10, help='walks per node')
parser.add_argument('--l', type=int, default=10, help='walk length')
parser.add_argument('--k', type=float, default=10, help='window size')
parser.add_argument('--input', type=str, required=True, help='blog')
parser.add_argument('--neighbor_epoch', type=int, required=True, help='num on epochs')
#parser.add_argument('--l_lit', type=str, required=True, help='l_list')
parser.add_argument('--position', type=str, required=True, help='If True -> position_ids else non position_ids')
parser.add_argument('--bert_layer', type=int, required=True, help='num of bert_layer')
parser.add_argument('--mlm_prob', type=float, required=True, help='masking probablity, 0.5 is best')
parser.add_argument('--hidden_size', type=int, required=True, help='hidden_size ex:768(BERT) or 128')
parser.add_argument('--block_size', type=int, required=True, help='block_size(hidden_size/block_size = block_size) ex:32(BERT) or 64')

args = parser.parse_args()
  
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    labels = [labels[row][indices[row]] for row in range(len(labels))]
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]

    results = metric.compute(predictions=predictions, references=labels)
    results["eval_accuracy"] = results["accuracy"]
    print(results)
    results.pop("accuracy")
    return results

maxlen = args.l+1
dfs_parameter_list = [[1000000 ,0.000001, args.l]]
graph_structure = 'undir'
epoch_num=args.neighbor_epoch

if args.position == 'True':
    HOP_NAME = f'position_paper_version/'
elif args.position == 'False':
    HOP_NAME = f'non_position_paper_version/'
mlm_prob = args.mlm_prob
bert_layer = args.bert_layer
hidden_size = args.hidden_size
block_size = args.block_size
#l_list = args.l
print(mlm_prob)
print(bert_layer)
print(hidden_size)
print(block_size)
print(args.l)
if block_size > hidden_size:
    print("ERROR, block size must be smaller than hidden_size)")
    exit() 
parameter_list_all = [str(mlm_prob), str(graph_structure), str(bert_layer), str(hidden_size),str(block_size),"PQ",str(args.l)]
project_name = HOP_NAME+args.input + "_".join(parameter_list_all)

# Graph generation and node token generation
G , _= read_graph(graph_structure, args.input)
nodes =  [x for x in list(G.nodes)]
nodes.sort()
special_tokens = [-2,-1]
nodes = special_tokens + nodes #[PAD,MASK]
vocab_size = len(nodes)
datasets = []
parameter_list = dfs_parameter_list
for parameters in parameter_list:
    temp_datasets = work_func(args,G,nodes,parameters)
    for i in temp_datasets:
        for j in i:
            if i == None:
                continue
            datasets.append(j)

    for _ in range(epoch_num): #How many times to insertthe node's neighbor information
        for i in nodes[len(special_tokens):]:
            example = dict()
            token_type_ids = []
            attention_mask = []
            label = []
            input_ids = [nodes.index(i), nodes.index(i)]
            position_ids = [1,1]
        
            embeddings = list(G.neighbors(i))
            random.shuffle(embeddings)
            for j in embeddings:
                input_ids.append(nodes.index(j))
                position_ids.append(1)
            if len(input_ids) > maxlen:
                temp_input_ids = [input_ids[0], input_ids[0]]
                random.shuffle(input_ids)
                input_ids = temp_input_ids + input_ids[:maxlen-2]
                position_ids = [1 for i in range(maxlen)]
            token_type_ids = [0 for i in range(len(input_ids))]
            attention_mask = [1 for i in range(len(input_ids))] 
            
            org_input_ids_len = len(input_ids)
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)
            token_type_ids.extend([0] * n_pad)
            attention_mask.extend([0] * n_pad)
            position_ids.extend([0] * n_pad)

            example = {
                                "input_ids": input_ids,
                                "token_type_ids": token_type_ids,
                                "attention_mask": attention_mask,
                                #"position_ids":position_ids,
                                #"labels" : label,
            }
            if args.position == 'True':
                example["position_ids"] = position_ids
            elif args.position == 'False':
                example["position_ids"] = [0 for i in range(len(input_ids))]
            elif args.position == 'Sequence':
                example["position_ids"] = [i for i in range(org_input_ids_len)]
                example["position_ids"].extend([0] * n_pad)
            datasets.append(example)


random.shuffle(datasets) 
createFolder(args.input) 

with open(args.input + os.sep + 'vocab.txt', 'w') as f:
    f.write("[PAD]" + '\n')
    f.write("[MASK]" + '\n')
    for i in list(G.nodes):
        f.write(str(i) + '\n')
                
config = BertConfig(   # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    vocab_size = vocab_size,
    hidden_size = hidden_size,
    num_hidden_layers = bert_layer,
    num_attention_heads = hidden_size//block_size,
    intermediate_size = hidden_size*4,
    max_position_embeddings = maxlen,
) 
tokenizer = BertTokenizer(
    vocab_file=args.input + os.sep +'vocab.txt',
    max_len=maxlen,
    do_lower_case=False,
)

dataset = TextDataset(dict_type_data = datasets)

data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
)
model = BertForMaskedLM(config=config)
model.num_parameters()

training_args = TrainingArguments(
    output_dir=project_name, 
    num_train_epochs=my_epoch,
    per_gpu_train_batch_size=batch_size,
    save_steps=600, # step 수마다 모델을 저장
    logging_steps=50,
    learning_rate=lr,
    dataloader_num_workers = 1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(project_name)

