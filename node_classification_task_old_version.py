import requests
from torch_geometric.datasets import Planetoid, WikipediaNetwork, PPI, Actor
from functools import partial
from node2vec import node2vec
import networkx as nx
import os
import parmap
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import BertTokenizerFast
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch.utils.data._utils.collate import default_convert
import torch
import pandas as pd
from torch import nn
from torch import optim
from transformers import BertForMaskedLM, BertConfig, BertForPreTraining
from torch.utils.data import Dataset, DataLoader, random_split
from node2vec import node2vec
import networkx as nx
import os
import networkx as nx
import pandas as pd
import os
import random
import argparse
import torch
from torch.utils.data.dataset import Dataset
from sklearn.metrics  import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import torch
import random
import random
random.seed(42)
batch_size = 128

def func_2(G, label, test_num, vec,  org_nodes, nodes):
    return_example = []
    test_datasets = []
    datasets = []
    for i in nodes:
        input_ids = [org_nodes.index(i)]
        position_ids = [1]
        token_type_ids = []
        embeddings = vec.node2vecWalk(i)
        for j in embeddings:
            input_ids.append(org_nodes.index(j))
            position_ids.append(len(nx.dijkstra_path(G,i,j)))

        token_type_ids = [0 for i in range(len(input_ids))]
        attention_mask = [1 for i in range(len(input_ids))]

        org_input_ids_len = len(input_ids)
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        position_ids.extend([0] * n_pad)
        token_type_ids.extend([0] * n_pad)
        attention_mask.extend([0] * n_pad)
        example = {
                            "input_ids": input_ids,
                            "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask,
                            #"position_ids":position_ids,
                            "label": label[i],
                            "node": i
        }
        if args.position == "True":
            example["position_ids"] = position_ids
        elif args.position == "False":
            example["position_ids"] = [0 for i in range(len(input_ids))]
        elif args.position == 'Sequence':
            example["position_ids"] = [i for i in range(org_input_ids_len)]
            example["position_ids"].extend([0] * n_pad)
            example["position_ids"] = example["position_ids"]
            
        if org_nodes.index(i) in test_num:
            test_datasets.append(example)
        else:
            datasets.append(example)
    return_example = [test_datasets,datasets]
    #return_example.append()
    #return_example.append()
    return return_example

def get_clf_eval(y_test, pred, ave):
    #confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred,average=ave)
    recall = recall_score(y_test, pred,average=ave)
    #print('Confusion Matrix')
    #print(confusion)
    print('{}, 정확도:{}, 정밀도:{}, 재현율:{}'.format(ave,accuracy, precision, recall))
    return accuracy

class TextDataset(Dataset):
    def __init__(
        self,
        dict_type_data,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        self.examples = dict_type_data
        for i in range(len(dict_type_data)):
            self.examples[i]["input_ids"] = torch.tensor(self.examples[i]["input_ids"], dtype=torch.long)
            self.examples[i]["position_ids"] = torch.tensor(self.examples[i]["position_ids"], dtype=torch.long)
            self.examples[i]["attention_mask"] = torch.tensor(self.examples[i]["attention_mask"], dtype=torch.long)
            self.examples[i]["token_type_ids"] = torch.tensor(self.examples[i]["token_type_ids"], dtype=torch.long)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def calc_accuracy(X,Y):
    X = X.to("cpu")
    Y = Y.to("cpu")
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data/max_indices.size()[0]
    return train_acc

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.bert = bert

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids = None):
        if position_ids != None:
            out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, position_ids = position_ids).pooler_output
        else:
            out = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).pooler_output
        return out

def read_graph(data_name):
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
    node_labels = data.y[list(G.nodes)].numpy().tolist()
    node_class =dict()
    for k,v  in enumerate(node_labels):
        node_class[k] = v
 
    G = G.to_undirected()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G, node_class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=float, default=3, help='return parameter')
    parser.add_argument('--q', type=float, default=0.5, help='in-out parameter')
    parser.add_argument('--d', type=int, default=128, help='dimension')
    parser.add_argument('--r', type=int, default=10, help='walks per node')
    parser.add_argument('--l', type=int, default=10, help='walk length')
    parser.add_argument('--k', type=float, default=10, help='window size')
    parser.add_argument('--input', type=str, default='cham', help='cora')
    parser.add_argument('--steps', type=int, required=True, default='3600')
    #parser.add_argument('--project_name', type=str, required=True)
    #parser.add_argument('--hop', type=str, required=True, default='not_hop_model or hop_model')
    parser.add_argument('--device', type=str, default="CUDA")
    parser.add_argument('--neighbor_epoch', type=int, required=True, help='num on epochs')
    #parser.add_argument('--l_lit', type=str, required=True, help='l_list')
    parser.add_argument('--position', type=str, required=True, help='If True -> position_ids else non position_ids')
    parser.add_argument('--bert_layer', type=int, required=True, help='num of bert_layer')
    parser.add_argument('--mlm_prob', type=float, required=True, help='masking probablity, 0.5 is best')
    parser.add_argument('--hidden_size', type=int, required=True, help='hidden_size ex:768(BERT) or 128')
    parser.add_argument('--block_size', type=int, required=True, help='block_size(hidden_size/block_size = block_size) ex:32(BERT) or 64')
    parser.add_argument('--N', type=int, required=True, help='N')

    args = parser.parse_args()
    return args

maxlen = args.l+1
dfs_parameter_list = [[1000000 ,0.000001, args.l]]
epoch_num=args.neighbor_epoch
mlm_prob = args.mlm_prob
bert_layer = args.bert_layer
hidden_size = args.hidden_size
block_size = args.block_size

if args.position == 'True':
    HOP_NAME = f'position_paper_version/'
elif args.position == 'False':
    HOP_NAME = f'non_position_paper_version/'

if block_size > hidden_size:
    print("ERROR, block size must be smaller than hidden_size)")
    exit()

parameter_list_all = [str(mlm_prob),  str(bert_layer), str(hidden_size),str(block_size),"PQ",str(args.l)]
project_name = HOP_NAME+args.input + "_".join(parameter_list_all)

# Graph generation and node token generation
G , _= read_graph(args.input)
nodes =  [x for x in list(G.nodes)]
nodes.sort()
special_tokens = [-2,-1]
nodes = special_tokens + nodes #[PAD,MASK]
vocab_size = len(nodes)
datasets = []
parameter_list = dfs_parameter_list

result_dict = dict()

#MEAN F1, MEAN ACC, MAX F1, MAX ACC, MIN F1, MIN ACC

if args.device == "CUDA":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

N = args.N

parameter_list_str = [str(mlm_prob), str(bert_layer), str(hidden_size),str(block_size),"PQ",str(args.l)]
project_name = HOP_NAME+args.input + "_".join(parameter_list_str)
#    project_name = HOP_NAME+args.input + "_".join(parameter_list_all)
if args.steps != 0:
    project_name +='/checkpoint-'+str(args.steps)#hop+"/"+args.input + "_".join(parameter_list_all)

result_dict[project_name] = [0 for i in range(3*4)]
result_dict[project_name] += [1,1,1,1,1,1]


X_Train = dict()
Y_Train = dict()
X_Test = dict()
Y_Test = dict()
multi_thread_list = []

parameter_list = dfs_parameter_list
for now_index in range(N):
    parameter_list_str = [str(mlm_prob), str(bert_layer), str(hidden_size),str(block_size),"PQ",str(args.l)]
    project_name = HOP_NAME+args.input + "_".join(parameter_list_str)
    parameter_list_all = [mlm_prob, bert_layer, hidden_size, block_size, parameter_list,args.steps]
    if args.steps != 0:
        project_name +='/checkpoint-'+str(args.steps)#hop+"/"+args.input + "_".join(parameter_list_all)

    #project_name +='/checkpoint-'+str(args.steps)#hop+"/"+args.input + "_".join(parameter_list_all)

    G , label = read_graph(args.input)

    nodes =  [x for x in list(G.nodes)]
    nodes.sort()
    special_tokens = [-2,-1]
    nodes = special_tokens + nodes #[PAD, UNK,CLS,SEP,MASK]
    #print(label)
    vocab_size = len(nodes)

    model_path = project_name
    datasets = []
    test_datasets = []
    test_num = random.sample(range(len(special_tokens), len(nodes)),int(len(nodes)*0.4))

    vec = node2vec(args, G)
    target_nodes = nodes[len(special_tokens):]
    sampled_target_nodes = []
    num_cores = 32
    for i in range(num_cores):
        start_index = int(len(target_nodes)/num_cores)*i
        end_index = int(len(target_nodes)/num_cores)*(i+1)
        sampled_target_nodes.append(target_nodes[start_index:end_index])        
    print(len(sampled_target_nodes))

    func = partial(func_2, G, label,test_num,vec, nodes)
    temp_datasets = parmap.map(func, sampled_target_nodes, pm_pbar=True, pm_processes=num_cores)
    for i in temp_datasets:
        if i == None:
            continue
    
        test_v = i[0]
        normal_v = i[1]
        #print("len_test_v", len(test_v))
        #print(test_v)
        for k in test_v:
            test_datasets.append(k)
        for k in normal_v:
            datasets.append(k)
                                
        
                
    print(len(test_datasets))
    print(len(test_num))
    print("##############")
    print(len(datasets))
    if len(datasets) >= 1:
        print(datasets[0])
        print("#################")
        print(datasets[1])
        #vec =  

    for _ in range(epoch_num):
        for i in nodes[len(special_tokens):]:
            input_ids = [nodes.index(i),nodes.index(i)]
            position_ids = [1,1]
            token_type_ids = []
            embeddings = G.neighbors(i)
            #embeddings = vec.node2vecWalk(i)
            embeddings = list(embeddings)
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
                                "label": label[i],
                                "node": i
            }
            if args.position == "True":
                example["position_ids"] = position_ids
            elif args.position == "False":
                example["position_ids"] = [0 for i in range(len(input_ids))]
            elif args.position == 'Sequence':
                example["position_ids"] = [i for i in range(org_input_ids_len)]
                example["position_ids"].extend([0] * n_pad)
                example["position_ids"] = example["position_ids"]
            """
            if (nodes.index(i) in test_num) :
                #datasets.append(example)
                test_datasets.append(example)
            else:
                datasets.append(example)

            """
            if not (nodes.index(i) in test_num) :
                datasets.append(example)
                #test_datasets.append(example)
            #else:
                #datasets.append(example)
            
    print(len(datasets))
    print(datasets[-1])
    print("#################")
    print(datasets[2])
    random.shuffle(datasets)
    random.shuffle(test_datasets)

    dataset = TextDataset(dict_type_data = datasets)
    test_dataset = TextDataset(dict_type_data = test_datasets)   

    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    bert_model = BertModel.from_pretrained(model_path)
    #bert_model = BertModel.from_pretrained("cora_sample_model")
    model = BERTClassifier(bert_model).to(device)

    svm_data = []
    svm_label = []
    test_svm_data = []
    test_svm_label = []
    num_epochs = 1
    for e in range(num_epochs):
        model.eval()
        print("start")
        temp_output_data = []
        for batch_id, samples in enumerate(train_dataloader):
            #print(se)
            #optimizer.zero_grad()
            ids = samples['input_ids']
            tok_type = samples['token_type_ids']
            label = samples['label']
            atten = samples['attention_mask']
            posid = samples['position_ids']
            out = model(input_ids = ids.to(device),token_type_ids = tok_type.to(device), attention_mask = atten.to(device), position_ids = posid.to(device))
            
            
            output = out.cpu().detach().numpy()
            labels = label.detach().numpy()
            #print(len(labels))
            #print(len(output))
            for i in range(len(labels)):
                svm_data.append(output[i])
                svm_label.append(labels[i])
                
    for e in range(num_epochs):
        model.eval()
        print("start")
        temp_output_data = []
        for batch_id, samples in enumerate(test_dataloader):
            ids = samples['input_ids']
            tok_type = samples['token_type_ids']
            label = samples['label']
            atten = samples['attention_mask']
            posid = samples['position_ids']
            out = model(input_ids = ids.to(device),token_type_ids = tok_type.to(device), attention_mask = atten.to(device), position_ids = posid.to(device))
        
            
            output = out.cpu().detach().numpy()
            labels = label.detach().numpy()
            for i in range(len(labels)):
                test_svm_data.append(output[i])
                test_svm_label.append(labels[i])
            
    import numpy as np
    if not (project_name in X_Train):
        X_Train[project_name] = [0 for i in range(N)]
        Y_Train[project_name] = [0 for i in range(N)]
        X_Test[project_name] = [0 for i in range(N)]
        Y_Test[project_name] = [0 for i in range(N)]
    X_Train[project_name][now_index] = np.array(svm_data)
    Y_Train[project_name][now_index] = np.array(svm_label)

    X_Test[project_name][now_index] = np.array(test_svm_data)
    Y_Test[project_name][now_index] = np.array(test_svm_label)
    multi_thread_list.append([project_name, now_index])
print(multi_thread_list)
 
 ##################################################################################################################
def work_func(X_Test, X_Train, Y_Test, Y_Train, parameters):
    project_name = parameters[0]
    now_index = parameters[1]
    #result_dict = []
    result_dict = [0,0,0,0,0,0]
    result_dict += [project_name, now_index]

    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_Train[project_name][now_index], Y_Train[project_name][now_index])
    pred = classifier.predict(X_Test[project_name][now_index])
    from sklearn.metrics  import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

    ave_list = ["micro", "weighted","macro"]
    for idx, ave in enumerate(ave_list):
        my_acc_score = get_clf_eval(Y_Test[project_name][now_index],pred,ave)
        #print(ave + ":", end="")
        #print(f1_score(Y_Test[project_name][now_index], pred, average=ave))
        my_f1_score = f1_score(Y_Test[project_name][now_index], pred, average=ave)
        result_dict[idx] += my_f1_score
        result_dict[idx+3] += my_acc_score
    return result_dict

func = partial(work_func, X_Test, X_Train, Y_Test, Y_Train)
num_cores = 28
temp_datasets = parmap.map(func, multi_thread_list, pm_pbar=True, pm_processes=num_cores)

for item in temp_datasets:
    project_name = item[-2]
    for idx in range(3):
        my_f1_score = item[idx]
        my_acc_score = item[idx+3]
        result_dict[project_name][idx] += my_f1_score
        result_dict[project_name][idx+3] += my_acc_score
        if my_f1_score > result_dict[project_name][idx+6]:
            result_dict[project_name][idx+6] = my_f1_score
        if my_acc_score > result_dict[project_name][idx+9]:
            result_dict[project_name][idx+9] = my_acc_score
        if my_f1_score < result_dict[project_name][idx+12]:
            result_dict[project_name][idx+12] = my_f1_score
        if my_acc_score < result_dict[project_name][idx+15]:
            result_dict[project_name][idx+15] = my_acc_score


print(temp_datasets)
print("########################################################")
print(result_dict)



            #print(args.p, args.q)
            #print(classification_report(Y_Test, pred))





