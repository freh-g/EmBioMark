#!/usr/bin/env python
import pandas as pd
import pickle
import os
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm

#BIOKG2VEC IMPORTS
from gensim.models import Word2Vec
import K2V_Walkers as w

# TRANSE DISTMULT IMPORTS 
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

#DLEMB IMPORTS
from keras.layers import Input, Embedding, Dot, Reshape
from keras.models import Model

#GCN IMPORTS
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

#EVALUATIONS
from sklearn.metrics import roc_auc_score

parser=argparse.ArgumentParser(description='Create embeddings given a knowledgegraph')
parser.add_argument('-k','--KnowledgeGraph',help='path of the KG in pickle format (networkx Digraph Object)')
parser.add_argument('-m','--model',help='model to use to produce embeddings possible values are: DLemb, BioKG2Vec, DistMult, TransE, GCN, N2V',type=str)
parser.add_argument('-e','--epochs',help='number of epochs to train the model for',type=int)
parser.add_argument('-o','--output',help = 'path in which to save the embeddings')

args=parser.parse_args()

def LoadData():
    with open(args.KnowledgeGraph,'rb') as f:
        kg = pickle.load(f)
        
    return kg
#BIOKG2VEC
def RunBioKG2Vec(kg):
    probabilities={('drug','protein'):100,
               ('protein','function'):0,
               ('function','phenotype'):1000}

    all_nodes=list(kg.nodes)
    random_walks = []
    print(" |================>   WALKING")
    for n in tqdm(all_nodes):
        biorw=w.KRW(n,kg,Iterations=5,Depth=50,
                NodeAttributeName='tipo',
                EdgeAttributeName='rel_type',
                DictOfProb=probabilities,
                directed='True',
                )
        random_walks.extend(biorw)
    model = Word2Vec(window = 5, sg = 1, hs = 0,
                 negative = 5, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14,workers=2)

    model.build_vocab(random_walks, progress_per=2)
    
    print(" |================>   TRAINING THE MODEL")

    model.train(random_walks, total_examples = model.corpus_count, epochs=args.epochs, report_delay=1)
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    return Id2Vec

def generate_batch(triplets, n_positive, negative_ratio ):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
            
    entities=set([t[0] for t in triplets]+[t[2] for t in triplets])
    pairs = [(t[0],t[2]) for t in triplets]
    set_of_pairs = set(pairs)
    
    batch = np.zeros((batch_size, 3))

 
    neg_label = -1

    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (head, tail) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (head, tail, 1)

        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # random selection
            random_head = random.randrange(len(entities))
            random_tail = random.randrange(len(entities))


            # Check to make sure this is not a positive example
            if (random_head, random_tail) not in set_of_pairs:

                # Add to batch and increment index
                batch[idx, :] = (random_head, random_tail, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
    
        yield {'head': batch[:, 0], 'tail': batch[:, 1]}, batch[:, 2]
#DLEMB
def PrepareDataForDLemb(kg):
    edgelist=nx.to_pandas_edgelist(kg)
    entities= list(set(edgelist.source.tolist()+edgelist.target.tolist()))
    interactions=list(set(edgelist.rel_type.tolist()))


    id_to_entity=dict(enumerate(entities))
    id_to_relation=dict(enumerate(interactions))

    entity_to_id={v:k for k,v in id_to_entity.items()}
    relation_to_id={v:k for k,v in id_to_relation.items()}

    heads_ids=list(map(entity_to_id.get,edgelist.source.tolist()))

    relations_ids=list(map(relation_to_id.get,edgelist.rel_type.tolist()))

    tails_ids=list(map(entity_to_id.get,edgelist.target.tolist()))


    triplets=list(zip(heads_ids,relations_ids,tails_ids))
    return entity_to_id,id_to_entity,relation_to_id,triplets

def DLemb(node_emb_size,input_dimension):   
    # Inputs for h,t
    h = Input(name = 'head', shape = [1])
    t = Input(name = 'tail',shape = [1])
        
    nodes_embedding_layer = Embedding(name = 'node_embeddings',
                                input_dim = input_dimension,
                                output_dim = node_emb_size)

    head_embedding = nodes_embedding_layer(h)
    tail_embedding = nodes_embedding_layer(t)
    
    dotted_embedding = Dot(name = 'dot_product',normalize = True, axes = 2)([head_embedding,
                                                                                         tail_embedding])
    merged = Reshape(target_shape = [1])(dotted_embedding)

            
    model = Model(inputs = [h, t], outputs = merged)

    model.compile(optimizer = 'Adam', loss = 'mse')

    return model


#NODE2VEC

def RunN2V(kg):
    # Produce the Node2Vec inputs
    Num2Node=dict(enumerate(list(kg.nodes)))
    Node2Num={v:k for k,v in Num2Node.items()}

    EdgeList=pd.DataFrame(([(Node2Num[s],Node2Num[t]) for (s,t,a) in list(kg.edges)]))
    EdgeList.to_csv('../tools/node2vec/Node2Vec_kg_input.txt',sep=' ',header=False,index=False)


    #Run Node2Vec
    os.system(f'../Tools/node2vec/./node2vec -i:../Tools/node2vec/Node2Vec_kg_input.txt -o:../Tools/node2vec/Node2Vec_kg_output.txt -e:{str(args.epochs)} -l:50 -d:100 -r:5 -p:0.3 -dr -v')
    with open('../Tools/node2vec/Node2Vec_kg_output.txt','r') as f:
        NodEmbs=f.readlines()

    NodEmbs=[s.split('\n') for s in NodEmbs ]
    NodEmbs=dict(zip([s[0].split(' ')[0] for s in NodEmbs[1:]],[s[0].split(' ')[1:] for s in NodEmbs[1:]]))
    Id2Vec={Num2Node[int(NodeNumber)]:np.array([float(number) for number in v]) for NodeNumber,v in NodEmbs.items()}
    os.system('rm ../Tools/node2vec/Node2Vec_kg_input.txt')
    os.system('rm ../Tools/node2vec/Node2Vec_kg_output.txt')
    return Id2Vec

#DISTMULT
    
def RunDistmult(kg):
    #CREATE MAPPING DICITONARIES
    Id2Node=dict(enumerate(list(kg.nodes)))
    Node2Id={v:k for k,v in Id2Node.items()}
    Id2Int=dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id={v:k for k,v in Id2Int.items()}


    # CREATE TRIPLES FACTORY 
    edgelist=list(nx.to_edgelist(kg))
    set_of_triples=[[Node2Id[node1],Int2Id[interaction['rel_type']],Node2Id[node2]] for (node1,node2,interaction) in edgelist]
    set_of_triples_Long=torch.LongTensor(set_of_triples)
    tf=TriplesFactory(set_of_triples_Long,entity_to_id=Node2Id,relation_to_id=Int2Id)
    training, testing, validation = tf.split([.8, .1, .1])
    
    result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='DistMult',
    model_kwargs=dict(embedding_dim=100),
    epochs=args.epochs)
    model=result.model
    entity_tensor= model.entity_representations[0]().detach().cpu().numpy()
    Id2Vec=dict(zip(Id2Node.values(),entity_tensor))
    return Id2Vec

# TRANSE

def RunTransE(kg):
    #CREATE MAPPING DICITONARIES
    Id2Node=dict(enumerate(list(kg.nodes)))
    Node2Id={v:k for k,v in Id2Node.items()}
    Id2Int=dict(enumerate(list(set([edge[2]['rel_type'] for edge in kg.edges(data=True)]))))
    Int2Id={v:k for k,v in Id2Int.items()}


    # CREATE TRIPLES FACTORY 
    edgelist=list(nx.to_edgelist(kg))
    set_of_triples=[[Node2Id[node1],Int2Id[interaction['rel_type']],Node2Id[node2]] for (node1,node2,interaction) in edgelist]
    set_of_triples_Long=torch.LongTensor(set_of_triples)
    tf=TriplesFactory(set_of_triples_Long,entity_to_id=Node2Id,relation_to_id=Int2Id)
    training, testing, validation = tf.split([.8, .1, .1])
    
    result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    model_kwargs=dict(embedding_dim=100),
    epochs=args.epochs)
    model=result.model
    entity_tensor= model.entity_representations[0]().detach().cpu().numpy()
    Id2Vec=dict(zip(Id2Node.values(),entity_tensor))
    return Id2Vec


# GCN

def eval_link_predictor(model, data):
    with torch.no_grad():
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs=args.epochs
):

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

#         if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def ExtractEmbeddingsPyG(model,features,edge_index):
    with torch.no_grad():
        Embs = model.encode(features,edge_index)
    Embs = Embs.detach().numpy()
    return Embs
    

    
    
    

def Main():
    kg = LoadData()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.model == "DLemb":
        
        entity_to_id,id_to_entity ,relation_to_id, triplets = PrepareDataForDLemb(kg)
        model = DLemb(100,len(entity_to_id))
        model.summary()
        gen = generate_batch(triplets, n_positive = 1500, negative_ratio=1)

        steps = len(triplets) // 1500
        
        # Train
        model.fit(gen, epochs = args.epochs, 
                                steps_per_epoch = steps,
                                verbose = 2)
        
        # Extract embeddings
        node_embeddings = model.get_layer('node_embeddings')
        node_embeddings = node_embeddings.get_weights()[0]
        node_embeddings.shape

        Id2Vec=dict(zip(id_to_entity.values(),node_embeddings))
        
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
    
    elif args.model == "GCN":
        pykg=Data()
        #Map Kg nodes to numbers
        id2node=dict(list(enumerate(kg.nodes())))
        node2id={v:k for k,v in id2node.items()}
        edge_index=torch.LongTensor([[node2id[n1], node2id[n2]] for (n1,n2) in kg.edges()])
        pykg.edge_index=edge_index.T
        #RANDOM FEATURES
        random_features = torch.rand(kg.number_of_nodes(),100)
        pykg.x=random_features
        train, validation, test = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1,)(pykg)
        gcn=GCN(pykg.num_features,100,100).to(device)
        optimizer = torch.optim.Adam(params=gcn.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        gcn_trained = train_link_predictor(gcn, train, validation, optimizer, criterion, n_epochs=args.epochs)
        Embs=ExtractEmbeddingsPyG(gcn_trained,pykg.x,pykg.edge_index)
        
        Id2Vec=dict(zip(list(kg.nodes()),Embs))
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
        
    elif args.model == "BioKG2Vec":
        Id2Vec = RunBioKG2Vec(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
    elif args.model == "N2V":
        Id2Vec = RunN2V(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
    elif args.model == "DistMult":
        Id2Vec = RunDistmult(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
    elif args.model == "TransE":
        Id2Vec = RunTransE(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)


if __name__ == '__main__':
    Main()
    
        
