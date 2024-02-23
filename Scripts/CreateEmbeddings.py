#!/usr/bin/env python
import pandas as pd
import pickle
import os
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import torch
from itertools import product
import subprocess


#BIOKG2VEC IMPORTS
from gensim.models import Word2Vec

# TRANSE DISTMULT IMPORTS 
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline





#STELLARGRAPH
from stellargraph import StellarGraph
from stellargraph.data import UniformRandomMetaPathWalk

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


def RunBioKG2Vec(outto,par = "{('drug','protein'):0,('protein','function'):10,('function','phenotype'):100}",ep = args.epochs):
    
    stri = f"""../tools/knowalk-main/KW2VEC.py -e ../KGs/edgelist.csv -w {par} -s True -d True --epochs {ep} -c 1 -o {outto}"""
    print(stri)
    subprocess.call(stri.split(' '))

def RunDLemb(outto,ep = args.epochs):
    
    subprocess.call(f"../tools/DLemb-main/DLemb.py -i ../KGs/edgelist.csv -e {ep} -o {outto}".split(' '))



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


def ParseKG(kg,subnet = False):
    nodes = list(kg.nodes())
    node_attributes = nx.get_node_attributes(kg,'tipo')
    edge_attributes = nx.get_edge_attributes(kg,'rel_type')
    nx.set_node_attributes(kg,node_attributes,'label')
    nx.set_edge_attributes(kg,edge_attributes,'label')
    if subnet:
        randomnodes = random.sample(nodes,subnet)
        subnet_kg = kg.subgraph(randomnodes)
        subnet_kg = nx.Graph(subnet_kg)
        subnet_kg.remove_nodes_from(list(nx.isolates(subnet_kg)))
        return subnet_kg
    else:
        return kg


def typology(i):
    if i.isnumeric():
        return 'protein'
    elif 'GO' in i:
        return 'function'
    elif 'C' in i:
        return 'phenotype'
    elif 'DB' in i:
        return 'drug'
    else:
        print(i)

def Convertkgtostellar(kg):
    stellar_kg = StellarGraph.from_networkx(kg)
    return stellar_kg
    
    
def run_metapath2vec(graph):
    # specify the metapath schemas as a list of lists of node types.
    types = list(set([typology(h) for h in graph.nodes()]))
    metapaths= list(product(types,repeat = 3))
    metapaths = [list(m) for m in metapaths]
    metapaths = [a for a in metapaths if a[0]==a[2]]

    # Create the random walker
    rw = UniformRandomMetaPathWalk(graph)

    walks = rw.run(
        nodes=list(graph.nodes()),  # root nodes
        length=50,  # maximum length of a random walk
        n=3,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )
    model = Word2Vec(window = 5, sg = 1, hs = 0,
                 negative = 5, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,workers=7,min_count=1)

    
    model.build_vocab(walks, progress_per=2)

    model.train(walks, total_examples = model.corpus_count, epochs=15, report_delay=1)

    print("Number of random walks:{}".format(len(walks)))
    Id2Vec=dict(zip(model.wv.index_to_key,model.wv.vectors))
    return Id2Vec


def Main():
    kg = LoadData()
    kg_parsed = ParseKG(kg)
    stellar_kg = Convertkgtostellar(kg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    edgelist = nx.to_pandas_edgelist(kg_parsed)
    edgelist = edgelist[['source','rel_type','target']]
    edgelist['source_type'] =  edgelist.source.apply(lambda x: typology(x))
    edgelist['target_type'] =  edgelist.target.apply(lambda x: typology(x))
    edgelist.to_csv('../KGs/edgelist.csv',index = False)
    if args.model == "DLemb":
        print(" | ========== > RUNNING DLEMB")
        RunDLemb(args.output)
        
    elif args.model == "BioKG2Vec":
        print(" | ========== > RUNNING BIOKG2VEC")
        RunBioKG2Vec(args.output)
        
    elif args.model == "N2V":
        print(" | ========== > RUNNING NODE2VEC")
        Id2Vec = RunN2V(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
            
    elif args.model == "M2V":
        print(" | ========== > RUNNING METAPATH2VEC")
        id2vec_meta = run_metapath2vec(stellar_kg)
        with open(args.output,'wb') as f:
            pickle.dump(id2vec_meta,f)
    
    
    elif args.model == "DistMult":
        print(" | ========== > RUNNING DISTMULT")
        Id2Vec = RunDistmult(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)
    elif args.model == "TransE":
        print(" | ========== > RUNNING TRANSE")
        Id2Vec = RunTransE(kg)
        with open(args.output,'wb') as f:
            pickle.dump(Id2Vec,f)


if __name__ == '__main__':
    Main()
    
        
