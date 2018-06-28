import pickle
import os
import numpy as np
from tqdm import tqdm


gallery_names = pickle.load(open('name.pkl','rb'))
gallery_feats = pickle.load(open('features.pkl','rb')) 
query_names =gallery_names #pickle.load(open('query_name','rb'))
query_feats =gallery_feats #pickle.load(open('query_feat','rb'))   

query_names = np.array(query_names)
query_feats = np.array(query_feats)
gallery_names = np.array(gallery_names)
gallery_feats = np.array(gallery_feats)

def retrieval_k(query,query_name, feats,names,k=6):
    # score = np.sqrt(np.sum((feats-query)**2 ,axis=1))
    a=np.linalg.norm(feats,axis=1)
    b=np.linalg.norm(query,axis=1)
    score = np.dot(feats,query.T).flatten()
    score = 1-(score.flatten()/(a*b))
    
    rand_idx = np.argsort(score)
    result = names[rand_idx[:k]]
    real_label = query_name.split('/')[3]
    is_exist =False
    for idx,item in enumerate(result):
        if idx == 0:
            continue
        label = item.split('/')[3]
        if label == real_label:
            is_exist = True
            break
    return is_exist

result = []
cuple = [(f,n)for f, n in zip(query_feats,query_names)]
for f, n in tqdm(cuple):
    query,qname = np.expand_dims(f,axis=0),n
    # print(qname)
    output = retrieval_k(query,qname,gallery_feats,gallery_names,k=20)
    result.append(output)

print(sum(result)/len(names))    