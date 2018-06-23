import numpy as np
from collections import OrderedDict
import pickle
import pandas as pd


train_data = OrderedDict()
with open("emb_train.pkl", 'rb') as f:
    tmp_cl = pickle.load(f)
with open("emb_train1.pkl", 'rb') as f:
    tmp_e = np.load(f)
for i, (k, v) in enumerate(tmp_cl.items()):
    train_data[k] = (v, tmp_e[i])

val_data = OrderedDict()
with open("emb_test.pkl", 'rb') as f:
    tmp_cl = pickle.load(f)
with open("emb_test1.pkl", 'rb') as f:
    tmp_e = np.load(f)
for i, (k, v) in enumerate(tmp_cl.items()):
    val_data[k] = (v, tmp_e[i])

# print(len(val_data), len(train_data))
# for i, (k, v) in enumerate(train_data.items()):
#     print(k, v)
#     break
# for i, (k, v) in enumerate(val_data.items()):
#     print(k, v)
#     break

# dist = np.zeros(shape=(len(val_data), len(train_data)), dtype=np.float32)
# for vi, (vname, (vcls, vemb)) in enumerate(val_data.items()):
#     print(vi)
#     for ti, (tname, (tcls, temb)) in enumerate(train_data.items()):
#         dist[vi, ti] = np.linalg.norm(vemb-temb)

# with open('dist.npz', 'wb') as f:
#      np.save(f, dist)
# exit(0)

df = pd.read_csv("./data/train.csv")

with open('dist.npz', 'rb') as f:
    dist = np.load(f)

tnames = list(train_data.keys())
vnames = list(val_data.keys())
print("Image,Id")
for i in range(len(val_data)):
    ind = np.argsort(dist[i])[:5]
    labels = [0,0,0,0,0]
    for j, j_ind in enumerate(ind):
        _, lbl = df[df["Image"] == tnames[j_ind]].iloc[0].tolist()
        labels[j] = lbl
        #labels[j] = train_data[tnames[j_ind]][0]
    print('{},{} {} {} {} {}'.format(vnames[i], *labels))
