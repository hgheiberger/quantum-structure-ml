import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_scatter

import e3nn
from e3nn import rs, o3
from e3nn.point.data_helpers import DataPeriodicNeighbors
from e3nn.networks import GatedConvParityNetwork
from e3nn.kernel_mod import Kernel
from e3nn.point.message_passing import Convolution

import pymatgen
from pymatgen.core.structure import Structure
import numpy as np
import pickle
from mendeleev import element
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import io
import random
import math
import sys 
import time, os
import datetime


torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {'len_embed_feat': 64,
          'num_channel_irrep': 32,
          'num_e3nn_layer': 2,
          'max_radius': 5,
          'num_basis': 10,
          'adamw_lr': 0.007,
          'adamw_wd': 0.06
         }



print('Length of embedding feature vector: {:3d} \n'.format(params.get('len_embed_feat')) + 
      'Number of channels per irreducible representation: {:3d} \n'.format(params.get('num_channel_irrep')) +
      'Number of tensor field convolution layers: {:3d} \n'.format(params.get('num_e3nn_layer')) + 
      'Maximum radius: {:3.1f} \n'.format(params.get('max_radius')) +
      'Number of basis: {:3d} \n'.format(params.get('num_basis')) +
      'AdamW optimizer learning rate: {:.4f} \n'.format(params.get('adamw_lr')) + 
      'AdamW optimizer weight decay coefficient: {:.4f}'.format(params.get('adamw_wd'))
     )


run_name = (time.strftime("%y%m%d-%H%M", time.localtime()))

#Used for debugging
identification_tag = f"This is 6.py Trial 1 {run_name} 0.6, relu test"
print(identification_tag)

cif=[]
y_indices=[]
filepaths=[]
id_list=[]
import os,glob
import codecs

folder_path = 'data'

for filename in glob.glob(os.path.join(folder_path, '*.mcif')):
    if (str(filename))[15]=="0":
        y_indices.append(0)
    else:
        y_indices.append(1)
    with codecs.open(filename, 'rb', encoding='utf-8', errors='ignore') as structure:
        cif.append(structure.readlines())
    filepaths.append(filename)
    count=0
    str_file=(str(filename))[10:]
    while str_file[count]!="m":
        count+=1
    id_list.append(str_file[:count-1])

structures=[]
indices_to_delete=[]
for i,c in enumerate(cif):
    try:
        print(i)
        structures.append(Structure.from_str("\n".join(c), "CIF"))
        if len(structures[-1])>250:
            structures.pop()
            indices_to_delete.append(i)
            print(f"Structure>75: {filepaths[i]}")
    except NotImplementedError:
        print(i)
        print(f"NotImplementedError: {filepaths[i]}")
        indices_to_delete.append(i)
        continue
    except ValueError:
        print(i)
        print(f"Value Error: {filepaths[i]}")
        indices_to_delete.append(i)
        continue
    except AssertionError:
        print(i)
        print(f"Assertion Error: {filepaths[i]}")
        indices_to_delete.append(i)
        continue

cif_dictionary = dict()
for i in range (len(cif)):
    cif_dictionary[i]=cif[i]

y_dictionary = dict()
for i in range (len(y_indices)):
    y_dictionary[i]=y_indices[i]
    
id_dictionary = dict()
for i in range (len(id_list)):
    id_dictionary[i]=id_list[i]

for i in indices_to_delete:
    del cif_dictionary[i]
    del y_dictionary[i]
    del id_dictionary[i]
 
cif2=[]       
for i in range (len(cif)):
    if i in cif_dictionary.keys():
        cif2.append(cif_dictionary[i])
cif = cif2

y2=[]       
for i in range (len(y_indices)):
    if i in y_dictionary.keys():
        y2.append(y_dictionary[i])
y_indices = y2  

id2=[]       
for i in range (len(id_list)):
    if i in id_dictionary.keys():
        id2.append(id_dictionary[i])
id_list = id2  
    


species = set()
count =0
for struct in structures[:]:
    try:
        species = species.union(list(set(map(str, struct.species))))
        count+=1
    except:
        print(count)
        count+=1
        continue
species = sorted(list(species))
print("Distinct atomic species ", len(species))

len_element = 125
atom_types_dim = len_element
embedding_dim = params['len_embed_feat']
lmax = 1
n_norm = 35  # Roughly the average number (over entire dataset) of nearest neighbors for a given atom

Rs_in = [(embedding_dim, 0, 1)]  # num_atom_types scalars (L=0) with even parity
Rs_out = [(1,0,1)]  # len_dos scalars (L=0) with even parity

model_kwargs = {
    "convolution": Convolution,
    "kernel": Kernel,
    "Rs_in": Rs_in,
    "Rs_out": Rs_out,
    "mul": params['num_channel_irrep'], # number of channels per irrep (differeing L and parity)
    "layers": params['num_e3nn_layer'],
    "max_radius": params['max_radius'],
    "lmax": lmax,
    "number_of_basis": params['num_basis']
}
print(model_kwargs)
        
class AtomEmbeddingAndSumLastLayer(torch.nn.Module):
    def __init__(self, atom_type_in, atom_type_out, model):
        super().__init__()
        self.linear = torch.nn.Linear(atom_type_in, atom_type_out)
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        #self.relu=torch.nn.ReLU()
    def forward(self, x, *args, batch=None, **kwargs):
        output = self.linear(x)
        #output = self.relu(output)
        output=self.softmax(output)
        output = self.model(output, *args, **kwargs)
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        output = torch_scatter.scatter_add(output, batch, dim=0)
        output=torch.sigmoid(output)
        return output
        
model = AtomEmbeddingAndSumLastLayer(atom_types_dim, embedding_dim, GatedConvParityNetwork(**model_kwargs))
opt = torch.optim.AdamW(model.parameters(), lr=params['adamw_lr'], weight_decay=params['adamw_wd'])


data = []
count=0
indices_to_delete=[]
for i, struct in enumerate(structures):
    try:
        print(f"Encoding sample {i+1:5d}/{len(structures):5d}", end="\r", flush=True)
        input = torch.zeros(len(struct), len_element)
        for j, site in enumerate(struct):
            input[j, int(element(str(site.specie)).atomic_number)] = element(str(site.specie)).atomic_weight
        data.append(DataPeriodicNeighbors(
            x=input, Rs_in=None, 
            pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
            r_max=params['max_radius'],
            y = (torch.tensor([[y_indices[i]],])).to(torch.double),
            n_norm=n_norm,
        ))
        count+=1
    except:
        indices_to_delete.append(i)
        print(f"Error: {count}", end="\n")
        count+=1
        continue

cif_dictionary = dict()
for i in range (len(cif)):
    cif_dictionary[i]=cif[i]
    
struc_dictionary = dict()
for i in range (len(structures)):
    struc_dictionary[i]=structures[i]

id_dictionary = dict()
for i in range (len(id_list)):
    id_dictionary[i]=id_list[i]

for i in indices_to_delete:
    del cif_dictionary[i]
    del struc_dictionary[i]
    del id_dictionary[i]
 
cif2=[]       
for i in range (len(cif)):
    if i in cif_dictionary.keys():
        cif2.append(cif_dictionary[i])
cif = cif2

structures2=[]       
for i in range (len(structures)):
    if i in struc_dictionary.keys():
        structures2.append(struc_dictionary[i])
structures = structures2

id2=[]       
for i in range (len(id_list)):
    if i in id_dictionary.keys():
        id2.append(id_dictionary[i])
id_list = id2 

compound_list=[]
for i, struc in enumerate(structures):
    str_struc = (str(struc))
    count=0
    while str_struc[count]!=":":
        count+=1
    str_struc = str_struc[count+2:]
    count=0
    while str_struc[count:count+3]!="abc":
        count+=1
    str_struc = str_struc[:count]
    compound_list.append(str_struc) 

torch.save(data, run_name+'_data.pt')
    
indices = np.arange(len(structures))
np.random.shuffle(indices)
index_tr, index_va, index_te = np.split(indices, [int(.8 * len(indices)), int(.9 * len(indices))])
    
assert set(index_tr).isdisjoint(set(index_te))
assert set(index_tr).isdisjoint(set(index_va))
assert set(index_te).isdisjoint(set(index_va))

#To log the runtime parameters and mcif indices, uncomment these lines 
# with open('models/200801_trteva_indices.pkl', 'wb') as f: 
#     pickle.dump([index_tr, index_va, index_te], f)
#with open(run_name+'_miscinfo.pkl', 'wb') as f: 
#   pickle.dump([model_kwargs, params], f)

with open(run_name+'loss.txt', 'a') as f:
    f.write(f"Iteration: {identification_tag} \n")

batch_size = 1
dataloader = torch_geometric.data.DataLoader([data[i] for i in index_tr], batch_size=batch_size, shuffle=True)
dataloader_valid = torch_geometric.data.DataLoader([data[i] for i in index_va], batch_size=batch_size)

loss_fn = torch.nn.BCELoss()

scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.78)

def savedata(step):
    for i, index in enumerate(index_tr):
        with torch.no_grad():
            d = torch_geometric.data.Batch.from_data_list([data[index]])
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            output_ori=output
            if output>=0.6:
                output=1
            else:
                output=0
            with open(f'{run_name}{step}training_results.txt', 'a') as f:
                        f.write(f"{id_list[index]} {compound_list[index]} Prediction: {output} Actual: {d.y.item()} Yscore: {output_ori} \n")
     

    for i, index in enumerate(index_va):
        with torch.no_grad():
            d = torch_geometric.data.Batch.from_data_list([data[index]])
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            output_ori=output
            if output>=0.6:
                output=1
            else:
                output=0
            with open(f'{run_name}{step}validation_results.txt', 'a') as f:
                        f.write(f"{id_list[index]} {compound_list[index]} Prediction: {output} Actual: {d.y.item()} Yscore: {output_ori} \n")

    for i, index in enumerate(index_te): 
        with torch.no_grad():
            print(len(index_te))
            print(f"Index being tested: {index}") 
            d = torch_geometric.data.Batch.from_data_list([data[index]])
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            output_ori=output
            if output>=0.6:
                output=1
            else:
                output=0
            with open(f'{run_name}{step}testing_results.txt', 'a') as f:
                        f.write(f"{id_list[index]} {compound_list[index]} Prediction: {output} Actual: {d.y.item()} Yscore: {output_ori} \n")

    #average_precision = average_precision_score(y_test, y_score)
    #f1_score_test = f1_score(y_test, y_pred, average="binary")
    #accuracy_score_test = accuracy_score(y_test, y_pred) 
    #accuracy_score_valid = accuracy_score(y_true_valid, y_pred_valid) 
    #accuracy_score_train = accuracy_score(y_true_train, y_pred_train) 
    #classification_report(y_test, y_pred, target_names=["class 0","class 1"])



















def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))

def evaluate(model, dataloader, device):
    model.eval()
    loss_cumulative = 0.
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
    return loss_cumulative / len(dataloader)

def train(model, optimizer, dataloader, dataloader_valid, max_iter=70., device="cpu"):
    model.to(device)
    
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []
    accuracy_valid_list=[]
    
    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " +
                  f"batch loss = {loss.data}", end="\r", flush=True)
            loss_cumulative = loss_cumulative + loss.detach().item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        end_time = time.time()
        wall = end_time - start_time
        
        if step == checkpoint:
            savedata(step)
            #accuracy_valid_list.append(acc_temp)
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            
            valid_avg_loss = evaluate(model, dataloader_valid, device)
            train_avg_loss = evaluate(model, dataloader, device)

            dynamics.append({
                'step': step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                },
                'valid': {
                    'loss': valid_avg_loss,
                },
                'train': {
                    'loss': train_avg_loss,
                },
            })

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }
            
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " +
                  f"train loss = {train_avg_loss:8.3f}   " +
                  f"valid loss = {valid_avg_loss:8.3f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
            with open(run_name+'loss.txt', 'a') as f:
                f.write(f"train average loss: {str(train_avg_loss)} \n")
                f.write(f" validation average loss: {str(valid_avg_loss)} \n")
        scheduler.step()

 

for results in train(model, opt, dataloader, dataloader_valid, device=device, max_iter=70):
    with open(run_name+'_trial_run_full_data.torch', 'wb') as f:
        results['model_kwargs'] = model_kwargs
        torch.save(results, f)
    
saved = torch.load(run_name+'_trial_run_full_data.torch')
steps = [d['step'] + 1 for d in saved['dynamics']]
valid = [d['valid']['loss'] for d in saved['dynamics']]
train = [d['train']['loss'] for d in saved['dynamics']]

plt.plot(steps, train, 'o-', label="train")
plt.plot(steps, valid, 'o-', label="valid")
plt.legend()
plt.savefig(run_name+'_hist.png',dpi=300)

