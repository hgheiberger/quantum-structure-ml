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

import pymatgen as mg
import pymatgen.io
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
import pymatgen.analysis.magnetism.analyzer as pg
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
#from pathlib import Path



#%% Process Materials Project Data
order_list_mp =  []
structures_list_mp = []
formula_list_mp = []
sites_list = []
id_list_mp = []
y_values_mp = []
order_encode = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}


mp_structures_dict = torch.load('/preload_data/mp_structures_2021-10-24_14-52.pt')
structures = mp_structures_dict['structures']

structures_copy = structures.copy()
for struc in structures_copy:
    if len(struc["structure"])>250:
        structures.remove(struc)
        print("MP Structure Deleted")

#%%
order_list = []
for i in range(len(structures)):
    order = pg.CollinearMagneticStructureAnalyzer(structures[i]["structure"])
    order_list.append(order.ordering.name)
id_NM = []
id_FM = []
id_AFM = []
for i in range(len(structures)):
    if order_list[i] == 'NM':
        id_NM.append(i)
    if order_list[i] == 'AFM':
        id_AFM.append(i)
    if order_list[i] == 'FM' or order_list[i] == 'FiM':
        id_FM.append(i)
np.random.shuffle(id_FM)
np.random.shuffle(id_NM)
np.random.shuffle(id_AFM)
id_AFM, id_AFM_to_delete = np.split(id_AFM, [int(len(id_AFM))])
id_NM, id_NM_to_delete = np.split(id_NM, [int(1.2*len(id_AFM))])
id_FM, id_FM_to_delete = np.split(id_FM, [int(1.2*len(id_AFM))])

structures_mp = [structures[i] for i in id_NM] + [structures[j] for j in id_FM] + [structures[k] for k in id_AFM]
np.random.shuffle(structures_mp)


for structure in structures_mp:
    analyzed_structure = pg.CollinearMagneticStructureAnalyzer(structure["structure"])
    order_list_mp.append(analyzed_structure.ordering)
    structures_list_mp.append(structure["structure"])
    formula_list_mp.append(structure["pretty_formula"])
    id_list_mp.append(structure["material_id"])
    sites_list.append(structure["nsites"])
    
for order in order_list_mp:
    y_values_mp.append(order_encode[order.name])    

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {'len_embed_feat': 64,
          'num_channel_irrep': 32,
          'num_e3nn_layer': 2,
          'max_radius': 5,
          'num_basis': 10,
          'adamw_lr': 0.005,
          'adamw_wd': 0.03
         }

#Used for debugging
identification_tag = "1:1:1.1 Relu wd:0.03 4 Linear"
cost_multiplier = 1.0

print('Length of embedding feature vector: {:3d} \n'.format(params.get('len_embed_feat')) + 
      'Number of channels per irreducible representation: {:3d} \n'.format(params.get('num_channel_irrep')) +
      'Number of tensor field convolution layers: {:3d} \n'.format(params.get('num_e3nn_layer')) + 
      'Maximum radius: {:3.1f} \n'.format(params.get('max_radius')) +
      'Number of basis: {:3d} \n'.format(params.get('num_basis')) +
      'AdamW optimizer learning rate: {:.4f} \n'.format(params.get('adamw_lr')) + 
      'AdamW optimizer weight decay coefficient: {:.4f}'.format(params.get('adamw_wd'))
     )


run_name = (time.strftime("%y%m%d-%H%M", time.localtime()))



structures = structures_list_mp
y_values =  y_values_mp
id_list = id_list_mp


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

len_element = 118
atom_types_dim = 3*len_element
embedding_dim = params['len_embed_feat']
lmax = 1
n_norm = 35  # Roughly the average number (over entire dataset) of nearest neighbors for a given atom

Rs_in = [(45, 0, 1)]  # num_atom_types scalars (L=0) with even parity
Rs_out = [(3,0,1)]  # len_dos scalars (L=0) with even parity

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
        self.linear = torch.nn.Linear(atom_type_in, 128)
        self.model = model
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 96)
        self.linear3 = torch.nn.Linear(96, 64)
        self.linear4 = torch.nn.Linear(64, 45)
        #self.linear5 = torch.nn.Linear(45, 32)
        #self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x, *args, batch=None, **kwargs):
        output = self.linear(x)
        output = self.relu(output)
        print(f"Input: {x}")
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        #output = self.linear5(output)
        output = self.relu(output)
        output = self.model(output, *args, **kwargs)
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        output = torch_scatter.scatter_add(output, batch, dim=0)
        print(f"Output: {output}")
        #output = self.softmax(output)
        return output

model = AtomEmbeddingAndSumLastLayer(atom_types_dim, embedding_dim, GatedConvParityNetwork(**model_kwargs))
opt = torch.optim.AdamW(model.parameters(), lr=params['adamw_lr'], weight_decay=params['adamw_wd'])

data = []
count=0
indices_to_delete=[]
for i, struct in enumerate(structures):
    try:
        print(f"Encoding sample {i+1:5d}/{len(structures):5d}", end="\r", flush=True)
        input = torch.zeros(len(struct), 3*len_element)
        for j, site in enumerate(struct):
            input[j, int(element(str(site.specie)).atomic_number)] = element(str(site.specie)).atomic_radius
            #input[j, len_element + int(element(str(site.specie)).atomic_number) +1] = element(str(site.specie)).atomic_weight
            input[j, len_element + int(element(str(site.specie)).atomic_number) +1] = element(str(site.specie)).en_pauling
            input[j, 2*len_element + int(element(str(site.specie)).atomic_number) +1] = element(str(site.specie)).dipole_polarizability
        data.append(DataPeriodicNeighbors(
            x=input, Rs_in=None, 
            pos=torch.tensor(struct.cart_coords.copy()), lattice=torch.tensor(struct.lattice.matrix.copy()),
            r_max=params['max_radius'],
            y = (torch.tensor([y_values[i]])).to(torch.long),
            n_norm=n_norm,
        ))

        count+=1
    except Exception as e:
        indices_to_delete.append(i)
        print(f"Error: {count} {e}", end="\n")
        count+=1
        continue

    
struc_dictionary = dict()
for i in range (len(structures)):
    struc_dictionary[i]=structures[i]

id_dictionary = dict()
for i in range (len(id_list)):
    id_dictionary[i]=id_list[i]

for i in indices_to_delete:
    del struc_dictionary[i]
    del id_dictionary[i] 

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


with open(run_name+'loss.txt', 'a') as f:
    f.write(f"Iteration: {identification_tag}")

batch_size = 1
dataloader = torch_geometric.data.DataLoader([data[i] for i in index_tr], batch_size=batch_size, shuffle=True)
dataloader_valid = torch_geometric.data.DataLoader([data[i] for i in index_va], batch_size=batch_size)

loss_fn = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.78)


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
            if d.y.item() == 2:
                 loss = cost_multiplier*loss_fn(output, d.y).cpu()
                 print("Multiplied Loss Index \n")
            elif d.y.item() == 0 or d.y.item() == 1:
                 loss = loss_fn(output, d.y).cpu()
                 print("Standard Loss Index \n")
            else:
                 print("Lost datapoint \n")
            loss_cumulative = loss_cumulative + loss.detach().item()
    return loss_cumulative / len(dataloader)





#run_name='abc'
#with open (run_name+'aa.txt','a') as f:
#	x=1

def savedata(step):
	x_test = []
	y_test = []
	y_score = []
	y_pred = []

	letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}

	training_composition_dict = {}
	training_sites_dict = {}
	
	for i, index in enumerate(index_tr):
	    d = torch_geometric.data.Batch.from_data_list([data[index]])
	    d.to(device)
	    output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
	
	    if max(output[0][0], output[0][1], output[0][2]) == output[0][0]:
	        output = 0
	    elif max(output[0][0], output[0][1], output[0][2]) == output[0][1]:
	        output = 1
	    else:
	        output = 2
	    with open(f'{run_name}{step}training_results.txt', 'a') as f:
	                f.write(f"{id_list[index]} {formula_list_mp[index]} Prediction: {output} Actual: {d.y} \n")
	     
	    
	    correct_flag = d.y.item() == output
	    
	    #Accuracy per element calculation
	    current_element = ""
	    for char_index in range(len(formula_list_mp[index])):
	        print("Entered Loop")
	        formula = formula_list_mp[index]
	
	        if formula[char_index] in letters:
	            current_element += formula[char_index]
	            print(f"Using char: {formula[char_index]}")
	            if char_index + 1 == len(formula) or formula[char_index + 1].isupper() or formula[char_index + 1] not in letters: 
	                print(f"printing to dict {current_element}")
	                if correct_flag:
	                    current_entry = training_composition_dict.get(current_element, [0,0])
	                    current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	                else:
	                    current_entry = training_composition_dict.get(current_element, [0,0])
	                    current_entry = [current_entry[0], current_entry[1] + 1]
	                training_composition_dict[current_element] = current_entry
	                current_element = ""
	  
	    #Accuracy per nsites calculation
	    current_nsites = sites_list[index]
	    if correct_flag:
	        current_entry = training_sites_dict.get(current_nsites, [0,0])
	        current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	    else:
	        current_entry = training_sites_dict.get(current_nsites, [0,0])
	        current_entry = [current_entry[0], current_entry[1] + 1]
	    training_sites_dict[current_nsites] = current_entry
	    
	#Accuracy per element depiction
	with open(f'{run_name}{step}training_composition_info.txt', 'a') as f:
	    f.write("Training Composition Ratios: \n")            
	    for key, value in training_composition_dict.items():
	        f.write(f"Element: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	
	#Accuracy per nsites depiction
	with open(f'{run_name}{step}training_nsites_info.txt', 'a') as f:
	    f.write("Training Nsites Info: \n")
	    for key, value in training_sites_dict.items():
	        f.write(f"nsites: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	  
	validation_composition_dict = {}
	validation_sites_dict = {}  
	  
	for i, index in enumerate(index_va):
	    d = torch_geometric.data.Batch.from_data_list([data[index]])
	    d.to(device)
	    output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
	    
	    with open(f'{run_name}{step}validation_results.txt', 'a') as f:
	        f.write(f"Output for below sample: {torch.exp(output)} \n")
	
	    if max(output[0][0], output[0][1], output[0][2]) == output[0][0]:
	        output = 0
	    elif max(output[0][0], output[0][1], output[0][2]) == output[0][1]:
	        output = 1
	    else:
	        output = 2
	    with open(f'{run_name}{step}validation_results.txt', 'a') as f:
	                f.write(f"{id_list[index]} {formula_list_mp[index]} Prediction: {output} Actual: {d.y} \n")
	    
	    correct_flag = d.y.item() == output
	    
	    #Accuracy per element calculation
	    current_element = ""
	    for char_index in range(len(formula_list_mp[index])):
	        print("Entered Loop")
	        formula = formula_list_mp[index]
	
	        if formula[char_index] in letters:
	            current_element += formula[char_index]
	            print(f"Using char: {formula[char_index]}")
	            if char_index + 1 == len(formula) or formula[char_index + 1].isupper() or formula[char_index + 1] not in letters: 
	                print(f"printing to dict {current_element}")
	                if correct_flag:
	                    current_entry = validation_composition_dict.get(current_element, [0,0])
	                    current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	                else:
	                    current_entry = validation_composition_dict.get(current_element, [0,0])
	                    current_entry = [current_entry[0], current_entry[1] + 1]
	                validation_composition_dict[current_element] = current_entry
	                current_element = ""
	 
	    #Accuracy per nsites calculation
	    current_nsites = sites_list[index]
	    if correct_flag:
	        current_entry = validation_sites_dict.get(current_nsites, [0,0])
	        current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	    else:
	        current_entry = validation_sites_dict.get(current_nsites, [0,0])
	        current_entry = [current_entry[0], current_entry[1] + 1]
	    validation_sites_dict[current_nsites] = current_entry
	
	#Accuracy per element depiction
	with open(f'{run_name}{step}validation_composition_info.txt', 'a') as f:
	    f.write("Validation Composition Ratios: \n")            
	    for key, value in validation_composition_dict.items():
	        f.write(f"Element: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	
	#Accuracy per nsites depiction
	with open(f'{run_name}{step}validation_nsites_info.txt', 'a') as f:
	    f.write("Validation Nsites Info: \n")
	    for key, value in validation_sites_dict.items():
	        f.write(f"nsites: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	
	testing_composition_dict = {}
	testing_sites_dict = {}
	letters = {"a",'b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A', 'B', 'C', 'D', 'E','F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
	
	for i, index in enumerate(index_te): 
	    with torch.no_grad():
	        print(len(index_te))
	        print(f"Index being tested: {index}") 
	        d = torch_geometric.data.Batch.from_data_list([data[index]])
	        d.to(device)
	        output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
	
	        y_test.append(d.y.item())
	        
	        y_score.append(output)
	       
	        with open(f'{run_name}{step}testing_results.txt', 'a') as f:
	            f.write(f"Output for below sample: {torch.exp(output)} \n")
	       
	        if max(output[0][0], output[0][1], output[0][2]) == output[0][0]:
	            output = 0
	        elif max(output[0][0], output[0][1], output[0][2]) == output[0][1]:
	            output = 1
	        else:
	            output = 2
	        y_pred.append(output)
	        with open(f'{run_name}{step}testing_results.txt', 'a') as f:
	            f.write(f"{id_list[index]} {formula_list_mp[index]} Prediction: {output} Actual: {d.y} \n")
	        
	        correct_flag = d.y.item() == output
	       
	        # Accuracy per element calculation
	        current_element = ""
	        for char_index in range(len(formula_list_mp[index])):
	            print("Entered Loop")
	            formula = formula_list_mp[index]
	
	            if formula[char_index] in letters:
	                current_element += formula[char_index]
	                print(f"Using char: {formula[char_index]}")
	                if char_index + 1 == len(formula) or formula[char_index + 1].isupper() or formula[char_index + 1] not in letters: 
	                    print(f"printing to dict {current_element}")
	                    if correct_flag:
	                        current_entry = testing_composition_dict.get(current_element, [0,0])
	                        current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	                    else:
	                        current_entry = testing_composition_dict.get(current_element, [0,0])
	                        current_entry = [current_entry[0], current_entry[1] + 1]
	                    testing_composition_dict[current_element] = current_entry
	                    current_element = ""
	                    
	        #Accuracy per nsites calculation
	        current_nsites = sites_list[index]
	        if correct_flag:
	            current_entry = testing_sites_dict.get(current_nsites, [0,0])
	            current_entry = [current_entry[0] + 1, current_entry[1] + 1]
	        else:
	            current_entry = testing_sites_dict.get(current_nsites, [0,0])
	            current_entry = [current_entry[0], current_entry[1] + 1]
	        testing_sites_dict[current_nsites] = current_entry
	
	#Accuracy per element depiction
	with open(f'{run_name}{step}testing_composition_info.txt', 'a') as f:
	    f.write("Testing Composition Ratios: \n")            
	    for key, value in testing_composition_dict.items():
	        f.write(f"Element: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	
	#Accuracy per nsites depiction
	with open(f'{run_name}{step}testing_nsites_info.txt', 'a') as f:
	    f.write("Testing Nsites Info: \n")
	    for key, value in testing_sites_dict.items():
	        f.write(f"nsites: {key} Ratio: {value[0]}/{value[1]} Fraction: {value[0]/value[1]}\n")
	
	#accuracy_score = accuracy_score(y_test, y_pred)
	
	with open(f'{run_name}{step}y_pred.txt', 'a') as f:
	    f.write("Predicted Values \n")
	    f.write(str(y_pred))
	
	with open(f'{run_name}{step}y_test.txt', 'a') as f:
	    f.write("Actual Values \n")
	    f.write(str(y_test))
	
	with open(f'{run_name}{step}statistics.txt', 'a') as f:
	    f.write("\n")
	    f.write("Network Analytics: \n")
	    f.write(f"Identification tag: {identification_tag}\n")
	    #f.write(f"Accuracy score: {accuracy_score}\n")
	    #f.write("Classification Report: \n")  
	    #f.write(classification_report(y_test, y_pred, target_names=["NM","AFM", "FM"]))
	    f.write("\n")

	
	


def train(model, optimizer, dataloader, dataloader_valid, max_iter=101, device="cpu"):
    model.to(device)
    
    checkpoint_generator = loglinspace(3.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []
    
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
            with open('loss.txt', 'a') as f:
                f.write(f"train average loss: {str(train_avg_loss)} \n")
                f.write(f" validation average loss: {str(valid_avg_loss)} \n")
        scheduler.step()
 

for results in train(model, opt, dataloader, dataloader_valid, device=device, max_iter=45):
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















