import os
import csv
import json
import random
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from p_tqdm import p_umap  # 使用 p_umap 来进行并行化
from tqdm import tqdm  # 使用 tqdm 进度条
import warnings

# 忽略 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*fractional coordinates.*")

# 设置随机种子
random_seed = 999
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)

def smooth(array, sigma = 3):
    return gaussian_filter1d(array, sigma=sigma)
        
class CIFData(Dataset):
    def __init__(self, cif_folder_path, radius=2.5, dmin=0, step=0.2):

        self.radius = radius

        self.cif_folder_path = cif_folder_path

        dic = {}
        for i in os.listdir(self.cif_folder_path):
            structure_path = os.path.join(self.cif_folder_path, i)

            structure = Structure.from_file(structure_path)
            dic[i] = structure
        
        self.dic = dic

        with open('orbital_electrons.json') as f:
            self.emb = json.load(f)
        self.emb2 = {element: [
                    info.get('1s', 0), 
                    info.get('2s', 0), info.get('2p', 0), 
                    info.get('3s', 0), info.get('3p', 0), info.get('3d', 0), 
                    info.get('4s', 0), info.get('4p', 0), info.get('4d', 0), info.get('4f', 0),
                    info.get('5s', 0), info.get('5p', 0), info.get('5d', 0), info.get('5f', 0),
                    info.get('6s', 0), info.get('6p', 0), info.get('6d', 0), info.get('6f', 0),
                    info.get('7s', 0), info.get('7p', 0)
                ] for element, info in self.emb.items()}

        self.orbital_counts = {}
        for element, info in self.emb.items():
            # 计算s, p, d, f轨道的电子数
            s_count = info.get('1s', 0) + info.get('2s', 0) + info.get('3s', 0) + info.get('4s', 0) + info.get('5s', 0) + info.get('6s', 0) + info.get('7s', 0)
            p_count = info.get('2p', 0) + info.get('3p', 0) + info.get('4p', 0) + info.get('5p', 0) + info.get('6p', 0) + info.get('7p', 0)
            d_count = info.get('3d', 0) + info.get('4d', 0) + info.get('5d', 0) + info.get('6d', 0)
            f_count = info.get('4f', 0) + info.get('5f', 0) + info.get('6f', 0)
        
            # 将计算结果保存到字典中
            self.orbital_counts[element] = [s_count, p_count, d_count, f_count]
        

            #total_sum = sum(sum(vals) for vals in self.emb.values())
            #self.emb = {element: [val / 14 for val in vals] for element, vals in self.emb.items()}
            
        #elif embedding == 'atomic_number':
            #self.emb = nn.embedding(num_class = 118, dim = 18)  

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)


    def structure_to_graph(self, structure):
        """
        Convert pymatgen Structure object to a NetworkX graph.
        Nodes represent atoms, and edges represent bonds (bidirectional).
        """
        G = nx.Graph()  # Use undirected graph to ensure bidirectional edges
        
        # Add nodes with atomic information
        for i, site in enumerate(structure):
            G.add_node(i, element=site.species_string, coords=site.coords)
        
        # Add edges based on distance (simple cutoff method)
        cutoff = self.radius  # Use the specified radius as the cutoff
        for i, site1 in enumerate(structure):
            for j, site2 in enumerate(structure):
                if i < j and site1.distance(site2) < cutoff:
                    G.add_edge(i, j, weight=site1.distance(site2))  # Store the distance as the edge weight
        
        return G

    def load_data_single(self, name_id, crystal):
        """
        单个样本的加载任务，供p_umap并行化调用
        """

        sga = SpacegroupAnalyzer(crystal)
        
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()
        
        # Convert structure to graph (bidirectional edges)
        G = self.structure_to_graph(crystal)
        
        # Initialize edge_index and edge_attr
        edge_index = []
        edge_attr = []
        atom_fea = []
        elec_conf = []
        orbital_counts = []
        
        # Extract node features
        for _, node in G.nodes(data=True):
            element = node['element']
            # print(element)
            elec_conf.append(self.emb2[element])  # Making it a list to be compatible with PyTorch Geometric
            orbital_counts.append(self.orbital_counts[element])
            
            # Use atomic number as the feature
            element_symbol = Element(element)  # For Hydrogen
            atomic_number = element_symbol.number
            atom_fea.append([atomic_number])  # Making it a list to be compatible with PyTorch Geometric
        
        # Extract edges from the graph and create edge_index for bidirectional edges
        for edge in G.edges():
            # Add both directions for undirected edges
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # Add reverse direction as well
            edge_attr.append(G[edge[0]][edge[1]]['weight'])
            edge_attr.append(G[edge[1]][edge[0]]['weight'])  # Same weight for both directions

        #print(edge_attr)    
        #print(type(G[edge[0]][edge[1]]['weight'])) 

        # Convert lists to numpy arrays
        edge_index = np.array(edge_index).T  # Transpose for PyTorch format
        edge_attr = np.array(edge_attr)

        # Expand the distance features using Gaussian expansion
        edge_attr = self.gdf.expand(edge_attr)
        #print(edge_attr.shape)


        # Convert to tensors for PyTorch
        space_group_number = torch.Tensor([space_group_number])
        space_group_number = space_group_number.unsqueeze(dim=0)
        atom_fea = torch.Tensor(atom_fea)
        elec_conf = torch.Tensor(elec_conf)
        orbital_counts = torch.Tensor(orbital_counts)
        
        edge_attr = torch.Tensor(edge_attr)
        edge_index = torch.LongTensor(edge_index)
        
        energies = np.arange(-10, 10.1, 0.1)
        energies = torch.Tensor(energies)
        energies = energies.unsqueeze(dim=0)

        
        # Create the Data object for torch_geometric
        data = Data(name_id=name_id, x=atom_fea, edge_index=edge_index, edge_attr=edge_attr, energies = energies, space_group_number = space_group_number, elec_conf = elec_conf, orbital_counts = orbital_counts)

        return data

    def load_data(self):
        """
        使用tqdm并行加载数据并显示进度条
        """
        # 使用tqdm包裹并行处理数据加载
        data_list = []
        with tqdm(total=len(self.dic), unit="sample") as pbar:
            for name, crystal in self.dic.items():
                #print(name, crystal)
                data = self.load_data_single(name, crystal)
                data_list.append(data)
                pbar.update(1)  # 更新进度条
        return data_list


def data_loader(dataset, batch_size=128, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    

'''
if __name__ == '__main__':

    
    root_dir = 'C:/Users/11095/Desktop/第二项工作(新一轮)'
    elem_embedding_file = 'electron_configurations.json'
    dataset = CIFData(root_dir=root_dir, embedding_filename=elem_embedding_file)
    
    # 确保数据对象正确传递
    print('Loading Training Set')
    train_list = dataset.load_data(dataset.train_data)
    print('Loading Validation Set')
    val_list = dataset.load_data(dataset.val_data)
    print('Loading Test Set')
    test_list = dataset.load_data(dataset.test_data)
    
    train_loader = data_loader(train_list, batch_size=128, shuffle=True)
    val_loader = data_loader(val_list, batch_size=64, shuffle=False)
    test_loader = data_loader(test_list, batch_size=64, shuffle=False)
    
    # Save the datasets as .pth files
    torch.save(train_loader, 'train_data.pth')
    torch.save(val_loader, 'val_data.pth')
    torch.save(test_loader, 'test_data.pth')
'''