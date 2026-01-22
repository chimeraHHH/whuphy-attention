
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter
import math

# =============================================================================
# 第一部分：核心组件模块 (Core Components)
# 这里定义了模型搭建所需要的“积木”，包括：
# 1. RBFExpansion: 用来精细地描述原子之间的距离（不仅是远近，还能区分化学键类型）。
# 2. MatformerConv: 模型的大脑，专门用来处理原子如何“交谈”和“互动”。
# 3. VirtualNodeBlock: 虚拟原子模块，像一个总指挥，负责协调全局信息。
# =============================================================================

class RBFExpansion(nn.Module):
    """
    [层级 1：几何结构编码] (Geometric Encoding)
    功能：将原子之间的“距离数值”转换成一组“高维特征向量”。
    通俗解释：
    这就好比我们不只是告诉模型“A和B距离1.5埃”，而是用一组高斯函数去扫描这个距离。
    这样模型能更敏锐地感知到微小的距离变化（比如化学键的拉伸或压缩），这对于捕捉晶体结构细节非常重要。
    """
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.center = torch.linspace(vmin, vmax, bins)
        self.sigma = (vmax - vmin) / bins
        if lengthscale is not None:
            self.sigma = lengthscale
        self.register_buffer('centers', self.center)

    def forward(self, distance):
        return torch.exp(-(distance.unsqueeze(1) - self.centers) ** 2 / self.sigma ** 2)

class MatformerConv(MessagePassing):
    """
    [层级 2：局部结构交互] (Local Structural Interaction)
    功能：实现原子与其邻居原子之间的信息交换（注意力机制）。
    优化点：采用了 Beta 衰减残差连接和全向量门控机制。
    
    通俗解释：
    这是模型的“社交网络”。每个原子都会查看它周围的邻居：
    1. 它是谁？(Key)
    2. 我们离得有多远？(Edge)
    3. 我需要关注它吗？(Attention)
    
    我们引入了“Beta衰减”技术，这就像一个智能调节阀，如果新学到的信息太嘈杂，
    模型可以选择忽略它，保留原来的记忆。这让模型即使堆叠得很深，也不会“学傻了”（过平滑）。
    """
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=None, dropout=0.0, beta=True):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.beta = beta

        # 定义 Query, Key, Value 的转换层（Transformer 的标配）
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # 专门处理边（距离）信息的层
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        
        # [优化] Beta 残差参数：用于智能调节新旧信息的比例
        if self.beta:
            self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
        else:
            self.register_parameter('lin_beta', None)

        # 消息更新层
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels), 
            nn.LayerNorm(out_channels)
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.lin_skip = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        # 1. 将原子特征转换为 Query, Key, Value
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        # 2. 开始消息传递（每个原子收集邻居的信息）
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        
        out = out.view(-1, H * C)
        out = self.lin_concate(out)
        out = F.silu(self.bn(out))
        
        # [优化] Beta-Decay 残差连接
        # 这里决定了：是多听听邻居的（新信息），还是多相信自己原来的（旧信息）
        x_r = self.lin_skip(x)
        if self.beta and self.lin_beta is not None:
            # 计算融合比例 beta
            beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
            beta = beta.sigmoid()
            # 动态融合
            out = beta * x_r + (1 - beta) * out
        else:
            out = out + x_r
            
        return out

    def message(self, query_i, key_i, key_j, value_j, value_i, edge_attr, index):
        # 这个函数定义了两个原子之间具体如何“交流”
        
        if self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            edge_emb = 0
            
        # [优化] Matformer 核心逻辑：三元组交互 (Triplet Interaction)
        # 我们不仅看 A 和 B，还看 A-B 之间的距离。
        # query_i: 中心原子 A
        # key_j:   邻居原子 B
        # edge_emb: 它们之间的距离
        
        # 1. 增强 Query 和 Key，让它们包含彼此的信息
        query_i_aug = torch.cat((query_i, query_i, query_i), dim=-1) 
        key_j_aug = torch.cat((key_i, key_j, edge_emb), dim=-1)      
        
        # 2. 计算注意力分数 (Attention Score)
        # 这里的乘法是在衡量：A 和 B 的关系有多紧密？
        alpha = (query_i_aug * key_j_aug) / math.sqrt(self.out_channels * 3)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 3. 准备要传递的信息 (Value)
        out = torch.cat((value_i, value_j, edge_emb), dim=-1) 
        
        # 4. 全向量门控机制 (Full Gating)
        # 这是一个精细的过滤器，它可以针对每一个特征维度决定是否放行。
        # 比如：也许 A 只关心 B 的电子数，但不关心 B 的质量，门控机制可以自动学会这一点。
        gate = self.sigmoid(self.layer_norm(alpha)) 
        out = self.lin_msg_update(out) * gate
        
        # 压缩信息维度，准备发送
        out = self.msg_layer(out) 
        
        return out

class VirtualNodeBlock(nn.Module):
    """
    [层级 3：虚拟原子与全局交互] (Virtual Atom / Global Interaction)
    功能：模拟一个看不见的“超级原子”，它能瞬间连接所有真实原子。
    
    通俗解释：
    在普通的晶体中，相距很远的两个原子很难直接交流（就像住在地球两端的人）。
    我们引入了一个“虚拟原子”（类似于互联网），它能瞬间收集所有人的信息，
    然后把总结后的全局信息广播给每一个人。这大大增强了模型捕捉长程效应的能力。
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        # 用于更新虚拟原子状态的神经网络
        self.mlp_virtual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        # 用于更新真实原子状态的神经网络
        self.mlp_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, batch, vx):
        # x: 真实原子特征
        # vx: 虚拟原子特征
        
        # 1. 真实 -> 虚拟 (汇报工作)
        # 所有真实原子把自己的信息加起来，汇报给虚拟原子
        vx_agg = global_add_pool(x, batch) 
        vx = vx + vx_agg
        vx = self.mlp_virtual(vx) # 虚拟原子消化这些信息，更新自己
        
        # 2. 虚拟 -> 真实 (下达指令)
        # 虚拟原子把更新后的全局信息，广播回给每一个真实原子
        vx_broadcast = vx[batch] 
        x = x + vx_broadcast
        x = self.mlp_real(x) # 真实原子根据全局信息调整自己
        
        return x, vx

# =============================================================================
# 第二部分：分层混合模型主体 (The Hierarchical Hybrid Model)
# 这里是将上述组件组装成最终成品的工厂。
# =============================================================================

class HierarchicalHybridMatformer(nn.Module):
    """
    [最终项目模型：分层混合 Matformer]
    
    架构设计完全对标申报书创新点：
    1. **局部层** (Local): 利用 RBF + MatformerConv 捕捉原子间的精细几何作用。
    2. **全局层** (Global): 利用 虚拟原子 (Virtual Atom) 捕捉长程相互作用（如缺陷引起的晶格畸变）。
    3. **物理层** (Physical): 融合轨道电子数和能量信息，注入物理先验知识。
    
    目标任务：同时预测有缺陷的二维材料的 PDOS（态密度）和标量性质（如形成能）。
    """
    def __init__(self, 
                 node_input_dim=118, 
                 edge_dim=128, 
                 hidden_dim=256, 
                 out_dim=201*4, 
                 num_layers=3, 
                 heads=4,
                 dropout=0.1,
                 use_vacancy_feature=True):
        super().__init__()
        
        self.use_vacancy_feature = use_vacancy_feature

        # --- A. 输入嵌入层 (Input Embeddings) ---
        # 将原子序号转换为向量
        self.node_embedding = nn.Embedding(node_input_dim, hidden_dim)
        
        # [高保真边编码] RBF
        # 将距离转换为向量
        self.rbf = RBFExpansion(vmin=0, vmax=8, bins=edge_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # [空位缺陷特征编码] (Vacancy Defect Feature)
        # 专门处理缺陷：计算原子距离空位中心的距离
        if self.use_vacancy_feature:
            self.vacancy_dist_rbf = RBFExpansion(vmin=0, vmax=15, bins=32) # 空位影响范围较大，设为15埃
            self.vacancy_dist_embedding = nn.Linear(32, hidden_dim)
            # 可选：如果知道空位类型，也可以编码
            self.vacancy_type_embedding = nn.Embedding(10, hidden_dim) 

        # [虚拟原子] 初始化状态
        self.virtual_node_embedding = nn.Embedding(1, hidden_dim)
        
        # --- B. 分层编码器 (Hierarchical Encoder Layers) ---
        # 这里是模型的主干，交替堆叠 Matformer 层和 虚拟原子层
        self.layers = nn.ModuleList()
        self.virtual_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            # 局部交互 (Matformer)
            self.layers.append(
                MatformerConv(in_channels=hidden_dim, 
                              out_channels=hidden_dim, 
                              heads=heads, 
                              edge_dim=hidden_dim,
                              dropout=dropout)
            )
            # 全局交互 (虚拟原子)
            self.virtual_blocks.append(
                VirtualNodeBlock(hidden_dim, dropout)
            )
            
        # --- C. 物理特征融合分支 (Physical Feature Fusion) ---
        # [物理先验] 能量编码：让模型理解“能量”这个物理量
        self.fc_energies = nn.Sequential(
            nn.Linear(201, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        
        # [物理先验] 轨道计数：告诉模型这个材料里有多少个 s, p, d, f 电子
        self.fc_orbital_counts = nn.Sequential(
            nn.Linear(4, 128, bias=False),
            nn.LayerNorm(128)
        )
        
        # --- D. 预测头 (Prediction Heads) ---
        # 1. PDOS 预测头：预测态密度曲线
        # 输入融合了：图特征 + 能量特征 + 轨道特征
        self.fc_pdos = nn.Sequential(
            nn.Linear(hidden_dim*3 + 128 + 128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, out_dim)
        )
        
        # 2. 标量性质预测头：预测形成能、带隙等
        # 输入融合了：图特征 + 轨道特征 (不需要能量网格特征)
        self.fc_scalar = nn.Sequential(
            nn.Linear(hidden_dim*3 + 128, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3) # 输出: [形成能, 带隙, p带中心]
        )

    def forward(self, data):
        # 1. 解包数据
        x, edge_index, edge_attr, energies, orbital_counts = \
            data.x.long(), data.edge_index, data.edge_attr, data.energies, data.orbital_counts
        batch = data.batch
        num_graphs = batch.max().item() + 1
        device = batch.device
        
        # 2. 初始嵌入
        x = self.node_embedding(x.squeeze(1))
        
        # 处理边特征 (RBF)
        if edge_attr.dim() > 1:
            distances = edge_attr[:, 0]
        else:
            distances = edge_attr
        edge_feat = self.rbf(distances)
        edge_feat = self.edge_embedding(edge_feat)
        
        # [空位缺陷特征注入]
        if self.use_vacancy_feature:
            # 1. 局部：距离空位的距离
            if hasattr(data, 'vacancy_dists') and data.vacancy_dists is not None:
                v_dists = data.vacancy_dists
                if v_dists.dim() == 1:
                    v_dists = v_dists.unsqueeze(-1) 
                # 计算距离编码并加到原子特征上
                v_feat = self.vacancy_dist_rbf(v_dists.squeeze(-1))
                v_feat_emb = self.vacancy_dist_embedding(v_feat)
                x = x + v_feat_emb 
            
        # 初始化虚拟原子
        vx = self.virtual_node_embedding(torch.zeros(num_graphs, dtype=torch.long, device=device))
        
        if self.use_vacancy_feature:
            # 2. 全局：空位类型
            if hasattr(data, 'vacancy_type') and data.vacancy_type is not None:
                v_type_emb = self.vacancy_type_embedding(data.vacancy_type)
                vx = vx + v_type_emb # 将空位类型信息注入给虚拟原子
        
        # 3. 分层消息传递 (核心循环)
        for layer, v_block in zip(self.layers, self.virtual_blocks):
            # A. 局部 Matformer 步骤 (原子间交互)
            x = layer(x, edge_index, edge_feat)
            
            # B. 全局虚拟原子步骤 (统筹全局)
            x, vx = v_block(x, batch, vx)
            
        # 4. 全局池化 (Readout)
        # 将所有原子的信息压缩成一个图向量
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        graph_feat = torch.cat((mean_pool, max_pool, sum_pool), dim=-1) 
        
        # 5. 物理特征融合
        energies_feat = self.fc_energies(energies) 
        
        # 聚合轨道计数信息
        cell_orbital_totals = torch.zeros((num_graphs, 4), dtype=orbital_counts.dtype, device=device)
        cell_orbital_totals = cell_orbital_totals.scatter_add(
            dim=0,
            index=batch.unsqueeze(1).expand(-1, 4),
            src=orbital_counts
        )
        orbital_feat = self.fc_orbital_counts(cell_orbital_totals) 
        
        # 6. 最终预测
        
        # A. PDOS 预测
        combined_feat_pdos = torch.cat((graph_feat, energies_feat, orbital_feat), dim=-1) 
        out_pdos = self.fc_pdos(combined_feat_pdos)
        out_pdos = out_pdos.reshape(num_graphs, 4, 201)
        
        # B. 标量预测
        combined_feat_scalar = torch.cat((graph_feat, orbital_feat), dim=-1) 
        out_scalar = self.fc_scalar(combined_feat_scalar) 
        
        return out_pdos, out_scalar
