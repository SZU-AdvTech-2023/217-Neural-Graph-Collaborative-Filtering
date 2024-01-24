'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

"导入pythorch库"


class NGCF(nn.Module):
    "继承自nn.modole类"

    def __init__(self, n_user, n_item, norm_adj, args):
        """n_user和n_item分别表示用户和物品的数量
        norm_adj是规范化的邻接矩阵
        args是一个包含模型参数的对象"""

        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        """
                self.emb_size = args.embed_size 这一行代码表示在 NGCF 模型中保存了嵌入的大小（embedding size）
                嵌入的大小是指学习到的用户和物品的表示向量的维度
                在协同过滤模型中每个用户和物品都被映射到一个固定大小的向量，这个向量的维度就是嵌入的大小
        """
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj
        "将传递给模型的规范化邻接矩阵保存为模型的属性"

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        """
        layers是模型中的每一层的大小，通过eval函数从字符串中获取
        这个字符串通常是形如"[64, 32, 16]"的列表
        decay是正则项的衰减率，同样通过eval函数从字符串中获取
        这个字符串通常是形如"[1e-5, 1e-5, 1e-2]"的列表，其中[0]是L2正则项      
        """

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        """
        self._convert_sp_mat_to_sp_tensor 是一个函数,用于邻接矩阵转换为 PyTorch 中稀疏 Tensor 的表示形式，以便在模型的前向传播中使用
        在图卷积网络中，使用稀疏 Tensor 通常能够更高效地处理大规模图数据
        """

    def init_weight(self):
        # xavier init
        "Xavier初始化的一种实现，它通过将权重初始化为均匀分布的方式来执行初始化"
        initializer = nn.init.xavier_uniform_
        "嵌入矩阵初始化"
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                             self.emb_size))),
            "user_emb 是用户嵌入矩阵，大小为 (n_user, emb_size)"
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                             self.emb_size)))
        })
        "神经网络层的权重和偏置初始化"
        weight_dict = nn.ParameterDict()
        "用于存储神经网络各层的权重和偏置。"
        layers = [self.emb_size] + self.layers
        "通过将嵌入层的大小 self.emb_size 和神经网络的隐藏层大小 self.layers 组合成一个列表 layers，得到了网络的结构。"
        "这里使用 self.layers 是因为它应该是一个包含神经网络各隐藏层大小的列表。"
        for k in range(len(self.layers)):
            "循环初始化每一层的权重和偏置"
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                    layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                    layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        "将稀疏矩阵转化为稀疏张量"
        coo = X.tocoo()
        "将输入的稀疏矩阵 X 转换为坐标格式coo"
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        "稀疏张量的 dropout ——正则化。随机将一部分神经元的输出置为零，来减少过拟合的风险"
        "生成Dropout掩码"
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        "提取输入稀疏张量的值和索引"
        i = x._indices()
        v = x._values()
        "应用Dropout 掩码"
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        " 创建稀疏张量"
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    """
    这里 (1. / (1 - rate)) 是一个缩放因子，其中 rate 是 Dropout 的概率。在训练时，" 
    由于一部分神经元被随机丢弃为了保持期望输出的一致性，需要对保留下来的元素进行缩放。具体而言，缩放因子是为了平衡丢弃的元素所造成的期望值的缩小" \
    """

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        #点积的方式计算正样本和负样本的分数
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)
        #BPR损失函数的核心是最大化正样本得分与负样本得分的差异的对数概率。nn.LogSigmoid() 是对对数sigmoid函数的应用，mf_loss 计算了损失的均值
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        #正则化项用于控制模型的复杂度，防止过拟合。这里采用了L2正则化，将用户和物品嵌入的L2范数加和，然后除以批次大小，以得到平均正则化项
        return mf_loss + emb_loss, mf_loss, emb_loss
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t() )
    '''输出:这个函数用于计算模型的评分，它是用户嵌入和正样本嵌入的点积。这个评分可以用于预测用户对正样本的喜好程度'''


    def forward(self, users, pos_items, neg_items, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        '''如果 drop_flag 为 True，则进行 Dropout 处理，否则直接使用原始的邻接矩阵'''
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            #计算邻居节点的嵌入 side_embeddings

            # transformed sum messages of neighbors.
            #通过权重矩阵进行线性变换，并通过 LeakyReLU 非线性激活函数进行激活
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.计算邻居节点嵌入的 element-wise 乘积，并通过权重矩阵进行线性变换
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.使用 LeakyReLU 非线性激活函数对 sum_embeddings + bi_embeddings 进行激活。
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.通过 Message Dropout 对结果进行随机 Dropout
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.使用 L2 归一化（normalize）操作，得到 norm_embeddings
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.将所有层的嵌入拼接在一起，然后提取出最终的用户和物品嵌入
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
