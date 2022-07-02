# AM-GCN-Reproduct

在这里面包含了对比实验和消融实验的实验结果，结果存放在txt文件中



对于对比实验，文件名命名为所用数据集+训练集标签率

对于消融实验，wo表示使用不加一致性约束和差异性约束的loss, d表示使用只加上差异性约束的loss，c表示使用只加上一致性约束的loss,full表示使用两种约束都加上的loss





dataloader.py中是处理数据集，并返回邻接矩阵，features矩阵，以及knn-graph邻接矩阵





Model里是AM-GCN的模型