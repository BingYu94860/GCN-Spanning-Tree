# GCN-Spanning-Tree

使用 Tensorflow 2.x



###### 資料集：(dataset.py)

1. mnist
2. fashion_mnist
3. cifar10
4. cifar100

# Part1 查看鄰接矩陣，在2D網格的拓撲連接

執行「Show adj.ipynb」的 Jupyter Notebook的檔案，需要將「graph_adj.py」放在同一資料夾下。

###### 鄰接矩陣類型：

1. 四連通、八連通: ['4C', '8C']

2. 直線型: ['0', '90', '45', '135', '22.5', '67.5', '112.5', '157.5']

3. 遞迴型: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']



# Part2 查看 GCN Model 的執行結果 

執行「Run GCN Model.ipynb」的 Jupyter Notebook的檔案，需要將「dataset.py、graph_adj.py、gclayer.py、layer.py」放在同一資料夾下。



# Part3 查看 Chebyshev GCN Model 的執行結果

執行「Run Chebyshev GCN Model.ipynb」的 Jupyter Notebook的檔案，需要將「dataset.py、graph_adj.py、gclayer.py、layer.py」放在同一資料夾下。



