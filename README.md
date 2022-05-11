# Cross Self-Attention Networks for 3D Point Cloud
It is a challenge that design a deep neural network for raw point cloud, which
is disordered and unstructured data. In this paper, we introduce a cross selfattention networks (CSANet) to solve raw point cloud classification and segmentation tasks. It has permutation invariance and can learn the coordinates and
features of point cloud at the same time. To better capture features of different scales, a multi-scale fusion (MF) module is proposed, which can adaptively
consider the information of different scales and establish a fast descent branch
to bring richer gradient information to the network.


Note that during actual training, we set the model's learning rate to 0.01！
