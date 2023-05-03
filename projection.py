import numpy as np
from sklearn.neighbors import kneighbors_graph
from numpy import linalg as la

def project_func(data, Gc, n_neighbors=10, output_dim=30, Lambda=1.0): #[1919, 50], [1989, 50]
	#print('data:', data)
	#print(data[0].shape, data[1].shape)
	#print('Gc:', Gc)

	n_datasets = len(data) #2
	H0 = []
	L = []
	for i in range(n_datasets-1):
		Gc[i] = Gc[i]*np.shape(data[i])[0]

	for i in range(n_datasets):    
		graph_data = kneighbors_graph(data[i], n_neighbors, mode="distance")
		graph_data = graph_data + graph_data.T.multiply(graph_data.T > graph_data) - \
			graph_data.multiply(graph_data.T > graph_data)
		W = np.array(graph_data.todense())
		index_pos = np.where(W>0)
		W[index_pos] = 1/W[index_pos] 
		D = np.diag(np.dot(W, np.ones(np.shape(W)[1])))
		L.append(D - W)
            
	#print('L:', L)    

	Sigma_x = []
	Sigma_y = []
	for i in range(n_datasets-1):
		Sigma_y.append(np.diag(np.dot(np.transpose(np.ones(np.shape(Gc[i])[0])), Gc[i])))
		Sigma_x.append(np.diag(np.dot(Gc[i], np.ones(np.shape(Gc[i])[1]))))

	S_xy = Gc[0]  # 1919 x 1989
	S_xx = L[0] + Lambda*Sigma_x[0]  # 1919 x 1919
	S_yy = L[-1] + Lambda*Sigma_y[0] # 1989 x 1989
	#for i in range(1, n_datasets-1): #如果只有两个数据集，该for循环不运行
	#	print('i:', i)
	#		S_xy = np.vstack((S_xy, self.Gc[i]))
	#	S_xx = block_diag(S_xx, L[i] + self.Lambda*Sigma_x[i])
	#		S_yy = S_yy + self.Lambda*Sigma_y[i]
            
	#print('S_xy:', S_xy.shape)
	#print('S_xx:', S_xx.shape)
	#print('S_yy:', S_yy.shape)

	v, Q = la.eig(S_xx)
	v = v + 1e-12   
	V = np.diag(v**(-0.5))
	H_x = np.dot(Q, np.dot(V, np.transpose(Q))) #new latent representation

	v, Q = la.eig(S_yy)
	v = v + 1e-12      
	V = np.diag(v**(-0.5))
	H_y = np.dot(Q, np.dot(V, np.transpose(Q)))  #new latent representation

	H = np.dot(H_x, np.dot(S_xy, H_y))
	U, sigma, V = la.svd(H) #奇异值分解，分成三个矩阵相乘

	num = [0]
	for i in range(n_datasets-1):
		num.append(num[i]+len(data[i]))

	U, V = U[:,:output_dim], np.transpose(V)[:,:output_dim]

	fx = np.dot(H_x, U)
	fy = np.dot(H_y, V)

	integrated_data = []
	for i in range(n_datasets-1):
		integrated_data.append(fx[num[i]:num[i+1]])

	integrated_data.append(fy)
        
	#print('integrated_data:', integrated_data)
	print(integrated_data[0].shape, integrated_data[1]) #[1919, 30], [1989, 30]

	return integrated_data