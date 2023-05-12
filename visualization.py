import torch
import pickle
import matplotlib.pyplot as plt

file = open('jac.pkl', 'rb')
jacobian = pickle.load(file)
file.close()

file = open('pos.pkl', 'rb')
pos = pickle.load(file)
file.close()

node_jacobian = jacobian[0]
total_nodes = node_jacobian.size(0)

source = -1
source_pos = pos[source].unsqueeze(0)

influence_scores = []
euclidean_distance = []

for target in range(total_nodes):
    h_x_y = node_jacobian[target,:,source,:].mean()
    h_x_all = node_jacobian[:,:,source,:].mean()
    I_x_y = h_x_y / h_x_all

    influence_scores.append(I_x_y.abs().item())

    D_x_y = torch.cdist(source_pos, pos[target].unsqueeze(0), p=2)
    euclidean_distance.append(D_x_y.item())


fig, ax = plt.subplots()
ax.scatter(euclidean_distance,
           influence_scores)
ax.set_xlabel('dist')
ax.set_ylabel('inf score')
# ax.legend()
# plt.show()
plt.savefig('foo.png')
