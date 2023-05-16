import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

file = open('inf_scores_egnn.pkl', 'rb')
entries = pickle.load(file)
file.close()

influence_scores = entries[0]["influence_score"]
distances = entries[0]["distances"]

# source = -1
# source_pos = pos[source].unsqueeze(0)

# influence_scores = []
# euclidean_distance = []

# for target in range(total_nodes):
#     h_x_y = node_jacobian[target,:,source,:].mean()
#     h_x_all = node_jacobian[:,:,source,:].mean()
#     I_x_y = h_x_y / h_x_all

#     influence_scores.append(I_x_y.abs().item())

#     D_x_y = torch.cdist(source_pos, pos[target].unsqueeze(0), p=2)
#     euclidean_distance.append(D_x_y.item())


fig, ax = plt.subplots()
ax.scatter(distances.flatten(),influence_scores.flatten())
ax.set_xlabel('dist')
ax.set_ylabel('inf score')
# ax.legend()
# plt.show()
plt.savefig('inf_scores_egnn.png')