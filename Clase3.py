import torch 

import numpy as np
import pandas as pd

numpy_array = np.random.randn(2, 2)
tensor_np = torch.from_numpy(numpy_array)
# print(tensor_np)

# sacando media de los tensores
median_tensor = torch.mean(tensor_np)
# print(torch.mean(tensor_np, dim=0))
# print(torch.mean(tensor_np, dim=1))
# print(median_tensor)

# sacando desviacion estandar
# print(torch.std(tensor_np))
# print(torch.std(tensor_np, dim=0))
# print(torch.std(tensor_np, dim=1))

# guardando el tensor
torch.save(tensor_np, "tensor.t")

# cargando tensor
torch.load("tensor.t")



url = "https://raw.githubusercontent.com/amanthedorkknight/fifa18-all-player-statistics/master/2019/data.csv"

data_frame = pd.read_csv(url)
# print(data_frame)

subset = data_frame[['Overall', 'Age', 'International Reputation', 'Weak Foot', 'Skill Moves']].dropna(axis=0, how="any")

columns = subset.columns[1:]
players = torch.tensor(subset.values).float()
print(players.shape)
print(players.type())

data = players[:, 1:]
print(data)

target = players[:, 0]
print(target)


mean = torch.mean(data, dim=0)
print(mean)


std = torch.std(data, dim=0)
print(std)

norm = (data - mean)/torch.sqrt(std)

print(norm)


good = data[torch.ge(target, 85)]
avg = data[torch.gt(target, 75) & torch.lt(target, 85)]
not_so_good = data[torch.le(target, 70)]


good_mean = torch.mean(good, dim=0)
avg_mean = torch.mean(avg, dim=0)
not_so_good_mean = torch.mean(not_so_good, dim=0)

for i, args in enumerate(zip(columns, good_mean, avg_mean, not_so_good_mean)):
    print('{:25} {:6.2f} {:6.2f} {:6.2f}'.format(*args))
