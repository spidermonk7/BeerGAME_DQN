import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
class QMLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=1):
        super(QMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
         

class Buffer():
    def __init__(self, ):
        # buffer of datas: [st-1, at-1, rt, st, pt]
        self.buffer = []
        self.pts = []
        
        
    def sample(self, weighted=False):
        if not weighted:
            idx = np.random.randint(0, len(self.buffer))
            return self.buffer[idx], idx
        
        else:
            # 将data和pr转换为numpy数组
            pr = np.array(self.pts)
            
            # 归一化权重数组，以确保其和为1
            pr = pr / np.sum(pr)
            
            # 使用numpy的random.choice进行采样，返回采样结果的索引
            sampled_indices = np.random.choice(len(self.buffer), size=1, p=pr, replace=True)
            
            sampled_indice = sampled_indices[0]
            
            # 根据索引获取采样结果
            sampled_data = self.buffer[sampled_indice]
    
            return sampled_data, sampled_indice
    

    def get_max(self):
        pt = max(self.pts)
        idx = self.pts.index(pt)
        return pt, idx

        

def inference(model, obs, action, device=device):
    input = np.concatenate([obs, np.array([action])])
    return model(torch.tensor(input, dtype=torch.float32).to(device)).item()


loss_fn = torch.nn.MSELoss()


def best_action(model, obs, device=device):
    for action in range(5):
        input = np.concatenate([obs, np.array([action])])
        output = model(torch.tensor(input, dtype=torch.float32).to(device))
        if action == 0:
            max_output = output
            best_action = action
        else:
            if output > max_output:
                max_output = output
                best_action = action
    return best_action, max_output.item()


def train(model, data, label, weighted_opt = False, device=device):
    input = np.concatenate([data[0], np.array([data[1]])])
    output = model(torch.tensor(input, dtype=torch.float32).to(device))
    loss = loss_fn(output, torch.tensor(label, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if not weighted_opt:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model, loss.item()