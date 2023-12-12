# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np

# 示例数据集
# 假设这是一些天气数据：每日温度
# 可以根据实际数据集进行替换
temperature_data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

# 将数据标准化
mean_temp = np.mean(temperature_data)
std_temp = np.std(temperature_data)
temperature_data = (temperature_data - mean_temp) / std_temp

# 准备数据
input_data = []
output_data = []

seq_length = 3  # 序列长度，假设我们使用前三天的温度来预测第四天的温度
for i in range(len(temperature_data) - seq_length):
    input_seq = temperature_data[i:i + seq_length]
    output_seq = temperature_data[i + seq_length]
    input_data.append(input_seq)
    output_data.append(output_seq)

input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
output_data = torch.tensor(output_data, dtype=torch.float32).unsqueeze(1)

# 示例的神经网络模型
class WeatherPredictor(nn.Module):
    def __init__(self):
        super(WeatherPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

# 创建模型和优化器
model = WeatherPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型（这里只是一个简单的示例，实际上需要更多的训练和调整）
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, output_data)
    loss.backward()
    optimizer.step()

# 保存模型到文件
torch.save(model, 'weather_predictor_model.pth')