import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                                            [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                                            [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model= nn.Linear(input_size, output_size)
###########################################################

#criterion= nn.MSELoss()
#optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
#
#for epoch in range(num_epochs):
#    inputs= torch.from_numpy(x_train)
#    labels= torch.from_numpy(y_train)
#
#    outputs= model(inputs)
#    loss= criterion(outputs, labels)
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    print('loss ', loss.item())
#
#
#torch.save(model.state_dict(), 'lin_regression.ckpt')

###################################################################
model.load_state_dict(torch.load('lin_regression.ckpt'))

##################################################################
# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

