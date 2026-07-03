import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


#This is all rubbish, we actually need to use Q learning to train the fish to avoid the shark, but for now we will just use a simple neural network to control the fish's movement based on the distance and angle to the shark





#Create a small neural network to control the fish's movement based on the distance and angle to the shark
class FishNN(nn.Module):
    #Input distance and angle to shark, output turn direction and speed
    #Think we will start with a simple feedforward network with one hidden layer and 5 neurons in the hidden layer
    def __init__(self):
        super(FishNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
    
torch.manual_seed(41)
model = FishNN()   

#We need to train the model to avoid the shark
#We know the input is the distance and angle to the shark, and the output is the turn direction and speed
#So we must multiply the input by some weights to get the output, and then we can use the output to update the fish's position and heading
#The output will be a vector of length 2, where the first element is the turn direction and the second element is the speed

def train_model(model, optimizer = torch.optim.SGD(model.parameters(), lr=0.01), criterion = nn.MSELoss(), num_epochs=1000):
    for epoch in range(num_epochs):
        #Randomly generate a distance and angle to the shark
        distance = np.random.uniform(0, 400)
        angle = np.random.uniform(-math.pi, math.pi)
        input_data = torch.tensor([[distance, angle]], dtype=torch.float32)

        #Randomly generate a turn direction and speed
        turn_direction = np.random.uniform(-1, 1)
        speed = np.random.uniform(0, 1)
        target_data = torch.tensor([[turn_direction, speed]], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        
        loss.append(loss.detach().numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train_model(model)        