
# Importing all the necessary packages
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

# Downloading the data and processing it.
dataset=pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
year_month=dataset['Month'].str.split('-',n=1,expand=True)
dataset['Passengers']=dataset['Passengers'].astype(dtype='float32')
dataset.drop('Month',inplace=True,axis=1)
dataset["Year"]=year_month[0].astype(dtype='float32')
dataset["Month"]=year_month[1].astype(dtype='float32')

# Finding and storing the number of rows in a dataset into a constant
NUMBER_OF_ROWS=dataset.shape[0]

# Turning the data into X and Y Labels
x=dataset.iloc[:,1:3]
y=dataset['Passengers']

# Creating the training and test data in 4:1 ratio. 
# Since this a very small set of data, I am not using train_test_split function of scikit-learn
x_train_data,x_test_data=x[:int((NUMBER_OF_ROWS*4)/5)],x[int((NUMBER_OF_ROWS*4)/5):]
y_train_data,y_test_data=y[:int((NUMBER_OF_ROWS*4)/5)],y[int((NUMBER_OF_ROWS*4)/5):]

# Turning the training and test data into tensors
x_train_data_tensor,x_test_data_tensor=torch.tensor(x_train_data.values),torch.tensor(x_test_data.values)
y_train_data_tensor,y_test_data_tensor=torch.tensor(y_train_data.values).view(115,1),torch.tensor(y_test_data.values)

# Defining the LSTM network
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTM,self).__init__()
        self.hidden_size=hidden_size
        self.first_layer=nn.LSTMCell(input_size,self.hidden_size)
        self.second_layer=nn.LSTMCell(self.hidden_size,self.hidden_size)
        self.output_layer=nn.Linear(self.hidden_size,1)
    def forward(self,x_data):
        output_list=[]
        sample_size=x_data.size(0)

        h_t=torch.zeros(sample_size,self.hidden_size,dtype=torch.float32)
        c_t=torch.zeros(sample_size,self.hidden_size,dtype=torch.float32)
        h_t2=torch.zeros(sample_size,self.hidden_size,dtype=torch.float32)
        c_t2=torch.zeros(sample_size,self.hidden_size,dtype=torch.float32)

        for input_tensor in x_train_data_tensor.split(2,dim=1):
            h_t,c_t=self.first_layer(input_tensor,(h_t,c_t))
            h_t2,c_t2=self.second_layer(h_t,(h_t2,c_t2))
            output=self.output_layer(h_t2)
            output_list.append(output)
        output_list=torch.cat(output_list,dim=1)
        return output_list

# Defining the hyperparameters
input_size=2
hidden_size=100
num_epochs=5
learning_rate=0.01

model=LSTM(input_size,hidden_size)

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    predictions=model(x_train_data_tensor)
    loss=criterion(predictions,y_train_data_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
