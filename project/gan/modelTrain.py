#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch
from modelDesign import DatasetFolder, NMSE #* AutoEncoder
from Model_Define import AutoEncoder
import os
import torch.nn as nn
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting for Data
NUM_FEEDBACK_BITS = 512
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
# Parameters Setting for Training
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
PRINT_RREQ = 100
torch.manual_seed(1)
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('/2020/data/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# data = np.transpose(data, (0, 3, 1, 2))
split = int(data.shape[0] * 0.7)
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS).cuda()
autoencoderModel = autoencoderModel
criterion = nn.MSELoss().cuda()
# criterion = NMSELoss(reduction='mean')
# criterion_test = NMSELoss(reduction='sum')
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1
for epoch in range(EPOCHS):
    autoencoderModel.train()
    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = autoencoderModel(autoencoderInput)
        loss = criterion(autoencoderOutput, autoencoderInput)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % PRINT_RREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t' .format(epoch, i, len(train_loader), loss=loss.item()))
    # Model Evaluating
    autoencoderModel.eval()
    totalLoss = 0
    totalNMSE = 0
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            autoencoderInput = autoencoderInput.cuda()
            autoencoderOutput = autoencoderModel(autoencoderInput)
            totalLoss += criterion(autoencoderOutput, autoencoderInput).item() * autoencoderInput.size(0)
            localNMSE = NMSE(autoencoderInput.to("cpu").numpy(), autoencoderOutput.to("cpu").numpy())
            totalNMSE += localNMSE
            if i % PRINT_RREQ == 0:
                print('NMSE %.4f' % np.mean(localNMSE))
        averageLoss = totalLoss / len(test_dataset)
        averageNMSE = totalNMSE / len(test_dataset)
        print('average NMSE {0:.6f}\t' 'average loss {1:.6f}'.format(averageNMSE * BATCH_SIZE, averageLoss))
        if averageLoss < bestLoss:
            # Model saving
            # Encoder Saving
            torch.save({'state_dict': autoencoderModel.encoder.state_dict(), }, './modelSubmit/encoder.pth.tar')
            # Decoder Saving
            torch.save({'state_dict': autoencoderModel.decoder.state_dict(), }, './modelSubmit/decoder.pth.tar')
            print("Model saved")
            bestLoss = averageLoss
#=======================================================================================================================
#=======================================================================================================================