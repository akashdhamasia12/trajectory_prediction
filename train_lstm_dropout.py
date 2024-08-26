import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
import config
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm


#set random seeds
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True

print("Random seed", config.seed)

learning_rate = config.LR

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    lr = config.LR * (0.1**(epoch % config.decay_rate))
    learning_rate = lr
    if lr < config.min_lr:
        lr = config.min_lr

    # lr = 0.01
    # if epoch == 10:
    #     lr = 0.001    
    # if epoch == 50:
    #     lr = 0.0001
    # if epoch == 80:
    #     lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_ade(outputs, targets):

    # batch_size = targets.size(0)
    outputs_ = outputs.view(outputs.size(0), -1, 2)
    targets_ = targets.view(outputs.size(0), -1, 2)
    displacement = torch.mean(torch.linalg.norm(outputs_ - targets_, dim=2), dim=1) 
    ade = torch.mean(displacement)   
    return ade

def calculate_fde(outputs, targets):

    # batch_size = targets.size(0)
    outputs_ = outputs.view(outputs.size(0), -1, 2)
    targets_ = targets.view(outputs.size(0), -1, 2)
    fde = torch.mean(torch.linalg.norm(outputs_[:, outputs_.size(1)-1, :] - targets_[:, outputs_.size(1)-1, :], dim=1))
    return fde

def evaluate(outputs, keypoints, availability):
    #denormalisation
    outputs_ = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
    keypoints_ = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

    availability_ = availability.bool()
    outputs_ = outputs_[availability_].view(outputs.size(0), -1)
    keypoints_ = keypoints_[availability_].view(outputs.size(0), -1)

    ade = calculate_ade(outputs_, keypoints_)
    fde = calculate_fde(outputs_, keypoints_)

    return ade, fde

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        # print(embedded.size())
        
        #embedded = [batch size, src len, emb dim] if batch_first = true
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Linear(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(1)
        
        #input = [1, batch size]

        # print("input", input.size())
        
        embedded = self.dropout(self.embedding(input))

        # print("embedded", embedded.size())

        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        # print(output.size())

        prediction = self.fc_out(output.squeeze(1))

        # print("prediction", prediction.size())

        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # assert encoder.hid_dim == decoder.hid_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # print(src.size())

        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        # print("input", input.size())

        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # print("output", output.size())
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # #get the highest predicted token from our predictions
            # top1 = output.argmax(1) 
            
            # print(top1.size())
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            # print("trg", trg[t].size())

            input = trg[:, t] if teacher_force else output

            # print("input", input.size())
        
        return outputs


# Model Creation
def build_model() -> torch.nn.Module:

    INPUT_DIM = 2
    OUTPUT_DIM = 2
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    HID_DIM = 128
    N_LAYERS = 2
    ENC_DROPOUT = 0 #0.2
    DEC_DROPOUT = 0 #0.2

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, config.DEVICE)

    return model

model = build_model().to(config.DEVICE)
# print(model)
if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()
    # model = torch.nn.DataParallel(model).cuda()
model.to(config.DEVICE)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


if config.LOSS_FUNCTION == "MSE":
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none") # criterion = nn.SmoothL1Loss()

begin_epoch = 0
f=open(f"{config.OUTPUT_PATH}/loss.txt","w+")
f.write("epoch, train_loss, validation_loss, learning_rate, best_epoch, best_val_loss, train_ade, train_fde, val_ade, val_fde\n")


# training function
def fit(model, dataloader, data, epoch):
    print('Training')
    model.train()
    train_running_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #print(i)
        counter += 1

        keypoints, availability, history_traj,  = torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE))

        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1, 2)
        history_traj = history_traj.view(history_traj.size(0), -1, 2)

        optimizer.zero_grad()

        outputs = model(history_traj, keypoints)
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.reshape(outputs.size(0), -1)

        keypoints = keypoints.reshape(outputs.size(0), -1)
        availability = availability.view(availability.size(0), -1)

        loss = criterion(outputs, keypoints)

        if config.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        loss = loss * availability
        
        loss = loss.mean() 
        train_running_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

    train_loss = train_running_loss/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter

    return train_loss, train_ade, train_fde


# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    total_ade_ = 0.0
    total_fde_ = 0.0

    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1

            keypoints, availability, history_traj,  = torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE))

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1, 2)
            history_traj = history_traj.view(history_traj.size(0), -1, 2)

            optimizer.zero_grad()

            outputs = model(history_traj, keypoints, 0)
            outputs = torch.transpose(outputs, 0, 1)
            outputs = outputs.reshape(outputs.size(0), -1)

            keypoints = keypoints.reshape(outputs.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            loss = criterion(outputs, keypoints)

            if config.train_evaluate:
                ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

            loss = loss * availability

            loss = loss.mean()
            valid_running_loss += loss.item()
            total_ade_ += ade.item()
            total_fde_ += fde.item()

    valid_loss = valid_running_loss/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter

    return valid_loss, val_ade, val_fde


train_loss = []
val_loss = []
prev_best_val_loss = float('inf')

for epoch in range(begin_epoch, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")

    if config.dynamic_lr:
        adjust_learning_rate(optimizer, epoch)

    train_epoch_loss, train_ade, train_fde = fit(model, train_loader, train_data, epoch)
    val_epoch_loss, val_ade, val_fde = validate(model, valid_loader, valid_data, epoch)

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    best_val_loss = min(val_loss)
    best_epoch_no = val_loss.index(min(val_loss))

    if best_val_loss < prev_best_val_loss:
        print("saving best epoch")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"{config.OUTPUT_PATH}/best_epoch_{config.LOSS_FUNCTION}_.pth")

        corresponding_train_loss = train_epoch_loss
        corresponding_train_ade = train_ade
        corresponding_train_fde = train_fde
        corresponding_val_ade = val_ade
        corresponding_val_fde = val_fde

    prev_best_val_loss = best_val_loss

    print("saving last epoch")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/last_epoch_{config.LOSS_FUNCTION}.pth")


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(learning_rate)+","+str(best_epoch_no)+","+str(best_val_loss)+","+str(train_ade)+","+str(train_fde)+","+str(val_ade)+","+str(val_fde)+"\n")
    print(f"Train Loss: {train_epoch_loss:.8f}")
    print(f'Val Loss: {val_epoch_loss:.8f}')
    print(f'train_ade: {train_ade:.8f}')
    print(f'train_fde: {train_fde:.8f}')
    print(f'val_ade: {val_ade:.8f}')
    print(f'val_fde: {val_fde:.8f}')
    print(f'Best Val Loss: {best_val_loss:.8f}')
    print('Corresponding Train Loss:', corresponding_train_loss)
    print('Corresponding train_ade', corresponding_train_ade)
    print('Corresponding train_fde', corresponding_train_fde)
    print('Corresponding val_ade:', corresponding_val_ade)
    print('Corresponding val_fde:', corresponding_val_fde)

    print('best_epoch_no:', best_epoch_no)

    f.close()
    f=open(f"{config.OUTPUT_PATH}/loss.txt","a+")

    # if epoch%config.SAVE_AFTER==0:
    #     print("SAVING")
        
    #     if config.LOSS_FUNCTION == "ADAPTIVE":
    #         torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'adaptive_state_dict': adaptive.state_dict(),
    #                 }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")
    #     elif config.LOSS_FUNCTION == "MSE":
    #         torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': criterion,
    #                 }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")


f.close()


    