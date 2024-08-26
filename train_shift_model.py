import torch
import random
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from torchvision.models.resnet import resnet50
from resnet50_dropout import resnet50
import torchvision.models as models

import config
if config.DATASET == "SHIFT":
    from dataset_shift import train_data, train_loader, valid_data, valid_loader, valid_loader_out
else:
    from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
# import utils
# import robust_loss_pytorch
from typing import Tuple


# from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use('ggplot')

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY

#set random seeds
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
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


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

# Model Creation
class MobileNetV2(nn.Module):
  """A `PyTorch Hub` MobileNetV2 model wrapper."""

  def __init__(
      self,
      num_classes: int,
      in_channels: int = 3,
  ) -> None:
    """Constructs a MobileNetV2 model."""
    super(MobileNetV2, self).__init__()

    self._model = torch.hub.load(
      'pytorch/vision:v0.9.0', 'mobilenet_v2', num_classes=num_classes,
      pretrained=False)

    # HACK(filangel): enables non-RGB visual features.
    _tmp = self._model.features._modules['0']._modules['0']
    self._model.features._modules['0']._modules['0'] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=_tmp.out_channels,
        kernel_size=_tmp.kernel_size,
        stride=_tmp.stride,
        padding=_tmp.padding,
        bias=_tmp.bias,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass from the MobileNetV2."""
    return self._model(x)


class BehaviouralModel(nn.Module):
    """A `PyTorch` implementation of a behavioural cloning model."""

    def __init__(
        self,
        in_channels: int,
        dim_hidden: int = 128,
        output_shape: Tuple[int, int] = (config.future_prediction if config.future_prediction > 0 else 30, 2),
        scale_eps: float = 1e-7,
        bc_deterministic: bool = True,
        generation_mode: str = 'sampling',
        **kwargs
    ) -> None:
        """Constructs a behavioural cloning model.

        Args:
            in_channels: Number of channels in image-featurized context
            dim_hidden: Hidden layer size of encoder output / GRU
            output_shape: The shape of the data distribution
                (a.k.a. event_shape).
            scale_eps: Epsilon term to avoid numerical instability by
                predicting zero Gaussian scale.
            generation_mode: one of {sampling, teacher-forcing}.
                In the former case, the autoregressive likelihood is formed
                    by conditioning on samples.
                In the latter case, the likelihood is formed by conditioning
                    on ground-truth.
        """
        super(BehaviouralModel, self).__init__()
        assert generation_mode in {'sampling', 'teacher-forcing'}

        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(
            in_channels=in_channels, num_classes=dim_hidden).to(config.DEVICE)

        # All inputs (including static HD map features)
        # have been converted to an image representation;
        # No need for an MLP merger.

        # The decoder recurrent network used for the sequence generation.
        self._decoder = nn.GRUCell(
            input_size=self._output_shape[-1], hidden_size=dim_hidden)

        self._scale_eps = scale_eps

        self._output = nn.Linear(
            in_features=dim_hidden,
            out_features=(self._output_shape[-1]))

        self.bc_deterministic = bc_deterministic
        self._generation_mode = generation_mode
        print(f'BC Model: using generation mode {generation_mode}.')

    def forward(self, image) -> torch.Tensor:
        """Returns the expert plan."""
        # Parses context variables.
        # feature_maps = context.get("feature_maps")

        # Encodes the visual input.
        z = self._encoder(image)

        # z is the decoder's initial state.

        # Output container.
        y = list()

        # Initial input variable.
        x = torch.zeros(  # pylint: disable=no-member
            size=(z.shape[0], self._output_shape[-1]),
            dtype=z.dtype,
        ).to(z.device)

        # if config.DATASET == "LYFT":
        #     x[:, 0] = -0.5
        #     x[:, 1] = 0.0

        # Autoregressive generation of plan.
        for _ in range(self._output_shape[0]):
            # Unrolls the GRU.
            z = self._decoder(x, z)

            # Predicts the displacement (residual).
            dx = self._output(z)
            x = dx + x

            # Updates containers.
            y.append(x)

        return torch.stack(y, dim=1)  # pylint: disable=no-member


def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    # '''
    if config.DATASET == "SHIFT":
        num_in_channels = 28
    else:
        num_in_channels = 3 + (2*past_trajectory)

    # change input channels number to match the rasterizer's output
    model = BehaviouralModel(in_channels=num_in_channels)

    return model

model = build_model().to(config.DEVICE)
# print(model)
if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()
    # model = torch.nn.DataParallel(model).cuda()
model.to(config.DEVICE)

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
        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)
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
            image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.view(keypoints.size(0), -1)
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
prev_best_val_loss = 1000000
val_ade_out = -999
val_fde_out = -999

for epoch in range(0, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")

    if config.dynamic_lr:
        adjust_learning_rate(optimizer, epoch)

    train_epoch_loss, train_ade, train_fde = fit(model, train_loader, train_data, epoch)
    val_epoch_loss, val_ade, val_fde = validate(model, valid_loader, valid_data, epoch)
    if config.DATASET == "SHIFT":
        val_epoch_loss_out, val_ade_out, val_fde_out = validate(model, valid_loader_out, valid_data, epoch)

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
        corresponding_val_ade_out = val_ade_out
        corresponding_val_fde_out = val_fde_out

    prev_best_val_loss = best_val_loss

    print("saving last epoch")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/last_epoch_{config.LOSS_FUNCTION}.pth")


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(learning_rate)+","+str(best_epoch_no)+","+str(best_val_loss)+","+str(train_ade)+","+str(train_fde)+","+str(val_ade)+","+str(val_fde)+","+str(val_ade_out)+","+str(val_fde_out)+"\n")
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
    print('Corresponding val_ade_out:', corresponding_val_ade_out)
    print('Corresponding val_fde_out:', corresponding_val_fde_out)

    print('best_epoch_no:', best_epoch_no)

    f.close()
    f=open(f"{config.OUTPUT_PATH}/loss.txt","a+")

    # if epoch%config.SAVE_AFTER==0:
    #     print("SAVING")        
    #     torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': criterion,
    #             }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")


f.close()
