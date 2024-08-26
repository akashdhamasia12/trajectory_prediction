import torch
import os

#METHOD_ = "train" #TODO need to seperate train, val and test dataloader (different folders)

model_name = "cnn_tuning" #cnn

DATASET = "LYFT" #NGSIM_SUBSET1" #"LYFT" #"NGSIM_SUBSET" #NGSIM #tryin to create shifted dataset with NGSIM_SUBSET1
MANEUVER_PRESENT = True 
IMAGE_SIZE = 224 #300 #112 #224 #300 #default: LYFT: 224 NGSIM: 300
num_sequences = 100 #percentage of all sequences to be used for train and val. (it may create unbalanced dataset)
MULTIPLE_GPU = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
cuda_device = 0
if MULTIPLE_GPU:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cuda:'+str(cuda_device) if torch.cuda.is_available() else 'cpu')

NEIGHBOUR_RADIUS = -1 #in pixels (if negative, then all neigbours are selected) (select it according to the image size)

seed = 42 #69 #1234 #42 #36 #69 #999 #12 #42 #999 #12 #56 #36 #20 #56 #42 #222 #900
clip=1

dynamic_lr = False
decay_rate = 10 # if dynamic_lr: after n epochs, decay lr by the factor of 10
min_lr = 0.0001

train_evaluate = True

BATCH_SIZE = 64 #64 #128 #128 #64 #64 #128(out of memory) #64 #8 #32 #reducing it to avoid overfitting
LR = 0.0001 #0.0001 #0.01 #0.0001 #0.01 #0.001
EPOCHS = 20 #2000 #30

noise = False #random normal distribution to the image.

#Bayesian Hyperparameters
delta = 0.5#0.5 #0.2 #0.1
prior_mu_ = 0.0
prior_sigma_ = 0.5#0.1 #should increase it. 0.1 to 0.5
posterior_mu_init_ = 0.0
posterior_rho_init_ = -9.0 #-1000.0 #-9.0 #-6.0
factor_kl = 0.01

#for uncertainty
num_monte_carlo = 10

#Multiple trajectory prediction
multimodal = 1 #number of output trajectories

if DATASET == "LYFT": #total_seq = 50000
    # DATASET_PATH = "/mnt/raid0/trajectorypred/lyft_10_30_50000"
    # DATASET_PATH = "/home/neslihan/TrajPred_Final/datasets/lyft_10_30_30000_balanced/"
    # DATASET_PATH = "/mnt/raid0/trajectorypred/lyft_10_30_30000_balanced"
    # DATASET_PATH = "/mnt/raid0/trajectorypred/lyft_10_30_9000_balanced"
    DATASET_PATH = "/home/akash/datasets/lyft_10_30_9000_balanced"
    # DATASET_PATH = "/home/akash/datasets/lyft_10_30_9000_balanced"
    # DATASET_PATH = "/dataset/lyft_10_30_9000_balanced"
    IMAGE_FACTOR = 224 / IMAGE_SIZE
    SEQ_LENGTH = 41 #(10+1=history + 30 future)
    HISTORY = 11
    TEST_SPLIT = 0.2 #increased validation dataset to stop volatililty of validation loss

elif DATASET == "NGSIM": #total_seq = 35000
    DATASET_PATH = "/dataset/ngsim2lyft_10_19"
    IMAGE_FACTOR = 300 / IMAGE_SIZE 
    SEQ_LENGTH = 29 #(9+1=history + 19 future))
    HISTORY = 10
    TEST_SPLIT = 0.2

elif DATASET == "NGSIM_SUBSET":
    DATASET_PATH = "dataset/ngsim2lyft_subset"
    IMAGE_FACTOR = 300 / IMAGE_SIZE 
    SEQ_LENGTH = 29 #(9+1=history + 19 future))
    HISTORY = 10
    TEST_SPLIT = 0.8

elif DATASET == "NGSIM_SUBSET1":
    DATASET_PATH = "dataset/ngsim2lyft_subset"
    IMAGE_FACTOR = 224 / IMAGE_SIZE 
    SEQ_LENGTH = 41 #(10+1=history + 30 future))
    HISTORY = 11
    TEST_SPLIT = 0.95


OUTPUT_PATH = DATASET_PATH + '/outputs_' + model_name + '_' + str(seed) 
plots = OUTPUT_PATH + "/plots"
if not os.path.exists(plots):
    os.makedirs(plots)

RESUME_PATH = None #OUTPUT_PATH + "/last_epoch_MSE.pth" #None #OUTPUT_PATH + "/last_epoch_MSE.pth" #"/nwstore/neslihan/trajpred/results/outputs_102_2/model_730.pth"
# DEVICE = torch.device('cpu')
ft_portion = 'complete' # "complete" or "last_layer"

#  Hyperparameters
# OTHER OPTION: DEFAULT: current to history, agent then target, alternative and at the end map
STACKING = "LYFT" #current to history, all agents first, all targets next then map, 
SAVE_AFTER = 5
LOSS_FUNCTION = "MSE" #MSE or ADAPTIVE

#TESTING PARAMETERS
model_path = "/home/akash/datasets/lyft_10_30_9000_balanced/best_epoch_MSE_.pth"
BEST_EPOCH = "best_epoch_MSE_.pth" #"model_30.pth" #"model_225.pth" #"best_epoch_MSE_.pth" #model_80.pth
TEST_DATALOADER = "val"
SHOW_DATASET_PLOT = False

#Ensembles
model_1_path = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_69/best_epoch_MSE_.pth"
model_2_path = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_999/best_epoch_MSE_.pth"
model_3_path = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_12/best_epoch_MSE_.pth"