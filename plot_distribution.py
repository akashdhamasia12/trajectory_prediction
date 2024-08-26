import numpy as np


# file_ade = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_25_42/plots_noise_8/ade-fde.txt"
file_ade = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_uncertainty_25_42/plots_noise_8/ade-fde.txt"
output_file = "retention_plot_cnn_dropout_calibrated_accurate.txt"
f_1=open(output_file,"w+")
f_1.write("ADE, uncertainty\n")

output_file1 = "retention_plot_cnn_dropout_calibrated_inaccurate.txt"
f_1_1=open(output_file1,"w+")
f_1_1.write("ADE_inaccurate, uncertainty_inaccurate\n")

f=open(file_ade,"r")
lines=f.readlines()
ade_var_cnn=[]
for x in lines[1:-7]:
    # print(x.split(',')[1])
    # ade_var = [float(x.split(',')[1]), float(x.split(',')[4][1:-1])]
    ade_var = [float(x.split(',')[1]), float(x.split(',')[5])]
    # print(ade_var)
    ade_var_cnn.append(ade_var)
f.close()

ade_var_cnn = np.array(ade_var_cnn)
accurate_mask = ade_var_cnn[:,0] <= 2.0 

ade_accurate = ade_var_cnn[accurate_mask]
ade_inaccurate = ade_var_cnn[~accurate_mask]

print(ade_accurate.shape)
print(ade_inaccurate.shape)

for i in range(0, ade_accurate.shape[0]):
    f_1.write(str(ade_accurate[i][0]) + "," + str(ade_accurate[i][1]) + "\n")

for i in range(0, ade_inaccurate.shape[0]):
    f_1_1.write(str(ade_inaccurate[i][0]) + "," + str(ade_inaccurate[i][1]) + "\n")

f_1.close()
f_1_1.close()
