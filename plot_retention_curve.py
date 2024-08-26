import numpy as np
from sdc.assessment import calc_uncertainty_regection_curve, f_beta_metrics
from sdc.constants import BASELINE_TO_COLOR_HEX
import matplotlib.pyplot as plt
# import seaborn as sns


def get_sparsification_factor(arr_len):
    """Determine with which multiple we
    subsample the array (for easier plotting)."""
    sparsification_factor = None
    if arr_len > 100000:
        sparsification_factor = 1000
    elif arr_len > 10000:
        sparsification_factor = 100
    elif arr_len > 1000:
        sparsification_factor = 10

    return sparsification_factor


def plot_retention_curve_with_baselines(
    uncertainty_scores: np.ndarray,
    losses: np.ndarray,
    uncertainty_scores1: np.ndarray,
    losses1: np.ndarray,
    uncertainty_scores2: np.ndarray,
    losses2: np.ndarray,
    uncertainty_scores3: np.ndarray,
    losses3: np.ndarray,
    # uncertainty_scores4: np.ndarray,
    # losses4: np.ndarray,
    model_key: str,
    metric_name: str = 'weightedADE'
):
    """
    Plot a retention curve with Random and Optimal baselines.

    Assumes that `uncertainty_scores` convey uncertainty (not confidence)
    for a particular point.

    Args:
        uncertainty_scores: np.ndarray, per--prediction request uncertainty
            scores (e.g., obtained by averaging per-plan confidence scores
            and negating).
        losses: np.ndarray, array of loss values, e.g., weightedADE.
        model_key: str, used for displaying the model configurations in the
            plot legend.
        metric_name: str, retention metric, displayed on y-axis.
    """
    M = len(losses)

    # methods = ['Random', 'Baseline', 'Optimal']
    # methods = ['Ensemble', 'Dropout', 'Variational-Bayesian', 'Quantile', 'Ensemble-Dropout', 'Optimal']
    # methods = ['Ensemble', 'Dropout', 'Quantile', 'Ensemble-Dropout', 'Optimal']
    methods = ['Quantile', 'Dropout', 'Ensemble', 'Ensemble-Dropout', 'Optimal-Dropout']
    # methods = ['Dropout', 'SVM', 'Optimal']
    # methods = ['EaUC', 'notEaUC', 'Optimal']
    aucs_retention_curves = []

    # Random results
    # random_indices = np.arange(M)
    # np.random.shuffle(random_indices)
    # retention_curve = calc_uncertainty_regection_curve(
    #     errors=losses, uncertainty=random_indices)
    # aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    # print('Computed Random curve.')

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses, uncertainty=uncertainty_scores)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Quantile curve.')

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses1, uncertainty=uncertainty_scores1)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Dropout curve.')

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses2, uncertainty=uncertainty_scores2)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Ensemble curve.')

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses3, uncertainty=uncertainty_scores3)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Ensemble-Dropout curve.')

    # Optimal results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses1, uncertainty=losses1)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Optimal curve.')

    plt.rcParams.update({'font.size': 6})

    fig, ax = plt.subplots()
    for b, (baseline, (auc, retention_values)) in enumerate(
            zip(methods, aucs_retention_curves)):
        color = BASELINE_TO_COLOR_HEX[baseline]
        if baseline == 'Baseline':
            baseline = ''

        # Subsample the retention value,
        # as there are likely many of them
        # (on any of the dataset splits)
        sparsification_factor = get_sparsification_factor(
            retention_values.shape[0])
        retention_values = retention_values[::sparsification_factor][::-1]
        retention_thresholds = np.arange(
            len(retention_values)) / len(retention_values)

        ax.plot(
            retention_thresholds,
            retention_values,
            #label=f'RIP ({model_key}) {baseline} [AUC: {auc:.3f}]',
            label=f'{baseline} [AUC: {auc:.3f}]',
            color=color)

    #ax.set(xlabel='Retention Fraction', ylabel=metric_name)
    #ax.legend()
    ax.set_xlabel(xlabel='Retention Fraction', fontsize = '14')
    ax.set_ylabel(ylabel='ADE', fontsize = '14')
    ax.legend(fontsize = 'x-large')
    fig.tight_layout()
    # sns.set_style('darkgrid')
    plt.grid()
    plt.show()
    return fig


# a = np.loadtxt("/home/akash/datasets/lyft_10_30_9000_balanced/outputs_999/ade-fde.txt")#[1:-7, 0]

# print(a)

# resnet_50_dropout_999 = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_999/ade-fde.txt"
# lstm_dropout_999 = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_lstm_999/ade-fde.txt"
# cnn_lstm_dropout_200 = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_lstm_cnn_200/ade-fde.txt"
# cnn_emsemble_69_999_12 = "/home/akash/datasets/lyft_10_30_9000_balanced/outputs_cnn_ensemble_69_999_12/ade-fde.txt"

# cnn_42_dropout = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_25_42/plots/ade-fde.txt"
cnn_ensembles = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_ensembles_25_42/ade-fde.txt"#"/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_25_42/plots/ade-fde.txt" #"/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_bayesian_25_42/ade-fde.txt" #"/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_25_42/plots_noise_10/ade-fde.txt"
cnn_dropout = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_25_42/plots_/ade-fde.txt"
cnn_dropout_svm = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_25_42/plots/ade-fde.txt"
cnn_dropout_eauc = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_eauc_25_42/plots/ade-fde.txt"
cnn_bayesian = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_bayesian_25_42/ade-fde.txt"
cnn_quantile = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_quantile_25_42/ade-fde.txt"
cnn_ensemble_dropout = "/home/akash/datasets/lyft_10_30_9000_balanced_new/outputs_cnn_dn_ensembles_dropout_25_42/ade-fde.txt"
# output_file = "RP_cnn_ensemble_42.txt"
# f_1=open(output_file,"w+")
# f_1.write("ade_cnn, fraction\n")

ade_var_cnn_ensembles =[]
ade_var_cnn_dropout =[]
ade_var_cnn_dropout_svm =[]
ade_var_cnn_dropout_eauc =[]
ade_var_cnn_bayesian =[]
ade_var_cnn_quantile =[]
ade_var_cnn_ensemble_dropout =[]

f=open(cnn_ensembles,"r")
lines=f.readlines()
for x in lines[1:-7]:
    # print(x.split(',')[1])
    # ade_var = [float(x.split(',')[1]), float(x.split(',')[4][1:-1])]
    ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
    # print(ade_var)
    ade_var_cnn_ensembles.append(ade_var)
f.close()

f=open(cnn_dropout,"r")
lines=f.readlines()
for x in lines[1:-7]:
    ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
    ade_var_cnn_dropout.append(ade_var)
f.close()

# f=open(cnn_dropout_svm,"r")
# lines=f.readlines()
# for x in lines[1:-1]:
#     ade_var = [float(x.split(',')[0]), float(x.split(',')[1])]
#     ade_var_cnn_dropout_svm.append(ade_var)
# f.close()

# f=open(cnn_dropout_eauc,"r")
# lines=f.readlines()
# for x in lines[1:-7]:
#     ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
#     ade_var_cnn_dropout_eauc.append(ade_var)
# f.close()
# f=open(cnn_bayesian,"r")
# lines=f.readlines()
# for x in lines[1:-7]:
#     ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
#     ade_var_cnn_bayesian.append(ade_var)
# f.close()

f=open(cnn_quantile,"r")
lines=f.readlines()
for x in lines[1:-7]:
    ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
    ade_var_cnn_quantile.append(ade_var)
f.close()

f=open(cnn_ensemble_dropout,"r")
lines=f.readlines()
for x in lines[1:-7]:
    ade_var = [int(x.split(',')[0]), float(x.split(',')[1]), float(x.split(',')[5])]
    ade_var_cnn_ensemble_dropout.append(ade_var)
f.close()

model_key = ''
ade_var_cnn_ensembles = np.array(ade_var_cnn_ensembles)
ade_var_cnn_dropout = np.array(ade_var_cnn_dropout)
# ade_var_cnn_dropout_svm = np.array(ade_var_cnn_dropout_svm)
# ade_var_cnn_dropout_eauc = np.array(ade_var_cnn_dropout_eauc)
# ade_var_cnn_bayesian = np.array(ade_var_cnn_bayesian)
ade_var_cnn_quantile = np.array(ade_var_cnn_quantile)
ade_var_cnn_ensemble_dropout = np.array(ade_var_cnn_ensemble_dropout)

plot_retention_curve_with_baselines(ade_var_cnn_quantile[:,2], ade_var_cnn_quantile[:,1], ade_var_cnn_dropout[:,2], ade_var_cnn_dropout[:,1], ade_var_cnn_ensembles[:,2], ade_var_cnn_ensembles[:,1], ade_var_cnn_ensemble_dropout[:,2], ade_var_cnn_ensemble_dropout[:,1], model_key=model_key)
# plot_retention_curve_with_baselines(ade_var_cnn_dropout[:,2], ade_var_cnn_dropout[:,1], ade_var_cnn_dropout_eauc[:,2], ade_var_cnn_dropout_eauc[:,1], model_key=model_key)
# plot_retention_curve_with_baselines(ade_var_cnn_dropout_eauc[:,2], ade_var_cnn_dropout_eauc[:,1], ade_var_cnn_dropout[:,2], ade_var_cnn_dropout[:,1], ade_var_cnn_dropout_svm[:,1], ade_var_cnn_dropout_svm[:,0], model_key=model_key)

# ade_var_cnn.sort(key = lambda i: i[2]) #sorting based on uncertainty.
# ade_var_cnn = np.array(ade_var_cnn)

# # print(ade_var_cnn[0:100])

# ade_var_cnn = ade_var_cnn[:,1]
# print(ade_var_cnn.shape)

# ade_mean = []

# for i in range(1, 101):
#     num_samples = int(ade_var_cnn.shape[0] * i/100)
#     ade = ade_var_cnn[0:num_samples]
#     ade_mean.append(sum(ade) / len(ade))

# ade_mean = np.array(ade_mean)
# # ade_mean = ade_mean/ade_mean.max()

# for i in range(1, 101):
#     f_1.write(str(ade_mean[i-1]) + "," + str(i) + "\n")

# f_1.close()
