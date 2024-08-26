
import pandas as pd
import os
import seaborn as sns
import matplotlib.pylab as plt
import os
from collections import defaultdict
import numpy as np
from ysdc_dataset_api.evaluation.metrics import compute_all_aggregator_metrics
from sdc.assessment import calc_uncertainty_regection_curve, f_beta_metrics
from pprint import pprint
from sdc.cache_metadata import load_dataset_key_to_arrs, construct_full_dev_sets
from sdc.analyze_metadata import compute_dataset_results


# Specify the metadata path for the cached predictions and per-plan confidence scores.
dir_metadata_cache = '/home/neslihan/VehicleMotionPrediction/shifts/sdc/data/metadata_cache/rip-bc-k_5-plan_ma-scene_ma/'

# Load in predictions, ground truths, per-plan confidence scores, request IDs for each dataset
dataset_key_to_arrs = load_dataset_key_to_arrs(metadata_cache_dir=dir_metadata_cache)

# Add a field for the full__validation dataset
dataset_key_to_arrs = construct_full_dev_sets(dataset_key_to_arrs)

# Based on our desired settings, retrieve the final model predictions given:
# * k: the ensemble size (must be less than or equal to the number of members that were used to generate
#       the predictions).
# * d: number of final predictions to sample from the ensemble
# * plan_agg: per-plan aggregation strategy
# * pred_req_agg: per--prediction request aggregation strategy
# * dataset_key: dataset to generate predictions, conf scores, etc. for
# * dataset_key_to_arrs_dict: contains cached data
# * n_pred_per_model: should be the same number as was set in the config parameter --rip_samples_per_model
#.      when caching the predictions
# * retention_column: metric on which we compute retention
# * return_preds_and_scores: boolean, if True, do not compute losses and instead return preds and 
#       scores for use with submission protobuf or more downstream analysis, as in our case.

# For more info, see sdc/analyze_metadata.py.

(model_preds, plan_conf_scores, pred_req_conf_scores, 
 request_ids, is_ood_arr) = (
    compute_dataset_results(
        k=5, d=5, plan_agg='MA', pred_req_agg='MA', 
        dataset_key='full__validation',
        dataset_key_to_arrs_dict=dataset_key_to_arrs, 
        n_pred_per_model=10, 
        retention_column='weightedADE', 
        return_preds_and_scores=True))

# Compute all metrics using our predictions.
metrics_dict = compute_all_aggregator_metrics(
    per_plan_confidences=plan_conf_scores,
    predictions=model_preds,
    ground_truth=dataset_key_to_arrs['full__validation']['gt_trajectories'],
    metric_name='weightedADE')

uncertainty_scores=(-pred_req_conf_scores)
plt.figure(1)
plt.subplot(211)
plt.plot(uncertainty_scores)
# Adding title
plt.title('Uncertainty Values and Histogram', fontsize=12)
# Adding axis title
plt.xlabel('Index', fontSsize=10)
plt.ylabel('Uncertainty', fontsize=10)
plt.text(350000, 300, "max:{:.3f}".format(np.max(uncertainty_scores)), fontsize = 8)
plt.text(350000, 250, "min:{:.3f}".format(np.min(uncertainty_scores)), fontsize = 8)
plt.subplot(212)
plt.hist(uncertainty_scores)
plt.xlabel('Index', fontsize=10)
plt.ylabel('Histogram', fontsize=10)
plt.savefig('/home/neslihan/VehicleMotionPrediction/shifts/sdc/data/metadata_cache/rip-bc-k_5-plan_ma-scene_ma/uncertainty_scores.png')
losses=metrics_dict['weightedADE']
plt.figure(2)
plt.subplot(411)
plt.plot(losses)
# Adding title
plt.title('Loss Values and Histogram', fontsize=12)
# Adding axis title
plt.ylabel('WeightedADE', fontsize=10)
plt.text(400000, 80, "max:{:.3f}".format(np.max(losses)), fontsize = 8)
plt.text(400000, 50, "min:{:.3f}".format(np.min(losses)), fontsize = 8)
plt.subplot(412)
plt.hist(losses, 1000)
plt.ylabel('Histogram', fontsize=10)
plt.savefig('/home/neslihan/VehicleMotionPrediction/shifts/sdc/data/metadata_cache/rip-bc-k_5-plan_ma-scene_ma/loss_scores2.png')



from plot_retention_curves import plot_retention_curve_with_baselines

model_key = 'BC, MA, K=5'
fig = plot_retention_curve_with_baselines(
    uncertainty_scores=(-pred_req_conf_scores),
    losses=metrics_dict['weightedADE'],
    model_key=model_key,
    metric_name='weightedADE')