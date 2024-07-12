import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from collections import defaultdict

import os

torch.cuda.is_available()
device = 'cpu'

def ev_id(evs, k):
        d = 0
        for i, ev in enumerate(evs):
            d += ev
            if d >= k:
                return i+1

def visualise_embeddings(all_X, all_y, output_figure_name):
    
    fig, axes = plt.subplots(1, 32, figsize=(128,4))

    for fig_i in range(32):
        visualiser = TSNE(n_components=2, random_state=42)

        X_tsne = visualiser.fit_transform(all_X[:,fig_i,:])

        df = []
        for i in range(len(all_y)):
            df.append([X_tsne[i,0], X_tsne[i,1], all_y[i]])
        df = pd.DataFrame(df, columns=["tsne-2d-one", "tsne-2d-two", "y"])  

        plt.figure(figsize=(6,6))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 5),
            data=df,
            legend='auto',
            alpha=0.3,
            ax=axes[fig_i],
        )
        axes[fig_i].set_title("layer"+str((fig_i)))

        sns.move_legend(
            axes[fig_i], "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
        )

    fig.savefig(output_figure_name, dpi=fig.dpi)

def get_heuristic_info(positives, negatives):
    
    heuristics_dict = defaultdict(dict)
    number_demonstrations, num_layers = positives.shape[0], positives.shape[1]
    for i_layer in tqdm(range(num_layers)):
        
        # Get representations for positives, negatives and bases with shape N*d
        positives_i_layer = positives[:,i_layer,:] # N, d
        negatives_i_layer = negatives[:,i_layer,:]
        
        # Create labels vector
        positive_labels = torch.tensor([[0, 1] for i in range(number_demonstrations)])
        negative_labels = torch.tensor([[1, 0] for i in range(number_demonstrations)])

        # Concate positives and negatives
        embeddings = torch.cat([positives_i_layer, negatives_i_layer], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0).float()

        embeddings = embeddings.T
        labels = labels.T

        # Calculating the corvariance
        info_cov = torch.matmul(embeddings, labels.T) / number_demonstrations # A=E[XZ^T], /Sigma in shape d*d, (4096*4096)
        _, s_, _ = torch.linalg.svd(info_cov, full_matrices=True) # dxd, d*N, N*N
        signature = torch.sum(s_)
        s_ = torch.pow(s_, 2)
        normalised_s_ = s_/torch.sum(s_)
        
        heuristics_dict[i_layer]['signature'] = signature.item()
        heuristics_dict[i_layer]['explained_variance'] = {i: v for i, v in enumerate(normalised_s_.tolist()[:10])}
    
    return heuristics_dict

def non_linear_feature_func(X, func='squared-exponential'):
    if func == 'squared-exponential':
        length_scale = 1
        return torch.exp(-1 * X**2 / (2 * length_scale**2)) # fix X to X**2 for gaussian
    if func == 'tanh':
        return torch.tanh(X)
    if func == 'elu':
        positive_X = (X) * (X >= 0)
        negative_X = (torch.exp(X) - 1) * (X < 0)
        return positive_X + negative_X
        

def inv_non_linear_feature_func(X, func='squared-exponential'):
    if func == 'squared-exponential':
        length_scale = 1
        eps = torch.ones(X.shape) * 1e-4
        return -torch.log(torch.max(X, eps)) * 2 * length_scale**2
    if func == 'tanh':
        eps = torch.ones(X.shape) * 1e-4
        X = torch.min(X, 1 - eps)
        X = torch.max(X, -1 + eps)
        return torch.atanh(X)
    if func == 'elu':
        eps = torch.ones(X.shape) * 1e-4
        positive_X = (X) * (X >= 0)
        negative_X = (torch.log(torch.max(X, -1+eps) + 1)) * (X < 0)
        return positive_X + negative_X

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demonstration-number", type=int, default=100)
    parser.add_argument("--dataset-name", type=str, default="truthfulqa")
    parser.add_argument("--positive-save-path", type=str)
    parser.add_argument("--negative-save-path", type=str)
    parser.add_argument("--base-save-path", type=str)

    parser.add_argument("--output-path", default="projections/llama-2-chat-7b-truthfulqa", type=str)

    # Hyperparameter for learning projections
    parser.add_argument("--positive-k-ratio", type=float, default=None)
    parser.add_argument("--negative-k-ratio", type=float, default=None)
    parser.add_argument("--positive-k-int", type=int, default=None)
    parser.add_argument("--negative-k-int", type=int, default=None)
    parser.add_argument("--do-mean-subtraction", action="store_true")
    parser.add_argument("--feature-function", type=str, default=None)

    # Experiment setup
    parser.add_argument("--visualisation", action="store_true")
    parser.add_argument("--get-layer-heuristic-info", action="store_true")
    
    args = parser.parse_args()
    
    assert (args.negative_k_int and args.positive_k_int) or (args.negative_k_ratio and args.positive_k_ratio)
    
    print("Learning parameters for projection matrics:")
    print(args)

    if not os.path.exists(args.output_path):
        print("Preparing output path...")
        os.makedirs(args.output_path)
    
    # Saving args and hyperparameters
    with open(args.output_path+"/sea-args.json", 'w+') as f:
        json.dump(vars(args), f, indent=4)

    positives = torch.load(args.positive_save_path+'.pt')[:args.demonstration_number].to(device).double()
    negatives = torch.load(args.negative_save_path+'.pt')[:args.demonstration_number].to(device).double()
    bases = torch.load(args.base_save_path+'.pt')[:args.demonstration_number].to(device).double()

    sea_pos, sea_neg, sea = [], [], []
    UU_pos, UU_neg = [], []

    if args.feature_function:
        nonlinear_pos, nonlinear_neg, nonlinear_base = [], [], []

    statistics_dict = defaultdict(dict)

    # For each layer, get necessary info for using heurstic to decide intervened layer
    # If a layer contains high signature to label matrix, it should be incldued for sea
    if args.get_layer_heuristic_info:
        heuristic_dict = get_heuristic_info(positives, negatives)
        with open(args.output_path+"/heuristic-dict.json", 'w+') as f:
            json.dump(heuristic_dict, f, indent=4)


    # For each layer, calculate and save UU^T projections for positives vs base and negatives vs base
    num_layers = positives.shape[1]
    for i_layer in tqdm(range(num_layers)):
        
        # Get representations for positives, negatives and bases with shape N*d
        positives_i_layer = positives[:,i_layer,:] # N, layer, d
        negatives_i_layer = negatives[:,i_layer,:]
        bases_i_layer = bases[:,i_layer,:]
        
        # Transpose embeddings to dxN, to align with the eqts in paper
        positives_i_layer = positives_i_layer.T
        negatives_i_layer = negatives_i_layer.T
        bases_i_layer = bases_i_layer.T

        norm_0 = torch.norm(bases_i_layer.float(),dim=-1).unsqueeze(-1)
        
        statistics_dict[i_layer]['positive_embeds_rank'] = positive_embeds_rank.item()
        statistics_dict[i_layer]['negative_embeds_rank'] = negative_embeds_rank.item()
        statistics_dict[i_layer]['base_embeds_rank'] = base_embeds_rank.item()

        # mean subtraction (Probably do not do mean subtraction, because adding mean might bring back rmoved info)
        if args.do_mean_subtraction:
            positives_i_layer_mean_subtracted = positives_i_layer - torch.mean(positives_i_layer, dim=0, keepdim=True)
            negatives_i_layer_mean_subtracted = negatives_i_layer - torch.mean(negatives_i_layer, dim=0, keepdim=True)
            bases_i_layer_mean_subtracted = bases_i_layer - torch.mean(bases_i_layer, dim=0, keepdim=True) # dxN
        else:
            positives_i_layer_mean_subtracted = positives_i_layer
            negatives_i_layer_mean_subtracted = negatives_i_layer
            bases_i_layer_mean_subtracted = bases_i_layer

        # Applying feature function to add non-linearity
        if args.feature_function:
            positives_i_layer_mean_subtracted = non_linear_feature_func(positives_i_layer_mean_subtracted, args.feature_function)
            negatives_i_layer_mean_subtracted = non_linear_feature_func(negatives_i_layer_mean_subtracted, args.feature_function)
            bases_i_layer_mean_subtracted = non_linear_feature_func(bases_i_layer_mean_subtracted, args.feature_function)
            nonlinear_pos.append(positives_i_layer_mean_subtracted.T)
            nonlinear_neg.append(negatives_i_layer_mean_subtracted.T)
            nonlinear_base.append(bases_i_layer_mean_subtracted.T)

        d, N = negatives_i_layer_mean_subtracted.size() # d: 4096, N: 100

        # Calculating cross covariance, select top k% elements
        cross_cov_pos = torch.matmul(bases_i_layer_mean_subtracted, positives_i_layer_mean_subtracted.T) / N # A=E[XZ^T], /Sigma in shape d*d, (4096*4096)
        pos_crosscov_rank = torch.linalg.matrix_rank(cross_cov_pos)
        
        statistics_dict[i_layer]['pos_crosscov_rank'] = pos_crosscov_rank.item()
        U_pos, s_pos, V_pos = torch.linalg.svd(cross_cov_pos, full_matrices=True) # dxd, d*N, N*N

        s_pos = torch.pow(s_pos, 2) # square to get the ratio of explained vairance 
        normalised_ev_pos = s_pos/torch.sum(s_pos) # normalised to get percentage of explained variance
        
        statistics_dict[i_layer]['pos_explained_ratio'] = {i: v for i, v in enumerate(normalised_ev_pos.tolist()[:10])}
        k_id_pos = ev_id(normalised_ev_pos, args.positive_k_ratio) # find the id for top-k% of explained vairance, pos_k = 0.99 for now
        
        statistics_dict[i_layer]['k_id_pos'] = k_id_pos
        ## Apply UU^T Transformation for positive
        U_pos_ = U_pos[:, :k_id_pos] # U: dxk
        UU_T = torch.matmul(U_pos_, U_pos_.T) # UU^T: dxd

        UU_pos.append(UU_T.float()) # Stacking UU_T for this layer to save
        b = torch.matmul(UU_T, bases_i_layer_mean_subtracted) # Applying UU_T to base representations
        if args.do_mean_subtraction:
            sea_outputs_pos = b + torch.mean(bases_i_layer, dim=0, keepdim=True) # Adding the subtracted mean back
        else:
            sea_outputs_pos = b

        # Calculating cross covariance, select top k% elements
        cross_cov_neg = torch.matmul(bases_i_layer_mean_subtracted, negatives_i_layer_mean_subtracted.T) / N
        neg_crosscov_rank = torch.linalg.matrix_rank(cross_cov_neg)

        statistics_dict[i_layer]['neg_crosscov_rank'] = neg_crosscov_rank.item()
        U_neg, s_neg, V_neg = torch.linalg.svd(cross_cov_neg, full_matrices=True) # dxd, d*N, N*N
        s_neg = torch.pow(s_neg, 2)
        normalised_ev_neg = s_neg/torch.sum(s_neg)

        statistics_dict[i_layer]['neg_explained_ratio'] = {i: v for i, v in enumerate(normalised_ev_neg.tolist()[:10])}
        
        # for negative we keep the last 1-k principles
        # k_id_neg = cross_cov_neg.size(0) - ev_id(normalised_ev_neg, args.negative_k_ratio) # neg_k = 0.99 for now
        k_id_neg = ev_id(normalised_ev_neg, args.negative_k_ratio)
        
        statistics_dict[i_layer]['k_id_neg'] = k_id_neg
        ## Apply UU^T Transformation for negative
        U_neg_ = U_neg[:, k_id_neg:] # sanity check with the smallest change, see model;s output
        UU_T = torch.matmul(U_neg_, U_neg_.T) # UU^T: dxd
        UU_neg.append(UU_T.float())
        b = torch.matmul(UU_T, bases_i_layer_mean_subtracted)

        if args.do_mean_subtraction:
            sea_outputs_neg = b + torch.mean(bases_i_layer, dim=0, keepdim=True)
        else:
            sea_outputs_neg = b

        # Inverse of non-linear feature func to get representations in original space
        if args.feature_function:
            sea_outputs_pos = inv_non_linear_feature_func(sea_outputs_pos, args.feature_function)
            sea_outputs_neg = inv_non_linear_feature_func(sea_outputs_neg, args.feature_function)
        
        # Combine pos adn neg representations by L2 norm
        sea_outputs_comb = sea_outputs_neg + sea_outputs_pos
        norm_comb = torch.norm(sea_outputs_comb.float(),dim=-1).unsqueeze(-1)
        sea_outputs_comb = sea_outputs_comb * norm_0 / norm_comb
        

        # Transpose representations back to N*d
        sea_outputs_pos = sea_outputs_pos.T
        sea_outputs_neg = sea_outputs_neg.T
        sea_outputs_comb = sea_outputs_comb.T
        
        sea_pos.append(sea_outputs_pos)
        sea_neg.append(sea_outputs_neg)
        sea.append(sea_outputs_comb)

    with open(args.output_path+"/sea-statistics.json", 'w+') as f:
        json.dump(statistics_dict, f, indent=4)

    sea_pos = torch.stack(sea_pos) # L, N, d
    sea_neg = torch.stack(sea_neg)
    sea = torch.stack(sea)
    UU_pos = torch.stack(UU_pos)
    UU_neg = torch.stack(UU_neg)

    print("Saving UU positive and negative transformations to local files.")
    if args.do_mean_subtraction:
        torch.save(UU_pos, args.output_path+'/uu_positive.pt')
        torch.save(UU_neg, args.output_path+'/uu_negative.pt')
    else:
        torch.save(UU_pos, args.output_path+'/no_mean_sub_uu_positive.pt')
        torch.save(UU_neg, args.output_path+'/no_mean_sub_uu_negative.pt')

    # Plot for pos and neg sea
    # Transpose for visualisation over each layer -> N, L, d
    if args.visualisation:
        sea_pos = sea_pos.transpose(0,1) 
        sea_neg = sea_neg.transpose(0,1)
        sea = sea.transpose(0,1)

        if args.feature_function:
            # Only plot positive and negative
            all_X = torch.cat([torch.stack(nonlinear_pos).transpose(0,1), torch.stack(nonlinear_neg).transpose(0,1), positives, negatives])
            all_y = ['nonlinear_positive'] * args.demonstration_number + ['nonlinear_negative'] * args.demonstration_number + ['positive'] * args.demonstration_number + ['negative'] * args.demonstration_number 
            visualise_embeddings(all_X, all_y, output_figure_name=args.output_path+'/embedding-visualisation-seperate.pdf')    
        else:
            all_X = torch.cat([positives, negatives])
            all_y = ['positive'] * args.demonstration_number + ['negative'] * args.demonstration_number 
            visualise_embeddings(all_X, all_y, output_figure_name=args.output_path+'/embedding-visualisation-seperate.pdf')


        all_X = torch.cat([positives, negatives, bases, sea])
        all_y = ['positive'] * args.demonstration_number + ['negative'] * args.demonstration_number + ['base'] * args.demonstration_number + ['sea'] * args.demonstration_number
        visualise_embeddings(all_X, all_y, output_figure_name=args.output_path+'/embedding-visualisation-combined.pdf')

    