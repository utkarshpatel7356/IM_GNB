import pandas as pd
import torch
import utlis
from bandit_algo import User_GNN_Bandit_Per_Arm
from parameters import get_GNB_parameters
import argparse
import numpy as np
import time
import signal
import sys
from multiprocessing import Pool
from load_data import Market_IM

# --- Global Configuration ---
Num_Seeds = 1    # Number of unique starting points
Num_Runs = 1     # Number of episodes/campaigns per seed
Num_cluster = 10 # Matches the number of lead stocks in Market_IM
Budget = 50      # Horizon (trading days/steps)
L = 1            # Number of lead stocks to select per step
data = 'market'  # Label for the dataset

# Initialize the Market Environment
b = Market_IM(Num_Seeds=Num_Seeds, Budget=Budget, Num_Runs=Num_Runs, Num_cluster=Num_cluster)

def train_model(seed):
    # Device Selection (GPU/CPU)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"Device: {dev} | Seed: {seed}")

    # Generate Market Contexts and Episodes
    contexts, episodes = b.generate(seed)
    activations_hist = []      # Distinct lag-stock responses
    all_activations_hist = []  # Total responses

    for p_i in range(Num_Runs):
        print(f"--- Current Run: {p_i + 1}/{Num_Runs} ---")
        
        # Load hyperparameters (Mapping 'market' to a config in parameters.py)
        # Note: You should add 'market' case to parameters.py similarly to 'weibo'
        parser = get_GNB_parameters(dataset='weibo') 
        args = parser.parse_args()

        # Initialize the GNB Model
        model = User_GNN_Bandit_Per_Arm(
            dim=b.dim, user_n=b.num_user, arm_n=b.n_arm, k=args.k,
            GNN_lr=args.GNN_lr, user_lr=args.user_lr,
            bw_reward=args.bw_reward, bw_conf_b=args.bw_conf_b,
            batch_size=args.batch_size,
            GNN_pooling_step_size=args.GNN_pool_step_size,
            user_pooling_step_size=args.user_pool_step_size,
            arti_explore_constant=args.arti_explore_constant,
            num_layer=-1, 
            explore_param=args.explore_param,
            separate_explore_GNN=args.separate_explore_GNN,
            train_every_user_model=args.train_every_user_model,
            device=device
        )

        prev_activated = set()

        for t in range(Budget):
            # 1. Get current market context (e.g., Sentiment/Volatility)
            context = contexts[p_i * Budget + t]

            # 2. Update Dual-Graphs (Exploitation & Exploration)
            model.update_user_graphs(
                contexts=np.hstack((b.influencer_emb, np.tile(context.reshape(1, -1), (b.n_arm, 1)))),
                user_i=1
            )

            # 3. Model Recommendation (Select Top L Lead Stocks)
            arm_select, point_est, whole_gradients = model.recommend(
                np.hstack((b.influencer_emb, np.tile(context.reshape(1, -1), (b.n_arm, 1)))), t, L
            )

            # 4. Observe Market Outcome
            # We look at the actual lag-stock responses for the selected lead stocks
            market_episode = episodes[p_i * Budget + t][
                episodes[p_i * Budget + t].influencer.isin([b.INFLUENCERS[a] for a in arm_select])
            ]
            
            # Remove duplicate lead signals if necessary
            if len(market_episode) > 0:
                market_episode = market_episode.groupby('influencer').sample()

            acts = set()
            acts_grouped = set()
            for row in market_episode.itertuples():
                acts.update(row.regular_node_set_unique)
                acts_grouped.update(row.regular_node_set_grouped)

            # Calculate Reward (New lag-stocks reacting to signal)
            reward = len(acts - prev_activated)
            distinct_acts = acts - prev_activated
            distinct_acts_grouped = set([b.mapping_dict[i] for i in distinct_acts])
            
            prev_activated.update(acts)
            activations_hist.append(reward)
            all_activations_hist.append(len(acts))

            # 5. Update Models based on Reward Feedback
            for arm in arm_select:
                if reward == 0:
                    # Case: Signal failed to produce a lag response
                    for u in np.arange(b.num_user):
                        if args.arti_explore_constant > 0:
                            model.update_artificial_explore_info(t, u, arm, whole_gradients)
                        model.update_info(
                            u_selected=u, a_selected=arm, 
                            contexts=np.hstack((b.influencer_emb, np.tile(context.reshape(1, -1), (b.n_arm, 1)))), 
                            reward=0,
                            GNN_gradient=whole_gradients[arm],
                            GNN_residual_reward=-point_est[arm][u]
                        )
                else:
                    # Case: Successful Lead-Lag signal identification
                    for u in distinct_acts_grouped:
                        GNN_residual_reward = (1 / len(arm_select)) - point_est[arm][u]
                        model.update_info(
                            u_selected=u, a_selected=arm, 
                            contexts=np.hstack((b.influencer_emb, np.tile(context.reshape(1, -1), (b.n_arm, 1)))),
                            reward=1 / len(arm_select),
                            GNN_gradient=whole_gradients[arm],
                            GNN_residual_reward=GNN_residual_reward
                        )

            # 6. Periodic Model Training
            u_exploit_loss, u_explore_loss = model.train_user_models(u=acts_grouped)
            GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()
            print(f"Step {t} Loss | User: {u_exploit_loss:.4f}, {u_explore_loss:.4f} | GNN: {np.mean(GNN_exploit_loss):.4f}")

    # Save Results
    reward_df = utlis.orgniaze_reward(activations_hist, all_activations_hist, seed, Budget, p_i)
    output_file = f"{data}_GNB_Results_Seed{seed}.csv"
    reward_df.to_csv(output_file, sep=";", index=False)
    print(f"Results saved to {output_file}")

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
    number_workers = 1
    seeds_list = list(b.seeds.keys())
    arg_list = [(seed,) for seed in seeds_list]

    with Pool(number_workers) as p:
        p.starmap(train_model, arg_list)