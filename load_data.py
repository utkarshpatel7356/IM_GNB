import pickle
import numpy as np
import pandas as pd
import os
import yfinance as yf
from utlis import load_pkl

class Market_IM:
    def __init__(self, Num_Seeds=5, Budget=100, Num_Runs=5, Num_cluster=50, save_data=True):
        """
        Stock Market Adaptation of Influence Maximization.
        Leads = Influencers (Arms)
        Lags/Sectors = Users (Nodes)
        """
        self.lead_symbols = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'GOOGL', 'META', 'JPM', 'GS', 'XOM']
        self.num_user = Num_cluster
        
        print("Fetching free market data via yfinance...")
        all_data = yf.download(self.lead_symbols, start="2024-01-01", end="2025-01-01")['Close']
        returns = all_data.pct_change().dropna()

        context_df = returns.rolling(window=5).mean().dropna()
        
        market_data = []
        for i in range(len(context_df)):
            current_ctx = context_df.iloc[i]
            ctx_str = ' '.join(current_ctx.values.astype(str))
            
            for arm_idx, lead in enumerate(self.lead_symbols):
                activated = [j for j, symbol in enumerate(self.lead_symbols) 
                             if returns[symbol].iloc[i] > 0.005]
                
                market_data.append({
                    'context': ctx_str,
                    'influencer': arm_idx,
                    # Convert sets to lists/strings for CSV compatibility
                    'regular_node_set_unique': list(set(activated)),
                    'regular_node_set_grouped': list(set([j % Num_cluster for j in activated])),
                    'new_activations': len(activated)
                })

        self.tweets = pd.DataFrame(market_data)
        
        # --- SAVE DATA TO CSV ---
        if save_data:
            csv_filename = f"market_{Num_cluster}clustered.csv"
            # Using sep=';' to match your previous codebase requirements
            self.tweets.to_csv(csv_filename, sep=';', index=False)
            print(f"Saved campaign data to: {csv_filename}")

        self.INFLUENCERS = list(range(len(self.lead_symbols)))
        self.n_arm = len(self.INFLUENCERS)
        
        # --- GENERATE AND SAVE EMBEDDINGS ---
        self.influencer_emb = np.random.normal(0, 1, (self.n_arm, 20))
        if save_data:
            emb_filename = "influencer_embedding_market.pkl"
            emb_dict = {i: self.influencer_emb[i] for i in range(self.n_arm)}
            with open(emb_filename, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"Saved influencer embeddings to: {emb_filename}")

        self.MAX_NA = float(self.tweets.new_activations.max())
        self.twitter_contexts = list(context_df.values)
        
        context_width = len(self.twitter_contexts[0]) 
        self.dim = 20 + context_width 

        np.random.seed(100)
        self.seeds = dict.fromkeys(list(set([np.random.randint(1000) 
                                   for _ in np.arange(Num_Seeds + 10)]))[:Num_Seeds])

        for seed in self.seeds.keys():
            np.random.seed(seed)
            context_idx = np.random.choice(len(self.twitter_contexts), 
                                           size=Budget*Num_Runs, replace=True)
            self.seeds[seed] = [self.twitter_contexts[idx] for idx in context_idx]

        self.mapping_dict = {i: i % Num_cluster for i in range(len(self.lead_symbols))}

    def generate(self, seed):
        campaign = []
        contexts = self.seeds[seed]
        for context in contexts:
            ctx_str = ' '.join(context.astype(str))
            campaign_temp = self.tweets.loc[self.tweets.context == ctx_str, :]
            campaign.append(campaign_temp)
        return contexts, campaign