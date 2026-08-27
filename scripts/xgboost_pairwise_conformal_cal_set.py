# In addition to main.py we introduce the exact same production refinement of xgboost with the only caveat
# that we allow the system to have pair-specific conformal calibration sets.
#
# Reminder: Under the tournament and production architecture the reference distribution a pair's M2 confidence is compared against is the same pooled mix for 
# all 11 instruments. If EM pairs generate far more trade rows (which they do), the pooled calibration set is dominated by EM-pair mistake statistics. A major's M2 confidence score 
# then gets benchmarked against "how confident is M2 typically when an EM-pair trade goes wrong" rather than against its own pair's baseline. 
# If majors' M1/M2 score distributions sit on a different scale (which they likely do, given fewer/weaker patterns to learn from), this pooling can make majors' p-values look 
# unremarkable almost by construction, thus gating them out regardless of whether they have a small but real edge.

import os
from utils import load_and_split
from forex_pairwise_conformal_cal_sets_experiment import run_wfv

# MINI COCKPIT #
winner_model = 'XGBoost'
n_inner_splits = 2
n_purged = 10
n_embargo = 10
opt_metric = 'sharpe'

# Low trials for speed
n_model_trials = 50 
n_trading_trials = 100

# Thresholds from Cockpit
mi_thresh = 0.005
pi_threshold = 0.001
corr_thresh = 0.90

# END MINI COCKPIT #

def pairwise_conformal_cal_sets_experiment():
    print("\n" + "="*80)
    print("PRODUCTION REFINEMENT PIPELINE WITH PAIRWISE CONFORMAL CALIBRATION SETS")
    print("="*80)

    # 1. Load available master data
    agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
    if not os.path.exists(agg_file):
        print(f"Error: {agg_file} not found. Please run the full pipeline or aggregation first.")
        return

    # Load and split (using 90% for training/tuning as per main)
    split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
    global_train_data = split[0]
    global_test_data = split[1]

    print(f"\nStarting Production Refinement for {winner_model}...")
    print(f"Model Trials: {n_model_trials}, Trading Trials: {n_trading_trials}")
    
    # 3. Execute Production Refinement
    try:
        run_wfv(data=global_train_data,
                global_test_data=global_test_data,
                winner_name=winner_model,
                n_inner_splits=n_inner_splits,
                n_purged=n_purged,
                n_embargo=n_embargo,
                opt_metric=opt_metric,
                n_model_trials=n_model_trials,
                n_trading_trials=n_trading_trials,
                mi_thresh=mi_thresh,
                pi_threshold=pi_threshold,
                corr_thresh=corr_thresh)
        
        print("\n" + "="*80)
        print(f"Artifact saved to: ./data/models/final_production/global_{winner_model}_prod.joblib")
        print("Check figures/optimization/ for Optuna plots.")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pairwise_conformal_cal_sets_experiment()