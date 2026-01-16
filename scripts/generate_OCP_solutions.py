import numpy as np
import json
import multiprocessing
import os
import pickle
import sys

# Import updated tools
# Ensure that OCP_solving_cpin_new_variable.py contains the modified version with ddq
sys.path.append(os.getcwd())
from tools.OCP_solving_cpin import solve_DOC
from tools.OCP_solving_cpin import plot_trajectory_q1, plot_trajectory_q2, plot_trajectory_ee

# --- CONFIGURATION ---
NUM_SAMPLES = 300
FREQ = 100.0
NUM_CORES = multiprocessing.cpu_count() - 2 
# NUM_CORES = 1

# Generation constants
# 3 initial joint angles configurations
Q_INIT_BASES_DEG = [[-90, 90], [-15, 105], [-115, 115]]
# noise standard deviation over the initial joint angles
Q_INIT_NOISE_STD_DEG = 7.0
# final a-axis position of the end-effector
X_FIN_BASE = 1.9
# noise standard deviation over the final a-axis position of the end-effector
X_FIN_NOISE_STD = 0.01

# --- WORKER FUNCTION ---
def generate_single_sample(seed):
    """
    Sample generation.
    Update: Retrieving acceleration (ddq)
    """
    np.random.seed(seed)
    
    # Randomization
    N_steps = np.random.randint(80, 201) 
    base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
    q_init_rad = np.deg2rad(np.array(Q_INIT_BASES_DEG[base_idx]) + np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2))
    x_fin = X_FIN_BASE + np.random.normal(0, X_FIN_NOISE_STD)
    
    # Weights (Normalized to sum to 1)
    w_matrix = np.random.rand(5, 3)
    w_matrix = w_matrix / w_matrix.sum(axis=0) 
    
    try:
        # Solver call (Retrieving 3 variables now)
        # res_ddq contains the optimal acceleration
        res_q, res_dq, res_ddq = solve_DOC(w_matrix, N_steps, x_fin=x_fin, q_init=q_init_rad, verbose=False)
        
        if res_q is not None:
            return {
                "status": "success",
                "w_matrix": w_matrix,
                "q": res_q,
                "dq": res_dq,
                "ddq": res_ddq,  # Adding acceleration to the result
                "params": {"N": N_steps, "q_init": q_init_rad, "x_fin": x_fin}
            }
        else:
            return {"status": "failed"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    
    seeds = [np.random.randint(0, 1000000) for _ in range(NUM_SAMPLES)]
    
    print(f"\nStarting generation on {NUM_CORES} cores.")
    print(f"Target: {NUM_SAMPLES} samples.")

    list_results_angles = []
    list_results_angular_velocities = []
    list_results_accelerations = []  # New list to store ddq
    list_w_matrices = []
    list_parameters = []
    valid_count = 0

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        try:
            from tqdm import tqdm
            iterator = tqdm(pool.imap_unordered(generate_single_sample, seeds), total=NUM_SAMPLES)
        except ImportError:
            print("Install tqdm for a progress bar.")
            iterator = pool.imap_unordered(generate_single_sample, seeds)

        for result in iterator:
            if result["status"] == "success":
                list_results_angles.append(result["q"])
                list_results_angular_velocities.append(result["dq"])
                list_results_accelerations.append(result["ddq"]) # Storage
                list_w_matrices.append(result["w_matrix"])
                list_parameters.append(result["params"])
                valid_count += 1
            
            if 'tqdm' not in locals() and valid_count % 100 == 0:
                print(f"\rSuccess: {valid_count}/{NUM_SAMPLES}", end="")

    print(f"\n\nGeneration completed. {valid_count}/{NUM_SAMPLES} valid.")

    # Final dictionary construction
    data_dict = {
        "w_matrices": list_w_matrices,
        "q_trajs": list_results_angles,
        "dq_trajs": list_results_angular_velocities,
        "ddq_trajs": list_results_accelerations, # Add to dataset
        "params": list_parameters
    }

    suffix = f"{valid_count}_samples"
    filepath = f'data/dataset_{suffix}.pkl'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dataset saved to: {filepath}")

    if valid_count > 0:
        subset_q = list_results_angles[:10]
        subset_dq = list_results_angular_velocities[:10]
        subset_ddq = list_results_accelerations[:10] # New list
        
        # Passing the new argument
        plot_trajectory_q1(subset_q, subset_dq, subset_ddq)
        plot_trajectory_q2(subset_q, subset_dq, subset_ddq)
        plot_trajectory_ee(subset_q, x_fin_target=X_FIN_BASE)