import os
import glob
import numpy as np
import pickle

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = f"{SCRIPT_DIRECTORY}/../data/dataset_unified.pkl"

def compute_x_fin(q_final, l1=1.0, l2=1.0):
    """
    Calculates the final x position based on angles (already shifted).
    """
    q1 = q_final[0]
    q2 = q_final[1]
    return l1 * np.cos(q1) + l2 * np.cos(q1 + q2)

def process_dataset(root_dir, output_filename=OUTPUT_PATH):
    full_dataset = {
        "w_matrices": [],
        "q_trajs": [],
        "dq_trajs": [],
        "ddq_trajs": [],
        "params": []
    }
    
    search_path = os.path.join(root_dir, "ns_*")
    folders = glob.glob(search_path)
    
    if not folders:
        print(f"No 'ns_*' folders found in {root_dir}")
        return

    print(f"Processing {len(folders)} folders (Atomic Mode)...")
    count_files = 0
    count_errors = 0
    
    for folder in folders:
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        npz_files.sort()
        
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                
                # --- Step 1: Temporary data preparation ---
                # We do not modify full_dataset yet
                
                # 1. W matrices
                w_original = data['W']
                w_T = w_original.T
                
                # 2. Processing q (WITH SHIFT) and dq
                xs = data['xs']
                q_raw = xs[:, :2]
                q_traj = q_raw - (np.pi / 2) # Shift
                dq_traj = xs[:, 2:]
                
                # 3. ddq
                ddq_traj = data['us']
                
                # 4. Params
                N = q_traj.shape[0]
                q_init = q_traj[0]
                q_final = q_traj[-1]
                x_fin = compute_x_fin(q_final)
                
                param_dict = {
                    "N": N,
                    "q_init": q_init,
                    "x_fin": x_fin
                }

                # --- Step 2: Atomic addition ---
                # If we get here, it means all keys existed and calculations are OK.
                full_dataset["w_matrices"].append(w_T)
                full_dataset["q_trajs"].append(q_traj)
                full_dataset["dq_trajs"].append(dq_traj)
                full_dataset["ddq_trajs"].append(ddq_traj)
                full_dataset["params"].append(param_dict)
                
                count_files += 1
                if count_files % 1000 == 0:
                    print(f"{count_files} trajectories processed...")

            except Exception as e:
                count_errors += 1
                print(f"Error ignored with {file_path} : {e}")
                # Thanks to this structure, if an error occurs, NOTHING is added.
                # The lists therefore remain perfectly synchronized.

    # --- Security Check ---
    n_w = len(full_dataset["w_matrices"])
    n_q = len(full_dataset["q_trajs"])
    print(f"\nVerification: {n_w} matrices vs {n_q} trajectories.")
    assert n_w == n_q, "Critical error: Lists are out of sync!"

    # --- Saving ---
    print(f"Saving {count_files} items (Errors: {count_errors}) in {output_filename}...")
    with open(output_filename, 'wb') as f:
        pickle.dump(full_dataset, f)
    
    print("Done!")

if __name__ == "__main__":
    pose = "P2"
    ROOT_DIRECTORY = f"{SCRIPT_DIRECTORY}/{pose}"
    process_dataset(ROOT_DIRECTORY)