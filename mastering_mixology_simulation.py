import random
from itertools import product
from collections import defaultdict
import csv
import argparse
import os
import shutil
import multiprocessing
from multiprocessing import Pool
# === Setup ===

class Potion:
    def __init__(self, id, mox, aga, lye, weight):
        self.id = id
        self.mox = mox
        self.aga = aga
        self.lye = lye
        self.weight = weight

potions = [
    Potion("AAA", 0, 20, 0, 5),
    Potion("MMM", 20, 0, 0, 5),
    Potion("LLL", 0, 0, 20, 5),
    Potion("MMA", 20, 10, 0, 4),
    Potion("MML", 20, 0, 10, 4),
    Potion("AAM", 10, 20, 0, 4),
    Potion("ALA", 0, 20, 10, 4),
    Potion("MLL", 10, 0, 20, 4),
    Potion("ALL", 0, 10, 20, 4),
    Potion("MAL", 20, 20, 20, 3),
]

potion_map = {p.id: p for p in potions}
potion_ids = list(potion_map.keys())
potion_weights = [potion_map[pid].weight for pid in potion_ids]

target = {"mox": 61050, "aga": 52550, "lye": 70500}


# === Helper Functions ===

def is_done(current):
    return current["mox"] >= target["mox"] and current["aga"] >= target["aga"] and current["lye"] >= target["lye"]

# Pre-compute bonus multipliers for faster lookup
BONUS_MULTIPLIERS = {1: 1.0, 2: 1.2, 3: 1.4}

def bonus_for_count(n):
    return BONUS_MULTIPLIERS[n]

def run_single_simulation(args_tuple):
    """Run a single simulation. This function is designed to be called by worker processes."""
    draw_to_choice_map, run_id = args_tuple
    current_mox, current_aga, current_lye = 0, 0, 0
    potion_counts = defaultdict(int)
    total_potions_used = 0
    
    # Cache target values for faster comparison
    target_mox, target_aga, target_lye = target["mox"], target["aga"], target["lye"]

    while current_mox < target_mox or current_aga < target_aga or current_lye < target_lye:
        draw = tuple(sorted(random.choices(potion_ids, weights=potion_weights, k=3)))
        chosen_potions = draw_to_choice_map.get(draw)

        if not chosen_potions:
            raise ValueError(f"No potion selection provided for draw: {draw}")

        count = len(chosen_potions)
        bonus = BONUS_MULTIPLIERS[count]
        for pid in chosen_potions:
            potion = potion_map[pid]
            current_mox += potion.mox * bonus
            current_aga += potion.aga * bonus
            current_lye += potion.lye * bonus
            potion_counts[pid] += 1
            total_potions_used += 1

    # Record run data - build dict more efficiently
    run_record = {
        "total_potions": total_potions_used,
        "mox": current_mox,
        "aga": current_aga,
        "lye": current_lye,
    }
    # Add potion counts (only for potions that were used)
    for pid in potion_ids:
        run_record[pid] = potion_counts[pid]
    return run_record

def run_baseline_simulation(draw_to_choice_map, runs=100000):
    aggregate_potion_counts = defaultdict(int)

    # Determine number of workers (up to CPU count)
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} worker processes for {runs} simulation runs\n")
    
    # Prepare arguments for worker processes
    # Each worker will receive (draw_to_choice_map, run_id)
    run_args = [(draw_to_choice_map, run_id) for run_id in range(runs)]
    
    # Run simulations in parallel using multiprocessing pool with progress tracking
    all_run_data = []
    completed = 0
    # Update every 1% or every 100 runs, whichever is smaller (for better responsiveness)
    update_interval = max(1, min(runs // 100, 100))
    
    print("Running simulations...")
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered to get results as they complete for better progress tracking
        for result in pool.imap_unordered(run_single_simulation, run_args):
            all_run_data.append(result)
            completed += 1
            # Update progress bar at intervals or on completion
            if completed % update_interval == 0 or completed == runs:
                percentage = (completed / runs) * 100
                bar_length = 40
                filled = int(bar_length * completed / runs)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\rProgress: [{bar}] {percentage:6.2f}% ({completed:,}/{runs:,} runs completed)", end='', flush=True)
    
    print("\n✓ All simulations completed!\n")  # New line after progress bar
    
    # === Summary Statistics ===
    # Compute all statistics in a single pass over the data
    total_potions_sum = 0
    mox_sum, aga_sum, lye_sum = 0, 0, 0
    total_potions_min = float('inf')
    total_potions_max = 0
    mox_min = aga_min = lye_min = float('inf')
    mox_max = aga_max = lye_max = 0
    min_run_total_potions = None
    max_run_total_potions = None
    min_run_mox = min_run_aga = min_run_lye = None
    max_run_mox = max_run_aga = max_run_lye = None
    
    # Per-potion statistics
    potion_mins = {pid: float('inf') for pid in potion_ids}
    potion_maxs = {pid: 0 for pid in potion_ids}
    
    # Single pass aggregation
    for run_record in all_run_data:
        # Aggregate potion counts
        for pid in potion_ids:
            count = run_record[pid]
            aggregate_potion_counts[pid] += count
            if count < potion_mins[pid]:
                potion_mins[pid] = count
            if count > potion_maxs[pid]:
                potion_maxs[pid] = count
        
        # Total potions statistics
        total_potions = run_record["total_potions"]
        total_potions_sum += total_potions
        if total_potions < total_potions_min:
            total_potions_min = total_potions
            min_run_total_potions = run_record
        if total_potions > total_potions_max:
            total_potions_max = total_potions
            max_run_total_potions = run_record
        
        # Resource statistics
        mox_val = run_record["mox"]
        aga_val = run_record["aga"]
        lye_val = run_record["lye"]
        mox_sum += mox_val
        aga_sum += aga_val
        lye_sum += lye_val
        
        if mox_val < mox_min:
            mox_min = mox_val
            min_run_mox = run_record
        if mox_val > mox_max:
            mox_max = mox_val
            max_run_mox = run_record
            
        if aga_val < aga_min:
            aga_min = aga_val
            min_run_aga = run_record
        if aga_val > aga_max:
            aga_max = aga_val
            max_run_aga = run_record
            
        if lye_val < lye_min:
            lye_min = lye_val
            min_run_lye = run_record
        if lye_val > lye_max:
            lye_max = lye_val
            max_run_lye = run_record
    
    # Calculate averages
    avg_potions_used = total_potions_sum / runs
    avg_per_potion = {pid: aggregate_potion_counts[pid] / runs for pid in potion_ids}
    avg_targets = {
        "mox": mox_sum / runs,
        "aga": aga_sum / runs,
        "lye": lye_sum / runs,
    }
    
    # Helper functions for min/max runs (now using pre-computed values)
    def min_run_by(key):
        if key == "total_potions":
            return min_run_total_potions
        elif key == "mox":
            return min_run_mox
        elif key == "aga":
            return min_run_aga
        elif key == "lye":
            return min_run_lye
        return min(all_run_data, key=lambda x: x[key])

    def max_run_by(key):
        if key == "total_potions":
            return max_run_total_potions
        elif key == "mox":
            return max_run_mox
        elif key == "aga":
            return max_run_aga
        elif key == "lye":
            return max_run_lye
        return max(all_run_data, key=lambda x: x[key])

    # === Prepare Output Directory ===
    strategy_basename = os.path.splitext(os.path.basename(args.strategy_file))[0]
    output_dir = os.path.join("strategies", strategy_basename)
    os.makedirs(output_dir, exist_ok=True)

    # Copy the strategy file into the output directory
    shutil.copy2(args.strategy_file, os.path.join(output_dir, os.path.basename(args.strategy_file)))

    # === Write Detailed Run Data ===
    run_data_path = os.path.join(output_dir, "run_data.csv")
    # === Write Detailed Run Data ===
    with open(run_data_path, "w", newline="") as csvfile:
        fieldnames = ["total_potions", "mox", "aga", "lye"] + potion_ids
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in all_run_data:
            writer.writerow(run)
        print(f"Run data saved to {run_data_path}")
    
    summary_path = os.path.join(output_dir, "summary.csv")
    # === Write Summary Statistics ===
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])

        # Overall potions used
        writer.writerow(["Average Potions Used", f"{avg_potions_used:.2f}"])
        writer.writerow(["Minimum Potions Used", total_potions_min])
        writer.writerow(["Maximum Potions Used", total_potions_max])

        writer.writerow([])
        writer.writerow(["Potion Type", "Average", "Minimum", "Maximum"])
        for pid in potion_ids:
            writer.writerow([pid, f"{avg_per_potion[pid]:.2f}", potion_mins[pid], potion_maxs[pid]])

        writer.writerow([])
        writer.writerow(["Target Resource", "Average", "Minimum (MOX,AGA,LYE)", "Maximum (MOX,AGA,LYE)"])
        for key in ["mox", "aga", "lye"]:
            min_run = min_run_by(key)
            max_run = max_run_by(key)
            writer.writerow([
                key.upper(),
                f"{avg_targets[key]:.2f}",
                f"{min_run['mox']},{min_run['aga']},{min_run['lye']}",
                f"{max_run['mox']},{max_run['aga']},{max_run['lye']}"
            ])
        print(f"Summary statistics saved to {summary_path}")

    # === Console Report ===
    print(f"\n=== Simulation Summary for Strategy File: {args.strategy_file} ===")
    print(f"Runs: {runs}")
    print(f"Average Potions Used to Reach Target: {avg_potions_used:.2f}")
    print("Average Potion Usage per Type:")
    for pid in potion_ids:
        print(f"  {pid}: {avg_per_potion[pid]:.2f}")
    print("Average Reached:")
    for key in ["mox", "aga", "lye"]:
        print(f"  {key.upper()}: {avg_targets[key]:.2f}")

# === Example Usage ===
def load_draw_choices_from_csv(filepath):
    draw_to_choice_map = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            draw = tuple(sorted(row['draw'].split('-')))
            choice = row['choice'].split('-')
            draw_to_choice_map[draw] = choice
    return draw_to_choice_map

def generate_draw_template(filepath="draw_choices.csv"):
    all_draws = set(tuple(sorted(draw)) for draw in product(potion_ids, repeat=3))
    with open(filepath, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["draw", "choice"])
        for draw in sorted(all_draws):
            draw_key = "-".join(draw)
            writer.writerow([draw_key, "-".join(draw)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mastering Mixology simulation.")
    parser.add_argument(
        "strategy_file",
        type=str,
        help="Path to your strategy CSV file (e.g., strategy.csv).\nAfter running it will create a folder in strategies and it will copy your current strategy to that folder and store the results."
    )
    parser.add_argument(
        "number_of_runs",
        type=int,
        nargs="?",
        default=100000,
        help="Number of simulation runs to perform (default: 100000)"
    )
    args = parser.parse_args()
    # generate_draw_template()
    # Print number of runs
    print(f"Using strategy file: {args.strategy_file}")
    print(f"Number of runs: {args.number_of_runs}")
    draw_to_choice_map = load_draw_choices_from_csv(args.strategy_file)
    run_baseline_simulation(draw_to_choice_map, runs=args.number_of_runs)
