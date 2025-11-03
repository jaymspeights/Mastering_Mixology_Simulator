"""
Genetic Algorithm Strategy Optimizer for Mastering Mixology Simulation

This module optimizes potion strategy files using a genetic algorithm approach.
It mutates strategy choices and evaluates them to find strategies that minimize
the average number of potions needed to reach the target.
"""

import random
import csv
import os
import copy
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
from itertools import product

# Import from the original simulation file
# We'll duplicate necessary code to avoid circular dependencies
# Define potions and constants
class Potion:
    def __init__(self, id, mox, aga, lye, weight, level):
        self.id = id
        self.mox = mox
        self.aga = aga
        self.lye = lye
        self.weight = weight
        self.level = level

potions = [
    Potion("AAA", 0, 20, 0, 5, 60),
    Potion("MMM", 20, 0, 0, 5, 60),
    Potion("LLL", 0, 0, 20, 5, 60),
    Potion("MMA", 20, 10, 0, 4, 63),
    Potion("MML", 20, 0, 10, 4, 66),
    Potion("AAM", 10, 20, 0, 4, 69),
    Potion("ALA", 0, 20, 10, 4, 72),
    Potion("MLL", 10, 0, 20, 4, 75),
    Potion("ALL", 0, 10, 20, 4, 78),
    Potion("MAL", 20, 20, 20, 3, 81),
]

potion_map = {p.id: p for p in potions}


def get_available_potions(level=99):
    """Get potions available at the given level.
    
    Args:
        level: Maximum level requirement (default: 99)
    
    Returns:
        Tuple of (potion_ids, potion_weights) filtered by level
    """
    available_potions = [p for p in potions if p.level <= level]
    available_ids = [p.id for p in available_potions]
    available_weights = [p.weight for p in available_potions]
    return available_ids, available_weights

target = {"mox": 61050, "aga": 52550, "lye": 70500}
BONUS_MULTIPLIERS = {1: 1.0, 2: 1.2, 3: 1.4}

# Cache target values
target_mox, target_aga, target_lye = target["mox"], target["aga"], target["lye"]

# Global variable for shared draws (used by worker processes)
_shared_draws = None

def init_worker(shared_draws):
    """Initialize worker process with shared draws."""
    global _shared_draws
    _shared_draws = shared_draws


def run_single_simulation_eval(draw_to_choice_map, pre_generated_draws, draw_index_start):
    """Run a single simulation using pre-generated draws and return total potions used.
    
    Args:
        draw_to_choice_map: Strategy mapping draws to choices
        pre_generated_draws: List of pre-generated draw tuples
        draw_index_start: Starting index in pre_generated_draws for this simulation
    
    Returns:
        Tuple of (total_potions_used, draw_index_end, success) where:
        - total_potions_used: Number of potions used (0 if ran out of draws)
        - draw_index_end: Next index to use (or len(pre_generated_draws) if ran out)
        - success: True if completed successfully, False if ran out of draws
    """
    current_mox, current_aga, current_lye = 0, 0, 0
    total_potions_used = 0
    draw_idx = draw_index_start

    while current_mox < target_mox or current_aga < target_aga or current_lye < target_lye:
        if draw_idx >= len(pre_generated_draws):
            # Ran out of draws - return failure
            return (0, draw_idx, False)
        
        draw = pre_generated_draws[draw_idx]
        draw_idx += 1
        
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
            total_potions_used += 1

    return (total_potions_used, draw_idx, True)


def evaluate_strategy_worker(args_tuple):
    """Worker function to evaluate a strategy with multiple runs using pre-generated draws.
    
    Uses global _shared_draws to avoid pickling overhead.
    
    Returns:
        Tuple of (child_id, score, avg_potions_used) where score = worst_avg - avg_potions_used
        If any simulation runs out of draws, score is 0.
    """
    global _shared_draws
    child_id, draw_to_choice_map, num_runs, worst_avg_reference = args_tuple
    total_potions_sum = 0
    draw_idx = 0
    
    # Use the same pre-generated draws for all runs (from global variable)
    for run_num in range(num_runs):
        potions_used, draw_idx, success = run_single_simulation_eval(
            draw_to_choice_map, _shared_draws, draw_idx
        )
        if not success:
            # Ran out of draws - score is 0
            return (child_id, 0.0, 0.0)
        total_potions_sum += potions_used
    
    avg_potions_used = total_potions_sum / num_runs
    score = worst_avg_reference - avg_potions_used  # Higher score is better
    return (child_id, score, avg_potions_used)


def load_strategy_from_csv(filepath):
    """Load strategy from CSV file and return as list of rows."""
    rows = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows


def load_strategy_as_dict(filepath):
    """Load strategy from CSV and return as dictionary for simulation."""
    draw_to_choice_map = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            draw = tuple(sorted(row['draw'].split('-')))
            choice = row['choice'].split('-')
            draw_to_choice_map[draw] = choice
    return draw_to_choice_map


def save_strategy_to_csv(rows, filepath):
    """Save strategy rows to CSV file."""
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['draw', 'choice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_valid_choices_for_draw(draw_str):
    """Get all 7 valid choices for a given draw string.
    
    Args:
        draw_str: String like "AAA-AAM-MAL"
    
    Returns:
        List of 7 valid choice strings
    """
    parts = draw_str.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid draw format: {draw_str}")
    
    p0, p1, p2 = parts[0], parts[1], parts[2]
    return [
        p0,                    # First potion alone
        p1,                    # Second potion alone
        p2,                    # Third potion alone
        f"{p0}-{p1}",          # First and second
        f"{p1}-{p2}",          # Second and third
        f"{p0}-{p2}",          # First and third
        f"{p0}-{p1}-{p2}",     # All three
    ]


def mutate_strategy(rows, num_mutations=1):
    """Mutate a strategy by changing random rows' choices.
    
    Args:
        rows: List of strategy rows (each is a dict with 'draw' and 'choice')
        num_mutations: Number of rows to mutate (default: 1)
    
    Returns:
        New list of rows with mutations applied
    """
    mutated_rows = copy.deepcopy(rows)
    
    # Apply mutations
    for _ in range(num_mutations):
        # Select random row
        row_idx = random.randint(0, len(mutated_rows) - 1)
        row = mutated_rows[row_idx]
        
        # Get valid choices for this draw
        valid_choices = get_valid_choices_for_draw(row['draw'])
        
        # Randomly select a new choice (may be same as current)
        new_choice = random.choice(valid_choices)
        row['choice'] = new_choice
    
    return mutated_rows


def select_parents(child_results, num_parents=2):
    """Select parents using roulette wheel selection based on scores.
    
    Args:
        child_results: List of (child_id, score, avg_potions_used, rows) tuples
        num_parents: Number of parents to select (default: 2)
    
    Returns:
        List of parent rows (strategy rows) selected
    """
    # Filter to only children with score > 0
    valid_children = [(score, rows) for _, score, _, rows in child_results if score > 0]
    
    if not valid_children:
        # If no valid children, use all children (but score 0 means low probability)
        valid_children = [(max(score, 0.01), rows) for _, score, _, rows in child_results]
    
    # Calculate total score
    total_score = sum(score for score, _ in valid_children)
    
    if total_score == 0:
        # Fallback: equal probability
        return [random.choice(valid_children)[1] for _ in range(num_parents)]
    
    # Select parents using roulette wheel
    parents = []
    for _ in range(num_parents):
        r = random.uniform(0, total_score)
        cumulative = 0
        for score, rows in valid_children:
            cumulative += score
            if r <= cumulative:
                parents.append(rows)
                break
    
    return parents


def crossover_strategies(parent1_rows, parent2_rows):
    """Create a child by randomly combining genes from two parents.
    
    For each row, randomly choose the choice from parent1 or parent2.
    
    Args:
        parent1_rows: Strategy rows from first parent
        parent2_rows: Strategy rows from second parent
    
    Returns:
        New strategy rows combining both parents
    """
    # Ensure both parents have the same draws (should always be true)
    if len(parent1_rows) != len(parent2_rows):
        raise ValueError("Parents must have the same number of rows")
    
    child_rows = []
    for i, row1 in enumerate(parent1_rows):
        row2 = parent2_rows[i]
        # Ensure same draw
        if row1['draw'] != row2['draw']:
            raise ValueError(f"Mismatched draws at index {i}")
        
        # Randomly choose choice from parent1 or parent2
        child_row = {'draw': row1['draw'], 'choice': random.choice([row1['choice'], row2['choice']])}
        child_rows.append(child_row)
    
    return child_rows


def generate_children_from_population(previous_generation_results, num_children=10, 
                                     max_mutations=10):
    """Generate new children from previous generation using selection, crossover, and mutation.
    
    Args:
        previous_generation_results: List of (child_id, score, avg_potions_used, rows) tuples
        num_children: Number of children to generate (default: 100)
        max_mutations: Maximum number of mutations per child (default: 10). 
                      Each child gets 0 to max_mutations mutations randomly.
    
    Returns:
        List of child strategy rows
    """
    children = []
    
    for _ in range(num_children):
        # Select two parents (can be the same)
        parents = select_parents(previous_generation_results, num_parents=2)
        parent1, parent2 = parents[0], parents[1]
        
        # Crossover to create base child
        child = crossover_strategies(parent1, parent2)
        
        # Apply additional mutations (0 to max_mutations)
        num_mutations = random.randint(0, max_mutations)
        child = mutate_strategy(child, num_mutations=num_mutations)
        
        children.append(child)
    
    return children


def rows_to_dict(rows):
    """Convert strategy rows to dictionary format for simulation."""
    draw_to_choice_map = {}
    for row in rows:
        draw = tuple(sorted(row['draw'].split('-')))
        choice = row['choice'].split('-')
        draw_to_choice_map[draw] = choice
    return draw_to_choice_map


def generate_random_strategy_from_template(template_rows):
    """Generate a completely random strategy from a template.
    
    For each row in the template, randomly selects one of the 7 valid choices.
    
    Args:
        template_rows: List of strategy rows from template (with draws)
    
    Returns:
        New strategy rows with random choices
    """
    random_strategy = []
    for row in template_rows:
        valid_choices = get_valid_choices_for_draw(row['draw'])
        random_choice = random.choice(valid_choices)
        random_strategy.append({'draw': row['draw'], 'choice': random_choice})
    return random_strategy


def generate_random_children_from_template(template_file, num_children=10):
    """Generate children with completely random genomes from a template.
    
    Args:
        template_file: Path to template CSV file
        num_children: Number of random children to generate (default: 10)
    
    Returns:
        List of child strategy rows, all randomly generated
    """
    print(f"Loading template from: {template_file}")
    template_rows = load_strategy_from_csv(template_file)
    print(f"Loaded {len(template_rows)} template rows\n")
    
    print(f"Generating {num_children} random children from template...")
    children = []
    for _ in range(num_children):
        random_strategy = generate_random_strategy_from_template(template_rows)
        children.append(random_strategy)
    
    print(f"✓ Generated {len(children)} random children\n")
    return children


def generate_children(parent_rows, num_children=10, max_mutations=10):
    """Generate children from parent strategy.
    
    Args:
        parent_rows: List of strategy rows from parent
        num_children: Total number of children to generate (default: 100)
        max_mutations: Maximum number of mutations per child (default: 10).
                      Each child gets 0 to max_mutations mutations randomly.
    
    Returns:
        List of child strategy rows (first child is unmodified parent)
    """
    children = []
    
    # Child 0: Unmodified parent
    children.append(copy.deepcopy(parent_rows))
    
    # Children 1+: Mutations (0 to max_mutations)
    for i in range(1, num_children):
        num_mutations = random.randint(0, max_mutations)
        mutated = mutate_strategy(parent_rows, num_mutations=num_mutations)
        children.append(mutated)
    
    return children


def run_generation(parent_strategy_file, num_children=10, evaluation_runs=100, 
                   output_dir="genetic_algorithm_runs/generation_0"):
    """Run a single generation of genetic algorithm.
    
    Args:
        parent_strategy_file: Path to parent strategy CSV file
        num_children: Number of children to generate (default: 100)
        evaluation_runs: Number of simulation runs per child (default: 100)
        output_dir: Directory to save results (default: "genetic_algorithm_runs/generation_0")
    
    Returns:
        List of tuples (child_id, score, avg_potions_used, rows) sorted by score (higher is better)
    """
    print(f"Loading parent strategy from: {parent_strategy_file}")
    parent_rows = load_strategy_from_csv(parent_strategy_file)
    print(f"Loaded {len(parent_rows)} strategy rows\n")
    
    print(f"Generating {num_children} children...")
    children = generate_children(parent_rows, num_children=num_children)
    print(f"✓ Generated {len(children)} children\n")
    
    # Pre-generate exactly 1,500,000 draws for all simulations
    # If a child needs more draws than this, its score will be 0
    global _shared_draws
    total_draws = 1_500_000
    print(f"Pre-generating {total_draws:,} draws for fair comparison...")
    
    # Generate draws and store in global variable
    _shared_draws = [
        tuple(sorted(random.choices(potion_ids, weights=potion_weights, k=3)))
        for _ in range(total_draws)
    ]
    
    print(f"✓ Pre-generated {len(_shared_draws):,} draws\n")
    
    # Try to use 'fork' start method on Unix/macOS for better performance
    # This allows workers to inherit the global variable without pickling
    try:
        if hasattr(multiprocessing, 'set_start_method'):
            # Only set if not already set
            try:
                multiprocessing.set_start_method('fork', force=False)
            except RuntimeError:
                # Start method already set, use current method
                pass
    except (AttributeError, ValueError):
        # Fork not available (e.g., on Windows), will use initializer
        pass
    
    # Prepare evaluation arguments (no need to pass draws - workers use global)
    num_workers = multiprocessing.cpu_count()
    print(f"Evaluating {num_children} children with {evaluation_runs} runs each")
    print(f"Using {num_workers} worker processes\n")
    
    eval_args = []
    for child_id, child_rows in enumerate(children):
        draw_to_choice_map = rows_to_dict(child_rows)
        eval_args.append((child_id, draw_to_choice_map, evaluation_runs))
    
    # Evaluate all children in parallel
    results = []
    completed = 0
    update_interval = max(1, num_children // 50)  # Update more frequently (every 2%)
    best_score_so_far = float('-inf')
    
    print("Running evaluations...")
    # Use initializer to ensure workers have access to draws (works with both fork and spawn)
    with Pool(processes=num_workers, initializer=init_worker, initargs=(_shared_draws,)) as pool:
        for child_id, score, avg_potions in pool.imap_unordered(evaluate_strategy_worker, eval_args):
            results.append((child_id, score, avg_potions))
            completed += 1
            
            # Track best score as we go
            if score > best_score_so_far:
                best_score_so_far = score
            
            # Update progress bar
            if completed % update_interval == 0 or completed == num_children:
                percentage = (completed / num_children) * 100
                bar_length = 40
                filled = int(bar_length * completed / num_children)
                bar = '█' * filled + '░' * (bar_length - filled)
                # Show current best score if we have results
                best_info = f" | Best: {best_score_so_far:.2f}" if best_score_so_far > float('-inf') else ""
                print(f"\rEpoch Progress: [{bar}] {percentage:6.2f}% ({completed}/{num_children} children){best_info}", 
                      end='', flush=True)
    
    print("\n✓ All evaluations completed!\n")
    
    # Map results back to children (since imap_unordered doesn't preserve order)
    child_results = []
    for child_id, score, avg_potions in results:
        child_results.append((child_id, score, avg_potions, children[child_id]))
    
    # Sort by score (higher is better)
    child_results.sort(key=lambda x: x[1], reverse=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best strategy
    best_child_id, best_score, best_avg_potions, best_rows = child_results[0]
    best_strategy_path = os.path.join(output_dir, "best_strategy.csv")
    save_strategy_to_csv(best_rows, best_strategy_path)
    print(f"Best strategy saved to: {best_strategy_path}")
    print(f"Best score: {best_score:.2f} (avg_potions_used: {best_avg_potions:.2f})\n")
    
    # Save results CSV
    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['child_id', 'score', 'avg_potions_used', 'rank'])
        for rank, (child_id, score, avg_potions, _) in enumerate(child_results, 1):
            writer.writerow([child_id, f"{score:.2f}", f"{avg_potions:.2f}", rank])
    print(f"Results saved to: {results_path}\n")
    
    # Print summary
    print("=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total children evaluated: {num_children}")
    print(f"Evaluations per child: {evaluation_runs}")
    print(f"\nTop 10 Children:")
    print(f"{'Rank':<6} {'Child ID':<10} {'Score':<12} {'Avg Potions Used':<20}")
    print("-" * 48)
    for rank, (child_id, score, avg_potions, _) in enumerate(child_results[:10], 1):
        marker = "★" if rank == 1 else " "
        print(f"{marker} {rank:<4} {child_id:<10} {score:>10.2f} {avg_potions:>15.2f}")
    
    avg_score = sum(score for _, score, _, _ in child_results) / len(child_results)
    avg_potions_all = sum(avg for _, _, avg, _ in child_results) / len(child_results)
    print(f"\nAverage score: {avg_score:.2f} (avg_potions_used: {avg_potions_all:.2f})")
    print(f"Best score: {child_results[0][1]:.2f} (avg_potions_used: {child_results[0][2]:.2f})")
    print(f"Worst score: {child_results[-1][1]:.2f} (avg_potions_used: {child_results[-1][2]:.2f})")
    print("=" * 70)
    
    return child_results


def evaluate_children(children, evaluation_runs=100, epoch_num=0):
    """Evaluate a list of children strategies.
    
    Args:
        children: List of child strategy rows
        evaluation_runs: Number of simulation runs per child
        epoch_num: Epoch number for display
    
    Returns:
        List of tuples (child_id, score, avg_potions_used, rows) sorted by score
        Score = worst_avg_of_current_epoch - avg_potions_used
    """
    num_children = len(children)
    num_workers = multiprocessing.cpu_count()
    
    # Evaluate all children to get averages (using dummy worst_avg for now)
    eval_args = []
    for child_id, child_rows in enumerate(children):
        draw_to_choice_map = rows_to_dict(child_rows)
        eval_args.append((child_id, draw_to_choice_map, evaluation_runs, 0.0))  # dummy, will recalculate
    
    # Evaluate all children in parallel
    avg_results = []
    completed = 0
    update_interval = max(1, num_children // 50)
    best_avg_so_far = float('inf')
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(_shared_draws,)) as pool:
        try:
            for child_id, _, avg_potions in pool.imap_unordered(evaluate_strategy_worker, eval_args):
                avg_results.append((child_id, avg_potions))
                completed += 1
                
                if avg_potions < best_avg_so_far:
                    best_avg_so_far = avg_potions
                
                if completed % update_interval == 0 or completed == num_children:
                    percentage = (completed / num_children) * 100
                    bar_length = 40
                    filled = int(bar_length * completed / num_children)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    best_info = f" | Best avg: {best_avg_so_far:.2f}" if best_avg_so_far < float('inf') else ""
                    print(f"\rEpoch {epoch_num} Progress: [{bar}] {percentage:6.2f}% ({completed}/{num_children} children){best_info}", 
                          end='', flush=True)
        except KeyboardInterrupt:
            # Suppress worker process messages when interrupting
            import sys
            import io
            old_stderr = sys.stderr
            # Keep stderr suppressed (already done by signal handler)
            try:
                pool.terminate()
                pool.join(timeout=1)
            except:
                pass
            raise  # Re-raise to be caught by outer handler
    
    # Print newline after progress bar completes
    print()
    
    # Find worst average from this epoch (highest avg_potions_used)
    avg_values = [avg for _, avg in avg_results]
    worst_avg_current = max(avg_values)
    
    # Use worst of current epoch as reference for scoring
    worst_avg_reference = worst_avg_current
    
    # Calculate scores from stored averages (no re-evaluation needed)
    # Score = (worst_avg_reference - avg_potions)²
    child_results = []
    for child_id, avg_potions in avg_results:
        base_score = worst_avg_reference - avg_potions
        score = base_score ** 2
        child_results.append((child_id, score, avg_potions, children[child_id]))
    
    # Sort by score (higher is better)
    child_results.sort(key=lambda x: x[1], reverse=True)
    
    return child_results


def run_multi_epoch(parent_strategy_file=None, template_file="strategy_template.csv", 
                    num_children=10, evaluation_runs=100,
                    base_output_dir="genetic_algorithm_runs", max_mutations=10, level=99):
    """Run epochs of genetic algorithm until Ctrl-C is pressed.
    
    Args:
        parent_strategy_file: Path to initial parent strategy CSV file (optional)
            If None, generates random children from template.
            Note: strategy_template.csv already has all potions selected for each draw.
        template_file: Path to template CSV file (default: "strategy_template.csv")
            Used when parent_strategy_file is None
        num_children: Number of children per generation (default: 100)
        evaluation_runs: Number of simulation runs per child (default: 100)
        base_output_dir: Base directory for output (default: "genetic_algorithm_runs")
        max_mutations: Maximum number of mutations per child (default: 10).
                      Each child gets 0 to max_mutations mutations randomly.
        level: Maximum level requirement for potions (default: 99)
    
    Returns:
        Best strategy across all epochs
    """
    # Set up signal handler and suppress multiprocessing logging
    import signal
    import sys
    import io
    import logging
    import os
    from multiprocessing import util
    
    # Suppress multiprocessing worker messages
    util.log_to_stderr(logging.WARNING)
    
    # Store original stderr for restoration
    _original_stderr = sys.stderr
    
    def signal_handler(signum, frame):
        # Suppress stderr immediately when SIGINT is received
        # Redirect stderr to /dev/null at OS level to suppress worker messages
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stderr.fileno())
        except:
            sys.stderr = io.StringIO()
        raise KeyboardInterrupt()
    
    # Register signal handler for SIGINT (Ctrl-C)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("GENETIC ALGORITHM - MULTI-EPOCH OPTIMIZATION")
    print("=" * 70)
    if parent_strategy_file:
        print(f"Parent strategy: {parent_strategy_file}")
    else:
        print(f"Starting from random strategies (template: {template_file})")
    print(f"Children per generation: {num_children}")
    print(f"Evaluations per child: {evaluation_runs}")
    print(f"Level requirement: {level}")
    print("Running until Ctrl-C is pressed...")
    print("=" * 70 + "\n")
    
    # Get available potions based on level
    available_potion_ids, available_potion_weights = get_available_potions(level)
    print(f"Available potions at level {level}: {', '.join(available_potion_ids)}")
    print(f"Total available: {len(available_potion_ids)} potions\n")
    
    if not available_potion_ids:
        raise ValueError(f"No potions available at level {level}")
    
    # Pre-generate draws once for all epochs
    global _shared_draws
    total_draws = 5000 * 3 * evaluation_runs
    print(f"Pre-generating {total_draws:,} draws for all epochs...")
    _shared_draws = [
        tuple(sorted(random.choices(available_potion_ids, weights=available_potion_weights, k=3)))
        for _ in range(total_draws)
    ]
    print(f"✓ Pre-generated {len(_shared_draws):,} draws\n")
    
    # Try to use 'fork' start method
    try:
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('fork', force=False)
            except RuntimeError:
                pass
    except (AttributeError, ValueError):
        pass
    
    # Track best across all epochs (by avg_potions_used, not score)
    best_overall = None
    best_overall_avg_potions = float('inf')  # Track lowest avg_potions_used
    current_generation = None
    last_epoch_num = 0
    
    try:
        # Run first epoch
        if parent_strategy_file:
            # Load initial parent strategy and generate children from it
            parent_rows = load_strategy_from_csv(parent_strategy_file)
            children = generate_children(parent_rows, num_children=num_children, 
                                         max_mutations=max_mutations)
        else:
            # Generate random children from template
            children = generate_random_children_from_template(template_file, num_children=num_children)
        
        current_generation = evaluate_children(children, evaluation_runs, epoch_num=0)
        last_epoch_num = 0
        
        # Check if any child has positive score
        best_score = current_generation[0][1]
        if best_score <= 0:
            print("\n⚠ No children have positive scores. Ending simulation.")
        else:
            # Update best overall (by avg_potions_used, lowest is best)
            best_child_id, best_score, best_avg_potions, best_rows = current_generation[0]
            if best_avg_potions < best_overall_avg_potions:
                best_overall = best_rows
                best_overall_avg_potions = best_avg_potions
        
        # Run subsequent epochs until Ctrl-C
        epoch = 1
        while True:
            # Check if previous generation had any positive scores
            if current_generation and current_generation[0][1] <= 0:
                print("\n⚠ No children have positive scores. Ending simulation.")
                break
            
            # Generate new children from previous generation using selection, crossover, and mutation
            children = generate_children_from_population(
                current_generation, 
                num_children=num_children,
                max_mutations=max_mutations
            )
            
            # Evaluate new generation
            current_generation = evaluate_children(children, evaluation_runs, epoch_num=epoch)
            last_epoch_num = epoch
            
            # Check if any child has positive score
            best_score = current_generation[0][1]
            if best_score <= 0:
                print("\n⚠ No children have positive scores. Ending simulation.")
                break
            
            # Update best overall (by avg_potions_used, lowest is best)
            best_child_id, best_score, best_avg_potions, best_rows = current_generation[0]
            if best_avg_potions < best_overall_avg_potions:
                best_overall = best_rows
                best_overall_avg_potions = best_avg_potions
            epoch += 1
            
    except KeyboardInterrupt:
        # Suppress stack trace and worker process messages on Ctrl-C
        # stderr should already be suppressed by signal handler, but ensure it's suppressed
        try:
            import os
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stderr.fileno())
        except:
            sys.stderr = io.StringIO()
        
        try:
            print("\n\n" + "=" * 70)
            print("INTERRUPTED BY USER (Ctrl-C)")
            print("=" * 70)
            print(f"Completed {epoch} epochs")
            if current_generation:
                best_child_id, best_score, best_avg_potions, best_rows = current_generation[0]
                if best_avg_potions < best_overall_avg_potions:
                    best_overall = best_rows
                    best_overall_avg_potions = best_avg_potions
        finally:
            # Restore stderr
            try:
                sys.stderr = _original_stderr
            except:
                pass
    
    # Save overall best strategy
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    if best_overall is None and current_generation:
        # Use best from current generation if no overall best was set
        best_overall = current_generation[0][3]
        best_overall_avg_potions = current_generation[0][2]
    
    if best_overall and best_overall_avg_potions != float('inf'):
        # Use the best strategy's avg_potions_used value (not an epoch average)
        filename = f"level_{level}_avg_{int(best_overall_avg_potions)}_runs_{evaluation_runs}.csv"
        output_path = os.path.join(base_output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        
        save_strategy_to_csv(best_overall, output_path)
        print(f"Best strategy saved to: {output_path}")
        print(f"Best overall avg_potions_used: {best_overall_avg_potions:.2f}")
    else:
        print("No valid strategies found.")
    print("=" * 70)
    
    return best_overall


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run genetic algorithm optimization on strategy file.")
    parser.add_argument(
        "--parent_strategy",
        type=str,
        nargs='?',
        default=None,
        help="Path to parent strategy CSV file (optional). If not provided, generates random children from template."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="strategy_template.csv",
        help="Path to template CSV file (default: strategy_template.csv). Used when parent_strategy is not provided."
    )
    parser.add_argument(
        "--children",
        type=int,
        default=100,
        help="Number of children per generation (default: 100)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Number of simulation runs per child (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="genetic_algorithm_runs",
        help="Base output directory for results (default: genetic_algorithm_runs)"
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=10,
        help="Maximum number of mutations per child (default: 10). Each child gets 0 to max_mutations mutations randomly."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=99,
        help="Herblore level (default: 99). Only potions with level <= this will be drawn."
    )
    
    args = parser.parse_args()
    
    run_multi_epoch(
        parent_strategy_file=args.parent_strategy,
        template_file=args.template,
        num_children=args.children,
        evaluation_runs=args.runs,
        base_output_dir=args.output_dir,
        max_mutations=args.max_mutations,
        level=args.level
    )


