import sys
import os
import time
import random
from typing import List, Dict, Set

# Add src to path
sys.path.append(os.path.abspath('src'))

from predictor import ConflictPredictor

def generate_mock_data(num_branches: int, num_files: int, files_per_branch: int):
    branches = [f"branch_{i}" for i in range(num_branches)]
    files = [f"file_{i}.py" for i in range(num_files)]

    branch_data = {}
    for branch in branches:
        selected_files = random.sample(files, files_per_branch)
        lines = {f: {f"line_{random.randint(0, 1000)}" for _ in range(10)} for f in selected_files}
        branch_data[branch] = {
            'files': set(selected_files),
            'lines': lines,
            'sorted_lines': {},
            'commit': f"hash_{branch}"
        }
    return branches, branch_data

def benchmark():
    num_branches = 200
    num_files = 1000
    files_per_branch = 20

    branches, branch_data = generate_mock_data(num_branches, num_files, files_per_branch)

    predictor = ConflictPredictor()
    main_branch = "main"

    # We want to benchmark the pairwise comparison logic
    start_time = time.perf_counter()

    # Simulate the loop in predict_conflicts
    file_to_branches = {}
    for branch, data in branch_data.items():
        for file in data['files']:
            if file not in file_to_branches:
                file_to_branches[file] = []
            file_to_branches[file].append(branch)

    seen_pairs = set()
    count = 0
    for common_file, sharers in file_to_branches.items():
        for i, branch_a in enumerate(sharers):
            for branch_b in sharers[i+1:]:
                pair = (branch_a, branch_b) if branch_a < branch_b else (branch_b, branch_a)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                data_a = branch_data[branch_a]
                data_b = branch_data[branch_b]
                overlap = data_a['files'] & data_b['files']

                line_conflicts = False
                for common_f in overlap:
                    if not data_a['lines'][common_f].isdisjoint(data_b['lines'][common_f]):
                        line_conflicts = True
                        break

                if not line_conflicts:
                    # Semantic check (which we suspect is slow or redundant)
                    for common_f in overlap:
                        # Simulated get_lazy_sorted_lines
                        if common_f not in data_a['sorted_lines']:
                            data_a['sorted_lines'][common_f] = tuple(sorted(data_a['lines'][common_f]))
                        if common_f not in data_b['sorted_lines']:
                            data_b['sorted_lines'][common_f] = tuple(sorted(data_b['lines'][common_f]))

                        content_a = data_a['sorted_lines'][common_f]
                        content_b = data_b['sorted_lines'][common_f]

                        # Call actual semantic_similarity
                        predictor._semantic_similarity(content_a, content_b, data_a['lines'][common_f], data_b['lines'][common_f])
                count += 1

    end_time = time.perf_counter()
    print(f"Processed {count} pairs in {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
