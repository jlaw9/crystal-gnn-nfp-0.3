import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from pymatgen.core import Structure
from tqdm.auto import tqdm

tqdm.pandas()

structure_dir = Path("/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures")
inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
out_dir = Path("outputs/20220205_cos_dist_bins")
os.makedirs(out_dir, exist_ok=True)

preprocessor = PymatgenPreprocessor()


def preprocess_structure(structure):
    scaled_struct = structure.copy()
    inputs = preprocessor(scaled_struct, train=True)

    # scale structures to a minimum of 1A interatomic distance
    min_distance = inputs["distance"].min()
    if np.isclose(min_distance, 0):
        raise RuntimeError(f"Error with {structure}")

    inputs["distance"] /= inputs["distance"].min()
    return inputs


def get_structures(filename):
    with gzip.open(Path(structure_dir, filename), "r") as f:
        for key, structure_dict in tqdm(json.loads(f.read().decode()).items()):
            structure = Structure.from_dict(structure_dict)
            inputs = preprocess_structure(structure)
            yield {"id": key, "inputs": inputs}


battery_structures_relaxed = pd.DataFrame(
    get_structures("battery_relaxed_structures.json.gz")
)

calc_energy = pd.read_csv(
    Path(inputs_dir, "20211223_deduped_energies_matminer_0.01.csv")
)
cos_dist = pd.read_csv(
    Path(structure_dir, "cos_distance/battery_unrel_rel_cos_dist.csv")
)

# limit the cos_dist measurements to the deduplicated set of structures
cos_dist = cos_dist[cos_dist.id.isin(calc_energy.id)]

#if experiment.get('feature_bins'):
# update the main feature to be categorical rather than continuous
print(f"splitting the cosine_dist into 2 classes using the splits (0.01, 0.05, 0.1)")
for x in ['0_01', '0_05', '0_1']:
    cos_dist[f'dist_class_{x}'] = 1 - np.digitize(cos_dist['cos_dist'], [float(x.replace('_','.'))])
    num_pos = len(cos_dist[cos_dist[f'dist_class_{x}'] == 1])
    num_neg = len(cos_dist[cos_dist[f'dist_class_{x}'] == 0])
    print(f"cutoff {x}: # neg: {num_neg}, "
          f"# pos: {num_pos}, frac: {num_pos / (len(cos_dist)):0.3f}") 

#icsd_structures = pd.DataFrame(get_structures("icsd_structures.json.gz"))
#icsd_energy = pd.read_csv(Path(structure_dir, "icsd_energies.csv"))

#icsd_energy["comp_type"] = icsd_energy.composition.apply(
#    lambda comp: int(
#        "".join(str(x) for x in sorted((int(x) for x in re.findall("(\d+)", comp))))
#    )
#)
#
#data = calc_energy.append(icsd_energy)
#structures = battery_structures_relaxed.append(icsd_structures)
#data = data.merge(structures, on="id", how="inner")

data = cos_dist.merge(battery_structures_relaxed, on="id", how="inner")
data.to_pickle(Path(out_dir, "all_data.p"))
preprocessor.to_json(Path(out_dir, "preprocessor.json"))
