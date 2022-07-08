import math
import os
import shutil
from collections import Counter
from pathlib import Path

import nfp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nfp.layers import RBFExpansion
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm.auto import tqdm

from preprocess import preprocessor

tqdm.pandas()

inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
data_file = Path(inputs_dir, "20220603_outliers_removed.p")
data = pd.read_pickle(data_file)
print(f"read {len(data)} structures from {data_file}")
print(data.head(2))
print(data.type.value_counts())
# drop the volume-only relaxation data
vol_data = data[data.type == "vol"].set_index('id')
data = data[data.type != "vol"].set_index('id')

# To repeat our original approach that did not work well with our current dataset:
# replace the 'relax' type inputs with the vol (unrelaxed) inputs
# to train the model to predict the relaxed energy from the unrelaxed structure
print("replacing relaxed structures with unrelaxed structures")
# make sure the indexes line up by removing ids that are unique to "relax"
ids_to_keep = set(data.index) - (
        set(data[data.type == 'relax'].index) - set(vol_data.index))
data = data[data.index.isin(ids_to_keep)]
data.loc[data.type == "relax", "inputs"] = vol_data.inputs
print(data.type.value_counts())

max_atomic_num = 84

composition_set = data.composition.isin(
    pd.Series(data.composition.unique()).sample(100, random_state=1)
)
test_composition = data[composition_set]
train_composition = data[~composition_set]

train, valid = train_test_split(
    train_composition,
    test_size=.1,
    random_state=1,
    stratify=train_composition["type"],
)
valid, test = train_test_split(
    valid,
    test_size=0.5,
    random_state=2,
    stratify=valid["type"],
)


def calculate_output_bias(train):
    """ We can get a reasonable guess for the output bias by just assuming the crystal's
     energy is a linear sum over it's element types """
    # This just converts to a count of each element by crystal
    site_counts = (
        train.inputs.progress_apply(lambda x: pd.Series(Counter(x["site"])))
        .reindex(columns=np.arange(max_atomic_num))
        .fillna(0)
    )
    # Linear regression assumes a sum, while we average over sites in the neural network
    # Here, we make the regression target the total energy, not the site-averaged energy
    num_sites = site_counts.sum(1)
    total_energies = train["energyperatom"] * num_sites

    # Do the least-squares regression, and stack on zeros for the mask and unknown
    # tokens
    output_bias = np.linalg.lstsq(site_counts, total_energies, rcond=None)[0]
    return output_bias


def build_dataset(split, batch_size):
    return (
        tf.data.Dataset.from_generator(
            lambda: ((row.inputs, row.energyperatom) for _, row in split.iterrows()),
            output_signature=(
                preprocessor.output_signature,
                tf.TensorSpec((), dtype=tf.float32),
            ),
        )
        .cache()
        .shuffle(buffer_size=len(split))
        .padded_batch(
            batch_size=batch_size,
            padding_values=(
                preprocessor.padding_values,
                tf.constant(np.nan, dtype=tf.float32),
            ),
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


# Calculate an initial guess for the output bias
output_bias = calculate_output_bias(train)

batch_size = 64
train_dataset = build_dataset(train, batch_size=batch_size)
valid_dataset = build_dataset(valid, batch_size=batch_size)


# Keras model
site_class = layers.Input(shape=[None], dtype=tf.int64, name="site")
distances = layers.Input(shape=[None], dtype=tf.float32, name="distance")
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
input_tensors = [site_class, distances, connectivity]

embed_dimension = 256
num_messages = 6

atom_state = layers.Embedding(
    max_atomic_num, embed_dimension, name="site_embedding", mask_zero=True
)(site_class)

atom_mean = layers.Embedding(
    max_atomic_num,
    1,
    name="site_mean",
    mask_zero=True,
    embeddings_initializer=tf.keras.initializers.Constant(output_bias),
)(site_class)

rbf_distance = RBFExpansion(
    dimension=128, init_max_distance=7, init_gap=30, trainable=True
)(distances)

bond_state = layers.Dense(embed_dimension)(rbf_distance)

for _ in range(num_messages):
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([bond_state, new_bond_state])
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([atom_state, new_atom_state])

# Reduce the atom state vector to a single energy prediction
atom_state = layers.Dense(
    1,
    name="site_energy_offset",
    kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=1e-6, seed=None
    ),
)(atom_state)

# Add this 'offset' prediction to the learned mean energy for the given element type
atom_state = layers.Add(name="add_energy_offset")([atom_state, atom_mean])

# Calculate a final mean energy per atom
out = tf.keras.layers.GlobalAveragePooling1D()(atom_state)

model = tf.keras.Model(input_tensors, [out])


# Train the model
STEPS_PER_EPOCH = math.ceil(len(train) / batch_size)  # number of training examples
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-4, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

wd_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-5, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

optimizer = tfa.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=wd_schedule, global_clipnorm=1.0
)

model.compile(loss="mae", optimizer=optimizer)

model_name = "./"
#model_name = "20220607_icsd_and_battery"
#
#if not os.path.exists(model_name):
#    os.makedirs(model_name)
#
## Make a backup of the job submission script
#shutil.copy(__file__, model_name)

filepath = model_name + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, save_best_only=True, verbose=0
)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + "/log.csv")

if __name__ == "__main__":
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=100,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
    )

    data["set"] = "train"
    data.loc[data.index.isin(valid.index), "set"] = "valid"
    data.loc[data.index.isin(test.index), "set"] = "test"
    data.loc[data.index.isin(test_composition.index), "set"] = "test_composition"

    dataset = (
        tf.data.Dataset.from_generator(
            lambda: (row.inputs for _, row in data.iterrows()),
            output_signature=preprocessor.output_signature,
        )
        .padded_batch(batch_size=128, padding_values=preprocessor.padding_values)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    predictions = model.predict(dataset, verbose=1)

    data["energy_predicted"] = predictions
    data.drop("inputs", axis=1).to_csv(
        model_name + "/predicted_energies.csv.gz", compression="gzip"
    )
