import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
#import nfp
from nfp import EdgeUpdate, NodeUpdate
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tqdm.auto import tqdm

tqdm.pandas()

# Some of these metrics work only on values in a range 0-1, 
# even if from_logits=True is set
# https://github.com/tensorflow/tensorflow/issues/42182#issuecomment-818777681
class FromLogitsMixin:
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


class AUC(FromLogitsMixin, tf.metrics.AUC):
    ...

class BinaryAccuracy(FromLogitsMixin, tf.metrics.BinaryAccuracy):
    ...

class TruePositives(FromLogitsMixin, tf.metrics.TruePositives):
    ...

class FalsePositives(FromLogitsMixin, tf.metrics.FalsePositives):
    ...

class TrueNegatives(FromLogitsMixin, tf.metrics.TrueNegatives):
    ...

class FalseNegatives(FromLogitsMixin, tf.metrics.FalseNegatives):
    ...

class Precision(FromLogitsMixin, tf.metrics.Precision):
    ...

class Recall(FromLogitsMixin, tf.metrics.Recall):
    ...

class F1Score(FromLogitsMixin, tfa.metrics.F1Score):
    ...


METRICS = [
      TruePositives(name='tp', from_logits=True),
      FalsePositives(name='fp', from_logits=True),
      TrueNegatives(name='tn', from_logits=True),
      FalseNegatives(name='fn', from_logits=True), 
      BinaryAccuracy(name='accuracy', from_logits=True),
      Precision(name='precision', from_logits=True),
      Recall(name='recall', from_logits=True),
      AUC(name='auroc', from_logits=True),
      AUC(name='auprc', curve='PR', from_logits=True), # precision-recall curve
]

#inputs_dir = Path("/projects/rlmolecule/pstjohn/crystal_inputs/")
inputs_dir = Path("outputs/20220205_cos_dist_bins")

bin_cutoff = "0_1"
train_col = f"dist_class_{bin_cutoff}"
run_id = f"model_b64_{train_col}"

model_dir = Path(inputs_dir, run_id)

data = pd.read_pickle(Path(inputs_dir, "all_data.p"))
preprocessor = PymatgenPreprocessor()
preprocessor.from_json(Path(inputs_dir, "preprocessor.json"))

train, valid = train_test_split(data, test_size=.1, random_state=1)
valid, test = train_test_split(valid, test_size=0.5)


def calculate_output_bias(train):
    """ Try setting the initial bias
    """
    num_pos = len(train[train[train_col] == 1])
    num_neg = len(train[train[train_col] == 0])
    initial_bias = np.log([num_pos / num_neg])
    return initial_bias


def build_dataset(split, batch_size):
    return (
        tf.data.Dataset.from_generator(
            lambda: ((row.inputs, row[train_col]) for _, row in split.iterrows()),
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
    preprocessor.site_classes, embed_dimension, name="site_embedding", mask_zero=True
)(site_class)

atom_mean = layers.Embedding(
    preprocessor.site_classes,
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
    new_bond_state = EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([bond_state, new_bond_state])
    new_atom_state = NodeUpdate()([atom_state, bond_state, connectivity])
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

#loss = 'mae'
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#weights = dict(zip(np.unique(train[train_col]), 
#                   compute_class_weight('balanced', 
#                                        classes=np.unique(train[train_col]), 
#                                        y=train[train_col])))
#print('Setting class weights: ', weights)

model.compile(loss=loss, optimizer=optimizer, metrics=METRICS)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Make a backup of the job submission script
dest_file = Path(model_dir, os.path.basename(__file__))
if not dest_file.is_file() or Path(__file__) != dest_file:
    shutil.copy(__file__, dest_file)

filepath = Path(model_dir, "best_model.hdf5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, save_best_only=True, verbose=0
)
csv_logger = tf.keras.callbacks.CSVLogger(Path(model_dir, "log.csv"))

if __name__ == "__main__":
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=100,
        callbacks=[checkpoint, csv_logger],
        #class_weight=weights,
        verbose=1,
    )
