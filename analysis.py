import pickle
from keras import layers, models, Input
from keras.src.models import Model
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import collections

from sklearn.cluster import KMeans

with open("my_eeg_epoched_data_30_inf.pkl", "rb") as f:
    epoched_data = pickle.load(f)

with open("my_raw_eeg_epoched_data_30_inf.pkl", "rb") as f:
    raw_epoched_data = pickle.load(f)

x1_test = epoched_data["x1_test"]
x2_test = epoched_data["x2_test"]
y_test = epoched_data["y_test"]

index_list_test = epoched_data["index_list_test"]


x1_test_raw = raw_epoched_data["x1_test"]
x2_test_raw = raw_epoched_data["x2_test"]
y_test_raw = raw_epoched_data["y_test"]

#index_list_test_raw = raw_epoched_data["index_list_test"]

from keras.src.saving import load_model

# ✅ Load the model, passing the Lambda function explicitly
import keras
#keras.config.enable_unsafe_deserialization()

import pickle as pk

@keras.saving.register_keras_serializable()
class L1DistanceLayer(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.math.abs(x1 - x2)


siamese_model = load_model(
    "saved_models/siamese_model_ker_61_inter_60epochs_30_inf_shuffled.keras",
    custom_objects={"L1DistanceLayer": L1DistanceLayer}# No unsafe mode needed!
)

test_loss, test_acc = siamese_model.evaluate([np.array(x1_test), np.array(x2_test)], np.array(y_test))

print(f"Test Accuracy: {test_acc:.4f}")

y_pred_probs_shuffled = siamese_model.predict([x1_test, x2_test])
y_pred_shuffled = (y_pred_probs_shuffled > 0.5).astype(int).flatten()
print(len(y_pred_shuffled))
pair_labels = (y_pred_shuffled == y_test)  # True where correct, False where wrong
correct_indices = np.where(pair_labels)[0]  # indices of correct predictions


subject_results = collections.defaultdict(lambda: {"y_true": [], "y_pred": []})
y_pred_probs = siamese_model.predict([x1_test, x2_test])
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(len(x1_test))
for i, (idx1, idx2, subj_name) in enumerate(index_list_test):
    subject_results[subj_name]["y_true"].append(y_test[i])
    subject_results[subj_name]["y_pred"].append(y_pred[i])

# Compute accuracy per subject
from sklearn.metrics import accuracy_score
subject_accuracies = {}
for subj, results in subject_results.items():
    acc = accuracy_score(results["y_true"], results["y_pred"])
    subject_accuracies[subj] = acc
    print(f"{subj}: {acc:.3f}")

import collections
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# STEP 1: Create the mapping from numeric ID → S1, S2, ...
original_ids = sorted(subject_accuracies.keys(), key=lambda x: int(x))  # Ensure numeric sort
id_mapping = {orig_id: f"S{i+1}" for i, orig_id in enumerate(original_ids)}

# STEP 2: Apply mapping and build new accuracy dict
renamed_accuracies = {id_mapping[subj]: acc for subj, acc in subject_accuracies.items()}

# STEP 3: Sort by accuracy descending for the bar plot
sorted_subjects = sorted(renamed_accuracies.items(), key=lambda x: int(x[0][1:]))  # removes 'S' and sorts by number
subjects = [s[0] for s in sorted_subjects]
accuracies = [s[1] for s in sorted_subjects]

# STEP 4: Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(subjects, accuracies, color='skyblue', edgecolor='black')
plt.axhline(y=0.5, color='red', linestyle='--', label='Chance Level')
plt.xticks(rotation=0, ha='center', fontsize=18)
plt.yticks(fontsize=16)
plt.ylabel('Accuracy', fontsize=24)
plt.title('Per-Subject Accuracy on Test Set (30+ Hz)', fontsize=24)
plt.ylim(0, 1.0)
plt.legend(fontsize=17)
plt.tight_layout()

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=18)
plt.savefig("subj_accuracy_30_inf.pdf", bbox_inches='tight')
plt.show()

#Embedding part of analysis
embedding_model = siamese_model.get_layer("functional")

embeddings_1 = embedding_model.predict(x1_test, batch_size=32)
embeddings_2 = embedding_model.predict(x2_test, batch_size=32)

l1_distances = np.abs(embeddings_1 - embeddings_2)

time_distances = np.array([abs(i1 - i2) for (i1, i2, _) in index_list_test]) * 2.5
time_distances = np.array(time_distances)

embedding_distances = np.sum(np.abs(embeddings_1 - embeddings_2), axis=1)

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

plt.scatter(time_distances, embedding_distances, alpha=0.3)
plt.xlabel("Temporal Distance (seconds)", fontsize=14)
plt.ylabel("Embedding Distance", fontsize=14)
plt.title("Embedding Distance vs. Temporal Distance (30+Hz)", fontsize=14)

# Fit a linear regression (1st degree polynomial)
coefficients = np.polyfit(time_distances, embedding_distances, deg=1)
poly_eq = np.poly1d(coefficients)

# Generate x values for the regression line (cover the range of your data)
x_line = np.linspace(time_distances.min(), time_distances.max(), 100)
y_line = poly_eq(x_line)

# Plot the regression line
plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')
plt.grid(True)
plt.legend()

plt.savefig("temporal_embedding.pdf", bbox_inches='tight')
plt.show()

rho, p_val = spearmanr(time_distances, embedding_distances)
print(f"Spearman correlation: rho = {rho:.3f}, p = {p_val:.3e}")
#for this you need umap-learn 
#the plot might be slightly different from the paper depending on version of packages and randomness, but same structure remains
import umap.umap_ as umap
#import matplotlib.pyplot as plt

# Combine the embeddings of both windows into one representation (e.g. absolute difference)
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

embedding_diff = np.abs(embeddings_1 - embeddings_2)  # (N, 128)

# Run UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding_umap = umap_reducer.fit_transform(embedding_diff)  # shape (N, 2)

# Plot UMAP
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1],
                      c=y_test, cmap='coolwarm', alpha=0.6, s=20)
# Get the colormap used in scatter plot
cmap = cm.get_cmap("coolwarm")

# Normalize label values between 0 and 1 for colormap lookup
norm = mcolors.Normalize(vmin=0, vmax=1)
labels = [0, 1]
# Get the exact colors from the colormap for each label
colors = [cmap(norm(label)) for label in labels]

# Create patches with exact colors from the colormap
patches = [mpatches.Patch(color=color, label=f"{'Far' if label==0 else 'Close'}")
           for label, color in zip(labels, colors)]

# Then add legend with these patches
plt.legend(handles=patches, fontsize=24)
plt.title("UMAP of Pairwise Embedding L1 Differences (30+ Hz)", fontsize=24)
plt.xlabel("UMAP 1", fontsize=24)
plt.ylabel("UMAP 2", fontsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig("umap_30.pdf", bbox_inches='tight')
plt.show()


def generate_temporal_gradcam(model, inputs, conv_layer_name="conv_temporal"):
    """Generate Grad-CAM heatmap for a single EEG input sample."""

    # Get the internal CNN model from the Siamese model
    cnn_model = model.get_layer("functional")

    # Sanity check: does the requested layer exist?
    if conv_layer_name not in [layer.name for layer in cnn_model.layers]:
        raise ValueError(f"No such layer: {conv_layer_name}")

    # Build a gradient model to extract features + gradients
    grad_model = Model(
        inputs=cnn_model.input,
        outputs=[cnn_model.get_layer(conv_layer_name).output, cnn_model.output]
    )

    # Ensure correct shape: (batch_size, channels, timepoints, 1)
    if len(inputs.shape) == 3:
        inputs = np.expand_dims(inputs, axis=-1)
    if len(inputs.shape) == 2:
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.expand_dims(inputs, axis=-1)

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_output, predictions = grad_model(inputs)
        loss = tf.reduce_sum(predictions)

    # Compute gradients
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Apply Grad-CAM formula
    conv_output = conv_output[0]
    heatmap = conv_output * pooled_grads
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.percentile(heatmap, 99) + 1e-8)

    return heatmap # Shape: (channels, timepoints, filters)

from tqdm import tqdm  # optional, for progress bar
import scipy
import matplotlib.pyplot as plt


def average_gradcam_psd_over_samples(model, x1_test, x2_test, conv_layer_name="conv_temporal", fs=250):
    print("starting gradcam psd calculations")
    """
    Compute average PSD of Grad-CAM weights across all samples and channels.

    Returns:
        f: frequency bins
        avg_psd: averaged PSD values
    """
    psd_accumulator = []

    num_samples = len(x1_test)

    for i in tqdm(range(num_samples)):
        # Compute Grad-CAM for x1 and x2
        gradcam_1 = generate_temporal_gradcam(model, x1_test[i], conv_layer_name=conv_layer_name)
        gradcam_2 = generate_temporal_gradcam(model, x2_test[i], conv_layer_name=conv_layer_name)

        # Average over filters to get (channels, timepoints)
        gradcam_1 = np.mean(gradcam_1, axis=-1)
        gradcam_2 = np.mean(gradcam_2, axis=-1)

        for gradcam_weights in [gradcam_1, gradcam_2]:
            for ch in range(gradcam_weights.shape[0]):
                #print(gradcam_weights[0])
                f, Pxx = scipy.signal.welch(gradcam_weights[ch], fs=fs, nperseg=128)
                psd_accumulator.append(Pxx)

    avg_psd = np.mean(psd_accumulator, axis=0)

    # Plotting
    freqs = np.array(f)
    power = np.array(avg_psd)

    # Mask for frequencies between 1 and 15 Hz
    mask = (freqs >= 0) & (freqs <= 80)
    filtered_freqs = freqs[mask]
    filtered_power = power[mask]
    plt.figure(figsize=(8, 4))
    plt.plot(filtered_freqs, filtered_power, linewidth=1)
    plt.title("Average PSD of Grad-CAM weights (30+ Hz)", fontsize=20)
    plt.xlabel("Frequency (Hz)", fontsize=20)
    plt.ylabel("Power (dB)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Gradcam_30.pdf", bbox_inches='tight')
    plt.show()

    return f, avg_psd

freqs_grad_psd, avg_gradcam_psd = average_gradcam_psd_over_samples(
    model=siamese_model,
    x1_test=x1_test,
    x2_test=x2_test,
    conv_layer_name="conv_temporal",
    fs=250
)

import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from typing import Tuple


def bandpass_filter(data, sfreq, low, high, order=4):
    nyq = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def extract_salient_eeg_windows_both_inputs(
        model,
        x1_eeg, x2_eeg,
        x1_normed, x2_normed,
        index_list,
        pred_labels=None, true_conditions=None,
        apply_mask="all",  # one of "all", "correct", "far", "close"
        conv_layer_name="conv_temporal",
        fs=250, low=1.5, high=4,
        window_size=0.2,
        top_percentile=99.99
):
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    '''
        all_saliencies = []

        print("Collecting saliency values to compute threshold...")
        for i in tqdm(range(n_samples)):
            for x_normed in [x1_normed[i], x2_normed[i]]:
                gradcam = generate_temporal_gradcam(model, x_normed, conv_layer_name=conv_layer_name)
                gradcam = np.mean(gradcam, axis=-1)  # [channels, timepoints]
                gradcam_filtered = bandpass_filter(gradcam, fs, low, high)
                all_saliencies.append(np.abs(gradcam_filtered))

        all_saliencies = np.concatenate(all_saliencies, axis=None)
        threshold = np.percentile(all_saliencies, top_percentile)
        print(f"Global saliency threshold (top {100 - top_percentile}%): {threshold:.4e}")
        '''


    x1_eeg, x2_eeg = np.asarray(x1_eeg), np.asarray(x2_eeg)
    x1_normed, x2_normed = np.asarray(x1_normed), np.asarray(x2_normed)
    n_samples, n_channels, n_times = x1_eeg.shape
    half_window = int((window_size * fs) // 2)

    raw_windows, normed_windows = [], []
    window_channels, window_samples_idx = [], []
    salient_sources = []  # (subject_id, channel, time_idx, absolute_time)
    all_salient_times = []  # for plotting histogram of absolute time points

    # Masks
    if apply_mask != "all":
        assert pred_labels is not None and true_conditions is not None
        correct_mask = pred_labels == 1
        if apply_mask == "correct":
            sample_mask = correct_mask
        elif apply_mask == "far":
            sample_mask = correct_mask & (true_conditions == 0)
        elif apply_mask == "close":
            sample_mask = correct_mask & (true_conditions == 1)
        else:
            raise ValueError(f"Unknown apply_mask: {apply_mask}")
    else:
        sample_mask = np.ones(n_samples, dtype=bool)

    # Precomputed saliency threshold (or compute it)
    threshold = 2.7080e-01  # adjust if needed

    print(f"Extracting salient EEG windows (mask: '{apply_mask}')...")
    for i in tqdm(range(n_samples)):
        if not sample_mask[i]:
            continue

        # Identify source window indices and subject ID
        idx1, idx2, subject_id = index_list[i]

        for x_raw, x_normed, window_index in zip(
                [x1_eeg[i], x2_eeg[i]],
                [x1_normed[i], x2_normed[i]],
                [idx1, idx2]
        ):
            gradcam = generate_temporal_gradcam(model, x_normed, conv_layer_name=conv_layer_name)
            gradcam = np.mean(gradcam, axis=-1)
            gradcam_filtered = bandpass_filter(gradcam, fs, low, high)

            for ch in range(n_channels):
                channel_saliency = np.abs(gradcam_filtered[ch])
                salient_indices = np.where(channel_saliency >= threshold)[0]

                for idx in salient_indices:
                    #print(window_index)
                    #print(salient_indices)
                    #print(ch)
                    absolute_time = window_index * 2.5 + (idx / fs)
                    all_salient_times.append(absolute_time)

                    start = idx - half_window
                    end = idx + half_window
                    if start >= 0 and end <= n_times:
                        raw_windows.append(x_raw[ch, start:end])
                        normed_windows.append(x_normed[ch, start:end])
                        window_channels.append(ch)
                        window_samples_idx.append(i)
                        salient_sources.append((subject_id, window_index, ch, idx, absolute_time))
                    else:
                        # Still store metadata even if no window extracted
                        salient_sources.append((subject_id, window_index, ch, idx, absolute_time))

    # Histogram of absolute salient time points
    plt.figure(figsize=(8, 4))
    plt.hist(all_salient_times, bins=100, color='tab:blue', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Salient point count")
    plt.title("Distribution of absolute GradCAM salient timepoints")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (
        np.array(raw_windows),
        np.array(normed_windows),
        np.array(window_channels),
        np.array(window_samples_idx),
        salient_sources
    )


raw_wins_correct, normed_wins_correct, win_channels, win_samples_idx, salient_sources = extract_salient_eeg_windows_both_inputs(
    siamese_model,
    x1_test_raw, x2_test_raw,
    x1_test, x2_test,
    apply_mask = "correct",
    pred_labels = pair_labels,
    true_conditions = y_test,
    conv_layer_name="conv_temporal",
    index_list = index_list_test,
    fs=250,
    low=1.5, high=4,
    window_size=0.4,
    top_percentile=99.99
)

with open("raw_windows_top0.01_correct_shuf_0_4.pkl", "wb") as f:
    pk.dump(raw_wins_correct, f)
    
with open("salient_sources_top0.01_correct_shuf.pkl", "wb") as f:
    pk.dump(salient_sources, f)

def remove_duplicate_windows_2(windows):
    unique_windows = []
    unique_indices = []
    seen = set()

    for i, w in enumerate(windows):
        # Use a hashable representation (e.g., bytes) to detect duplicates efficiently
        w_bytes = w.tobytes()
        if w_bytes not in seen:
            seen.add(w_bytes)
            unique_windows.append(w)
            unique_indices.append(i)
    return np.array(unique_windows), np.array(unique_indices)

raw_wins_correct_dedup, unique_idx = remove_duplicate_windows_2(raw_wins_correct)
print(f"Correct windows before dedup: {len(raw_wins_correct)}, after dedup: {len(raw_wins_correct_dedup)}")

# If you have labels for correct windows, apply indexing to them as well:
# For example, if you had labels_correct:
# labels_correct_dedup = labels_correct[unique_idx]

from scipy.signal import welch

#flat_data, freqs = compute_psd_features(raw_wins_correct_dedup, fs=250)
# Flatten windows for clustering
flat_data = raw_wins_correct_dedup.reshape(raw_wins_correct_dedup.shape[0], -1)
# ---------- KMeans Clustering ----------
n_clusters = 4  # You can tune this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(flat_data)

# ---------- Optional: Visualize with UMAP ----------
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
embedding = reducer.fit_transform(flat_data)
# Make sure cluster_labels range from 0 to n_clusters-1
n_clusters = 4

# Generate colors for each cluster label from tab10 colormap
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(n_clusters)]
labels_group = np.zeros(len(raw_wins_correct_dedup), dtype=int)
plt.figure(figsize=(8, 6))
for cluster in range(n_clusters):
    idx = cluster_labels == cluster
    plt.scatter(embedding[idx, 0], embedding[idx, 1], color=colors[cluster], label=f'Cluster {cluster + 1}', s=10)
plt.title("KMeans Clusters Visualized with UMAP (Window = 0.4s)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.stats import sem  # Standard error of the mean
#this will get you the kmeans cluster seen in the paper Figure 8
def plot_avg_waveform_per_cluster_per_group(windows, cluster_labels, group_labels, fs=250):
    unique_clusters = np.unique(cluster_labels)
    #time_axis = np.linspace(-0.1 / 2, 0.1 / 2, windows.shape[1])
    time_axis = np.linspace(-0.2, 0.2, windows.shape[1])
    group_colors = {0: 'royalblue', 1: 'crimson'}

    for cluster in unique_clusters:
        plt.figure(figsize=(10, 4))

        for group in [0, 1]:
            idx = (cluster_labels == cluster) & (group_labels == group)
            if np.sum(idx) == 0:
                continue

            group_waves = windows[idx]
            avg_wave = np.mean(group_waves, axis=0)
            error_sem = sem(group_waves, axis=0)
            #std_wave = np.std(group_waves, axis=0)

            ci95 = 1.96 * error_sem
            color = group_colors[group]
            #plt.fill_between(time_axis, avg_wave - std_wave, avg_wave + std_wave, color=color, alpha=0.15,
                             #label='_nolegend_')
            # Plot mean waveform
            plt.plot(time_axis, avg_wave, label=f"{'Correct' if group == 0 else 'Close'} (n={np.sum(idx)})", color=color)

            # Shaded SEM
            plt.fill_between(time_axis, avg_wave - error_sem, avg_wave + error_sem,
                             color=color, alpha=0.2, linewidth=0)

            # 95% CI as dashed lines (no label in legend)
            plt.plot(time_axis, avg_wave - ci95, linestyle='--', color=color, alpha=0.4, linewidth=1)
            plt.plot(time_axis, avg_wave + ci95, linestyle='--', color=color, alpha=0.4, linewidth=1)
        window_size = windows.shape[1] / fs
        plt.title(f"KMeans Cluster {cluster + 1} - Mean EEG ± SEM & 95% CI (Window = {window_size:.1f}s)",fontsize = 20)
        plt.xlabel("Time (s)", fontsize = 20)
        plt.ylabel("Amplitude (V)",fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize = 18)
        plt.grid(True)
        plt.tight_layout()
        ax = plt.gca()  # get current axes
        ax.yaxis.get_offset_text().set_fontsize(14)
        plt.savefig(f"avgwave_{cluster+1}.pdf", bbox_inches='tight')
        plt.show()

plot_avg_waveform_per_cluster_per_group(raw_wins_correct_dedup, cluster_labels, labels_group)


#now the last part of the analysis; which will 9-12 and 2

# Define the subject range and exclude 032309
subject_ids = [f"0323{str(i).zfill(2)}" for i in range(1, 45) if (i != 9 and i != 35)]

with open("preprocessed_data_EC.pkl", "rb") as f:
    pre_processed_data_EC = pk.load(f)

with open("salient_sources_top0.01_correct_shuf.pkl" , "rb") as f:
    salient_sources = pk.load(f)


import random
random.seed(42)  # Ensure reproducibility
random.shuffle(subject_ids)

n_train = int(0.6 * len(subject_ids))
n_val = int(0.2 * len(subject_ids))

train_subjects = subject_ids[:n_train]
val_subjects = subject_ids[n_train:n_train + n_val]
test_subjects = subject_ids[n_train + n_val:]

print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")


# Create epoch pairs
def create_pairs_3(subject_name, epochs, close_threshold, far_threshold, offset, subject_id):
    pairs_x1, pairs_x2, labels_y, idxs = [], [], [], []
    n_epochs = len(epochs)

    # Close pairs
    close_pairs = set()
    for idx1 in range(n_epochs):
        for idx2 in range(idx1 + 1, min(idx1 + close_threshold, n_epochs)):
            close_pairs.add((idx1, idx2))

    # Far pairs
    far_pairs = set()
    for idx1 in range(n_epochs):
        for idx2 in range(n_epochs):
            if abs(idx1 - idx2) > far_threshold and idx1 != idx2:
                far_pairs.add((idx1, idx2))

    far_pairs = list(far_pairs)
    np.random.seed(42)
    np.random.shuffle(far_pairs)
    far_pairs = far_pairs[:len(close_pairs)]

    # Add to lists
    for idx1, idx2 in close_pairs:
        pairs_x1.append(epochs[idx1])
        pairs_x2.append(epochs[idx2])

        labels_y.append(1)
        idxs.append((idx1 + offset, idx2 + offset, subject_name))

    for idx1, idx2 in far_pairs:
        # works

        pairs_x1.append(epochs[idx1])
        pairs_x2.append(epochs[idx2])
        labels_y.append(0)
        idxs.append((idx1 + offset, idx2 + offset, subject_name))

    return (
        np.array(pairs_x1, dtype=np.float32),
        np.array(pairs_x2, dtype=np.float32),
        np.array(labels_y, dtype=np.float32),
        idxs,
    )

epoch_length = int(2.5 * 250)
# 🛠️ Corrected helper function
def add_missing_channels(raw, target_channels, use_zero_padding=False):
    missing = [ch for ch in target_channels if ch not in raw.ch_names]
    if not missing:
        return raw, [], False  # nothing to do

    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    data = np.zeros((len(missing), n_times))

    # Create info for dummy channels
    import mne
    dummy_info = mne.create_info(missing, sfreq, ch_types='eeg')
    dummy_raw = mne.io.RawArray(data, dummy_info)

    # Don't set any montage here; instead, we align channel names later
    # This avoids 'dig' mismatches
    dummy_raw.set_montage(None)  # ensure no dig present

    # Also remove dig from original raw before merging
    raw.set_montage(None)  # ensure both raw and dummy_raw are clean

    # Now merge safely
    raw.add_channels([dummy_raw])

    # Set montage after merge, only once, for consistent positions
    raw.set_montage("standard_1020", on_missing='ignore')

    if use_zero_padding:
        raw.info['bads'] = []
        return raw, missing, True
    else:
        raw.info['bads'] = missing
        return raw, missing, False


# Initialize datasets
from scipy.signal import welch
from fooof import FOOOF
import numpy as np
import matplotlib.pyplot as plt


def compute_avg_psd(raw_data, sfreq, freq_range=(1, 120)):
    psds = []
    freqs = None
    for ch_data in raw_data:
        f, Pxx = welch(ch_data, fs=sfreq, nperseg=sfreq * 2)
        if freqs is None:
            freqs = f
        psds.append(Pxx)
    psds = np.array(psds)
    avg_psd = np.mean(psds, axis=0)
    return freqs, avg_psd


def process_subject_psd(raw, normalize=False):
    data = raw.get_data()
    if normalize:
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    return compute_avg_psd(data, sfreq=raw.info['sfreq'])


def fooof_1f_removal(freqs, psd, freq_range=(1, 120)):
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[freq_mask]
    psd_fit = psd[freq_mask]

    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1)
    fm.fit(freqs_fit, psd_fit)

    offset, slope = fm.get_params('aperiodic_params')
    print(f"FOOOF aperiodic fit: offset = {offset:.3f}, slope = {slope:.3f}")

    ap_fit_linear = 10 ** fm._ap_fit
    corrected_psd = psd.copy()
    corrected_psd[freq_mask] = psd_fit / ap_fit_linear
    return corrected_psd, fm


def plot_psd(freqs, psd, corrected_psd=None, fooof_fit=None, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 10 * np.log10(psd), label='Original PSD')
    if corrected_psd is not None:
        plt.plot(freqs, 10 * np.log10(corrected_psd), label='1/f Removed PSD', color='orange')
    if fooof_fit is not None:
        plt.plot(freqs[(freqs >= fooof_fit.freq_range[0]) & (freqs <= fooof_fit.freq_range[1])],
                 fooof_fit._ap_fit, '--', label='FOOOF Fit', color='gray')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


from fooof import FOOOF


def apply_fooof_to_power(power, freqs):
    from fooof import FOOOF
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1)
    power = np.array(power)
    power[power <= 0] = 1e-20  # Avoid zeros or negative values
    fm.fit(freqs, power)
    aperiodic_fit = fm._ap_fit
    flattened_spectrum = fm.flattened_spectrum_
    return power, aperiodic_fit, flattened_spectrum, fm


import tensorpac

print(tensorpac.__file__)
print(tensorpac.__version__)
from tensorpac import Pac
import inspect

valid_channels_name = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4',
                       'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
                       'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4',
                       'FT8', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
                       'PO3', 'POz', 'PO4', 'PO8', 'TP7', 'T7']

# === Main analysis ===
fooof_results_per_subject = {}
x1_train, x2_train, y_train, index_list_train = [], [], [], []
x1_val, x2_val, y_val, index_list_val = [], [], [], []
x1_test, x2_test, y_test, index_list_test = [], [], [], []

interpolation_log = {}

# Get reference channel order
ref_raw = pre_processed_data_EC[subject_ids[0]]["raw"].copy()

ref_raw.set_montage("standard_1020")
ref_raw.pick_channels(valid_channels_name)
channel_order = ref_raw.ch_names


from scipy.signal import welch

from fooof import FOOOF

all_psds_pre = []
all_psds_post = []
control_all_subjects_power = []
sfreq = 250  # or set dynamically based on raw.info['sfreq']

with open("salient_sources_top0.01_correct_shuf.pkl", "rb") as f:
    salient_sources = pk.load(f)
from scipy.stats import sem  # Standard error of the mean
from tensorpac import Pac

from scipy.stats import sem, ttest_1samp
import numpy as np
from tqdm import tqdm


def compute_surrogate_pac(pac, sfreq, signal, n_surrogates=100):
    surrogate_vals = []
    n_samples = len(signal)

    for _ in range(n_surrogates):
        shift = np.random.randint(sfreq, n_samples - sfreq)  # shift by at least 1s
        surrogate_signal = np.roll(signal, shift)
        try:
            val = pac.filterfit(sfreq, surrogate_signal[np.newaxis, :])[0]
            surrogate_vals.append(val)
        except Exception:
            continue
    return np.array(surrogate_vals)


def sample_control_windows(data, salient_times, n_controls, sfreq, window_length_sec, buffer_sec=0.5):
    total_samples = data.shape[1]
    half_window = int((window_length_sec / 2) * sfreq)
    buffer_samples = int(buffer_sec * sfreq)
    valid_indices = []
    salient_sample_indices = [int(t * sfreq) for t in salient_times]

    for _ in range(n_controls * 5):  # Oversample then trim
        idx = np.random.randint(half_window, total_samples - half_window)
        too_close = any(abs(idx - s_idx) < (half_window + buffer_samples) for s_idx in salient_sample_indices)
        if not too_close:
            valid_indices.append(idx)
        if len(valid_indices) >= n_controls:
            break

    return valid_indices[:n_controls]


from scipy.stats import ttest_1samp, ttest_rel, sem
from scipy.signal import hilbert, butter, filtfilt


def extract_and_average_morlet_from_data(
        subj_id,
        data,
        salient_sources,
        channel_order,
        window_length_sec=4.0,
        freq_min=2,
        freq_max=100,
        n_freqs=50,
        sfreq=250
):
    """
        Compute time–frequency representations (Morlet wavelets) and PAC measures
        from EEG/MEG data centered on salient timepoints, with comparison to control windows.

        Parameters
        ----------
        subj_id : str or int
            Identifier for the subject (to filter salient_sources).
        data : ndarray
            Continuous EEG data for this subject.
        salient_sources : list of tuples
            List of salient events, each tuple: (subj, window_idx, ch_idx, timepoint_within_window, absolute_time).
        channel_order : list
        window_length_sec : float
            Duration (in seconds) of each analysis window (centered on salient event).
        freq_min, freq_max : float
            Frequency range for Morlet TFR.
        n_freqs : int
            Number of frequency points for Morlet decomposition.
        sfreq : int
            Sampling frequency in Hz.

        Returns
        -------
        mean_power : ndarray
            Average Morlet power across salient windows.
        sem_power : ndarray
            Standard error of the mean of Morlet power.
        times : ndarray
            Time axis corresponding to mean_power (in seconds, centered on event).
        freqs : ndarray
            Frequencies used in Morlet decomposition.
        used_timepoints : int
            Number of valid salient windows used.
        pac_mean : float or None
            Average PAC value across salient windows.
        pac_zscore_mean : float or None
            Average PAC z-score across salient windows.
        subject_significant : (float, bool)
            Tuple of (p-value, significance) for PAC z-scores > 0.
        control_pac_vals : list
            PAC values from control windows.
        control_z_scores : list
            PAC z-scores from control windows.
        pac_zscores : ndarray
            All PAC z-scores for salient windows.
        """
    import mne
    import numpy as np
    from tensorpac import Pac
    from scipy.stats import ttest_1samp, sem

    half_window_samples = int((window_length_sec / 2) * sfreq)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    n_cycles = freqs / 2.0

    subj_salient = [s for s in salient_sources if s[0] == subj_id]
    salient_times = [s[4] for s in subj_salient]
    all_powers = []
    pac_values = []
    pac_zscores = []
    seen = set()
    used_timepoints = 0
    # Initialize PAC estimator: delta/theta phase (1.5–4 Hz) vs gamma amplitude (25–45 Hz)
    pac = Pac(idpac=(1, 1, 4), f_pha=(1.5, 4), f_amp=(25, 45), dcomplex='hilbert')

    # -------- LOOP THROUGH SALIENT EVENTS --------

    for (subj, window_idx, ch_idx, timepoint_within_window, absolute_time) in subj_salient:
        sample_index = int(absolute_time * sfreq)
        key = (ch_idx, sample_index)
        if key in seen:
            continue
        seen.add(key)

        if ch_idx >= len(channel_order) or ch_idx >= data.shape[0]:
            continue

        start = sample_index - half_window_samples
        end = sample_index + half_window_samples
        if start < 0 or end > data.shape[1]:
            continue

        # Morlet
        '''
        #this for envelope
        sig = data[ch_idx, start:end]
        envelope = np.abs(hilbert(sig))
        data_window = envelope[np.newaxis, np.newaxis, :]
        power = mne.time_frequency.tfr_array_morlet(
            data_window, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', zero_mean=True
        )
        '''

        # use this for EEG data
        data_window = data[ch_idx, start:end][np.newaxis, np.newaxis, :]
        power = mne.time_frequency.tfr_array_morlet(
            data_window, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', zero_mean=True
        )

        all_powers.append(power[0, 0])
        used_timepoints += 1

        # PAC computation without infer argument

        try:
            sig = data[ch_idx, start:end]
            pac.filterfit(sfreq, sig[np.newaxis, :])
            # Now infer p-values explicitly:
            pac.infer_pvalues(p=0.05, mcp='maxstat')

            pac_val = pac.pac  # shape: (n_amp, n_pha, n_epochs)
            surro = pac.surrogates  # shape: (n_perm, n_amp, n_pha, n_epochs)

            # Average across amplitude, phase, epochs for simplicity:
            pac_mean_val = pac_val.mean()
            surro_mean = surro.mean()
            surro_std = surro.std()
            z_score = (pac_mean_val - surro_mean) / surro_std if surro_std > 0 else 0

            pac_values.append(pac_mean_val)
            pac_zscores.append(z_score)
        except Exception as e:
            print(f"PAC failed at idx {sample_index}: {e}")

    from tensorpac import Pac
    # -------- CONTROL WINDOWS --------
    pac_control = Pac(idpac=(1, 1, 4), f_pha=(1.5, 4), f_amp=(25, 45), dcomplex='hilbert')
    control_pac_vals = []
    control_z_scores = []

    control_indices = sample_control_windows(data, salient_times, n_controls=used_timepoints, sfreq=sfreq,
                                             window_length_sec=window_length_sec)

    for idx in control_indices:
        start = idx - half_window_samples
        end = idx + half_window_samples
        if start < 0 or end > data.shape[1]:
            continue
        ch_idx = np.random.choice(data.shape[0])  # random channel
        sig = data[ch_idx, start:end]

        try:
            pac_control.filterfit(sfreq, sig[np.newaxis, :])
            pac_val = pac_control.pac.mean()
            surro_mean = pac_control.surrogates.mean()
            surro_std = pac_control.surrogates.std()
            z = (pac_val - surro_mean) / surro_std if surro_std > 0 else 0
            control_pac_vals.append(pac_val)
            control_z_scores.append(z)
        except Exception as e:
            print(f"PAC control failed at {idx}: {e}")

    if len(all_powers) == 0:
        print(f"⚠️ No valid windows found for subject {subj_id}")
        return None, None, None, None, 0, None, None, (None, None), None, None, None

    sem_power = sem(all_powers, axis=0)
    mean_power = np.mean(all_powers, axis=0)

    # Control timepoint extraction
    n_controls = used_timepoints
    # control_indices = sample_control_windows(data, salient_times, n_controls, sfreq, window_length_sec=4.0)

    # Morlet on control windows
    control_powers = []
    for idx in control_indices:
        start = idx - int((4.0 / 2) * sfreq)
        end = idx + int((4.0 / 2) * sfreq)
        if start < 0 or end > data.shape[1]:
            continue
        ch_idx = np.random.choice(len(channel_order))  # random channel
        '''
        #this is to check the envelope
        sig = data[ch_idx, start:end]
        envelope = np.abs(hilbert(sig))
        data_window = envelope[np.newaxis, np.newaxis, :]
        power = mne.time_frequency.tfr_array_morlet(
            data_window, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', zero_mean=True
        )
        '''

        # use this for normal EEG data

        data_window = data[ch_idx, start:end][np.newaxis, np.newaxis, :]
        power = mne.time_frequency.tfr_array_morlet(
            data_window, sfreq=sfreq, freqs=freqs, n_cycles=freqs / 2.0, output='power', zero_mean=True
        )

        control_powers.append(power[0, 0])

    if len(control_powers) > 0:
        control_mean_power = np.mean(control_powers, axis=0)
        control_sem_power = sem(control_powers, axis=0)
        control_all_subjects_power.append(control_mean_power)
    else:
        print(f"No valid control windows for subject {subj_id}")

    # Flatten for scalar-level comparison
    salient_flat = mean_power.flatten()
    control_flat = control_mean_power.flatten()
    df = len(salient_flat) - 1
    if len(salient_flat) == len(control_flat):
        t_stat, p_val = ttest_rel(salient_flat, control_flat)
        is_significant = p_val < 0.05

        salient_vs_control_stats.append({
            "subject": subj_id,
            "t": t_stat,
            "p": p_val,
            "significant": is_significant,
            "n_salient": used_timepoints,
            "n_control": len(control_powers)
        })

        print(
            f"📊 Subject {subj_id} salient vs control power: t({df})={t_stat:.3f}, p={p_val:.4f} → {'✅ significant' if is_significant else '❌ not significant'}")
    else:
        print(f"⚠️ Mismatched power array lengths for subject {subj_id}, skipping significance test.")
    times = np.linspace(-window_length_sec / 2, window_length_sec / 2, mean_power.shape[1])
    # -------- STATISTICS ON PAC --------
    pac_values = np.array(pac_values)
    pac_zscores = np.array(pac_zscores)
    pac_zscore_mean = pac_zscores.mean() if len(pac_zscores) > 0 else None
    pac_mean = pac_values.mean() if len(pac_values) > 0 else None

    if len(pac_zscores) > 0:
        t_stat, p_val = ttest_1samp(pac_zscores, popmean=0.0, alternative='greater')
        is_significant = p_val < 0.05
        subject_significant = (p_val, is_significant)
    else:
        subject_significant = (None, False)
    # control_pac_vals = None
    # control_z_scores = None
    # pac_zscores = None
    return mean_power, sem_power, times, freqs, used_timepoints, pac_mean, pac_zscore_mean, subject_significant, control_pac_vals, control_z_scores, pac_zscores


salient_vs_control_stats = []  # list of dicts per subject
all_subjects_power = []
pac_zscore_all = []
significant_per_person = []
times, freqs = None, None
# the right preprocessing
pac_all = []
sample_counts = []
epochsss = []
pac_salient_all = []
pac_control_all = []
z_salient_all = []
z_control_all = []
pac_significance_per_subject = []


from mne.time_frequency import psd_array_welch

for subj_id in tqdm(test_subjects, desc="Subjects"):  # subject_ids:

    print(f"\n🔄 Processing subject {subj_id}")
    raw = pre_processed_data_EC[subj_id]["raw"].copy()
    raw.set_montage("standard_1020")

    # Add missing channels and optionally zero pad
    raw, missing_channels, used_zeros = add_missing_channels(raw, channel_order, use_zero_padding=False)
    print(f"  Missing channels: {missing_channels}")
    data = raw.get_data()
    print("Raw data stats:", data.min(), data.max(), data.mean(), data.std())
    # raw.filter(30, None, skip_by_annotation='edge')
    check_filter = False
    if check_filter:
        # Take first 10 seconds of EEG after filtering
        fs = int(raw.info['sfreq'])
        data = raw.get_data(picks=['Cz'])  # or any channel
        segment = data[0, :]  # 10 seconds
        # Compute PSD
        f, Pxx = welch(segment, fs=fs, nperseg=fs * 2)
        # Plot
        plt.semilogy(f, Pxx)
        plt.axvline(1.0, color='red', linestyle='--', label='1 Hz cutoff')
        plt.title("PSD After Filtering (Before Epoching)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.xlim(0, 120)
        plt.legend()
        plt.show()
    if not used_zeros and (missing_channels != []):
        try:
            raw.interpolate_bads(reset_bads=True)
        except Exception as e:
            print(f"  ❌ Skipping subject {subj_id} due to interpolation error: {e}")
            continue

    try:
        raw.pick_channels(channel_order, ordered=True)
    except Exception as e:
        print(f"  ❌ Skipping subject {subj_id} due to missing channels after interpolation: {e}")
        continue

    # Confirm channel order
    assert raw.ch_names == channel_order, f"❗ Channel order mismatch in subject {subj_id}"
    print(f"  ✅ Channel order confirmed")
    print(channel_order)


    # Log interpolation info
    interpolation_log[subj_id] = {
        "missing": missing_channels,
        "used_zero_padding": used_zeros,
        "count": len(missing_channels)
    }

    # Diagnostics (optional)
    data = raw.get_data()
    for ch in missing_channels:
        idx = raw.ch_names.index(ch)
        std = np.std(data[idx])
        print(f"    ⚠️ Std for interpolated/zeroed channel {ch}: {std:.4f}")
        if std < 0.01:
            print(f"      🔇 Channel {ch} has near-zero variance")
    morlet_wave = True
    if morlet_wave:
        mean_power, sem_power, times, freqs, n_used, pac_mean, pac_zscore_mean, (
        p_val, is_significant), control_pac_vals, control_z_scores, pac_zscores = extract_and_average_morlet_from_data(
            subj_id=subj_id,
            data=data,
            salient_sources=salient_sources,
            channel_order=channel_order,
            window_length_sec=4.0,
            freq_min=1,
            freq_max=100,
            n_freqs=50,
            sfreq=sfreq
        )
        z = float(pac_zscore_mean) if pac_zscore_mean is not None else None
        p = float(p_val) if p_val is not None else None
        if z is not None and p is not None:
            print(f"PAC mean z-score: {z:.2f} | p = {p:.3e} | significant: {is_significant}")
        else:
            print("PAC stats unavailable for this subject.")

        if pac_mean is not None:
            pac_salient_all.append(pac_mean)
        if pac_zscore_mean is not None:
            z_salient_all.append(pac_zscore_mean)

        if control_pac_vals is not None:
            pac_control_all.append(np.mean(control_pac_vals))
        if control_z_scores is not None:
            z_control_all.append(np.mean(control_z_scores))
        if pac_zscores is not None:
            if len(pac_zscores) == len(control_z_scores) and len(pac_zscores) > 3:
                t_pac_subj, p_pac_subj = ttest_rel(pac_zscores, control_z_scores)
                pac_significant = p_pac_subj < 0.05
                df = len(pac_zscores) - 1
                print(
                    f"  🧪 PAC z-score t-test: t({df}) = {t_pac_subj:.2f}, p = {p_pac_subj:.4f} → {'✅ significant' if pac_significant else '❌ not significant'}")
                pac_significance_per_subject.append({
                    "subject": subj_id,
                    "t": t_pac_subj,
                    "p": p_pac_subj,
                    "significant": pac_significant
                })
            else:
                print("  ⚠️ Not enough PAC z-scores for per-subject t-test.")

        if mean_power is not None:

            significant_per_person.append((is_significant, subj_id, p))
            if times is not None and freqs is not None:
                times_all = times
                freqs_all = freqs

            sample_counts.append(n_used)
            all_subjects_power.append(mean_power)
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.pcolormesh(times, freqs, mean_power, shading='gouraud')
            plt.title(f'Subject {subj_id} Average Morlet Power (n={n_used})', fontsize=18)
            plt.xlabel('Time (s)', fontsize=18)
            plt.ylabel('Frequency (Hz)', fontsize=18)
            cbar = plt.colorbar(label='Power')
            cbar.set_label('Power', fontsize=16)
            cbar.ax.tick_params(labelsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()

            plt.figure(figsize=(10, 5))
            plt.pcolormesh(times, freqs, sem_power, shading='gouraud', cmap='Reds')
            plt.title(f'Subject {subj_id} SEM of Morlet Power (n={n_used})', fontsize=18)
            plt.xlabel('Time (s)', fontsize=18)
            plt.ylabel('Frequency (Hz)', fontsize=18)
            cbar = plt.colorbar(label='SEM Power')
            cbar.set_label('SEM Power', fontsize=16)
            cbar.ax.tick_params(labelsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()

            gamma_freq_mask = freqs >= 27


            if subj_id == "032316":
                plt.figure(figsize=(10, 5))
                plt.pcolormesh(times, freqs[gamma_freq_mask], mean_power[gamma_freq_mask, :], shading='gouraud')
                plt.title(f'Subject 6 Average Morlet Power (n={n_used}) (30+ Hz)', fontsize=17)
                plt.xlabel('Time (s)', fontsize=18)
                plt.ylabel('Frequency (Hz)', fontsize=18)
                cbar = plt.colorbar(label='Power')
                cbar.set_label('Power', fontsize=16)
                cbar.ax.tick_params(labelsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                print(subj_id)
                print("ok")
                plt.savefig("Average_Power_30_S032316.pdf", bbox_inches='tight')
            # plt.show()


            if subj_id == "032316":
                plt.figure(figsize=(10, 5))
                plt.pcolormesh(times, freqs[gamma_freq_mask], sem_power[gamma_freq_mask, :], shading='gouraud',
                               cmap='Reds')
                plt.title(f'Subject 6 SEM of Morlet Power (n={n_used}) (30+ Hz)', fontsize=17)
                plt.xlabel('Time (s)', fontsize=18)
                plt.ylabel('Frequency (Hz)', fontsize=18)
                cbar = plt.colorbar(label='SEM Power')
                cbar.set_label('SEM Power', fontsize=16)
                cbar.ax.tick_params(labelsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                print("ok")
                plt.savefig("AverageSEM_subject032316.pdf", bbox_inches='tight')
            # plt.show()

    psd_of_all_test = True
    if psd_of_all_test:  # and subj_id in test_subjects:
        data_pre = raw.get_data()
        data_pre = data_pre * 1e6
        psd_pre, freqs = psd_array_welch(
            data_pre, sfreq=sfreq, fmin=1, fmax=80, n_fft=sfreq * 2, average='mean'
        )
        # DON'T convert to dB here for FOOOF fitting
        avg_psd_pre = np.mean(psd_pre, axis=0)
        all_psds_pre.append(avg_psd_pre)

    # Normalize
    norm = True
    if norm:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-8)

    if psd_of_all_test and subj_id in test_subjects:
        # data = data * 1e6
        psd_post, freqs = psd_array_welch(
            data, sfreq=sfreq, fmin=1, fmax=80, n_fft=sfreq * 2, average='mean'
        )
        freqs_psd = freqs
        # Keep linear PSD for FOOOF
        avg_psd_post = np.mean(psd_post, axis=0)
        all_psds_post.append(avg_psd_post)

    # Epoching
    n_epochs = data.shape[1] // epoch_length
    data_trimmed = data[:, :n_epochs * epoch_length]
    epochs = data_trimmed.reshape(len(channel_order), n_epochs, epoch_length)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, epoch_length)
    epochsss.append(n_epochs)
    print(f"Number of epochs per subject: {n_epochs}")
    # Pair generation
    pairs_x1, pairs_x2, labels_y, idxs = create_pairs_3(subj_id, epochs, 3, 10, 0, subject_id=subj_id)

    # Assign to dataset
    if subj_id in train_subjects:
        x1_train.extend(pairs_x1)
        x2_train.extend(pairs_x2)
        y_train.extend(labels_y)
    elif subj_id in val_subjects:
        x1_val.extend(pairs_x1)
        x2_val.extend(pairs_x2)
        y_val.extend(labels_y)
    else:
        x1_test.extend(pairs_x1)
        x2_test.extend(pairs_x2)
        y_test.extend(labels_y)
        index_list_test.extend(idxs)

# Group-level stats on z-scored PAC
# Compare salient vs. control power across subjects
if morlet_wave == True:
    print("\n📋 PAC z-score significance per subject:")
    for stat in pac_significance_per_subject:
        print(
            f"{stat['subject']}: t={stat['t']:.2f}, p={stat['p']:.4f} → {'✅ significant' if stat['significant'] else '❌ not significant'}")

    salient_power = np.stack(all_subjects_power)  # shape: (subjects, freqs, times)
    control_power = np.stack(control_all_subjects_power)

    power_diff = salient_power - control_power  # shape: (subjects, freqs, times)
    tmap, pmap = ttest_1samp(power_diff, 0.0, axis=0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_all, freqs_all, tmap, shading='gouraud', cmap='RdBu_r')
    plt.title("t-statistics (salient - control)")
    plt.colorbar(label="t")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    # plt.show()

    # plt.show()
    print("\n🧾 Summary: Salient vs. Control Power Significance per Subject")
    for stat in salient_vs_control_stats:
        s = stat['subject']
        p = stat['p']
        t = stat['t']
        sig = "✅" if stat['significant'] else "❌"
        print(f"  {s}: t={t:.3f}, p={p:.4e} {sig}")

    pac_zscore_all = np.array(pac_zscore_all)

    if len(pac_salient_all) == len(pac_control_all) > 3:
        t_pac, p_pac = ttest_rel(pac_salient_all, pac_control_all)
        df_pac = len(pac_salient_all) - 1
        print(f"\nPAC (raw) salient vs control: t({df_pac}) = {t_pac:.3f}, p = {p_pac:.4f}")
    else:
        print("⚠️ Not enough PAC data for raw comparison.")

    # PAC z-score
    if len(z_salient_all) == len(z_control_all) > 3:
        t_z, p_z = ttest_rel(z_salient_all, z_control_all)
        df_z = len(z_salient_all) - 1
        print(f"\nPAC z-score salient vs control: t({df_z}) = {t_z:.3f}, p = {p_z:.4f}")
    else:
        print("⚠️ Not enough PAC z-score data for comparison.")

weighted = False
if len(all_subjects_power) > 0:
    subject_powers = np.stack(all_subjects_power, axis=0)  # (n_subjects, n_freqs, n_times)

    if weighted:
        sample_counts = np.array(sample_counts)
        weights = sample_counts / np.sum(sample_counts)  # normalized weights

        # Weighted mean
        global_mean_power = np.tensordot(weights, subject_powers, axes=([0], [0]))

        # Weighted SEM
        diffs = subject_powers - global_mean_power
        weighted_var = np.tensordot(weights, diffs ** 2, axes=([0], [0]))
        global_sem_power = np.sqrt(weighted_var / len(sample_counts))
    else:
        # Unweighted mean and SEM
        global_mean_power = np.mean(subject_powers, axis=0)
        global_sem_power = sem(subject_powers, axis=0)
    # Then you can plot global mean:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_all, freqs_all, global_mean_power, shading='gouraud')
    plt.title("Average Morlet Power across subjects", fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    cbar1 = plt.colorbar(label='Power')
    cbar1.set_label('Power', fontsize=16)  # Colorbar label fontsize
    cbar1.ax.tick_params(labelsize=14)  # Colorbar ticks fontsize
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("avg_morlet_power.pdf", bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_all, freqs_all, global_sem_power, shading='gouraud', cmap='Reds')
    plt.title("SEM of Morlet Power across subjects", fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    cbar2 = plt.colorbar(label='SEM')
    cbar2.set_label('SEM', fontsize=16)
    cbar2.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("avg_morlet_SEM.pdf", bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_all, freqs_all[gamma_freq_mask], global_mean_power[gamma_freq_mask, :], shading='gouraud')
    plt.title("Average Morlet Power across subjects (30Hz+)", fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    cbar3 = plt.colorbar(label='Power')
    cbar3.set_label('Power', fontsize=16)
    cbar3.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("avg_morlet_power_30.pdf", bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_all, freqs_all[gamma_freq_mask], global_sem_power[gamma_freq_mask, :], shading='gouraud',
                   cmap='Reds')
    plt.title("SEM of Morlet Power across subjects (30Hz+)", fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    cbar4 = plt.colorbar(label='SEM')
    cbar4.set_label('SEM', fontsize=16)
    cbar4.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("avg_morlet_SEM_30.pdf", bbox_inches='tight')
    plt.show()

else:
    print("No subject Morlet power data collected.")

# Bandpass filter helper
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import ttest_ind
from mne.time_frequency import tfr_array_morlet, psd_array_welch
from tqdm import tqdm
import pickle
import os

# === Helper Functions ===

def bandpass_filter(data, sfreq, low=1.5, high=4.0):
    b, a = butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype='band')
    return filtfilt(b, a, data)

def compute_envelope_power_and_psd(window, sfreq, plot_example=False):
    """
        Compute the low-frequency power of the gamma-band envelope and its PSD.
    """
    powers = []
    psds = []
    example_plotted = False

    for ch in range(window.shape[0]):
        try:
            gamma_sig = window[ch]
            envelope = np.abs(hilbert(gamma_sig))
            lf_fluct = bandpass_filter(envelope, sfreq)
            power = np.mean(lf_fluct ** 2)
            powers.append(power)

            f, Pxx = welch(envelope, fs=sfreq, nperseg=256)
            psds.append(Pxx)
            if plot_example and not example_plotted:
                print(" d")
                time = np.arange(len(gamma_sig)) / sfreq
                plt.figure(figsize=(12, 4))
                plt.plot(time, gamma_sig, label="Gamma", alpha=0.4)
                plt.plot(time, envelope, label="Envelope", linewidth=2)
                plt.plot(time, lf_fluct, label="1.5–4 Hz Band", linewidth=2)
                plt.xlabel("Time (s)")
                plt.title("Gamma Envelope & Low-Frequency Fluctuation")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 4))
                plt.semilogy(f, Pxx)
                plt.title("PSD of Gamma Envelope")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power")
                plt.axvline(2, color='r', linestyle='--', label='~2 Hz')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

                example_plotted = True

        except Exception:
            continue

    avg_power = np.mean(powers) if powers else np.nan
    avg_psd = np.mean(psds, axis=0) if psds else None
    return avg_power, f, avg_psd

def compute_envelope_tfr(window, sfreq, freqs=np.arange(1.5, 4.1, 0.5), n_cycles=3):
    tfrs = []
    for ch in range(window.shape[0]):
        try:
            gamma_sig = window[ch]
            envelope = np.abs(hilbert(gamma_sig))
            data = envelope[np.newaxis, np.newaxis, :]
            tfr = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output='power')
            tfrs.append(tfr[0, 0])
        except Exception:
            continue
    return np.mean(tfrs, axis=0) if tfrs else None, freqs

# Save/load data helpers
pickle_path = "envelope_analysis_data.pkl"

def save_data(data, path=pickle_path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {path}")

def load_data(path=pickle_path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {path}")
        return data
    else:
        print(f"No saved data found at {path}")
        return None

# === Main Processing ===

# Provide your own x1_test, x2_test, y_test, sfreq before running
# Example:
# x1_test, x2_test, y_test, sfreq = ...

saved_data = load_data()
if saved_data is None:
    env_close, psd_close, tfr_close = [], [], []
    env_far, psd_far, tfr_far = [], [], []
    all_envs_close, all_envs_far = [], []

    for i in tqdm(range(len(y_test))):
        env1, f, psd1 = compute_envelope_power_and_psd(x1_test[i], sfreq)
        env2, _, psd2 = compute_envelope_power_and_psd(x2_test[i], sfreq)
        tfr1, freqs = compute_envelope_tfr(x1_test[i], sfreq)
        tfr2, _ = compute_envelope_tfr(x2_test[i], sfreq)

        diff_env = np.abs(env1 - env2)
        diff_psd = np.abs(psd1 - psd2) if psd1 is not None and psd2 is not None else None
        diff_tfr = np.abs(tfr1 - tfr2) if tfr1 is not None and tfr2 is not None else None

        if y_test[i] == 1:
            env_close.append(diff_env)
            if diff_psd is not None:
                psd_close.append(diff_psd)
            if diff_tfr is not None:
                tfr_close.append(diff_tfr)
            all_envs_close.extend([env1, env2])
        else:
            env_far.append(diff_env)
            if diff_psd is not None:
                psd_far.append(diff_psd)
            if diff_tfr is not None:
                tfr_far.append(diff_tfr)
            all_envs_far.extend([env1, env2])

    env_close = np.array(env_close)
    env_far = np.array(env_far)
    env_close = env_close[~np.isnan(env_close)]
    env_far = env_far[~np.isnan(env_far)]

    all_envs_close = np.array(all_envs_close)
    all_envs_close = all_envs_close[~np.isnan(all_envs_close)]

    all_envs_far = np.array(all_envs_far)
    all_envs_far = all_envs_far[~np.isnan(all_envs_far)]

    save_data({
        "env_close": env_close,
        "env_far": env_far,
        "psd_close": psd_close,
        "psd_far": psd_far,
        "tfr_close": tfr_close,
        "tfr_far": tfr_far,
        "f": f,
        "freqs": freqs,
        "sfreq": sfreq,
        "all_envs_close": all_envs_close,
        "all_envs_far": all_envs_far
    })
else:
    env_close = saved_data["env_close"]
    env_far = saved_data["env_far"]
    psd_close = saved_data["psd_close"]
    psd_far = saved_data["psd_far"]
    tfr_close = saved_data["tfr_close"]
    tfr_far = saved_data["tfr_far"]
    f = saved_data["f"]
    freqs = saved_data["freqs"]
    sfreq = saved_data["sfreq"]
    all_envs_close = saved_data["all_envs_close"]
    all_envs_far = saved_data["all_envs_far"]

# === Statistical Analysis ===

t_stat, p_val = ttest_ind(env_close, env_far)
df = len(env_close) + len(env_far) - 2
print(len(x1_test))
print(f"\n📊 Envelope Power DIFFERENCE (1.5–4 Hz) — close vs far: t({df}) = {t_stat:.3f}, p = {p_val:.9f}")

# === PSD Plot with SEM ===

if len(psd_close) > 0 and len(psd_far) > 0:
    psd_close = np.array(psd_close)
    psd_far = np.array(psd_far)

    mean_psd_close = np.nanmean(psd_close, axis=0)
    mean_psd_far = np.nanmean(psd_far, axis=0)

    sem_psd_close = np.nanstd(psd_close, axis=0) / np.sqrt(psd_close.shape[0])
    sem_psd_far = np.nanstd(psd_far, axis=0) / np.sqrt(psd_far.shape[0])

    if f.shape[0] == mean_psd_close.shape[0]:
        plt.figure(figsize=(10, 5))

        # Plot lines and fill_between as before
        plt.plot(f, mean_psd_close, label="Close", color="blue")
        plt.fill_between(f, mean_psd_close - sem_psd_close, mean_psd_close + sem_psd_close,
                         color="blue", alpha=0.3)
        plt.plot(f, mean_psd_far, label="Far", color="orange")
        plt.fill_between(f, mean_psd_far - sem_psd_far, mean_psd_far + sem_psd_far,
                         color="orange", alpha=0.3)

        # Shade 1.5-4 Hz band
        plt.axvspan(1.5, 4.0, color='gray', alpha=0.2, label="1.5–4 Hz Band")

        # Significance star
        plt.text(2.5, plt.ylim()[1] * 0.9, "***", fontsize=24, ha='center', va='center', color='black')

        # Increase font sizes here:
        plt.title("Mean PSD of Gamma Envelope Differences ± SEM", fontsize=24)
        plt.xlabel("Frequency (Hz)", fontsize=18)
        plt.ylabel("Power", fontsize=18)

        plt.xlim(0, 10)
        plt.legend(fontsize=16)
        plt.grid(True)

        # Increase tick label sizes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Mismatch in PSD frequency dimension: f.shape={f.shape}, psd.shape={mean_psd_close.shape}")


# === Combine all individual PSDs from both conditions ===
all_psds = np.array(psd_close + psd_far)

# Check shape
if all_psds.shape[1] != f.shape[0]:
    print("⚠️ Frequency mismatch: f.shape != psd.shape")
else:
    mean_psd = np.nanmean(all_psds, axis=0)
    sem_psd = np.nanstd(all_psds, axis=0) / np.sqrt(all_psds.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(f, mean_psd, color="purple", label="All Test Windows")
    plt.fill_between(f, mean_psd - sem_psd, mean_psd + sem_psd, color="purple", alpha=0.3)

    # Highlight the 1.5–4 Hz band
    #plt.axvspan(1.5, 4.0, color='gray', alpha=0.2, label="1.5–4 Hz Band")

    # Formatting
    plt.title("Average PSD of Gamma Envelope", fontsize=24)
    plt.xlabel("Frequency (Hz)", fontsize=18)
    plt.ylabel("Power", fontsize=18)
    plt.xlim(0, 10)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

#getting the psd of all test windows (in the paper it is of all the windows; and can be made by changing the loop above to all subjects instead of just test)

def fit_and_return_fooof(freqs, psd, label):
    freq_mask = (freqs >= 1) & (freqs <= 80)

    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1)
    fm.fit(freqs[freq_mask], psd[freq_mask])

    offset, slope = fm.get_params('aperiodic_params')
    print(f"{label} → FOOOF offset: {offset:.3f}, slope: {slope:.3f}")

    aperiodic_only = fm._ap_fit             # already in dB
    periodic_only = 10 * np.log10(np.clip(fm._peak_fit, 1e-12, None))  # convert linear peak to dB
    full_model = aperiodic_only + periodic_only  # sum in dB space

    full_model_full = np.zeros_like(freqs)
    periodic_only_full = np.zeros_like(freqs)
    aperiodic_only_full = np.zeros_like(freqs)

    full_model_full[freq_mask] = full_model
    periodic_only_full[freq_mask] = periodic_only
    aperiodic_only_full[freq_mask] = aperiodic_only

    return {
        "psd": 10 * np.log10(psd),              # your original PSD converted to dB for plotting
        "periodic": periodic_only_full,
        "aperiodic": aperiodic_only_full,
        "full_model": full_model_full
    }

all_psds_pre = np.array(all_psds_pre)
all_psds_post = np.array(all_psds_post)

mean_psd_pre_db = np.mean(all_psds_pre, axis=0)
mean_psd_post_db = np.mean(all_psds_post, axis=0)

pre = fit_and_return_fooof(freqs_psd, mean_psd_pre_db, label='Pre-Norm')
post = fit_and_return_fooof(freqs_psd, mean_psd_post_db, label='Post-Norm')

print(f"Pre aperiodic min/max: {pre['aperiodic'].min():.2f} / {pre['aperiodic'].max():.2f}")
print(f"Post aperiodic min/max: {post['aperiodic'].min():.2f} / {post['aperiodic'].max():.2f}")


# Plot

import numpy as np
import matplotlib.pyplot as plt

f = freqs_psd

# Convert dB back to linear power for linear-linear plot
pre_psd_linear = 10 ** (pre["psd"] / 10)
post_psd_linear = 10 ** (post["psd"] / 10)
pre_periodic_linear = 10 ** (pre["periodic"] / 10)
post_periodic_linear = 10 ** (post["periodic"] / 10)
pre_aperiodic_linear = 10 ** (pre["aperiodic"] / 10)
post_aperiodic_linear = 10 ** (post["aperiodic"] / 10)

plt.figure(figsize=(10, 6))

# Updated plot lines with new labels
plt.plot(f, pre["psd"], label='Average Power Spectrum Density over subjects', color='blue')
plt.plot(f, pre["periodic"], label='Periodic FOOOF component', linestyle='--', color='blue')
plt.plot(f, pre["aperiodic"], label='Aperiodic FOOOF component', linestyle=':', color='blue')

# Axis labels with larger font size
plt.xlabel("Frequency (Hz)", fontsize=22)
plt.ylabel("Power (dB)", fontsize=22)

# Title
plt.title("Average PSD Resting State Data", fontsize=26)

# Legend with increased font size
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=20)
# Grid and layout
plt.grid(True)
plt.tight_layout()
plt.xlim(1, 60)
plt.ylim(-40)
plt.savefig("resting_state_psd.pdf", bbox_inches='tight')
plt.show()





