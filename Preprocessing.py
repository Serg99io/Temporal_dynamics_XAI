from mne.io import read_raw_eeglab
import mne
import tqdm
import random
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle

# after downloading the set 1 of https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html

# Define the subject range and exclude 032309 and 032335
subject_ids = [f"0323{str(i).zfill(2)}" for i in range(1, 45) if (i != 9 and i != 35)]

# Dictionary to store preprocessed data
pre_processed_data_EC = {}

for subj in subject_ids:
    file_path_EC = f"subject_data/sub-{subj}/sub-{subj}/sub-{subj}_EC.set"

    try:
        raw_EC = read_raw_eeglab(file_path_EC, preload=True)
        pre_processed_data_EC[subj] = {
    "raw": raw_EC,  # Full Raw object
    "data": raw_EC.get_data()  # NumPy array
}
        print(f"Successfully loaded subject {subj}")
    except Exception as e:
        print(f"Error loading subject {subj}: {e}")

with open("preprocessed_data_EC.pkl", "wb") as f:
    pickle.dump(pre_processed_data_EC, f)

random.seed(42)  # Ensure reproducibility
random.shuffle(subject_ids)

n_train = int(0.6 * len(subject_ids))
n_val = int(0.2 * len(subject_ids))

train_subjects = subject_ids[:n_train]
val_subjects = subject_ids[n_train:n_train + n_val]
test_subjects = subject_ids[n_train + n_val:]

print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")


# Corrected helper function which adds missing channels to the subjects who miss channels with the help of interpolation

def add_missing_channels(raw, target_channels, use_zero_padding=False):
    missing = [ch for ch in target_channels if ch not in raw.ch_names]
    if not missing:
        return raw, [], False  # nothing to do

    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    data = np.zeros((len(missing), n_times))

    # Create info for dummy channels
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


def create_pairs_3(subject_name, epochs, close_threshold, far_threshold, offset):
    """
    Create pairs of EEG epochs with a relative positioning task for training the models (e.g., Siamese networks).

    Parameters
    ----------
    subject_name : str
        Identifier for the subject (used in the index tracking).
    epochs : list or np.ndarray
        Collection of EEG epochs (time-series segments).
    close_threshold : int
        Maximum distance (in epochs) to consider two epochs as a "close" pair.
    far_threshold : int
        Minimum distance (in epochs) required to consider two epochs as a "far" pair.
    offset : int
        Index offset for keeping track of epoch indices across subjects.


    Returns
    -------
    pairs_x1, pairs_x2 : np.ndarray
        Arrays of paired epochs (X1 and X2).
    labels_y : np.ndarray
        Labels for pairs (1 = close/similar, 0 = far/dissimilar).
    idxs : list of tuples
        Metadata with (epoch_index1, epoch_index2, subject_name).
    """

    # Lists to hold results
    pairs_x1, pairs_x2, labels_y, idxs = [], [], [], []
    n_epochs = len(epochs)

    # ---------- Create "close" pairs ----------
    close_pairs = set()
    for idx1 in range(n_epochs):
        # Pair each epoch with nearby epochs within close_threshold
        for idx2 in range(idx1 + 1, min(idx1 + close_threshold, n_epochs)):
            close_pairs.add((idx1, idx2))

    # ---------- Create "far" pairs ----------
    far_pairs = set()
    for idx1 in range(n_epochs):
        for idx2 in range(n_epochs):
            # Far = epochs separated by more than far_threshold
            if abs(idx1 - idx2) > far_threshold and idx1 != idx2:
                far_pairs.add((idx1, idx2))

    # Randomly sample far pairs to balance dataset size with close pairs
    far_pairs = list(far_pairs)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(far_pairs)
    far_pairs = far_pairs[:len(close_pairs)]

    # ---------- Add close pairs to final lists ----------
    for idx1, idx2 in close_pairs:
        pairs_x1.append(epochs[idx1])
        pairs_x2.append(epochs[idx2])
        labels_y.append(1)  # label = 1 (close/similar)
        idxs.append((idx1 + offset, idx2 + offset, subject_name))

    # ---------- Add far pairs to final lists ----------
    for idx1, idx2 in far_pairs:
        pairs_x1.append(epochs[idx1])
        pairs_x2.append(epochs[idx2])
        labels_y.append(0)  # label = 0 (far/dissimilar)
        idxs.append((idx1 + offset, idx2 + offset, subject_name))

    # ---------- Return arrays ----------
    return (
        np.array(pairs_x1, dtype=np.float32),
        np.array(pairs_x2, dtype=np.float32),
        np.array(labels_y, dtype=np.float32),
        idxs,
    )


def preprocess_and_create_pairs(
        subject_ids,
        pre_processed_data_EC,
        train_subjects,
        val_subjects,
        create_pairs_fn,
        low_filter,
        high_filter,
        norm=True,
        epoch_length_sec=2.5,
        sfreq=250,
        close_threshold=3,
        far_threshold=10,
        use_zero_padding=False,
        save_results=False,
        save_path="processed_data.pkl"

):
    """
    Preprocess EEG data for multiple subjects, interpolate missing channels,
    epoch the data, normalize (optional), and generate pairs for training.
    Optionally saves the results to a pickle file.

    Parameters
    ----------
    subject_ids : list
        List of subject IDs to process.
    pre_processed_data_EC : dict
        Dictionary with subject EEG data: pre_processed_data_EC[subj_id]["raw"] = mne.Raw object.
    train_subjects : list
        IDs of subjects used for training.
    val_subjects : list
        IDs of subjects used for validation.
    create_pairs_fn : function
        Function used to generate pairs (e.g., create_pairs_3).
    low_filter : int or None
        Everything below will be filtered out
    high_filter : int or None
        Everything above will be filtered out
    norm : bool, default=True
        If True, apply z-normalization to data per channel.
    epoch_length_sec : float, default=2.5
        Length of each epoch in seconds.
    sfreq : int, default=250
        Sampling frequency of the EEG data.
    close_threshold : int, default=3
        Max distance between epochs to be considered "close".
    far_threshold : int, default=10
        Min distance between epochs to be considered "far".
    use_zero_padding : bool, default=False
        If True, pad missing channels with zeros.
    save_results : bool, default=False
        If True, saves results with pickle.
    save_path : str, default="processed_data.pkl"
        Path where pickle file is saved.

    Returns
    -------
    results : dict
        Dictionary containing datasets, logs, and channel info.
    """

    # ---------- valid channels order utilized in study ----------

    valid_channels_name = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4',
                           'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
                           'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4',
                           'FT8', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
                           'PO3', 'POz', 'PO4', 'PO8', 'TP7', 'T7']



    # ---------- Initialize containers ----------
    x1_train, x2_train, y_train = [], [], []
    x1_val, x2_val, y_val = [], [], []
    x1_test, x2_test, y_test, index_list_test = [], [], [], []
    interpolation_log = {}

    # ---------- Reference channel order ----------
    ref_raw = pre_processed_data_EC[subject_ids[0]]["raw"].copy()
    ref_raw.set_montage("standard_1020")
    ref_raw.pick_channels(valid_channels_name)
    channel_order = ref_raw.ch_names
    print(channel_order)


    # ---------- Loop over subjects ----------
    for subj_id in tqdm(subject_ids, desc="Subjects"):
        print(f"\n🔄 Processing subject {subj_id}")
        raw = pre_processed_data_EC[subj_id]["raw"].copy()
        raw.set_montage("standard_1020")

        # Add missing channels
        raw, missing_channels, used_zeros = add_missing_channels(raw, channel_order, use_zero_padding=use_zero_padding)
        print(f"  Missing channels: {missing_channels}")
        data = raw.get_data()
        print("Raw data stats:", data.min(), data.max(), data.mean(), data.std())

        # Apply high-pass filter (30 Hz cutoff)
        raw.filter(low_filter, high_filter, skip_by_annotation='edge')

        # Interpolate if needed
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

        # Log interpolation info
        interpolation_log[subj_id] = {
            "missing": missing_channels,
            "used_zero_padding": used_zeros,
            "count": len(missing_channels)
        }

        # Diagnostics
        data = raw.get_data()
        for ch in missing_channels:
            idx = raw.ch_names.index(ch)
            std = np.std(data[idx])
            print(f"    ⚠️ Std for interpolated/zeroed channel {ch}: {std:.4f}")
            if std < 0.01:
                print(f"      🔇 Channel {ch} has near-zero variance")

        # ---------- Normalize ----------
        if norm:
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            data = (data - mean) / (std + 1e-8)

        # ---------- Epoching ----------
        epoch_length = int(epoch_length_sec * sfreq)
        n_epochs = data.shape[1] // epoch_length
        data_trimmed = data[:, :n_epochs * epoch_length]
        epochs = data_trimmed.reshape(len(channel_order), n_epochs, epoch_length)
        epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, epoch_length)

        print(f"Number of epochs per subject: {n_epochs}")

        # ---------- Pair generation ----------
        pairs_x1, pairs_x2, labels_y, idxs = create_pairs_fn(
            subj_id, epochs, close_threshold, far_threshold, 0)

        # ---------- Assign to dataset ----------
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
            index_list_test.extend(idxs)  # keep track of test idxs


    #random_shuffle everything

    x1_train, x2_train, y_train = shuffle(x1_train, x2_train, y_train, random_state=42)

    # Shuffle validation data
    x1_val, x2_val, y_val = shuffle(x1_val, x2_val, y_val, random_state=42)

    # Shuffle test data
    x1_test, x2_test, y_test = shuffle(x1_test, x2_test, y_test, random_state=42)
    index_list_test = shuffle(index_list_test, random_state=42)
    # ---------- Package results ----------
    results = {
        "x1_train": np.array(x1_train, dtype=np.float32),
        "x2_train": np.array(x2_train, dtype=np.float32),
        "y_train": np.array(y_train, dtype=np.float32),
        "x1_val": np.array(x1_val, dtype=np.float32),
        "x2_val": np.array(x2_val, dtype=np.float32),
        "y_val": np.array(y_val, dtype=np.float32),
        "x1_test": np.array(x1_test, dtype=np.float32),
        "x2_test": np.array(x2_test, dtype=np.float32),
        "y_test": np.array(y_test, dtype=np.float32),
        "index_list_test": index_list_test,
        "interpolation_log": interpolation_log,
        "valid_channels_name": valid_channels_name,
        "channel_order": channel_order
    }

    # ---------- Save if requested ----------
    if save_results:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"💾 Results saved to {save_path}")

    return results

epoched_data = preprocess_and_create_pairs(
    subject_ids,
    pre_processed_data_EC,
    train_subjects,
    val_subjects,
    create_pairs_3,
    low_filter=30,
    high_filter=None,
    norm=True,
    save_results=True,
    save_path="my_eeg_epoched_data_30_inf.pkl"
)

raw_epoched_data = preprocess_and_create_pairs(
    subject_ids,
    pre_processed_data_EC,
    train_subjects,
    val_subjects,
    create_pairs_3,
    low_filter=30,
    high_filter=None,
    norm=False,
    save_results=True,
    save_path="my_raw_eeg_epoched_data_30_inf.pkl"
)

with open("my_eeg_epoched_data_30_inf.pkl", "rb") as f:
    epoched_data = pickle.load(f)

with open("my_raw_eeg_epoched_data_30_inf.pkl", "rb") as f:
    raw_epoched_data = pickle.load(f)






