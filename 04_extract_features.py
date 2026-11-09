
#!/usr/bin/env python3

"""
Stage 4: Biological feature extraction.

Inputs:
    prepared_dataset/transcript_sequences.fa.gz
    prepared_dataset/transcript_labels.fa.gz
    partitioned_dataset/partitioned_window_metadata.csv

Outputs:
    feature_dataset/features.h5

Stored data:
    sequence: encoded DNA, shape (N, 2048)
    labels: E=0, I=1, X=2, shape (N, 2048)
    composition: GC content and GC skew, shape (N, 2048, 2)
    kmer: 16 dinucleotide + 64 trinucleotide channels,
          shape (N, 2048, 80)

The engineered features are calculated using the full
transcript sequence before extracting each window.

Feature context:
    Centered 101-bp window.

K-mer order:
    A, C, G, T (base-4 encoding).

Ambiguous k-mers containing N are excluded.
"""

import argparse
import csv
import gzip
from pathlib import Path

import h5py
import numpy as np


WINDOW_LENGTH = 2048
CONTEXT_LENGTH = 101
HALF_CONTEXT = CONTEXT_LENGTH // 2

# DNA encoding.
BASE_CODES = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4
}

LABEL_CODES = {
    "E": 0,
    "I": 1,
    "X": 2
}


def read_fasta(path):
    """Read a compressed FASTA file."""

    sequences = {}

    with gzip.open(path, "rt", encoding="ascii") as handle:
        current_id = None
        parts = []

        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(parts)

                current_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper())

        if current_id is not None:
            sequences[current_id] = "".join(parts)

    return sequences


def encode_sequence(sequence):
    """Convert DNA characters into integer codes."""

    try:
        return np.fromiter(
            (BASE_CODES[base] for base in sequence),
            dtype=np.uint8,
            count=len(sequence)
        )
    except KeyError as exc:
        raise ValueError(
            f"Unexpected DNA character: {exc}"
        ) from exc


def encode_labels(labels):
    """Convert E/I labels into integer codes."""

    try:
        return np.fromiter(
            (LABEL_CODES[label] for label in labels),
            dtype=np.uint8,
            count=len(labels)
        )
    except KeyError as exc:
        raise ValueError(
            f"Unexpected label: {exc}"
        ) from exc


def prefix_sum(values):
    """Prefix sum with a leading zero."""

    return np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(values, dtype=np.int64)
    ))


def centered_bounds(length, context_length):
    """
    Return left and right bounds for every sequence position.

    Bounds are left-inclusive and right-exclusive.
    At transcript edges, the context is truncated.
    """

    positions = np.arange(length)

    left = np.maximum(
        0, positions - context_length // 2
    )

    right = np.minimum(
        length, positions + context_length // 2 + 1
    )

    return left, right


def extract_composition(codes):
    """
    Extract GC content and GC skew.

    GC content = (G + C) / (A + C + G + T)

    GC skew = (G - C) / (G + C)

    N bases are excluded.
    """

    length = len(codes)

    left, right = centered_bounds(
        length, CONTEXT_LENGTH
    )

    gc = np.zeros(length, dtype=np.uint8)
    skew = np.zeros(length, dtype=np.uint8)

    valid = (codes < 4)
    c_count = prefix_sum(codes == 1)
    g_count = prefix_sum(codes == 2)
    valid_count = prefix_sum(valid)

    local_c = c_count[right] - c_count[left]
    local_g = g_count[right] - g_count[left]
    local_valid = valid_count[right] - valid_count[left]

    local_gc = local_c + local_g

    # Quantize GC content from [0,1] to [0,255].
    gc_mask = local_valid > 0

    gc[gc_mask] = np.rint(
        255.0 * local_gc[gc_mask] /
        local_valid[gc_mask]
    ).astype(np.uint8)

    # Quantize GC skew from [-1,1] to [0,255].
    # 0 corresponds to -1, 127/128 approximately to 0,
    # and 255 corresponds to +1.
    skew_mask = local_gc > 0

    skew[skew_mask] = np.rint(
        127.5 * (
            1.0 +
            (local_g[skew_mask] - local_c[skew_mask]) /
            local_gc[skew_mask]
        )
    ).astype(np.uint8)

    return np.stack((gc, skew), axis=-1)


def extract_kmer_features(codes, k):
    """
    Calculate local k-mer frequencies at every position.

    For each position, count complete k-mers whose full
    sequence lies inside its centered context.

    The frequency channels are normalized over valid
    A/C/G/T k-mers and quantized to uint8 [0,255].
    """

    length = len(codes)
    channel_count = 4 ** k

    output = np.zeros(
        (length, channel_count),
        dtype=np.uint8
    )

    if length < k:
        return output

    # Encode every k-mer using base-4 notation.
    kmers = np.lib.stride_tricks.sliding_window_view(
        codes, k
    )

    valid_kmers = np.all(kmers < 4, axis=1)

    weights = 4 ** np.arange(k - 1, -1, -1)

    kmer_codes = np.sum(
        kmers.astype(np.int64) * weights,
        axis=1
    )

    kmer_codes[~valid_kmers] = -1

    number_of_kmers = len(kmer_codes)

    positions = np.arange(length)

    # A k-mer start must be inside the 101-bp context,
    # and the entire k-mer must fit inside that context.
    left = np.maximum(
        0, positions - HALF_CONTEXT
    )

    right = np.minimum(
        number_of_kmers,
        positions + HALF_CONTEXT + 2 - k
    )

    right = np.maximum(right, left)

    denominator_prefix = prefix_sum(valid_kmers)

    denominator = (
        denominator_prefix[right] -
        denominator_prefix[left]
    )

    valid_positions = denominator > 0

    # One frequency channel at a time to limit RAM use.
    for channel in range(channel_count):

        channel_prefix = prefix_sum(
            kmer_codes == channel
        )

        counts = (
            channel_prefix[right] -
            channel_prefix[left]
        )

        mask = valid_positions

        output[mask, channel] = np.rint(
            255.0 * counts[mask] / denominator[mask]
        ).astype(np.uint8)

    return output


def load_metadata(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():

    parser = argparse.ArgumentParser(
        description="Extract biological features for genomic windows."
    )

    parser.add_argument("--sequences", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="feature_dataset/features.h5")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading transcript sequences...")
    sequences = read_fasta(args.sequences)

    print("Loading transcript labels...")
    labels = read_fasta(args.labels)

    if sequences.keys() != labels.keys():
        raise ValueError("Sequence and label IDs do not match.")

    print("Loading partitioned window metadata...")
    metadata = load_metadata(args.metadata)

    number_of_windows = len(metadata)

    if number_of_windows == 0:
        raise ValueError("No windows found in metadata.")

    # Preserve metadata row order exactly.
    windows_by_transcript = {}

    for row_index, row in enumerate(metadata):

        transcript_id = row["TranscriptID"]

        windows_by_transcript.setdefault(
            transcript_id, []
        ).append((row_index, row))

    if set(windows_by_transcript) != set(sequences):
        raise ValueError(
            "Metadata transcript IDs do not match FASTA IDs."
        )

    print(f"Transcripts: {len(sequences):,}")
    print(f"Windows: {number_of_windows:,}")

    print("\nCreating HDF5 datasets...")

    with h5py.File(output_path, "w") as h5:

        seq_ds = h5.create_dataset(
            "sequence",
            shape=(number_of_windows, WINDOW_LENGTH),
            dtype=np.uint8,
            chunks=(1, WINDOW_LENGTH),
            compression="gzip",
            compression_opts=1
        )

        label_ds = h5.create_dataset(
            "labels",
            shape=(number_of_windows, WINDOW_LENGTH),
            dtype=np.uint8,
            chunks=(1, WINDOW_LENGTH),
            compression="gzip",
            compression_opts=1
        )

        composition_ds = h5.create_dataset(
            "composition",
            shape=(number_of_windows, WINDOW_LENGTH, 2),
            dtype=np.uint8,
            chunks=(1, WINDOW_LENGTH, 2),
            compression="gzip",
            compression_opts=1
        )

        kmer_ds = h5.create_dataset(
            "kmer",
            shape=(number_of_windows, WINDOW_LENGTH, 80),
            dtype=np.uint8,
            chunks=(1, WINDOW_LENGTH, 80),
            compression="gzip",
            compression_opts=1
        )

        # Save metadata fields in the same row order.
        string_dtype = h5py.string_dtype(encoding="utf-8")

        for field in metadata[0].keys():
            values = [row[field] for row in metadata]

            if field in (
                "WindowNumber", "TranscriptStart",
                "TranscriptEnd", "GenomicStart",
                "GenomicEnd", "ValidLength",
                "ExonBases", "IntronBases",
                "OverlapGroup"
            ):
                h5.create_dataset(
                    f"metadata/{field}",
                    data=np.asarray(values, dtype=np.int64)
                )
            else:
                h5.create_dataset(
                    f"metadata/{field}",
                    data=np.asarray(values, dtype=object),
                    dtype=string_dtype
                )

        h5.attrs["window_length"] = WINDOW_LENGTH
        h5.attrs["context_length"] = CONTEXT_LENGTH
        h5.attrs["dinucleotide_channels"] = 16
        h5.attrs["trinucleotide_channels"] = 64
        h5.attrs["composition_channels"] = 2
        h5.attrs["feature_encoding"] = "uint8, normalized to 0-255"
        h5.attrs["sequence_encoding"] = "A=0,C=1,G=2,T=3,N=4"
        h5.attrs["label_encoding"] = "E=0,I=1,X=2"
        h5.attrs["kmer_order"] = "A,C,G,T base-4"

        print("\nExtracting features transcript by transcript...")

        processed = 0

        for transcript_index, (
            transcript_id, window_rows
        ) in enumerate(windows_by_transcript.items(), start=1):

            sequence = sequences[transcript_id]
            transcript_labels = labels[transcript_id]

            if len(sequence) != len(transcript_labels):
                raise ValueError(
                    f"Length mismatch: {transcript_id}"
                )

            codes = encode_sequence(sequence)
            label_codes = encode_labels(transcript_labels)

            composition = extract_composition(codes)

            dinucleotide = extract_kmer_features(codes, 2)
            trinucleotide = extract_kmer_features(codes, 3)

            kmer_features = np.concatenate(
                (dinucleotide, trinucleotide),
                axis=1
            )

            if kmer_features.shape != (len(sequence), 80):
                raise RuntimeError("Unexpected k-mer dimensions.")

            # Slice transcript-level features into the exact
            # windows specified by the partitioned metadata.
            for row_index, row in window_rows:

                start = int(row["TranscriptStart"]) - 1
                valid_length = int(row["ValidLength"])
                end = start + valid_length

                if end > len(sequence):
                    raise ValueError(
                        f"Window exceeds transcript: {transcript_id}"
                    )

                if valid_length <= 0 or valid_length > WINDOW_LENGTH:
                    raise ValueError(
                        f"Invalid window length: {transcript_id}"
                    )

                seq_window = codes[start:end]
                label_window = label_codes[start:end]

                comp_window = composition[start:end]
                kmer_window = kmer_features[start:end]

                padding = WINDOW_LENGTH - valid_length

                if padding:
                    seq_window = np.pad(
                        seq_window, (0, padding),
                        constant_values=4
                    )

                    label_window = np.pad(
                        label_window, (0, padding),
                        constant_values=2
                    )

                    comp_window = np.pad(
                        comp_window,
                        ((0, padding), (0, 0)),
                        constant_values=0
                    )

                    kmer_window = np.pad(
                        kmer_window,
                        ((0, padding), (0, 0)),
                        constant_values=0
                    )

                seq_ds[row_index] = seq_window
                label_ds[row_index] = label_window
                composition_ds[row_index] = comp_window
                kmer_ds[row_index] = kmer_window

                processed += 1

            if transcript_index % 100 == 0:
                print(
                    f"Processed {transcript_index:,}/"
                    f"{len(sequences):,} transcripts; "
                    f"{processed:,} windows"
                )

            del codes
            del label_codes
            del composition
            del dinucleotide
            del trinucleotide
            del kmer_features

        if processed != number_of_windows:
            raise RuntimeError(
                f"Expected {number_of_windows} windows, "
                f"processed {processed}."
            )

        h5.flush()

    print("\n" + "=" * 52)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 52)
    print(f"Transcripts: {len(sequences):,}")
    print(f"Windows: {number_of_windows:,}")
    print("DNA channels: 5 (generated later by PyTorch)")
    print("Composition channels: 2")
    print("Dinucleotide channels: 16")
    print("Trinucleotide channels: 64")
    print("Engineered feature channels: 82")
    print("Total model input channels: 87")
    print(f"Output: {output_path}")
    print("=" * 52)
    print("Stage 4 completed successfully.")


if __name__ == "__main__":
    main()
