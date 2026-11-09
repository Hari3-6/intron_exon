
#!/usr/bin/env python3

"""
Stage 2: Generate overlapping transcript windows.

Input:
    transcript_sequences.fa.gz
    transcript_labels.fa.gz
    transcript_metadata.csv

Output:
    window_sequences.fa.gz
    window_labels.fa.gz
    window_metadata.csv

Configuration:
    Window length = 2048
    Window stride = 1024
    Sequence padding = N
    Label padding = X
"""

import argparse
import csv
import gzip
from pathlib import Path


WINDOW_LENGTH = 2048
WINDOW_STRIDE = 1024


def read_fasta(path):
    """Read a FASTA file and return {sequence_id: sequence}."""

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
                if current_id is None:
                    raise ValueError("FASTA sequence found before header.")

                parts.append(line)

        if current_id is not None:
            sequences[current_id] = "".join(parts)

    return sequences


def generate_starts(transcript_length):
    """Generate window starts, including the final full window."""

    if transcript_length <= WINDOW_LENGTH:
        return [0]

    starts = list(
        range(
            0,
            transcript_length - WINDOW_LENGTH + 1,
            WINDOW_STRIDE
        )
    )

    final_start = transcript_length - WINDOW_LENGTH

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def main():

    parser = argparse.ArgumentParser(
        description="Generate overlapping transcript windows."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="window_dataset")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_file = input_dir / "transcript_sequences.fa.gz"
    label_file = input_dir / "transcript_labels.fa.gz"
    metadata_file = input_dir / "transcript_metadata.csv"

    print("Loading transcript sequences...")
    sequences = read_fasta(sequence_file)

    print("Loading transcript labels...")
    labels = read_fasta(label_file)

    print(f"Sequences: {len(sequences):,}")
    print(f"Labels: {len(labels):,}")

    if sequences.keys() != labels.keys():
        raise ValueError("Sequence and label transcript IDs do not match.")

    with open(metadata_file, newline="", encoding="utf-8") as handle:
        transcript_metadata = {
            row["TranscriptID"]: row
            for row in csv.DictReader(handle)
        }

    if sequences.keys() != transcript_metadata.keys():
        raise ValueError("Transcript metadata IDs do not match FASTA IDs.")

    sequence_output = output_dir / "window_sequences.fa.gz"
    label_output = output_dir / "window_labels.fa.gz"
    metadata_output = output_dir / "window_metadata.csv"

    metadata_columns = [
        "TranscriptID",
        "GeneID",
        "Strand",
        "WindowNumber",
        "TranscriptStart",
        "TranscriptEnd",
        "GenomicStart",
        "GenomicEnd",
        "ValidLength",
        "ExonBases",
        "IntronBases"
    ]

    total_windows = 0
    total_valid_bases = 0
    total_exon_bases = 0
    total_intron_bases = 0
    total_padding = 0

    print("\nGenerating windows...")

    with gzip.open(
        sequence_output, "wt", encoding="ascii"
    ) as seq_out, gzip.open(
        label_output, "wt", encoding="ascii"
    ) as label_out, open(
        metadata_output, "w", newline="", encoding="utf-8"
    ) as metadata_out:

        writer = csv.DictWriter(
            metadata_out,
            fieldnames=metadata_columns
        )
        writer.writeheader()

        for transcript_index, (transcript_id, sequence) in enumerate(
            sequences.items(), start=1
        ):

            label = labels[transcript_id]

            if len(sequence) != len(label):
                raise ValueError(
                    f"Length mismatch for {transcript_id}"
                )

            if set(sequence) - set("ACGTN"):
                raise ValueError(
                    f"Invalid DNA character in {transcript_id}"
                )

            if set(label) - set("EI"):
                raise ValueError(
                    f"Invalid label character in {transcript_id}"
                )

            transcript_length = len(sequence)
            transcript_info = transcript_metadata[transcript_id]

            starts = generate_starts(transcript_length)

            for window_number, start in enumerate(starts, start=1):

                end = min(start + WINDOW_LENGTH, transcript_length)

                sequence_window = sequence[start:end]
                label_window = label[start:end]

                valid_length = len(sequence_window)
                padding_length = WINDOW_LENGTH - valid_length

                sequence_window += "N" * padding_length
                label_window += "X" * padding_length

                window_id = f"{transcript_id}_W{window_number}"

                seq_out.write(f">{window_id}\n")
                label_out.write(f">{window_id}\n")

                for pos in range(0, WINDOW_LENGTH, 80):
                    seq_out.write(sequence_window[pos:pos + 80] + "\n")
                    label_out.write(label_window[pos:pos + 80] + "\n")

                exon_bases = label_window.count("E")
                intron_bases = label_window.count("I")

                # Coordinates are 1-based and inclusive.
                genomic_start = int(transcript_info["GenomicStart"])
                genomic_end = int(transcript_info["GenomicEnd"])
                strand = transcript_info["Strand"]

                if strand == "+":
                    window_genomic_start = genomic_start + start
                    window_genomic_end = genomic_start + end - 1
                else:
                    window_genomic_start = genomic_end - end + 1
                    window_genomic_end = genomic_end - start

                writer.writerow({
                    "TranscriptID": transcript_id,
                    "GeneID": transcript_info["GeneID"],
                    "Strand": strand,
                    "WindowNumber": window_number,
                    "TranscriptStart": start + 1,
                    "TranscriptEnd": end,
                    "GenomicStart": window_genomic_start,
                    "GenomicEnd": window_genomic_end,
                    "ValidLength": valid_length,
                    "ExonBases": exon_bases,
                    "IntronBases": intron_bases
                })

                total_windows += 1
                total_valid_bases += valid_length
                total_exon_bases += exon_bases
                total_intron_bases += intron_bases
                total_padding += padding_length

            if transcript_index % 500 == 0:
                print(
                    f"Processed {transcript_index:,}/"
                    f"{len(sequences):,} transcripts"
                )

    print("\n" + "=" * 48)
    print("WINDOW GENERATION SUMMARY")
    print("=" * 48)
    print(f"Transcripts: {len(sequences):,}")
    print(f"Windows: {total_windows:,}")
    print(f"Valid base occurrences: {total_valid_bases:,}")
    print(f"Exon occurrences: {total_exon_bases:,}")
    print(f"Intron occurrences: {total_intron_bases:,}")
    print(f"Padding positions: {total_padding:,}")
    print(f"Window length: {WINDOW_LENGTH}")
    print(f"Window stride: {WINDOW_STRIDE}")
    print("=" * 48)
    print("Stage 2 completed successfully.")


if __name__ == "__main__":
    main()
