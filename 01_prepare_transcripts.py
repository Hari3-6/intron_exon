
#!/usr/bin/env python3
"""
Stage 1: Prepare transcript sequences and exon/intron labels.

Input:
    GRCh38.primary_assembly.genome.fa
    gencode.v50.primary_assembly.annotation.gtf

Output:
    transcript_sequences.fa.gz
    transcript_labels.fa.gz
    transcript_metadata.csv

Reference:
    Human GRCh38, GENCODE v50
    Chromosome 22, protein-coding transcripts

Labels:
    E = exon
    I = intron

All output sequences are in transcription orientation.
"""

import argparse
import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_TRANSCRIPTS = 6730
CHROMOSOME = "chr22"

# Complement translation table for DNA.
COMPLEMENT = str.maketrans(
    "ACGTNacgtn",
    "TGCANtgcan"
)


def parse_attributes(attribute_text):
    """Parse GENCODE GTF attributes into a dictionary."""
    return dict(
        re.findall(r'(\S+)\s+"([^"]*)"', attribute_text)
    )


def load_chromosome_sequence(fasta_path, chromosome):
    """
    Read only the requested chromosome from a FASTA file.
    The chromosome sequence is held in memory.
    """
    print(f"Reading {chromosome} from genome FASTA...")

    sequence_parts = []
    collecting = False

    with open(fasta_path, "r") as fasta:
        for line in fasta:
            if line.startswith(">"):
                header = line[1:].strip().split()[0]
                collecting = (header == chromosome)
                continue

            if collecting:
                sequence_parts.append(line.strip().upper())

    if not sequence_parts:
        raise ValueError(
            f"Chromosome {chromosome} was not found in {fasta_path}"
        )

    sequence = "".join(sequence_parts)

    print(f"Chromosome length: {len(sequence):,} bp")

    return sequence


def extract_protein_coding_exons(gtf_path, chromosome):
    """
    Read exon records from the selected chromosome.

    Only records with gene_type = protein_coding are retained.
    Exons are grouped by transcript ID.
    """
    print("\nReading protein-coding exon annotations...")

    transcripts = defaultdict(
        lambda: {
            "gene_id": None,
            "strand": None,
            "exons": []
        }
    )

    exon_records = 0

    with open(gtf_path, "r") as gtf:
        for line_number, line in enumerate(gtf, start=1):

            if not line or line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) != 9:
                continue

            seqname = fields[0]
            feature = fields[2]

            if seqname != chromosome or feature != "exon":
                continue

            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]

            attributes = parse_attributes(fields[8])

            if attributes.get("gene_type") != "protein_coding":
                continue

            transcript_id = attributes.get("transcript_id")
            gene_id = attributes.get("gene_id")

            if not transcript_id or not gene_id:
                continue

            if strand not in ("+", "-"):
                continue

            record = transcripts[transcript_id]

            if record["gene_id"] is None:
                record["gene_id"] = gene_id
                record["strand"] = strand

            elif (
                record["gene_id"] != gene_id
                or record["strand"] != strand
            ):
                raise ValueError(
                    f"Inconsistent transcript annotation: {transcript_id}"
                )

            record["exons"].append((start, end))
            exon_records += 1

            if line_number % 1_000_000 == 0:
                print(f"  Read {line_number:,} GTF lines...")

    print(f"Protein-coding exon records: {exon_records:,}")
    print(f"Unique transcripts: {len(transcripts):,}")

    return transcripts


def reconstruct_transcripts(genome, transcripts):
    """
    Reconstruct each transcript's genomic span.

    Genomic coordinates in GTF are 1-based and inclusive.
    Python sequence indexing is 0-based and end-exclusive.

    Exon positions receive E.
    Positions between exons receive I.

    Negative-strand sequences and labels are reversed into
    transcription orientation.
    """
    print("\nReconstructing transcript sequences and labels...")

    results = []

    overlap_count = 0
    invalid_count = 0

    for index, (transcript_id, record) in enumerate(
        transcripts.items(), start=1
    ):

        exons = sorted(record["exons"])

        if not exons:
            continue

        genomic_start = min(start for start, end in exons)
        genomic_end = max(end for start, end in exons)

        if genomic_start < 1 or genomic_end > len(genome):
            invalid_count += 1
            continue

        # Convert 1-based inclusive genomic coordinates
        # into Python's 0-based, end-exclusive coordinates.
        sequence = genome[
            genomic_start - 1 : genomic_end
        ]

        labels = bytearray(b"I") * len(sequence)

        previous_end = 0

        for exon_start, exon_end in exons:

            if exon_start < genomic_start or exon_end > genomic_end:
                raise ValueError(
                    f"Invalid exon coordinates: {transcript_id}"
                )

            if exon_start > exon_end:
                raise ValueError(
                    f"Invalid exon interval: {transcript_id}"
                )

            # Position within the transcript's genomic span.
            local_start = exon_start - genomic_start
            local_end = exon_end - genomic_start + 1

            if local_start < previous_end:
                overlap_count += 1

            labels[local_start:local_end] = b"E"
            previous_end = max(previous_end, local_end)

        if len(sequence) != len(labels):
            raise ValueError(
                f"Sequence-label length mismatch: {transcript_id}"
            )

        if record["strand"] == "-":
            sequence = sequence.translate(COMPLEMENT)[::-1]
            labels = labels[::-1]

        exon_bases = labels.count(ord("E"))
        intron_bases = labels.count(ord("I"))

        results.append({
            "transcript_id": transcript_id,
            "gene_id": record["gene_id"],
            "strand": record["strand"],
            "genomic_start": genomic_start,
            "genomic_end": genomic_end,
            "length": len(sequence),
            "exon_bases": exon_bases,
            "intron_bases": intron_bases,
            "sequence": sequence,
            "labels": labels.decode("ascii")
        })

        if index % 1000 == 0:
            print(
                f"  Reconstructed {index:,}/"
                f"{len(transcripts):,} transcripts..."
            )

    print(f"\nReconstructed transcripts: {len(results):,}")
    print(f"Overlapping exon intervals: {overlap_count}")
    print(f"Invalid transcripts skipped: {invalid_count}")

    return results


def write_outputs(results, output_dir):
    """Write compressed FASTA files and transcript metadata CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_path = output_dir / "transcript_sequences.fa.gz"
    labels_path = output_dir / "transcript_labels.fa.gz"
    metadata_path = output_dir / "transcript_metadata.csv"

    print("\nWriting output files...")

    with gzip.open(
        sequence_path, "wt", encoding="ascii"
    ) as sequence_file, gzip.open(
        labels_path, "wt", encoding="ascii"
    ) as label_file:

        for item in results:

            transcript_id = item["transcript_id"]

            sequence_file.write(f">{transcript_id}\n")
            label_file.write(f">{transcript_id}\n")

            sequence = item["sequence"]
            labels = item["labels"]

            for start in range(0, len(sequence), 80):
                sequence_file.write(
                    sequence[start:start + 80] + "\n"
                )
                label_file.write(
                    labels[start:start + 80] + "\n"
                )

    metadata_columns = [
        "TranscriptID",
        "GeneID",
        "Strand",
        "GenomicStart",
        "GenomicEnd",
        "Length",
        "ExonBases",
        "IntronBases"
    ]

    with open(
        metadata_path, "w", newline="", encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=metadata_columns
        )

        writer.writeheader()

        for item in results:
            writer.writerow({
                "TranscriptID": item["transcript_id"],
                "GeneID": item["gene_id"],
                "Strand": item["strand"],
                "GenomicStart": item["genomic_start"],
                "GenomicEnd": item["genomic_end"],
                "Length": item["length"],
                "ExonBases": item["exon_bases"],
                "IntronBases": item["intron_bases"]
            })

    print("\nOutput files:")
    print(f"  {sequence_path}")
    print(f"  {labels_path}")
    print(f"  {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare protein-coding transcript sequences."
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to GRCh38 genome FASTA"
    )

    parser.add_argument(
        "--gtf",
        required=True,
        help="Path to GENCODE annotation GTF"
    )

    parser.add_argument(
        "--output",
        default="prepared_dataset",
        help="Output directory"
    )

    parser.add_argument(
        "--expected-transcripts",
        type=int,
        default=EXPECTED_TRANSCRIPTS,
        help="Expected transcript count for validation"
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    gtf_path = Path(args.gtf)
    output_dir = Path(args.output)

    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    if not gtf_path.is_file():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    genome = load_chromosome_sequence(
        fasta_path, CHROMOSOME
    )

    transcripts = extract_protein_coding_exons(
        gtf_path, CHROMOSOME
    )

    if len(transcripts) != args.expected_transcripts:
        raise RuntimeError(
            "\nTranscript count does not match the expected value.\n"
            f"Expected: {args.expected_transcripts:,}\n"
            f"Observed: {len(transcripts):,}\n"
            "Check the genome assembly, GENCODE release, "
            "chromosome name, and annotation filter."
        )

    results = reconstruct_transcripts(
        genome, transcripts
    )

    if len(results) != args.expected_transcripts:
        raise RuntimeError(
            "Some transcripts were skipped during reconstruction. "
            "Output files were not written."
        )

    write_outputs(results, output_dir)

    total_bases = sum(x["length"] for x in results)
    total_exon = sum(x["exon_bases"] for x in results)
    total_intron = sum(x["intron_bases"] for x in results)

    print("\n" + "=" * 48)
    print("TRANSCRIPT PREPARATION SUMMARY")
    print("=" * 48)
    print(f"Transcripts: {len(results):,}")
    print(f"Total bases: {total_bases:,}")
    print(f"Exon bases: {total_exon:,}")
    print(f"Intron bases: {total_intron:,}")
    print(
        f"Exon percentage: "
        f"{100 * total_exon / total_bases:.4f}%"
    )
    print(
        f"Intron percentage: "
        f"{100 * total_intron / total_bases:.4f}%"
    )
    print("=" * 48)
    print("Stage 1 completed successfully.")


if __name__ == "__main__":
    main()
