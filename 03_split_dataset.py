
#!/usr/bin/env python3

"""
Stage 3: Gene-level train/validation/test splitting.

Inputs:
    transcript_metadata.csv
    window_metadata.csv

Output:
    partitioned_window_metadata.csv
    gene_overlap_groups.csv

Method:
    1. Calculate genomic span of each gene.
    2. Merge overlapping gene spans into connected groups.
    3. Sort groups by descending window count.
    4. Assign each group to train/validation/test using
       the greedy target-ratio balancing method.

No random window-level splitting is performed.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TARGET_RATIOS = {
    "Training": 0.70,
    "Validation": 0.15,
    "Testing": 0.15
}

EXPECTED_GENES = 434
EXPECTED_GROUPS = 339
EXPECTED_WINDOWS = 283174


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_gene_spans(transcript_rows):
    """Calculate one genomic span per gene."""

    spans = {}

    for row in transcript_rows:
        gene_id = row["GeneID"]
        start = int(row["GenomicStart"])
        end = int(row["GenomicEnd"])

        if gene_id not in spans:
            spans[gene_id] = {
                "GeneID": gene_id,
                "Start": start,
                "End": end
            }
        else:
            spans[gene_id]["Start"] = min(
                spans[gene_id]["Start"], start
            )
            spans[gene_id]["End"] = max(
                spans[gene_id]["End"], end
            )

    # Python sorting is stable, preserving original order for ties.
    return sorted(
        spans.values(),
        key=lambda x: x["Start"]
    )


def create_overlap_groups(gene_spans):
    """
    Merge overlapping gene spans into connected components.

    A new group begins only when:
        next_gene_start > current_group_end

    Strand is intentionally ignored.
    """

    groups = []
    current_group = []
    current_end = None

    for gene in gene_spans:

        if not current_group:
            current_group = [gene]
            current_end = gene["End"]

        elif gene["Start"] > current_end:
            groups.append(current_group)

            current_group = [gene]
            current_end = gene["End"]

        else:
            current_group.append(gene)
            current_end = max(current_end, gene["End"])

    if current_group:
        groups.append(current_group)

    return groups


def count_windows_per_gene(window_rows):
    """Count the number of windows belonging to each gene."""

    counts = defaultdict(int)

    for row in window_rows:
        counts[row["GeneID"]] += 1

    return counts


def assign_groups(groups, gene_window_counts, total_windows):
    """
    Reproduce the MATLAB greedy group assignment.

    Groups are processed from largest to smallest window count.

    For each group, choose the partition minimizing:

        actual_assigned_windows / target_windows

    Ties are resolved in this order:
        Training, Validation, Testing
    """

    partition_names = [
        "Training",
        "Validation",
        "Testing"
    ]

    target_counts = {
        name: TARGET_RATIOS[name] * total_windows
        for name in partition_names
    }

    actual_counts = {
        name: 0
        for name in partition_names
    }

    # Calculate the number of windows in each overlap group.
    group_records = []

    for group_id, genes in enumerate(groups, start=1):

        group_windows = sum(
            gene_window_counts[gene["GeneID"]]
            for gene in genes
        )

        group_records.append({
            "OverlapGroup": group_id,
            "Genes": genes,
            "WindowCount": group_windows
        })

    # Stable descending sort.
    group_records.sort(
        key=lambda x: x["WindowCount"],
        reverse=True
    )

    group_assignments = {}

    for group in group_records:

        ratios = {
            name: actual_counts[name] / target_counts[name]
            for name in partition_names
        }

        selected_partition = min(
            partition_names,
            key=lambda name: ratios[name]
        )

        group_id = group["OverlapGroup"]

        group_assignments[group_id] = selected_partition

        actual_counts[selected_partition] += group["WindowCount"]

    return group_records, group_assignments, actual_counts


def main():

    parser = argparse.ArgumentParser(
        description="Create gene-overlap-aware dataset partitions."
    )

    parser.add_argument(
        "--transcripts",
        required=True,
        help="Path to transcript_metadata.csv"
    )

    parser.add_argument(
        "--windows",
        required=True,
        help="Path to window_metadata.csv"
    )

    parser.add_argument(
        "--output",
        default="partitioned_dataset"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading transcript metadata...")
    transcript_rows = read_csv(args.transcripts)

    print("Loading window metadata...")
    window_rows = read_csv(args.windows)

    if len(window_rows) != EXPECTED_WINDOWS:
        raise ValueError(
            f"Expected {EXPECTED_WINDOWS:,} windows, "
            f"found {len(window_rows):,}."
        )

    # Step 1: Build gene genomic spans.
    print("\nBuilding gene genomic spans...")

    gene_spans = build_gene_spans(transcript_rows)

    print(f"Unique genes: {len(gene_spans):,}")

    if len(gene_spans) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} genes, "
            f"found {len(gene_spans)}."
        )

    # Step 2: Create overlap-connected groups.
    print("\nCreating genomic overlap groups...")

    groups = create_overlap_groups(gene_spans)

    print(f"Overlap-connected groups: {len(groups):,}")

    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_GROUPS} groups, "
            f"found {len(groups)}."
        )

    # Step 3: Count windows per gene.
    gene_window_counts = count_windows_per_gene(window_rows)

    for gene in gene_spans:
        if gene["GeneID"] not in gene_window_counts:
            raise ValueError(
                f"No windows found for gene {gene['GeneID']}."
            )

    # Step 4: Assign groups to partitions.
    print("\nAssigning overlap groups to partitions...")

    group_records, group_assignments, actual_counts = (
        assign_groups(
            groups,
            gene_window_counts,
            len(window_rows)
        )
    )

    # Map every gene to its overlap group and partition.
    gene_assignments = {}

    for group in group_records:

        group_id = group["OverlapGroup"]
        partition = group_assignments[group_id]

        for gene in group["Genes"]:
            gene_assignments[gene["GeneID"]] = {
                "OverlapGroup": group_id,
                "Split": partition
            }

    # Step 5: Add partition information to every window.
    print("\nAssigning windows...")

    output_columns = list(window_rows[0].keys()) + [
        "OverlapGroup",
        "Split"
    ]

    output_path = (
        output_dir / "partitioned_window_metadata.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=output_columns
        )

        writer.writeheader()

        for row in window_rows:

            gene_id = row["GeneID"]

            if gene_id not in gene_assignments:
                raise ValueError(
                    f"Gene missing from grouping: {gene_id}"
                )

            assignment = gene_assignments[gene_id]

            writer.writerow({
                **row,
                "OverlapGroup": assignment["OverlapGroup"],
                "Split": assignment["Split"]
            })

    # Step 6: Save gene/group information.
    group_summary_path = (
        output_dir / "gene_overlap_groups.csv"
    )

    with open(
        group_summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        columns = [
            "GeneID",
            "GenomicStart",
            "GenomicEnd",
            "OverlapGroup",
            "Split",
            "WindowCount"
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns
        )

        writer.writeheader()

        for gene in gene_spans:

            gene_id = gene["GeneID"]
            assignment = gene_assignments[gene_id]

            writer.writerow({
                "GeneID": gene_id,
                "GenomicStart": gene["Start"],
                "GenomicEnd": gene["End"],
                "OverlapGroup": assignment["OverlapGroup"],
                "Split": assignment["Split"],
                "WindowCount": gene_window_counts[gene_id]
            })

    # Step 7: Verify no group or gene crosses partitions.
    group_splits = defaultdict(set)
    gene_splits = defaultdict(set)
    split_genes = defaultdict(set)

    for row in window_rows:
        gene_id = row["GeneID"]
        assignment = gene_assignments[gene_id]

        group_splits[assignment["OverlapGroup"]].add(
            assignment["Split"]
        )

        gene_splits[gene_id].add(assignment["Split"])

        split_genes[assignment["Split"]].add(gene_id)

    if any(len(x) != 1 for x in group_splits.values()):
        raise RuntimeError("Overlap-group leakage detected.")

    if any(len(x) != 1 for x in gene_splits.values()):
        raise RuntimeError("Gene leakage detected.")

    total_assigned = sum(actual_counts.values())

    if total_assigned != len(window_rows):
        raise RuntimeError("Window count mismatch.")

    # Final report.
    print("\n" + "=" * 52)
    print("DATASET SPLIT SUMMARY")
    print("=" * 52)

    for name in ["Training", "Validation", "Testing"]:

        count = actual_counts[name]
        percentage = 100 * count / total_assigned

        print(
            f"{name:12s}: {count:>9,} windows "
            f"({percentage:.2f}%), "
            f"{len(split_genes[name]):,} genes"
        )

    print("-" * 52)
    print(f"Total windows: {total_assigned:,}")
    print(f"Total genes: {len(gene_spans):,}")
    print(f"Overlap groups: {len(groups):,}")
    print("Gene leakage: NONE")
    print("Overlap-group leakage: NONE")
    print("=" * 52)

    print("\nOutput files:")
    print(f"  {output_path}")
    print(f"  {group_summary_path}")

    print("\nStage 3 completed successfully.")


if __name__ == "__main__":
    main()
