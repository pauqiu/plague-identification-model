"""Asigna train, validation y test, y comprueba que no haya fuga.

Parte de metadata.csv, quita reports/exclusions.csv y reparte cada etiqueta
final en 70/15/15. La semilla es 42. Escribe reports/split.csv con el SHA-256
de cada archivo conservado.
"""

import hashlib
import random
from pathlib import Path

import pandas as pd

SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def find_project_root():
    start = Path(__file__).resolve().parent

    for folder in (start, *start.parents):
        needed = (
            folder / "data" / "metadata.csv",
            folder / "reports" / "exclusions.csv",
        )
        if all(path.is_file() for path in needed):
            return folder

    raise FileNotFoundError(
        "No se encontraron metadata.csv y reports/exclusions.csv "
        f"a partir de {start}."
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sizes(count):
    validation = int(round(count * VALIDATION_FRACTION))
    test = int(round(count * TEST_FRACTION))
    train = count - validation - test
    if min(train, validation, test) <= 0:
        raise ValueError(f"Una partición quedó vacía para {count} imágenes.")
    return train, validation, test


def assign_split(paths):
    ordered = sorted(paths)
    random.Random(SEED).shuffle(ordered)
    train_count, validation_count, _ = split_sizes(len(ordered))
    assigned = {}
    for index, path in enumerate(ordered):
        if index < train_count:
            assigned[path] = "train"
        elif index < train_count + validation_count:
            assigned[path] = "validation"
        else:
            assigned[path] = "test"
    return assigned


def main():
    root = find_project_root()
    metadata = pd.read_csv(root / "data" / "metadata.csv")
    exclusions = set(pd.read_csv(root / "reports" / "exclusions.csv")["path"])

    kept = metadata[~metadata["path"].isin(exclusions)].copy()
    if len(kept) + len(exclusions) != len(metadata):
        missing = exclusions - set(metadata["path"])
        raise ValueError(f"Exclusiones que no están en el inventario: {sorted(missing)}")

    assignment = {}
    for _, group in kept.groupby("final_label"):
        assignment.update(assign_split(group["path"]))

    kept["split"] = kept["path"].map(assignment)
    kept["sha256"] = [
        file_sha256(root / path) for path in kept["path"]
    ]

    if kept["path"].duplicated().any():
        raise ValueError("Hay rutas repetidas dentro de la asignación.")
    if set(kept["split"]) != {"train", "validation", "test"}:
        raise ValueError("Falta alguna partición.")
    if kept["sha256"].duplicated().any():
        raise ValueError("Hay un SHA-256 repetido dentro del conjunto conservado.")

    shared = (
        kept.groupby("sha256")["split"].nunique()
    )
    if (shared > 1).any():
        raise ValueError("Un mismo SHA-256 aparece en más de una partición.")

    destination = root / "reports" / "split.csv"
    kept[
        ["path", "split", "final_label", "source", "crop", "sha256"]
    ].sort_values(["split", "final_label", "path"]).to_csv(destination, index=False)

    counts = (
        kept.groupby(["final_label", "source", "split"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["final_label", "source", "split"])
    )
    counts_path = root / "reports" / "split_counts.csv"
    counts.to_csv(counts_path, index=False)

    print(f"Semilla: {SEED}")
    print(f"Imágenes: {len(kept)}")
    print(kept["split"].value_counts().to_string())
    print("--- etiqueta x partición ---")
    print(pd.crosstab(kept["final_label"], kept["split"], margins=True).to_string())
    print("--- fuente x partición ---")
    print(pd.crosstab(kept["source"], kept["split"], margins=True).to_string())
    print(destination)
    print(counts_path)


if __name__ == "__main__":
    main()
