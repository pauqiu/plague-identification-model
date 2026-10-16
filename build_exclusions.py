"""Arma reports/exclusions.csv a partir de los reportes de duplicados.

Une las copias exactas y las copias perceptuales confirmadas en una fila
por imagen. Las rutas quedan relativas a data/raw, como en metadata.csv.
"""

from pathlib import Path

import pandas as pd


def find_project_root():
    start = Path(__file__).resolve().parent

    for folder in (start, *start.parents):
        reports = folder / "reports"
        needed = (
            reports / "excluded_exact_duplicates.csv",
            reports / "similar_candidates.csv",
            folder / "data" / "metadata.csv",
        )
        if all(path.is_file() for path in needed):
            return folder

    raise FileNotFoundError(
        "No se encontraron los reportes de duplicados y metadata.csv "
        f"a partir de {start}."
    )


def to_relative(value):
    text = str(value).replace("\\", "/")
    marker = "data/raw/"
    index = text.find(marker)
    if index < 0:
        raise ValueError(f"La ruta no contiene data/raw: {value}")
    return text[index:]


def main():
    root = find_project_root()
    reports = root / "reports"

    metadata = pd.read_csv(root / "data" / "metadata.csv")
    metadata["path"] = metadata["path"].map(to_relative)

    exact = pd.read_csv(reports / "excluded_exact_duplicates.csv")
    exact["path"] = exact["image_path"].map(to_relative)
    exact["keep"] = exact["keep_path"].map(to_relative)

    similar = pd.read_csv(reports / "similar_candidates.csv")
    confirmed = similar[similar["decision"] == "duplicate_confirmed"].copy()
    confirmed["path"] = confirmed["exclude_path"].map(to_relative)
    confirmed["image_a"] = confirmed["image_a"].map(to_relative)
    confirmed["image_b"] = confirmed["image_b"].map(to_relative)
    confirmed["partner"] = confirmed.apply(
        lambda row: row.image_a if row.path == row.image_b else row.image_b,
        axis=1,
    )

    rows = []

    for row in exact.itertuples(index=False):
        rows.append(
            {
                "path": row.path,
                "reason": "exact_duplicate",
                "related_paths": row.keep,
                "n_pairs": 1,
                "phash_distance_min": "",
                "sha256": row.sha256,
            }
        )

    for path, group in confirmed.groupby("path"):
        rows.append(
            {
                "path": path,
                "reason": "confirmed_perceptual_duplicate",
                "related_paths": ";".join(sorted(group["partner"].unique())),
                "n_pairs": int(len(group)),
                "phash_distance_min": int(group["phash_distance"].min()),
                "sha256": "",
            }
        )

    exclusions = pd.DataFrame(rows)
    labels = metadata.set_index("path")[["source", "crop", "final_label"]]
    exclusions = exclusions.join(labels, on="path")
    exclusions = exclusions.sort_values(["reason", "path"]).reset_index(drop=True)

    missing = [
        path
        for path in exclusions["path"]
        if path not in set(metadata["path"]) or not (root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Hay rutas excluidas que no están en el inventario o en disco: "
            + ", ".join(missing)
        )

    overlap = set(exact["path"]) & set(confirmed["path"])
    if overlap:
        raise ValueError(
            "Una misma ruta está en copias exactas y en copias perceptuales: "
            + ", ".join(sorted(overlap))
        )

    destination = reports / "exclusions.csv"
    exclusions.to_csv(destination, index=False)
    print(f"Filas escritas: {len(exclusions)}")
    print(destination)


if __name__ == "__main__":
    main()
