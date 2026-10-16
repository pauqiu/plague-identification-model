from pathlib import Path, PurePosixPath

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def find_project_root():
    start = (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else Path.cwd().resolve()
    )

    for folder in (start, *start.parents):
        csv_path = folder / "reports" / "similar_candidates.csv"

        if csv_path.is_file():
            return folder

    raise FileNotFoundError(
        "No se encontró reports/similar_candidates.csv "
        f"en {start} ni en sus carpetas superiores."
    )


PROJECT_ROOT = find_project_root()
CSV_PATH = PROJECT_ROOT / "reports" / "similar_candidates.csv"

# Raíz original de las imágenes en Colab.
OLD_ROOT = PurePosixPath("/content/plague-identification-model")

START_AT = 145

candidates = pd.read_csv(CSV_PATH).fillna("")

required = {"image_a", "image_b", "phash_distance"}
missing = required - set(candidates.columns)

if missing:
    raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

if candidates.empty:
    raise ValueError("El CSV no contiene candidatos.")

position = max(0, min(START_AT, len(candidates) - 1))


def resolve_path(value):
    original = str(value).strip().replace("\\", "/")

    if original.startswith("/content/"):
        colab_path = PurePosixPath(original)

        try:
            relative = colab_path.relative_to(OLD_ROOT)
        except ValueError as error:
            raise ValueError(
                f"La ruta del CSV no comienza con {OLD_ROOT}: "
                f"{original}"
            ) from error

        return PROJECT_ROOT.joinpath(*relative.parts)

    path = Path(original)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path