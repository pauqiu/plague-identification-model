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


fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.08)


def show_current():
    row = candidates.iloc[position]

    for ax, column, label in zip(
        axes,
        ["image_a", "image_b"],
        ["Imagen A", "Imagen B"],
    ):
        ax.clear()
        ax.axis("off")

        path = resolve_path(row[column])

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1200, 1200))
                ax.imshow(image.copy())

            ax.set_title(f"{label}\n{path.name}", fontsize=10)

        except Exception as error:
            ax.set_title(label)
            ax.text(
                0.5,
                0.5,
                f"No se pudo abrir:\n{path}\n\n{error}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )

    decision = row.get("decision", "") or "Sin registrar"

    fig.suptitle(
        f"Par {position + 1} de {len(candidates)} "
        f"| Índice: {position}\n"
        f"pHash distance: {row['phash_distance']} "
        f"| Decisión: {decision}",
        fontsize=12,
    )

    footer.set_text(
        f"A: {row['image_a']}\n"
        f"B: {row['image_b']}\n\n"
        "→ / Espacio: siguiente    ←: anterior    Esc: salir"
    )

    fig.canvas.draw_idle()


def on_key(event):
    global position

    if event.key in ("right", " ", "space"):
        position = min(position + 1, len(candidates) - 1)
        show_current()

    elif event.key == "left":
        position = max(position - 1, 0)
        show_current()

    elif event.key == "escape":
        plt.close(fig)


footer = fig.text(0.02, 0.02, "", fontsize=8, va="bottom")
fig.canvas.mpl_connect("key_press_event", on_key)

# Evita que las flechas activen también la navegación de Matplotlib.
for shortcut in ("keymap.back", "keymap.forward"):
    plt.rcParams[shortcut] = []

show_current()
plt.show()