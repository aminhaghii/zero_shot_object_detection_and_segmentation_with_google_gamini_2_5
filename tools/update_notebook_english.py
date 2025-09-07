import json, os, re

REPO_OWNER = "aminhaghii"
REPO_NAME = "zero_shot_object_detection_and_segmentation_with_google_gamini_2_5"
NB_NAME = "zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb"

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(repo_dir, NB_NAME)

assert os.path.exists(nb_path), f"Notebook not found: {nb_path}"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def contains_persian(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", s or ""))

# Build badges
colab_url = (
    f"https://colab.research.google.com/github/{REPO_OWNER}/{REPO_NAME}/blob/main/{NB_NAME}"
)
kaggle_url = (
    f"https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{NB_NAME}"
)

badge_md = (
    "# ZeroShot Object Detection & Segmentation with Google Gemini\n\n"
    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n"
    f"[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)]({kaggle_url})\n\n"
    "Note: This notebook is English-only. Set your GEMINI_API_KEY as an environment variable before running.\n"
)

# Ensure cells list exists
cells = nb.get("cells", [])

# Insert a top markdown cell with badges if not already present
if not cells or (cells[0].get("cell_type") != "markdown" or "Open In Colab" not in "".join(cells[0].get("source", []))):
    cells.insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [badge_md]
    })

# Replace Persian-only markdown/raw cells with English placeholder
for cell in cells:
    ctype = cell.get("cell_type")
    if ctype in ("markdown", "raw"):
        src = cell.get("source", [])
        if isinstance(src, str):
            src_lines = src.splitlines(keepends=True)
        else:
            src_lines = src
        # If most lines contain Persian characters, replace the whole cell
        if src_lines and sum(1 for ln in src_lines if contains_persian(ln)) >= max(1, len(src_lines)//2):
            cell["source"] = [
                "This section has been converted to English-only content. Please refer to the README for full instructions and workflow.\n"
            ]
        else:
            # Otherwise, only strip Persian lines
            new_lines = []
            replaced = False
            for ln in src_lines:
                if contains_persian(ln):
                    if not replaced:
                        new_lines.append("[Text removed: non-English content]\n")
                        replaced = True
                else:
                    new_lines.append(ln)
            cell["source"] = new_lines
    elif ctype == "code":
        # Avoid breaking code; leave as-is
        pass

nb["cells"] = cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"Notebook updated in English and badges added: {nb_path}")