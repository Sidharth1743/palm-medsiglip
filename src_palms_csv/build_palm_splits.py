from __future__ import annotations

import csv
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "Anemia Detection using Palpable Palm Image Datasets from Ghana"
    / "Palm"
)
OUTPUT_DIR = PROJECT_ROOT / "src" / "palms"


def label_from_name(name: str) -> int | None:
    lower = name.lower()
    if lower.startswith("anemic"):
        return 1
    if lower.startswith("non-anemic"):
        return 0
    if lower.startswith("non anemic"):
        return 0
    return None


def subject_id_from_name(name: str) -> str:
    stem = Path(name).stem
    pattern = re.compile(
        r"^(anemic|non[- ]?anemic)[-_]?(?P<sid>[^ (]+)", re.IGNORECASE
    )
    match = pattern.match(stem)
    if match:
        prefix = match.group(1).lower().replace(" ", "-")
        return f"{prefix}:{match.group('sid')}"
    # Fallback: first token before space or '(' if prefix is unexpected
    for sep in (" (", "(", " "):
        if sep in stem:
            return stem.split(sep, 1)[0].lower()
    return stem.lower()


def write_csv(rows: list[tuple[Path, int]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for path, label in rows:
            writer.writerow([str(path), label])


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Dataset not found: {DATA_DIR}")

    rows: list[tuple[Path, int, str]] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        label = label_from_name(path.name)
        if label is None:
            continue
        subject_id = subject_id_from_name(path.name)
        rows.append((path, label, subject_id))

    if not rows:
        raise SystemExit("No labeled palm images found.")

    random.seed(42)
    random.shuffle(rows)

    # Stratified split
    anemia = [r for r in rows if r[1] == 1]
    non = [r for r in rows if r[1] == 0]

    def group_by_subject(items: list[tuple[Path, int, str]]):
        subjects: dict[str, list[tuple[Path, int]]] = {}
        for path, label, sid in items:
            subjects.setdefault(sid, []).append((path, label))
        subject_ids = list(subjects.keys())
        return subjects, subject_ids

    def split_subjects(subject_ids: list[str]):
        n = len(subject_ids)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train = subject_ids[:n_train]
        val = subject_ids[n_train : n_train + n_val]
        test = subject_ids[n_train + n_val :]
        return train, val, test

    anemia_groups, anemia_subjects = group_by_subject(anemia)
    non_groups, non_subjects = group_by_subject(non)

    random.shuffle(anemia_subjects)
    random.shuffle(non_subjects)

    a_train_ids, a_val_ids, a_test_ids = split_subjects(anemia_subjects)
    n_train_ids, n_val_ids, n_test_ids = split_subjects(non_subjects)

    def expand_subjects(subject_ids: list[str], groups: dict[str, list[tuple[Path, int]]]):
        rows_out: list[tuple[Path, int]] = []
        for sid in subject_ids:
            rows_out.extend(groups[sid])
        return rows_out

    a_train = expand_subjects(a_train_ids, anemia_groups)
    a_val = expand_subjects(a_val_ids, anemia_groups)
    a_test = expand_subjects(a_test_ids, anemia_groups)

    n_train = expand_subjects(n_train_ids, non_groups)
    n_val = expand_subjects(n_val_ids, non_groups)
    n_test = expand_subjects(n_test_ids, non_groups)

    train = a_train + n_train
    val = a_val + n_val
    test = a_test + n_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(train, OUTPUT_DIR / "train.csv")
    write_csv(val, OUTPUT_DIR / "val.csv")
    write_csv(test, OUTPUT_DIR / "test.csv")
    write_csv([(p, l) for p, l, _ in rows], OUTPUT_DIR / "all.csv")

    print("Palms subject-level split complete:")
    print(f"  total={len(rows)}")
    print(f"  train={len(train)} val={len(val)} test={len(test)}")
    print(
        f"  subjects: train={len(a_train_ids) + len(n_train_ids)} "
        f"val={len(a_val_ids) + len(n_val_ids)} test={len(a_test_ids) + len(n_test_ids)}"
    )
    print(
        f"  anemia: train={len(a_train)} val={len(a_val)} test={len(a_test)} total={len(anemia)}"
    )
    print(
        f"  non-anemia: train={len(n_train)} val={len(n_val)} test={len(n_test)} total={len(non)}"
    )


if __name__ == "__main__":
    main()
