"""
Dataset access layer (Section 3.4 of the methodology), scoped to
exactly the collections downloaded for this study:

  Collection "youtube": original_sequences/youtube  (real, 1000 clips)
                         + manipulated_sequences/Deepfakes (fake, 1000 clips)
  Collection "actors":  original_sequences/actors   (real, 363 clips)
                         + manipulated_sequences/DeepFakeDetection (fake)

Both are distributed through the same FaceForensics++ download script
and licence (Roessler et al., 2019). "actors" originals come from
Google/Jigsaw's DeepFakeDetection (DFD) release; "youtube" originals
are the original FaceForensics++ pristine videos. They use different
filename conventions, so samples are identity-matched only *within* a
collection, never across the two (see `_match_identity`).

DEFAULT_ROOT_DIR below points at this machine's actual local copy, so
`load_faceforensics()` works with no arguments. Override it (pass
`root_dir=...` explicitly) if you move the data or run this on a
different machine.

Request access: https://github.com/ondyari/FaceForensics

If you later add more methods paired with "youtube" (Face2Face,
FaceSwap, NeuralTextures), just add them to
`COLLECTIONS["youtube"]["methods"]` below; nothing else in this file
needs to change.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

Label = Literal["real", "fake"]
Compression = Literal["raw", "c23", "c40"]

# This machine's local copy, populated by download-FaceForensics.py.
DEFAULT_ROOT_DIR = "/Users/ronaldkato/Downloads/Annet Research/FaceForensics++_data"

# Which manipulation methods belong to which original-video collection.
# An empty "methods" list means: load this collection's real videos only.
COLLECTIONS = {
    "youtube": {
        "original": ("original_sequences", "youtube"),
        "methods": ["Deepfakes"],  # add "Face2Face", "FaceSwap", "NeuralTextures" here if downloaded later
    },
    "actors": {
        "original": ("original_sequences", "actors"),
        "methods": ["DeepFakeDetection"],
    },
}

# Flat list of every manipulation method in scope, derived from COLLECTIONS
# so it can never drift out of sync with the mapping above.
FF_METHODS = [m for c in COLLECTIONS.values() for m in c["methods"]]


@dataclass
class Sample:
    path: str
    label: Label
    collection: str  # "youtube" or "actors" -- never mixed together during splitting
    dataset: str = "FaceForensics++"
    manipulation_method: str = "pristine"
    identity: str = ""  # filename stem of the matching *original* video; see load_faceforensics


def _list_videos(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted(f for f in os.listdir(dir_path) if f.lower().endswith((".mp4", ".avi", ".mov")))


def load_faceforensics(
    root_dir: str = DEFAULT_ROOT_DIR,
    compression: Compression = "c23",
    collections: Optional[Sequence[str]] = None,
) -> List[Sample]:
    """Load real (pristine) and fake (manipulated) samples from a local
    FaceForensics++/DeepFakeDetection download, restricted to the
    collections and methods declared in COLLECTIONS above.

    Args:
        root_dir: path to the root of the extracted download (the folder
            containing `original_sequences/` and `manipulated_sequences/`).
            Defaults to DEFAULT_ROOT_DIR (this machine's local copy).
        compression: "raw", "c23" (light compression), or "c40" (heavy
            compression) -- c23 is used by default as it best matches
            media re-shared on social platforms (Section 3.5).
        collections: which entries of COLLECTIONS to load, e.g.
            `("actors",)` to load only the actors/DeepFakeDetection pair
            and skip youtube/Deepfakes entirely (useful when only a
            subset of the full FaceForensics++ download is actually
            present on disk, or when a study is deliberately scoped to
            one collection). Defaults to every collection in COLLECTIONS.
            Unlike editing COLLECTIONS directly, this doesn't change
            `FF_METHODS` or `leave_one_method_out_split`'s notion of
            what's globally in scope -- use `methods_present()` on the
            returned samples if you need the methods actually loaded.

    Each fake sample is tagged with the filename stem of the pristine
    original it was matched against (`identity`), determined by prefix
    matching against the real filenames in the same collection (see
    `_match_identity`) rather than by parsing a manipulated filename's
    internal structure -- this avoids assuming a specific separator
    convention, which differs between the "youtube" and "actors"
    collections.
    """
    if collections is None:
        selected = dict(COLLECTIONS)
    else:
        unknown = [c for c in collections if c not in COLLECTIONS]
        if unknown:
            raise ValueError(
                f"Unknown collection(s) {unknown}; valid options are {list(COLLECTIONS)}."
            )
        selected = {name: COLLECTIONS[name] for name in collections}

    samples: List[Sample] = []

    for collection_name, spec in selected.items():
        orig_subdir, orig_name = spec["original"]
        real_dir = os.path.join(root_dir, orig_subdir, orig_name, compression, "videos")
        real_files = _list_videos(real_dir)
        real_identities = [_extract_leading_id(os.path.splitext(f)[0]) for f in real_files]

        if not real_files:
            continue  # this collection isn't downloaded (yet) -- skip silently

        for fname, identity in zip(real_files, real_identities):
            samples.append(Sample(
                path=os.path.join(real_dir, fname), label="real",
                collection=collection_name, identity=identity,
            ))

        for method in spec["methods"]:
            fake_dir = os.path.join(root_dir, "manipulated_sequences", method, compression, "videos")
            fake_files = _list_videos(fake_dir)
            unmatched = 0
            for fname in fake_files:
                identity = _match_identity(fname, real_identities)
                if identity is None:
                    unmatched += 1
                    identity = os.path.splitext(fname)[0]  # fall back to its own stem; isolated in its own split bucket
                samples.append(Sample(
                    path=os.path.join(fake_dir, fname), label="fake",
                    collection=collection_name, manipulation_method=method, identity=identity,
                ))
            if unmatched:
                warnings.warn(
                    f"{unmatched}/{len(fake_files)} files in {method!r} did not match any "
                    f"original identity in the {collection_name!r} collection by filename "
                    f"prefix; they were each treated as their own identity, which is safe "
                    f"for split integrity but means they won't be grouped with their real "
                    f"counterpart. Inspect a few filenames if this count looks high."
                )

    return samples


def _extract_leading_id(stem: str) -> str:
    """Leading run of digits at the start of a filename stem, used as the
    identity-grouping key: "000" -> "000", "000_007" -> "000",
    "04__walk_down_hall_angry" -> "04", "04_12__scene__HASH" -> "04"
    (the first of the two actor IDs DeepFakeDetection's naming
    convention encodes for an identity-swap pair). Grouping consistently
    on the leading digits keeps every clip belonging to a given actor in
    one identity bucket regardless of collection-specific naming
    conventions, which is the whole point of identity-stratified
    splitting (Section 3.4) -- falls back to the full stem if it
    doesn't start with digits, so loading never crashes on an
    unexpected filename, it just won't group that one usefully.
    """
    match = re.match(r"\d+", stem)
    return match.group(0) if match else stem


def _match_identity(fake_filename: str, real_identities: List[str]) -> str | None:
    """Match a manipulated filename to the real identity it was derived
    from, via the leading numeric ID shared by both filename
    conventions currently in scope: "000_007.mp4" -> "000" (youtube/
    Deepfakes) and "04_12__walking_..._HASH.mp4" -> "04" (actors/
    DeepFakeDetection). `real_identities` is expected to already be
    leading-ID form (see `_extract_leading_id`, applied to every real
    filename in `load_faceforensics` before this is called) -- matching
    on extracted IDs rather than substring-matching full stems is what
    makes this work for "actors", where the real filename's descriptive
    suffix ("04__walk_down_hall_angry") never appears verbatim in the
    fake filename's own (different) description, so no full-stem
    prefix would ever match.
    """
    leading_id = _extract_leading_id(os.path.splitext(fake_filename)[0])
    return leading_id if leading_id in real_identities else None


def stratified_identity_split(
    samples: List[Sample], train: float = 0.70, val: float = 0.15, seed: int = 42
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """70/15/15 split, stratified so that (a) every clip sharing a source
    identity -- the pristine clip and every manipulated version matched
    to it via `_match_identity` -- lands in the same split, and (b) the
    split is computed independently per collection ("youtube" vs
    "actors") so the two never share identity space and both are
    proportionally represented in every split (Section 3.4).
    """
    import random

    rng = random.Random(seed)
    train_set, val_set, test_set = [], [], []

    for collection_name in COLLECTIONS:
        collection_samples = [s for s in samples if s.collection == collection_name]
        if not collection_samples:
            continue

        by_identity: Dict[str, List[Sample]] = {}
        for s in collection_samples:
            by_identity.setdefault(s.identity, []).append(s)

        identities = list(by_identity.keys())
        rng.shuffle(identities)

        n = len(identities)
        n_train = int(n * train)
        n_val = int(n * val)
        train_ids = set(identities[:n_train])
        val_ids = set(identities[n_train:n_train + n_val])

        for identity, items in by_identity.items():
            bucket = train_set if identity in train_ids else (val_set if identity in val_ids else test_set)
            bucket.extend(items)

    return train_set, val_set, test_set


def methods_present(samples: List[Sample]) -> List[str]:
    """Manipulation methods actually present in a given sample list
    (excludes "pristine"), sorted for deterministic output. Use this --
    not the global `FF_METHODS` -- to decide whether cross-manipulation
    (leave-one-method-out) evaluation is meaningful for a particular
    `load_faceforensics(collections=...)` call, since `FF_METHODS` always
    reflects every method declared in COLLECTIONS regardless of which
    collections were actually loaded.
    """
    return sorted({s.manipulation_method for s in samples if s.manipulation_method != "pristine"})


def leave_one_method_out_split(
    samples: List[Sample], held_out_method: str
) -> Tuple[List[Sample], List[Sample]]:
    """Cross-manipulation generalisation split (Section 3.11): train on
    every method except `held_out_method` (plus all pristine footage),
    evaluate on pristine + the held-out method.

    Requires at least two manipulation methods to be present in `samples`
    -- with only one, holding it out would leave zero fake samples to
    train on, which isn't a meaningful generalisation test. In that
    situation, rely on the in-distribution test split from
    `stratified_identity_split` instead.
    """
    available = methods_present(samples)
    if len(available) < 2:
        raise ValueError(
            f"leave_one_method_out_split needs at least 2 manipulation methods present in "
            f"`samples` to be meaningful, but only {available} is present. Use the "
            f"in-distribution test split instead for now."
        )
    if held_out_method not in available:
        raise ValueError(f"held_out_method must be one of {available}, got {held_out_method!r}")

    train_set = [s for s in samples if s.manipulation_method != held_out_method]
    held_out_set = [s for s in samples if s.manipulation_method in ("pristine", held_out_method)]
    return train_set, held_out_set


def sample_for_quick_run(samples: List[Sample], max_total: int = 80, seed: int = 42) -> List[Sample]:
    """Stratified subsample for a fast, exploratory run (e.g. under 100
    videos total), instead of processing the entire downloaded dataset.

    Rationale: face detection + caching is the slow part of this
    pipeline (each video needs its frames decoded and run through
    MTCNN), so runtime scales almost linearly with video count. Running
    on the full dataset before confirming the pipeline works end to end
    is how a first run turns into many hours; this keeps a run fast
    enough to iterate on while still being representative.

    Technique: proportional stratified sampling by (collection,
    manipulation_method) group -- e.g. ("youtube", "pristine"),
    ("youtube", "Deepfakes"), ("actors", "pristine"),
    ("actors", "DeepFakeDetection") are each sampled down to roughly
    their original share of `max_total`, with a floor of
    `min_per_group` (so no class/method disappears entirely just
    because it was a small slice of the full dataset) and a seeded RNG
    (so the same subset is reproducible across runs, which matters for
    comparing hyperparameter changes). If `max_total` is at least as
    large as the dataset, every sample is kept and the RNG isn't used.
    """
    if max_total <= 0:
        raise ValueError("max_total must be positive")
    if len(samples) <= max_total:
        return list(samples)

    groups: Dict[Tuple[str, str], List[Sample]] = {}
    for s in samples:
        groups.setdefault((s.collection, s.manipulation_method), []).append(s)

    min_per_group = min(4, max_total // max(len(groups), 1))
    rng = __import__("random").Random(seed)

    # Proportional target count per group, respecting the floor above.
    total = len(samples)
    targets = {}
    for key, group in groups.items():
        share = round(max_total * (len(group) / total))
        targets[key] = max(min_per_group, min(share, len(group)))

    # Proportional rounding can overshoot max_total; trim the largest
    # groups first (they can most afford to lose a sample) until we're
    # at or under budget, never going below min_per_group.
    while sum(targets.values()) > max_total:
        key = max(targets, key=lambda k: targets[k])
        if targets[key] <= min_per_group:
            break  # every group is already at its floor -- accept going slightly over
        targets[key] -= 1

    picked: List[Sample] = []
    for key, group in groups.items():
        k = min(targets[key], len(group))
        picked.extend(rng.sample(group, k))

    picked.sort(key=lambda s: s.path)  # deterministic order regardless of dict/set iteration
    return picked


def class_balance(samples: List[Sample]) -> Dict[str, int]:
    """Simple real/fake count, used to set `pos_weight` for the loss
    function (Section 3.5's class weighting for imbalance)."""
    counts = {"real": 0, "fake": 0}
    for s in samples:
        counts[s.label] += 1
    return counts


def collection_summary(samples: List[Sample]) -> Dict[str, Dict[str, int]]:
    """Per-collection, per-label/method counts -- handy for a quick sanity
    check after loading (e.g. confirming DeepFakeDetection finished
    downloading before you kick off training)."""
    summary: Dict[str, Dict[str, int]] = {}
    for s in samples:
        c = summary.setdefault(s.collection, {})
        key = s.label if s.label == "real" else s.manipulation_method
        c[key] = c.get(key, 0) + 1
    return summary


if __name__ == "__main__":
    if not os.path.isdir(DEFAULT_ROOT_DIR):
        print(f"DEFAULT_ROOT_DIR does not exist on this machine: {DEFAULT_ROOT_DIR}")
        print("Pass root_dir=... explicitly to load_faceforensics() if your data lives elsewhere.")
    else:
        print(f"Loading from {DEFAULT_ROOT_DIR} ...")
        samples = load_faceforensics()
        print(f"Loaded {len(samples)} samples total.")
        for collection, counts in collection_summary(samples).items():
            print(f"  {collection}: {counts}")
        print(f"Manipulation methods present: {methods_present(samples)}")
