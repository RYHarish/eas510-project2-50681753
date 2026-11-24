"""Turn EMBER raw-feature JSON records into fixed-length numeric vectors.

This is a standalone re-implementation of the EMBER (feature version 2) vectorizer
described in Anderson & Roth (2018). It works on the raw JSON dicts that ship with
the dataset, so it needs neither the `ember` package nor `lief`. It never touches an
executable file.

Layout (2,381 columns):
    histogram        256  normalized byte histogram
    byteentropy      256  normalized byte/entropy 2-D histogram
    strings          104  string counts, printable-char distribution, entropy, IOC counts
    general           10  file size, virtual size, import/export counts, flags
    header            62  COFF + optional header fields (categoricals feature-hashed)
    section          255  section counts + hashed per-section size/entropy/vsize, entry section
    imports         1280  hashed imported libraries (256) and library:function pairs (1024)
    exports          128  hashed exported function names
    datadirectories   30  size and virtual address of the 15 PE data directories
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import FeatureHasher

GENERAL_KEYS = ["size", "vsize", "has_debug", "exports", "imports", "has_relocations",
                "has_resources", "has_signature", "has_tls", "symbols"]
OPTIONAL_NUM_KEYS = ["major_image_version", "minor_image_version", "major_linker_version",
                     "minor_linker_version", "major_operating_system_version",
                     "minor_operating_system_version", "major_subsystem_version",
                     "minor_subsystem_version", "sizeof_code", "sizeof_headers",
                     "sizeof_heap_commit"]
STRING_TAIL_KEYS = ["entropy", "paths", "urls", "registry", "MZ"]
N_DATADIR = 15

# (group name, width) in column order
GROUPS = [("histogram", 256), ("byteentropy", 256), ("strings", 104), ("general", 10),
          ("header", 62), ("section", 255), ("imports", 1280), ("exports", 128),
          ("datadirectories", 30)]
N_FEATURES = sum(w for _, w in GROUPS)  # 2381


def _hash(n: int, rows, input_type: str = "string") -> np.ndarray:
    return FeatureHasher(n, input_type=input_type).transform(rows).toarray().astype(np.float32)


def _normalized(rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float32)
    s = a.sum(axis=1, keepdims=True)
    return np.divide(a, s, out=np.zeros_like(a), where=s > 0)


def _strings(recs) -> np.ndarray:
    out = np.zeros((len(recs), 104), dtype=np.float32)
    for i, r in enumerate(recs):
        s = r["strings"]
        div = float(s["printables"]) if s["printables"] > 0 else 1.0
        out[i, 0:3] = [s["numstrings"], s["avlength"], s["printables"]]
        out[i, 3:99] = np.asarray(s["printabledist"], dtype=np.float32) / div
        out[i, 99:104] = [s[k] for k in STRING_TAIL_KEYS]
    return out


def _general(recs) -> np.ndarray:
    return np.array([[r["general"][k] for k in GENERAL_KEYS] for r in recs], dtype=np.float32)


def _header(recs) -> np.ndarray:
    coff = [r["header"]["coff"] for r in recs]
    opt = [r["header"]["optional"] for r in recs]
    return np.hstack([
        np.array([[c["timestamp"]] for c in coff], dtype=np.float32),
        _hash(10, [[c["machine"]] for c in coff]),
        _hash(10, [c["characteristics"] for c in coff]),
        _hash(10, [[o["subsystem"]] for o in opt]),
        _hash(10, [o["dll_characteristics"] for o in opt]),
        _hash(10, [[o["magic"]] for o in opt]),
        np.array([[o[k] for k in OPTIONAL_NUM_KEYS] for o in opt], dtype=np.float32),
    ])


def _section(recs) -> np.ndarray:
    general, sizes, entropy, vsize, entry, entry_props = [], [], [], [], [], []
    for r in recs:
        secs = r["section"]["sections"]
        ent = r["section"]["entry"]
        general.append([
            len(secs),
            sum(s["size"] == 0 for s in secs),
            sum(s["name"] == "" for s in secs),
            sum("MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"] for s in secs),
            sum("MEM_WRITE" in s["props"] for s in secs),
        ])
        sizes.append([(s["name"], s["size"]) for s in secs])
        entropy.append([(s["name"], s["entropy"]) for s in secs])
        vsize.append([(s["name"], s["vsize"]) for s in secs])
        entry.append([ent])
        entry_props.append([p for s in secs if s["name"] == ent for p in s["props"]])
    return np.hstack([
        np.array(general, dtype=np.float32),
        _hash(50, sizes, "pair"), _hash(50, entropy, "pair"), _hash(50, vsize, "pair"),
        _hash(50, entry), _hash(50, entry_props),
    ])


def _imports(recs) -> np.ndarray:
    libs = [sorted({lib.lower() for lib in r["imports"]}) for r in recs]
    funcs = [[f"{lib.lower()}:{fn}" for lib, fns in r["imports"].items() for fn in fns] for r in recs]
    return np.hstack([_hash(256, libs), _hash(1024, funcs)])


def _datadirectories(recs) -> np.ndarray:
    out = np.zeros((len(recs), 2 * N_DATADIR), dtype=np.float32)
    for i, r in enumerate(recs):
        for j, d in enumerate(r["datadirectories"][:N_DATADIR]):
            out[i, 2 * j] = d["size"]
            out[i, 2 * j + 1] = d["virtual_address"]
    return out


def vectorize(recs: list[dict]) -> np.ndarray:
    """Vectorize a batch of raw EMBER records -> float32 array of shape (n, 2381)."""
    X = np.hstack([
        _normalized([r["histogram"] for r in recs]),
        _normalized([r["byteentropy"] for r in recs]),
        _strings(recs),
        _general(recs),
        _header(recs),
        _section(recs),
        _imports(recs),
        _hash(128, [r["exports"] for r in recs]),
        _datadirectories(recs),
    ])
    assert X.shape[1] == N_FEATURES, X.shape
    return X


def feature_names() -> list[str]:
    """One readable name per column, prefixed by its feature group."""
    names = [f"histogram_{i}" for i in range(256)]
    names += [f"byteentropy_{i}" for i in range(256)]
    names += (["strings_numstrings", "strings_avlength", "strings_printables"]
              + [f"strings_printabledist_{i}" for i in range(96)]
              + [f"strings_{k}" for k in STRING_TAIL_KEYS])
    names += [f"general_{k}" for k in GENERAL_KEYS]
    names += (["header_timestamp"]
              + [f"header_{f}_hash{i}" for f in ["machine", "characteristics", "subsystem",
                                                 "dll_characteristics", "magic"] for i in range(10)]
              + [f"header_{k}" for k in OPTIONAL_NUM_KEYS])
    names += (["section_count", "section_zero_size", "section_empty_name", "section_rx", "section_w"]
              + [f"section_{f}_hash{i}" for f in ["size", "entropy", "vsize", "entry_name", "entry_props"]
                 for i in range(50)])
    names += [f"imports_lib_hash{i}" for i in range(256)] + [f"imports_func_hash{i}" for i in range(1024)]
    names += [f"exports_hash{i}" for i in range(128)]
    names += [f"datadir_{j}_{k}" for j in range(N_DATADIR) for k in ["size", "va"]]
    assert len(names) == N_FEATURES
    return names


def feature_groups() -> np.ndarray:
    """Group name for every column (same order as `vectorize`)."""
    return np.concatenate([[g] * w for g, w in GROUPS])
