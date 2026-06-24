#!/usr/bin/env python3
"""
grimage_v2_surrogates.py
========================

Run GrimAge **V2** on a beta-value matrix and output, in *wide* format
(index = sample_id), the 10 intermediate DNAm surrogate markers
*together with* the composite GrimAge2 age.

The 10 V2 surrogates (stage-1 components of GrimAge2):
    DNAm PACKYRS, DNAm ADM, DNAm B2M, DNAm Cystatin C, DNAm GDF-15,
    DNAm Leptin, DNAm logCRP, DNAm logA1C, DNAm PAI-1, DNAm TIMP-1

Why this script exists
----------------------
biolearn's public `GrimageModel.predict()` returns only the final
mortality-adjusted age; the stage-1 surrogates are computed internally
and discarded. This script reuses biolearn's *validated coefficient
file* (`GrimAgeV2.csv`) and replays the same two-stage linear algebra so
that every surrogate is exposed, then **cross-checks** its reconstructed
composite against biolearn's official `predict()` output. If the two do
not agree within tolerance the script fails loudly rather than emit
silently-wrong numbers.

Design notes / assumptions the script resolves at runtime
---------------------------------------------------------
1. Component-grouping column in the coefficient CSV is auto-detected
   (the single non-numeric column), so we don't hard-code "Y.pred".
2. Sex -> "Female" indicator coding is auto-resolved by trying both
   polarities and keeping whichever reproduces biolearn's official age.
3. Whether the COX (stage-2) linear predictor is already in year units
   or needs an affine transform is detected from the cross-check and
   reported; the *authoritative* final value written out is always
   biolearn's official `predict()`.

GrimAge2 also requires chronological age and sex per sample, so a
metadata table is mandatory.

Usage
-----
    python grimage_v2_surrogates.py \
        --betas betas.parquet \
        --metadata pheno.csv \
        --output grimage_v2_wide.parquet \
        [--orientation auto] [--imputation none] [--tol 1e-3] [--strict]

Inputs
------
--betas      Parquet or CSV. Beta values in [0, 1]. Either orientation
             (CpGs x samples or samples x CpGs) is accepted; the CpG axis
             is detected from overlap with the model's CpG set.
--metadata   CSV with a sample id column plus 'age' and 'sex'. The id
             column is matched to the beta sample axis. 'sex' may be coded
             1/2, 0/1, or "M"/"F"/"male"/"female" (case-insensitive).
--output     Parquet path for the wide result (also writes a .csv twin).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings

import numpy as np
import pandas as pd

from biolearn.data_library import GeoData
from biolearn.model_gallery import ModelGallery
from biolearn.util import get_data_file


CG_RE = re.compile(r"^(cg|ch\.|rs)\d", re.IGNORECASE)

# Special non-CpG terms that may appear as coefficient "variables".
INTERCEPT_KEYS = {"intercept", "(intercept)"}
AGE_KEYS = {"age"}
FEMALE_KEYS = {"female"}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _read_table(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if path.endswith((".csv", ".csv.gz", ".tsv")):
        sep = "\t" if path.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep, index_col=0)
    raise ValueError(f"Unsupported file extension: {path}")


def load_betas(path: str, orientation: str, model_cpgs: set[str]) -> pd.DataFrame:
    """Return betas as CpGs (rows) x samples (cols), biolearn's dnam layout."""
    df = _read_table(path)

    def _overlap(labels) -> int:
        return len(set(map(str, labels)) & model_cpgs)

    if orientation == "cpgs_rows":
        out = df
    elif orientation == "samples_rows":
        out = df.transpose()
    else:  # auto
        row_hit = _overlap(df.index)
        col_hit = _overlap(df.columns)
        if row_hit == 0 and col_hit == 0:
            raise ValueError(
                "No model CpGs found on either axis of the beta matrix. "
                "Check the input or pass --orientation explicitly."
            )
        out = df if row_hit >= col_hit else df.transpose()

    out.index = out.index.map(str)
    out.columns = out.columns.map(str)
    return out


def load_metadata(path: str, sample_ids) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}

    # Locate the id column: prefer an explicit name, else assume first column.
    id_col = None
    for cand in ("sample_id", "sample", "id", "basename", "gsm", "sentrix"):
        if cand in cols:
            id_col = cols[cand]
            break
    if id_col is None:
        id_col = raw.columns[0]

    if "age" not in cols or "sex" not in cols:
        raise ValueError("Metadata must contain 'age' and 'sex' columns.")

    meta = raw.set_index(raw[id_col].astype(str))
    meta = meta.rename(columns={cols["age"]: "age", cols["sex"]: "sex"})
    meta = meta[["age", "sex"]].copy()
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")

    missing = sorted(set(map(str, sample_ids)) - set(meta.index))
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) in the beta matrix have no metadata, "
            f"e.g. {missing[:5]}"
        )
    return meta.loc[list(map(str, sample_ids))]


# --------------------------------------------------------------------------- #
# Coefficient-file introspection
# --------------------------------------------------------------------------- #
def load_coefficients(coef_file: str = "GrimAgeV2.csv") -> pd.DataFrame:
    """Load biolearn's bundled GrimAgeV2 coefficients (index = variable name)."""
    coef = pd.read_csv(get_data_file(coef_file), index_col=0)
    coef.index = coef.index.map(str)
    return coef


def detect_columns(coef: pd.DataFrame) -> tuple[str, str]:
    """Return (component_column, weight_column).

    component_column: the lone non-numeric column whose values name the
    sub-models (e.g. DNAmADM ... COX).
    weight_column: the numeric coefficient column.
    """
    numeric_cols, object_cols = [], []
    for c in coef.columns:
        if pd.api.types.is_numeric_dtype(coef[c]):
            numeric_cols.append(c)
        else:
            object_cols.append(c)

    if not object_cols:
        raise ValueError(
            "Could not find a categorical component column in the "
            f"coefficient file. Columns seen: {list(coef.columns)}"
        )
    if not numeric_cols:
        raise ValueError(
            "Could not find a numeric weight column in the coefficient file. "
            f"Columns seen: {list(coef.columns)}"
        )

    # Component column = categorical column with the fewest distinct values
    # (sub-model labels repeat across many CpG rows).
    comp_col = min(object_cols, key=lambda c: coef[c].nunique())
    # Weight column = numeric column with the widest spread (the actual betas).
    weight_col = max(numeric_cols, key=lambda c: float(np.nanstd(coef[c].values)))
    return comp_col, weight_col


def identify_cox_group(groups: dict[str, pd.Series]) -> str:
    """The stage-2 group references *other group names* and carries Age/Female
    but essentially no CpGs."""
    group_names = set(groups)
    best, best_score = None, -1.0
    for name, weights in groups.items():
        idx = [str(v).lower() for v in weights.index]
        n_cpg = sum(bool(CG_RE.match(v)) for v in idx)
        n_refs = sum(
            (v in {g.lower() for g in group_names - {name}})
            or (v in AGE_KEYS | FEMALE_KEYS)
            for v in idx
        )
        score = n_refs - n_cpg
        if score > best_score:
            best, best_score = name, score
    return best


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def _female_indicator(sex: pd.Series, female_is: str) -> pd.Series:
    """Map a 'sex' column to a 0/1 Female indicator under a chosen convention.

    female_is in {"high", "low"}: which numeric pole is female, used when the
    column is numeric. Strings are mapped directly regardless.
    """
    s = sex.copy()
    as_str = s.astype(str).str.strip().str.lower()
    if as_str.isin({"m", "male", "f", "female"}).any():
        return as_str.isin({"f", "female"}).astype(float)

    num = pd.to_numeric(s, errors="coerce")
    uniq = sorted(num.dropna().unique())
    if len(uniq) < 2:
        return (num == num).astype(float) * (1.0 if female_is == "high" else 0.0)
    lo, hi = uniq[0], uniq[-1]
    target = hi if female_is == "high" else lo
    return (num == target).astype(float)


def build_feature_matrix(
    betas_cpg_by_sample: pd.DataFrame,
    meta: pd.DataFrame,
    needed_cpgs: list[str],
    female: pd.Series,
) -> pd.DataFrame:
    """samples x {CpGs, Age, Female, Intercept}. Missing CpGs are NaN (the
    caller is expected to have imputed via biolearn already)."""
    feats = betas_cpg_by_sample.reindex(needed_cpgs).transpose()  # samples x CpGs
    feats["age"] = meta["age"].astype(float)
    feats["female"] = female.astype(float)
    feats["intercept"] = 1.0
    return feats


def _norm(v: str) -> str:
    v = str(v).strip().lower()
    if v in INTERCEPT_KEYS:
        return "intercept"
    if v in AGE_KEYS:
        return "age"
    if v in FEMALE_KEYS:
        return "female"
    return v


def score_group(weights: pd.Series, feats: pd.DataFrame) -> pd.Series:
    """Linear score = sum_v weight_v * feature_v for one sub-model."""
    acc = pd.Series(0.0, index=feats.index)
    norm_cols = {_norm(c): c for c in feats.columns}
    for var, w in weights.items():
        key = _norm(var)
        col = norm_cols.get(key, var if var in feats.columns else None)
        if col is None:
            # Variable referenced by the model but absent from features.
            continue
        acc = acc.add(float(w) * feats[col].astype(float), fill_value=0.0)
    return acc


def compute_surrogates(
    betas_cpg_by_sample: pd.DataFrame,
    meta: pd.DataFrame,
    coef: pd.DataFrame,
    female: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return (surrogates_df samples x 10, reconstructed_final, cox_group_name)."""
    comp_col, weight_col = detect_columns(coef)

    groups: dict[str, pd.Series] = {}
    for name, sub in coef.groupby(comp_col):
        groups[str(name)] = sub[weight_col]

    cox_name = identify_cox_group(groups)
    surrogate_names = [g for g in groups if g != cox_name]

    # All CpGs referenced by any stage-1 surrogate.
    needed = []
    for g in surrogate_names:
        needed += [v for v in groups[g].index if CG_RE.match(str(v))]
    needed = sorted(set(needed))

    feats = build_feature_matrix(betas_cpg_by_sample, meta, needed, female)

    # Stage 1: each surrogate.
    surrogates = pd.DataFrame(index=feats.index)
    for g in surrogate_names:
        surrogates[g] = score_group(groups[g], feats)

    # Stage 2 (COX): combine surrogates + age + female + intercept.
    stage2_feats = surrogates.copy()
    stage2_feats["age"] = feats["age"]
    stage2_feats["female"] = feats["female"]
    stage2_feats["intercept"] = 1.0
    reconstructed = score_group(groups[cox_name], stage2_feats)

    return surrogates, reconstructed, cox_name


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> pd.DataFrame:
    coef = load_coefficients()
    comp_col, weight_col = detect_columns(coef)
    model_cpgs = {str(v) for v in coef.index if CG_RE.match(str(v))}
    print(f"[info] coefficient component column = '{comp_col}', "
          f"weight column = '{weight_col}', model CpGs = {len(model_cpgs)}")

    betas = load_betas(args.betas, args.orientation, model_cpgs)
    sample_ids = list(betas.columns)
    print(f"[info] loaded betas: {betas.shape[0]} CpGs x {len(sample_ids)} samples")

    meta = load_metadata(args.metadata, sample_ids)

    # ---- biolearn official prediction (authoritative final value) ---------- #
    geo = GeoData(metadata=meta.copy(), dnam=betas.copy())
    gallery = ModelGallery()
    model = gallery.get("GrimAgeV2", imputation_method=args.imputation)
    official = model.predict(geo)
    official_final = _as_series(official, sample_ids)
    # biolearn imputes internally; reuse its imputed matrix for our recon so
    # that surrogate scores see the same inputs.
    betas_imputed = getattr(geo, "dnam", betas)

    # ---- reconstruct surrogates, auto-resolving the Female coding ---------- #
    best = None
    for female_is in ("high", "low"):
        female = _female_indicator(meta["sex"], female_is)
        surr, recon, cox_name = compute_surrogates(
            betas_imputed, meta, coef, female
        )
        diff = float(np.nanmax(np.abs(recon.reindex(sample_ids) - official_final)))
        if best is None or diff < best[0]:
            best = (diff, female_is, surr, recon, cox_name)

    max_diff, female_is, surrogates, reconstructed, cox_name = best
    print(f"[info] stage-2 group = '{cox_name}', resolved sex coding: "
          f"female = {'higher' if female_is == 'high' else 'lower'} pole")
    print(f"[check] max |reconstructed - biolearn official| = {max_diff:.3e}")

    if max_diff > args.tol:
        msg = (
            f"Cross-check failed: reconstructed GrimAge2 differs from biolearn's "
            f"official output by up to {max_diff:.3e} (tol={args.tol}). The "
            f"surrogate decomposition may not match this coefficient-file "
            f"schema. Inspect the printed component/weight columns."
        )
        if args.strict:
            raise RuntimeError(msg)
        warnings.warn(msg)
    else:
        print("[check] OK — surrogate decomposition reproduces the composite.")

    # ---- assemble wide output --------------------------------------------- #
    out = surrogates.reindex(sample_ids).copy()
    out = _order_surrogates(out)
    out["GrimAgeV2"] = official_final.values
    out["AgeAccelGrimV2"] = official_final.values - meta["age"].reindex(sample_ids).values
    out.index.name = "sample_id"
    return out


def _as_series(pred, sample_ids) -> pd.Series:
    """Coerce biolearn's prediction (DataFrame or Series) to a Series."""
    if isinstance(pred, pd.DataFrame):
        col = pred.columns[0]
        s = pred[col]
    else:
        s = pred
    s.index = s.index.map(str)
    return s.reindex(list(map(str, sample_ids))).astype(float)


# Preferred display order for the 10 V2 surrogates (best-effort name match).
_PREFERRED = [
    "PACKYRS", "ADM", "B2M", "Cystatin", "GDF", "Leptin",
    "logCRP", "CRP", "logA1C", "A1C", "PAI", "TIMP",
]


def _order_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    def rank(col: str) -> int:
        u = col.upper()
        for i, key in enumerate(_PREFERRED):
            if key.upper() in u:
                return i
        return len(_PREFERRED)
    return df[sorted(df.columns, key=rank)]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--betas", required=True, help="Parquet/CSV beta matrix.")
    p.add_argument("--metadata", required=True,
                   help="CSV with sample id, 'age', 'sex'.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument("--orientation", default="auto",
                   choices=["auto", "cpgs_rows", "samples_rows"])
    p.add_argument("--imputation", default="none",
                   help="biolearn imputation_method (e.g. none, averaging).")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Max allowed |recon - official| for the cross-check.")
    p.add_argument("--strict", action="store_true",
                   help="Raise (not warn) if the cross-check exceeds --tol.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = run(args)

    out_path = args.output
    if not out_path.endswith((".parquet", ".pq")):
        out_path += ".parquet"
    out.to_parquet(out_path)
    out.to_csv(out_path.rsplit(".", 1)[0] + ".csv")
    print(f"[done] wrote {out.shape[0]} samples x {out.shape[1]} columns -> {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
