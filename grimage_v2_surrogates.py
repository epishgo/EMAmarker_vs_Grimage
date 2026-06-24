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
        [--orientation auto] [--missing-cpgs gold_standard] [--tol 1e-3] [--strict]

--missing-cpgs controls how model CpGs absent from the input are handled:
  gold_standard  impute missing required CpGs from the sesame 450k gold-standard
                 reference (biolearn hybrid_impute). [default]
  present_only   use only the CpGs present in your data; do not impute the
                 missing model CpGs.
(Advanced: --imputation passes a raw biolearn method and overrides the above.)

Inputs
------
--betas      Parquet or CSV. Beta values in [0, 1]. Either orientation
             (CpGs x samples or samples x CpGs) is accepted; the CpG axis
             is detected from overlap with the model's CpG set.
--metadata   CSV with a sample id column plus 'age' and 'sex'. The id
             column is matched to the beta sample axis. 'sex' may be coded
             1/2, 0/1, or "M"/"F"/"male"/"female" (case-insensitive).
--output     Output path for the wide result. The file format follows the
             extension: .csv/.csv.gz (CSV), .tsv (TSV), .parquet/.pq (Parquet).
             Anything else defaults to CSV.
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


# Column names that, when present, identify the sample axis of a beta table.
ID_COLUMN_KEYS = ("sample_id", "sample", "id", "basename", "gsm", "sentrix", "name")


def _promote_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Move an embedded sample-id column into the index.

    Some beta matrices are stored with samples on the rows but their sample
    identifiers kept in a dedicated column (e.g. a parquet written with a
    default RangeIndex plus a ``sample_id`` column) rather than on the index.
    Left untouched, the meaningless 0..N-1 RangeIndex would later be transposed
    into the *column* labels and mistaken for sample ids. We only intervene
    when the current index looks like a default integer range, so frames that
    already carry meaningful labels are left alone.
    """
    looks_default = (
        df.index.name is None
        and pd.api.types.is_integer_dtype(df.index)
        and list(df.index) == list(range(len(df)))
    )
    if not looks_default:
        return df

    # Prefer an explicitly named id column.
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ID_COLUMN_KEYS:
        if cand in lower:
            return df.set_index(lower[cand])

    # Fallback: a lone non-numeric, non-CpG column whose values are all unique.
    candidates = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not CG_RE.match(str(c))
        and df[c].is_unique
    ]
    if len(candidates) == 1:
        return df.set_index(candidates[0])
    return df


def load_betas(path: str, orientation: str, model_cpgs: set[str]) -> pd.DataFrame:
    """Return betas as CpGs (rows) x samples (cols), biolearn's dnam layout."""
    df = _promote_id_column(_read_table(path))

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


# Sex labels recognized when normalizing to biolearn's 0=female / 1=male code.
FEMALE_TOKENS = {"f", "female", "fem", "woman", "women", "girl", "女", "女性"}
MALE_TOKENS = {"m", "male", "man", "men", "boy", "男", "男性"}


def _normalize_sex(sex: pd.Series) -> pd.Series:
    """Normalize a sex column to biolearn's convention: 0 = female, 1 = male.

    Recognizes English and Japanese labels (e.g. ``F``/``M``, ``女``/``男``) as
    well as numeric 0/1 codings. biolearn derives the GrimAge ``Female``
    indicator as ``1 if sex == 0 else 0``, so female MUST map to 0 or the
    official prediction silently treats every sample as male.
    """
    s = sex.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=sex.index, dtype="float64")
    out[s.isin(FEMALE_TOKENS)] = 0.0
    out[s.isin(MALE_TOKENS)] = 1.0

    unresolved = out.isna()
    if unresolved.any():
        num = pd.to_numeric(sex, errors="coerce")
        ok = unresolved & num.isin([0, 1])
        out[ok] = num[ok]

    if out.isna().any():
        bad = sorted(set(sex[out.isna()].astype(str)))[:5]
        raise ValueError(
            f"Unrecognized sex value(s): {bad}. Expected female/male labels "
            "(e.g. F/M, female/male, 女/男) or numeric 0/1 "
            "(0 = female, 1 = male, biolearn convention)."
        )
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
    meta["sex"] = _normalize_sex(meta["sex"])

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


def _count_cpgs(values) -> int:
    return sum(bool(CG_RE.match(str(v))) for v in values)


def normalize_coefficients(coef: pd.DataFrame) -> pd.DataFrame:
    """Flatten the coefficient file into a tidy frame with columns
    ``['component', 'variable', 'weight']`` (one row per coefficient).

    Two on-disk schemas are supported transparently, regardless of whether the
    CpG / component labels live on the index or in a column:

    * **Schema A** (this script's original assumption): the index holds the
      *variable* names (CpGs / Age / Intercept / group references), an object
      column holds the *component* (sub-model) label, and a numeric column
      holds the weight.
    * **Schema B** (biolearn 0.9.1's bundled ``GrimAgeV2.csv``): the index
      (``Y.pred``) holds the *component* label, a ``var`` column holds the
      *variable* names, and a ``beta`` column holds the weight.

    Detection is content-based: the variable axis is whichever label source
    contains the most CpG-like tokens; the weight is the numeric column with
    the widest spread; the component is the remaining low-cardinality label
    source.
    """
    # Promote the index to an ordinary column so the index and the columns are
    # treated uniformly as candidate label sources.
    df = coef.reset_index()
    df.columns = [str(c) for c in df.columns]

    numeric_cols, object_cols = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            object_cols.append(c)

    if not numeric_cols:
        raise ValueError(
            "Could not find a numeric weight column in the coefficient file. "
            f"Columns seen: {list(df.columns)}"
        )

    # Variable column = the label source carrying the CpG identifiers.
    var_col = max(df.columns, key=lambda c: _count_cpgs(df[c]))
    if _count_cpgs(df[var_col]) == 0:
        raise ValueError(
            "Could not locate a variable column containing CpG identifiers "
            f"(cg.../ch.../rs...). Columns seen: {list(df.columns)}"
        )

    # Weight column = numeric column with the widest spread (the actual betas).
    weight_col = max(
        numeric_cols,
        key=lambda c: float(np.nanstd(pd.to_numeric(df[c], errors="coerce").values)),
    )

    # Component column = remaining object column with the fewest distinct
    # values (sub-model labels repeat across many variable rows).
    comp_candidates = [c for c in object_cols if c != var_col]
    if not comp_candidates:
        raise ValueError(
            "Could not find a categorical component column in the "
            f"coefficient file. Columns seen: {list(df.columns)}"
        )
    comp_col = min(comp_candidates, key=lambda c: df[c].nunique())

    tidy = pd.DataFrame(
        {
            "component": df[comp_col].map(str),
            "variable": df[var_col].map(str),
            "weight": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )
    return tidy


# Standardization parameters of the final affine transform (cox LP -> years).
# GrimAge: grimage = (cox - m_cox) / sd_cox * sd_age + m_age
TRANSFORM_KEYS = {"m_age", "sd_age", "m_cox", "sd_cox"}


def identify_transform_group(groups: dict[str, pd.Series]) -> str | None:
    """Return the name of the affine-transform group, if present.

    GrimAge2's coefficient file carries a small group (no CpGs, no group
    references) holding the standardization constants ``m_age/sd_age/m_cox/
    sd_cox`` used to map the stage-2 COX linear predictor into age units.
    """
    for name, weights in groups.items():
        idx = {str(v).strip().lower() for v in weights.index}
        if TRANSFORM_KEYS.issubset(idx):
            return name
    return None


def _apply_transform(cox_lp: pd.Series, params: pd.Series) -> pd.Series:
    """Map a stage-2 COX linear predictor into age units via the standardization
    constants. Returns ``cox_lp`` unchanged if the constants are unusable."""
    p = {str(k).strip().lower(): float(v) for k, v in params.items()}
    try:
        m_age, sd_age = p["m_age"], p["sd_age"]
        m_cox, sd_cox = p["m_cox"], p["sd_cox"]
    except KeyError:
        return cox_lp
    if sd_cox == 0:
        return cox_lp
    return (cox_lp - m_cox) / sd_cox * sd_age + m_age


def identify_cox_group(
    groups: dict[str, pd.Series], exclude: set[str] | None = None
) -> str:
    """The stage-2 group references *other group names* and carries Age/Female
    but essentially no CpGs."""
    exclude = exclude or set()
    group_names = set(groups)
    best, best_score = None, -1.0
    for name, weights in groups.items():
        if name in exclude:
            continue
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
    """Return (surrogates_df samples x 10, reconstructed_final, cox_group_name).

    ``coef`` is the tidy frame produced by :func:`normalize_coefficients`
    (columns ``component`` / ``variable`` / ``weight``).
    """
    groups: dict[str, pd.Series] = {}
    for name, sub in coef.groupby("component"):
        groups[str(name)] = pd.Series(
            sub["weight"].to_numpy(), index=sub["variable"].astype(str)
        )

    transform_name = identify_transform_group(groups)
    cox_name = identify_cox_group(
        groups, exclude={transform_name} if transform_name else None
    )
    # Stage-1 surrogates are every group except the stage-2 COX combiner and
    # the affine-transform constants.
    surrogate_names = [
        g for g in groups if g != cox_name and g != transform_name
    ]

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

    # Final affine transform: standardize the COX LP and rescale into years.
    if transform_name is not None:
        reconstructed = _apply_transform(reconstructed, groups[transform_name])

    return surrogates, reconstructed, cox_name


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
# User-facing strategies for model CpGs that are missing from the input matrix,
# mapped onto biolearn's imputation_method names.
MISSING_CPG_STRATEGIES = {
    # Fill missing required CpGs from the sesame 450k gold-standard reference
    # (biolearn hybrid_impute). Recommended when the input lacks model CpGs.
    "gold_standard": "sesame_450k",
    # Use only the CpGs present in the input; do NOT impute missing model CpGs.
    "present_only": "none",
}


def resolve_imputation(args: argparse.Namespace) -> str:
    """Resolve the biolearn imputation_method from the CLI options.

    A raw ``--imputation`` value (advanced override) wins; otherwise the
    friendly ``--missing-cpgs`` strategy is mapped to its biolearn method.
    """
    if args.imputation:
        return args.imputation
    return MISSING_CPG_STRATEGIES[args.missing_cpgs]


def run(args: argparse.Namespace) -> pd.DataFrame:
    coef = normalize_coefficients(load_coefficients())
    model_cpgs = {v for v in coef["variable"] if CG_RE.match(v)}
    print(f"[info] coefficient schema normalized: "
          f"{coef['component'].nunique()} components, {len(coef)} coefficients, "
          f"model CpGs = {len(model_cpgs)}")

    betas = load_betas(args.betas, args.orientation, model_cpgs)
    sample_ids = list(betas.columns)
    print(f"[info] loaded betas: {betas.shape[0]} CpGs x {len(sample_ids)} samples")

    present = len(model_cpgs & set(betas.index))
    missing = len(model_cpgs) - present
    method = resolve_imputation(args)
    strategy = (f"--imputation override '{method}'" if args.imputation
                else f"strategy='{args.missing_cpgs}'")
    print(f"[info] model CpGs present in input: {present}/{len(model_cpgs)} "
          f"({missing} missing) | {strategy} "
          f"-> biolearn imputation='{method}'")
    if args.missing_cpgs == "present_only" and missing:
        warnings.warn(
            f"{missing}/{len(model_cpgs)} model CpGs are absent from the input "
            "and are NOT imputed under 'present_only'; GrimAge is computed from "
            "the present subset only and may be unreliable."
        )

    meta = load_metadata(args.metadata, sample_ids)

    # ---- biolearn official prediction (authoritative final value) ---------- #
    geo = GeoData(metadata=meta.copy(), dnam=betas.copy())
    gallery = ModelGallery()
    model = gallery.get("GrimAgeV2", imputation_method=method)
    official = model.predict(geo)
    official_final = _as_series(official, sample_ids)
    # biolearn imputes on an internal copy (it does not mutate geo.dnam), so
    # reproduce that exact imputed matrix here for the reconstruction; otherwise
    # NaNs that biolearn fills with column means would be scored as 0.
    betas_imputed = _biolearn_imputed_dnam(model, betas)

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


def _select_grimage_column(pred: pd.DataFrame) -> pd.Series:
    """Pick the composite GrimAge column from biolearn's prediction frame.

    Recent biolearn returns the full surrogate table plus ``Age``, ``Female``,
    ``DNAmGrimAge`` and ``AgeAccelGrim``; the composite age is ``DNAmGrimAge``,
    *not* the first column (which is a surrogate). We match on name and avoid
    the age-acceleration / residual column.
    """
    lower = {str(c).lower(): c for c in pred.columns}
    for key in ("dnamgrimage2", "dnamgrimage", "grimage2", "grimagev2", "grimage"):
        if key in lower:
            return pred[lower[key]]
    grim = [
        c for c in pred.columns
        if "grim" in str(c).lower() and "accel" not in str(c).lower()
    ]
    if grim:
        return pred[grim[0]]
    return pred[pred.columns[0]]


def _biolearn_imputed_dnam(model, betas: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the imputed beta matrix biolearn feeds to its own predict().

    biolearn wraps the clock in an imputation decorator that calls
    ``imputation_method(dnam, needed_cpgs)`` on a copy. Replaying it here keeps
    the surrogate reconstruction byte-for-byte aligned with the official run.
    Falls back to the raw matrix if the model does not expose these hooks.
    """
    try:
        needed = model.methylation_sites()
        return model.imputation_method(betas.copy(), needed)
    except Exception:
        return betas


def _as_series(pred, sample_ids) -> pd.Series:
    """Coerce biolearn's prediction (DataFrame or Series) to a Series."""
    if isinstance(pred, pd.DataFrame):
        s = _select_grimage_column(pred)
    else:
        s = pred
    s = s.copy()
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
    p.add_argument("--output", required=True,
                   help="Output path. Format follows the extension: "
                        ".csv/.csv.gz, .tsv, or .parquet/.pq (default: .csv).")
    p.add_argument("--orientation", default="auto",
                   choices=["auto", "cpgs_rows", "samples_rows"])
    p.add_argument("--missing-cpgs", dest="missing_cpgs",
                   default="gold_standard",
                   choices=sorted(MISSING_CPG_STRATEGIES),
                   help="How to handle model CpGs absent from the input. "
                        "'gold_standard': impute missing required CpGs from the "
                        "sesame 450k gold-standard reference (biolearn "
                        "hybrid_impute). 'present_only': predict using only the "
                        "CpGs present in your data, with no imputation of the "
                        "missing ones. (default: gold_standard)")
    p.add_argument("--imputation", default=None,
                   help="Advanced override: a raw biolearn imputation_method "
                        "(none, averaging, sesame_450k, dunedin). When set, it "
                        "takes precedence over --missing-cpgs.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Max allowed |recon - official| for the cross-check.")
    p.add_argument("--strict", action="store_true",
                   help="Raise (not warn) if the cross-check exceeds --tol.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = run(args)

    out_path = args.output
    if out_path.endswith((".parquet", ".pq")):
        out.to_parquet(out_path)
    elif out_path.endswith(".tsv"):
        out.to_csv(out_path, sep="\t")
    else:
        if not out_path.endswith((".csv", ".csv.gz")):
            out_path += ".csv"
        out.to_csv(out_path)
    print(f"[done] wrote {out.shape[0]} samples x {out.shape[1]} columns -> {out_path}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
