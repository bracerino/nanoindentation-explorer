"""
Nanoindentation (nanoDMA) explorer
==================================

Upload the *_DYN.txt exports from the Hysitron/Bruker nanoDMA tool, pick which
measurements (per sample / fluence / grain) to overlay, and plot the depth
profiles of storage modulus, hardness and contact depth.

Filename convention (decoded automatically):
    s40_f46_g15_000_DYN.txt
     |   |   |   |
     |   |   |   +-- measurement (replicate) number
     |   |   +------ grain
     |   +---------- fluence
     +-------------- sample

Written for a recent Streamlit (>= 1.36). Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import re
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Nanoindentation explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The three measured quantities in every export, with display labels + units.
METRICS = {
    "StorageModulus": ("Storage modulus", "GPa"),
    "Hardness": ("Hardness", "GPa"),
    "ContactDepth": ("Contact depth", "nm"),
}

# Map the (slightly variable) export headers onto our canonical column names.
HEADER_ALIASES = {
    "storage mod. (gpa)": "StorageModulus",
    "storage modulus (gpa)": "StorageModulus",
    "hardness (gpa)": "Hardness",
    "contact depth (nm)": "ContactDepth",
}

FILENAME_RE = re.compile(
    r"^s(?P<sample>\d+)_f(?P<fluence>\d+)_g(?P<grain>\d+)_(?P<meas>\d+)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
@dataclass
class ParsedFile:
    name: str
    sample: str
    fluence: str
    grain: str
    meas: str
    label: str
    data: pd.DataFrame


def parse_filename(name: str) -> dict[str, str]:
    """Extract sample / fluence / grain / measurement from the file name."""
    stem = name.rsplit("/", 1)[-1]
    m = FILENAME_RE.match(stem)
    if not m:
        return {"sample": "?", "fluence": "?", "grain": "?", "meas": "?"}
    return {
        "sample": m.group("sample"),
        "fluence": m.group("fluence"),
        "grain": m.group("grain"),
        "meas": m.group("meas"),
    }


@st.cache_data(show_spinner=False)
def parse_file(name: str, raw: bytes) -> ParsedFile:
    """Read one nanoDMA text export into a tidy DataFrame."""
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Locate the header row (the one that names the columns).
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "hardness" in low and ("mod" in low or "depth" in low):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find a column header row in {name!r}.")

    raw_cols = [c.strip() for c in lines[header_idx].split("\t") if c.strip()]
    cols = [HEADER_ALIASES.get(c.lower(), c) for c in raw_cols]

    body = "\n".join(lines[header_idx + 1 :])
    df = pd.read_csv(
        io.StringIO(body),
        sep="\t",
        header=None,
        usecols=range(len(cols)),
        names=cols,
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    meta = parse_filename(name)
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if "?" in meta.values():
        # Filename doesn't follow s#_f#_g#_### — fall back to the bare file name.
        label = stem
    else:
        label = f"f{meta['fluence']} · g{meta['grain']} · #{meta['meas']}"
    return ParsedFile(name=name, label=label, data=df, **meta)


def grain_sort_key(g: str) -> tuple[int, str | int]:
    """Sort grains numerically when possible, otherwise alphabetically."""
    return (0, int(g)) if g.isdigit() else (1, g)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def bin_profile(df: pd.DataFrame, x: str, y: str, n_bins: int) -> pd.DataFrame:
    """Average y over evenly spaced bins of x (depth) -> a smooth profile."""
    sub = df[[x, y]].dropna()
    if sub.empty:
        return sub
    edges = np.linspace(sub[x].min(), sub[x].max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub = sub.assign(_bin=pd.cut(sub[x], bins=edges, labels=False, include_lowest=True))
    agg = (
        sub.groupby("_bin")[y]
        .agg(["mean", "std", "count"])
        .reindex(range(n_bins))
    )
    agg[x] = centers
    return agg.dropna(subset=["mean"])


def grain_profile(members, y: str, n_bins: int, stat: str) -> pd.DataFrame:
    """Bin each replicate over depth, then combine replicates per bin.

    `stat` is one of "mean_std", "median_std", "mean_sem", "median_sem".
    Returns a DataFrame with columns: ContactDepth, center, spread, n
    (replicates per bin).
    """
    allx = np.concatenate(
        [m.data["ContactDepth"].dropna().to_numpy() for m in members]
    )
    if allx.size == 0:
        return pd.DataFrame()
    edges = np.linspace(allx.min(), allx.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # One row per replicate: its binned-mean profile on the shared edges.
    mat = np.full((len(members), n_bins), np.nan)
    for i, m in enumerate(members):
        sub = m.data[["ContactDepth", y]].dropna()
        b = pd.cut(sub["ContactDepth"], bins=edges, labels=False, include_lowest=True)
        means = sub.groupby(b)[y].mean()
        idx = means.index.to_numpy().astype(int)
        mat[i, idx] = means.to_numpy()

    n = np.sum(~np.isnan(mat), axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # Bins with a single replicate give ddof<=0 (std undefined) — expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        median = np.nanmedian(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1)  # NaN when n < 2
    std = np.nan_to_num(std, nan=0.0)

    sem = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 0)
    center = median if stat.startswith("median") else mean
    spread = sem if stat.endswith("sem") else std

    out = pd.DataFrame(
        {"ContactDepth": centers, "center": center, "spread": spread, "n": n}
    )
    return out[out["n"] > 0].reset_index(drop=True)


def main() -> None:
    # --------------------------------------------------------------------------- #
    # Custom styling
    # --------------------------------------------------------------------------- #
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #0099ff;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 0.5em 1em;
            border: none;
            border-radius: 5px;
            height: 3em;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #007acc !important;
            color: white !important;
            border: none !important;
        }
        div.stButton > button:active,
        div.stButton > button:focus {
            background-color: #0099ff !important;
            color: white !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        div[data-testid="stDataFrameContainer"] table td {
             font-size: 22px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.15rem !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 20px !important;
        }

        .stTabs [data-baseweb="tab-list"] button {
            background-color: #f0f4ff !important;
            border-radius: 12px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
            border: none !important;
            color: #1e3a8a !important;
        }

        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #dbe5ff !important;
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #e0e7ff !important;
            color: #1e3a8a !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
        }

        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------- #
    # Sidebar — upload & global controls
    # --------------------------------------------------------------------------- #
    st.sidebar.title("🔬 Nanoindentation")
    st.sidebar.caption("nanoDMA depth-profile explorer")

    # Persistent {filename -> custom group name} assignments (survive reruns).
    file_groups: dict[str, str] = st.session_state.setdefault("file_groups", {})

    uploaded = st.sidebar.file_uploader(
        "Upload `*_DYN.txt` exports",
        type=["txt"],
        accept_multiple_files=True,
        help="Drag in as many files as you like — they're grouped by fluence and grain.",
    )

    if not uploaded:
        st.title("Nanoindentation explorer")
        st.info(
            "👈 Upload your `*_DYN.txt` files in the sidebar to get started.\n\n"
            "Each file holds a dynamic depth profile (storage modulus, hardness, "
            "contact depth). File names like `s40_f46_g15_000_DYN.txt` are decoded "
            "into **sample / fluence / grain / measurement**."
        )
        st.stop()

    parsed: list[ParsedFile] = []
    errors: list[str] = []
    for f in uploaded:
        try:
            parsed.append(parse_file(f.name, f.getvalue()))
        except Exception as exc:  # noqa: BLE001 — surface parse problems to the user
            errors.append(f"{f.name}: {exc}")

    if errors:
        st.sidebar.error("Some files could not be parsed:\n\n" + "\n".join(errors))
    if not parsed:
        st.stop()

    parsed.sort(key=lambda p: (p.sample, p.fluence, grain_sort_key(p.grain), p.meas))

    st.sidebar.success(f"Loaded **{len(parsed)}** file(s).")

    y_metric = st.sidebar.selectbox(
        "Quantity to plot (Y)",
        options=[k for k in METRICS if k != "ContactDepth"] + ["ContactDepth"],
        format_func=lambda k: f"{METRICS[k][0]} ({METRICS[k][1]})",
    )
    x_metric = st.sidebar.selectbox(
        "X axis",
        options=["ContactDepth", "index"],
        format_func=lambda k: "Contact depth (nm)" if k == "ContactDepth" else "Measurement point #",
        help="Plot against contact depth (depth profile) or against the raw row index.",
    )

    mode = st.sidebar.radio(
        "Plot mode",
        options=[
            "Per measurement",
            "Mean ± std per group",
            "Median ± std per group",
            "Mean ± SEM per group",
            "Median ± SEM per group",
        ],
        help=(
            "**Per measurement** overlays every selected indent.\n\n"
            "The **per group** modes bin each replicate over depth, then combine the "
            "members of a group per bin (group = grain, or a custom group you define "
            "in the selection table):\n"
            "- **Mean ± std** — average curve, band = spread between members.\n"
            "- **Median ± std** — robust centre (less sensitive to an outlier).\n"
            "- **Mean ± SEM** — band = standard error of the mean (std / √n), the "
            "uncertainty on the average curve.\n"
            "- **Median ± SEM** — robust centre with the same standard-error band."
        ),
    )

    n_bins = st.sidebar.slider(
        "Depth bins (smoothing / averaging)",
        min_value=20,
        max_value=400,
        value=120,
        step=10,
        help="Profiles are binned over depth to tame the ~17k raw points per file.",
    )

    # If the file names carry no usable metadata, default to colouring by file.
    color_opts = ["grain", "fluence", "meas", "name"]
    has_metadata = any(p.grain != "?" for p in parsed)
    color_by = st.sidebar.selectbox(
        "Colour curves by",
        options=color_opts,
        index=color_opts.index("name") if not has_metadata else 0,
        format_func={"grain": "Grain", "fluence": "Fluence", "meas": "Measurement", "name": "File"}.get,
    )

    # For the per-group modes: aggregate by grain or by the custom groups.
    has_custom_groups = any(file_groups.get(p.name) for p in parsed)
    agg_opts = ["Custom group", "Grain"] if has_custom_groups else ["Grain"]
    agg_by = st.sidebar.selectbox(
        "Group by (for per-group modes)",
        options=agg_opts,
        help="Define custom groups in the selection table to compare arbitrary sets of files.",
    )


    # --------------------------------------------------------------------------- #
    # Selection — an "Ungrouped" tab plus one tab per custom group
    # --------------------------------------------------------------------------- #
    st.title("Nanoindentation explorer")

    # What's ticked this run. The keyed data_editors own their own check state
    # (base column is a constant True); we just read it back here.
    show_now: dict[str, bool] = {}

    def selection_table(items: list[ParsedFile], key: str):
        """Render a Show/metadata editor for `items` and record the ticked rows."""
        if not items:
            st.caption("No files here (check the filters above).")
            return
        tbl = pd.DataFrame(
            {
                "show": True,
                "Sample": [p.sample for p in items],
                "Fluence": [p.fluence for p in items],
                "Grain": [p.grain for p in items],
                "Meas.": [p.meas for p in items],
                "Group": [file_groups.get(p.name, "") for p in items],
                "Points": [len(p.data) for p in items],
                "File": [p.name for p in items],
            }
        )
        edited = st.data_editor(
            tbl,
            hide_index=True,
            width="stretch",
            column_config={
                "show": st.column_config.CheckboxColumn("Show", help="Include this curve in the plot"),
                "Group": st.column_config.TextColumn("Group", help="Custom group assignment"),
                "Points": st.column_config.NumberColumn("Points", format="%d"),
                "File": st.column_config.TextColumn("File", width="large"),
            },
            disabled=["Sample", "Fluence", "Grain", "Meas.", "Group", "Points", "File"],
            key=key,
        )
        for fn, sh in zip(edited["File"], edited["show"]):
            show_now[fn] = bool(sh)

    def assignment_controls(candidates: list[ParsedFile]) -> None:
        """Name field + button to assign the currently-shown `candidates` to a group."""
        shown_now = [p.name for p in candidates if show_now.get(p.name, True)]
        a1, a2 = st.columns([3, 1])
        name_in = a1.text_input(
            "Group name",
            placeholder="e.g. batch A",
            label_visibility="collapsed",
            key="group_name_input",
        )
        if a2.button("➕ Add shown to group", type="primary", width="stretch", disabled=not shown_now):
            name = name_in.strip()
            if not name:
                st.warning("Type a group name first.")
            else:
                for fn in shown_now:
                    file_groups[fn] = name
                # Keep the new group visible in the group filter (don't auto-hide it).
                gf = st.session_state.get("flt_group")
                if gf is not None and name not in gf:
                    gf.append(name)
                st.rerun()

    group_names = sorted({g for g in file_groups.values() if g}, key=grain_sort_key)
    members_of = {g: [p for p in parsed if file_groups.get(p.name) == g] for g in group_names}
    ungrouped = [p for p in parsed if not file_groups.get(p.name)]

    with st.expander("①  Choose which measurements to plot", expanded=True):
        st.caption(
            "Tick **Show** to pick curves; use the filters to drop whole grains / "
            "fluences / groups from the plot. Assign shown files in the **Ungrouped** "
            "tab to a named group — each group then gets its own tab."
        )

        # ---- Global filters (gate both the listing and the plot) ----
        # Show a filter for any metadata field that actually carries values
        # (i.e. not all "?"), regardless of how many distinct values there are.
        samples = sorted({p.sample for p in parsed}, key=grain_sort_key)
        fluences = sorted({p.fluence for p in parsed}, key=grain_sort_key)
        grains = sorted({p.grain for p in parsed}, key=grain_sort_key)
        group_vals = sorted({file_groups.get(p.name) or "(ungrouped)" for p in parsed}, key=grain_sort_key)
        flt_specs = []
        if samples != ["?"]:
            flt_specs.append(("Filter samples", samples, "flt_sample", "sample"))
        if fluences != ["?"]:
            flt_specs.append(("Filter fluences", fluences, "flt_fluence", "fluence"))
        if grains != ["?"]:
            flt_specs.append(("Filter grains", grains, "flt_grain", "grain"))
        if group_names:
            flt_specs.append(("Filter groups", group_vals, "flt_group", "group"))

        # Did the number of ungrouped files grow since last run (i.e. new uploads)?
        prev_n_ungrouped = st.session_state.get("_n_ungrouped", 0)
        n_ungrouped = sum(1 for p in parsed if not file_groups.get(p.name))
        more_ungrouped = n_ungrouped > prev_n_ungrouped
        st.session_state["_n_ungrouped"] = n_ungrouped

        picks: dict[str, list] = {}
        if flt_specs:
            fcols = st.columns(len(flt_specs))
            for col, (label, opts, skey, _attr) in zip(fcols, flt_specs):
                prev_key = f"_opts_{skey}"
                prev_opts = st.session_state.get(prev_key, [])
                # Manage the selection through session_state (no `default=`, which
                # otherwise overrides our edits when the key already exists).
                if skey not in st.session_state:
                    st.session_state[skey] = list(opts)
                else:
                    sel = st.session_state[skey]
                    sel[:] = [s for s in sel if s in opts]  # drop values that vanished
                    # Auto-select values that appeared since last run (new uploads),
                    # keeping the user's explicit unchecks of already-seen values.
                    sel.extend(o for o in opts if o not in prev_opts and o not in sel)
                    # New ungrouped uploads should re-show the "(ungrouped)" group.
                    if skey == "flt_group" and more_ungrouped and "(ungrouped)" in opts and "(ungrouped)" not in sel:
                        sel.append("(ungrouped)")
                st.session_state[prev_key] = list(opts)
                picks[skey] = col.multiselect(label, opts, key=skey)

        def passes_meta(p: ParsedFile) -> bool:
            """Sample / fluence / grain filters — gate the tab listings."""
            if "flt_sample" in picks and p.sample not in picks["flt_sample"]:
                return False
            if "flt_fluence" in picks and p.fluence not in picks["flt_fluence"]:
                return False
            if "flt_grain" in picks and p.grain not in picks["flt_grain"]:
                return False
            return True

        def passes(p: ParsedFile) -> bool:
            """Full filter (incl. group) — gates what actually gets plotted."""
            if not passes_meta(p):
                return False
            if "flt_group" in picks and (file_groups.get(p.name) or "(ungrouped)") not in picks["flt_group"]:
                return False
            return True

        if group_names:
            tab_labels = [f"📋 Ungrouped ({len(ungrouped)})"] + [
                f"{g} ({len(members_of[g])})" for g in group_names
            ]
            tabs = st.tabs(tab_labels)
            with tabs[0]:
                # Tabs always list their files (group filter only affects the plot).
                disp = [p for p in ungrouped if passes_meta(p)]
                selection_table(disp, key="sel_ungrouped")
                assignment_controls(disp)
                if st.button("🗑️ Clear all groups", type="primary"):
                    file_groups.clear()
                    st.session_state.pop("flt_group", None)
                    st.rerun()
            for tab, g in zip(tabs[1:], group_names):
                with tab:
                    selection_table([p for p in members_of[g] if passes_meta(p)], key=f"sel_grp_{g}")
                    if st.button(f"🗑️ Disband “{g}”", key=f"disband_{g}", type="primary"):
                        for p in members_of[g]:
                            file_groups.pop(p.name, None)
                        gf = st.session_state.get("flt_group")
                        if gf is not None and g in gf:
                            gf.remove(g)
                        st.rerun()
        else:
            disp = [p for p in parsed if passes_meta(p)]
            selection_table(disp, key="sel_all")
            assignment_controls(disp)

    selected = [p for p in parsed if passes(p) and show_now.get(p.name, True)]

    if not selected:
        st.warning("No measurements selected — tick at least one row, or relax the filters.")
        st.stop()


    # --------------------------------------------------------------------------- #
    # Plot
    # --------------------------------------------------------------------------- #
    if x_metric == y_metric:
        st.warning(
            "The X and Y axes are both set to **contact depth** — that would just "
            "plot a straight line. Pick a different quantity (hardness or storage "
            "modulus) for the Y axis in the sidebar."
        )
        st.stop()

    y_label = f"{METRICS[y_metric][0]} ({METRICS[y_metric][1]})"
    x_label = "Contact depth (nm)" if x_metric == "ContactDepth" else "Measurement point #"

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    color_keys = sorted({getattr(p, color_by) for p in selected})
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(color_keys)}

    fig = go.Figure()
    # {filename -> DataFrame} of exactly what is drawn, for the bulk CSV download.
    export_curves: dict[str, pd.DataFrame] = {}

    if mode == "Per measurement":
        for p in selected:
            df = p.data.copy()
            df["index"] = np.arange(len(df))
            prof = (
                bin_profile(df, x_metric, y_metric, n_bins)
                if x_metric == "ContactDepth"
                else df.rename(columns={y_metric: "mean"})
            )
            if prof.empty:
                continue
            x_vals = prof[x_metric] if x_metric == "ContactDepth" else prof["index"]
            export_curves[f"{p.sample}_f{p.fluence}_g{p.grain}_{p.meas}.csv"] = pd.DataFrame(
                {x_label: np.asarray(x_vals), y_label: np.asarray(prof["mean"])}
            )
            key = getattr(p, color_by)
            fig.add_trace(
                go.Scatter(
                    x=prof[x_metric] if x_metric == "ContactDepth" else prof["index"],
                    y=prof["mean"],
                    mode="lines",
                    name=p.label,
                    legendgroup=str(key),
                    line=dict(color=color_map[key], width=1.6),
                    hovertemplate=f"{p.label}<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
                )
            )
    else:  # one of the per-group aggregation modes
        # Map the chosen mode to (statistic key, centre label, spread label).
        stat, center_lbl, spread_lbl = {
            "Mean ± std per group": ("mean_std", "mean", "std"),
            "Median ± std per group": ("median_std", "median", "std"),
            "Mean ± SEM per group": ("mean_sem", "mean", "SEM"),
            "Median ± SEM per group": ("median_sem", "median", "SEM"),
        }[mode]

        if x_metric != "ContactDepth":
            st.info("Group aggregation always uses contact depth as the X axis.")

        # Group label = custom group (falling back to the file name) or grain.
        def label_of(p: ParsedFile) -> str:
            if agg_by == "Custom group":
                return file_groups.get(p.name) or f"(ungrouped) {p.label}"
            return f"grain {p.grain}"

        groups: dict[str, list[ParsedFile]] = {}
        for p in selected:
            groups.setdefault(label_of(p), []).append(p)

        ordered = sorted(groups.items(), key=lambda kv: grain_sort_key(kv[0]))
        for gi, (gname, members) in enumerate(ordered):
            prof = grain_profile(members, y_metric, n_bins, stat)
            if prof.empty:
                continue
            n_rep = len(members)
            safe = re.sub(r"[^0-9A-Za-z._-]+", "_", gname).strip("_") or "group"
            export_curves[f"{safe}_{stat}.csv"] = pd.DataFrame(
                {
                    "Contact depth (nm)": np.asarray(prof["ContactDepth"]),
                    f"{y_label} {center_lbl}": np.asarray(prof["center"]),
                    f"{y_label} {spread_lbl}": np.asarray(prof["spread"]),
                    "n members": np.asarray(prof["n"]),
                }
            )
            color = palette[gi % len(palette)]
            rgb = px.colors.hex_to_rgb(color) if color.startswith("#") else (80, 120, 200)
            band = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)"

            upper = prof["center"] + prof["spread"]
            lower = prof["center"] - prof["spread"]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([prof["ContactDepth"], prof["ContactDepth"][::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=band,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=gname,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=prof["ContactDepth"],
                    y=prof["center"],
                    mode="lines",
                    name=f"{gname} ({center_lbl}, n={n_rep})",
                    legendgroup=gname,
                    line=dict(color=color, width=2.4),
                    customdata=np.stack([prof["spread"]], axis=-1),
                    hovertemplate=(
                        f"{gname}<br>{x_label}: %{{x:.3f}}<br>"
                        f"{y_label} ({center_lbl}): %{{y:.3f}}<br>"
                        f"{spread_lbl}: %{{customdata[0]:.3f}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=620,
        font=dict(size=18),  # base font for the whole figure
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title={"grain": "Grain", "fluence": "Fluence", "meas": "Measurement", "name": "File"}.get(color_by, color_by),
        legend=dict(font=dict(size=16), title=dict(font=dict(size=17))),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="closest",
        hoverlabel=dict(font=dict(size=16)),
        title=dict(
            text=f"{METRICS[y_metric][0]} profile — {len(selected)} measurement(s)",
            font=dict(size=24),
        ),
    )
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=16))

    st.subheader("②  Plot")
    st.plotly_chart(fig, width='stretch')

    # ----------------------------------------------------------------------- #
    # Bulk download of the plotted curves (sidebar)
    # ----------------------------------------------------------------------- #
    st.sidebar.divider()
    st.sidebar.subheader("⬇️ Download plotted data")
    if export_curves:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, frame in export_curves.items():
                zf.writestr(fname, frame.to_csv(index=False))
        st.sidebar.download_button(
            f"Download {len(export_curves)} curve(s) as ZIP",
            data=buf.getvalue(),
            file_name="nanoindentation_curves.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
            help="One CSV per plotted curve, exactly as shown (after depth binning).",
        )
    else:
        st.sidebar.caption("Nothing plotted yet.")


    # --------------------------------------------------------------------------- #
    # Summary statistics
    # --------------------------------------------------------------------------- #
    st.subheader("③  Summary statistics")
    st.caption("Per-measurement mean of each quantity (over all depth points in the file).")

    rows = []
    for p in selected:
        row = {"Fluence": p.fluence, "Grain": p.grain, "Meas.": p.meas}
        for col, (lbl, unit) in METRICS.items():
            if col in p.data:
                row[f"{lbl} ({unit})"] = p.data[col].mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    st.dataframe(summary, hide_index=True, width='stretch')

    grain_stats = (
        summary.drop(columns=["Meas."])
        .groupby(["Fluence", "Grain"], as_index=False)
        .mean(numeric_only=True)
    )
    with st.expander("Averages per grain"):
        st.dataframe(grain_stats, hide_index=True, width='stretch')

    st.download_button(
        "⬇️  Download summary as CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="nanoindentation_summary.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
