from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="China ESG Collaboration Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
.card {
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.65);
}
.small-note {color: #555; font-size: 0.9rem;}
.badge {
    display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
    border:1px solid rgba(49,51,63,.2); margin-right:0.35rem; font-size:0.83rem;
}
</style>
""",
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

EXPECTED_FILES = {
    "initiatives": "initiatives.csv",
    "company": "company_master_clean.csv",
    "cluster_assign": "patterns_topic_cluster_assignments.csv",
    "cluster_summary": "patterns_topic_clusters_summary.csv",
    "mechanisms": "patterns_mechanism_counts.csv",
    "partners": "patterns_partner_types_from_initiatives.csv",
    "top10": "generic_patterns_brief_top10.csv",
}


def load_csv(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


@st.cache_data(show_spinner=False)
def load_all():
    return {k: load_csv(v) for k, v in EXPECTED_FILES.items()}


files = load_all()
missing = [v for k, v in EXPECTED_FILES.items() if files[k] is None]


# ---------- Helpers ----------
def normalize_yes_no(v):
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "YES"
    if s in {"no", "n", "false", "0"}:
        return "NO"
    return "NOT STATED"


def clean_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def infer_initiatives(df: pd.DataFrame | None, files_dict: dict) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    out = df.copy()

    # Standardize some possible column name differences
    rename_map = {}
    if "ESG_block" in out.columns and "ESG_block_norm" not in out.columns:
        rename_map["ESG_block"] = "ESG_block_norm"
    if "company" in out.columns and "company_name_fixed" not in out.columns:
        rename_map["company"] = "company_name_fixed"
    if rename_map:
        out = out.rename(columns=rename_map)

    # Ensure key columns exist (so app won't break)
    for c in [
        "initiative_id",
        "report_id",
        "company_name_fixed",
        "industry_sector",
        "ownership_type",
        "year_of_report",
        "ESG_block_norm",
        "theme_tag",
        "collab_type_short",
        "initiative_title",
        "initiative_description",
        "outputs_or_outcomes",
        "KPI_present",
        "KPI_list",
        "geography",
        "actors_involved",
        "evidence_file_name",
        "page_primary",
        "evidence_quote_15w",
        "evidence_excerpt",
    ]:
        if c not in out.columns:
            out[c] = pd.NA

    # Fill important display fields
    if "company_canonical" in out.columns:
        out["company_name_fixed"] = out["company_name_fixed"].fillna(out["company_canonical"])
    out["company_name_fixed"] = out["company_name_fixed"].fillna("Unknown")
    out["collab_type_short"] = out["collab_type_short"].fillna("NOT STATED")
    out["ESG_block_norm"] = out["ESG_block_norm"].fillna("NOT STATED")
    out["theme_tag"] = out["theme_tag"].fillna("not_stated")
    out["KPI_present_norm"] = out["KPI_present"].apply(normalize_yes_no)
    out["year_of_report"] = pd.to_numeric(out["year_of_report"], errors="coerce")

    # Better display title
    out["initiative_title_display"] = out["initiative_title"].fillna("").astype(str)
    blank_mask = out["initiative_title_display"].str.strip().eq("")
    out.loc[blank_mask, "initiative_title_display"] = (
        out.loc[blank_mask, "theme_tag"].astype(str).str.replace("_", " ").str.title()
    )

    # Merge cluster assignments if available
    assign = files_dict.get("cluster_assign")
    if assign is not None and not assign.empty and "initiative_id" in assign.columns:
        keep_cols = [c for c in ["initiative_id", "cluster_id", "pattern_label"] if c in assign.columns]
        if keep_cols:
            out = out.merge(assign[keep_cols].drop_duplicates(), on="initiative_id", how="left")

    # Fill pattern_label from top10 or summary if not already present
    if "pattern_label" not in out.columns or out["pattern_label"].isna().all():
        for src_name in ["top10", "cluster_summary"]:
            src = files_dict.get(src_name)
            if src is not None and "cluster_id" in src.columns and "pattern_label" in src.columns:
                out = out.merge(
                    src[["cluster_id", "pattern_label"]].drop_duplicates(),
                    on="cluster_id",
                    how="left",
                    suffixes=("", "_src"),
                )
                if "pattern_label_src" in out.columns:
                    if "pattern_label" not in out.columns:
                        out["pattern_label"] = out["pattern_label_src"]
                    else:
                        out["pattern_label"] = out["pattern_label"].fillna(out["pattern_label_src"])
                    out = out.drop(columns=["pattern_label_src"])
                break

    return out


initiatives = infer_initiatives(files.get("initiatives"), files)
company = files.get("company")

# Merge company metadata (prefer company_master values)
if company is not None and not company.empty and "report_id" in company.columns and "report_id" in initiatives.columns:
    company_cols = [c for c in ["report_id", "company_name_clean", "industry_sector", "ownership_type", "year_of_report"] if c in company.columns]
    initiatives = initiatives.merge(company[company_cols].drop_duplicates(), on="report_id", how="left", suffixes=("", "_cm"))

    # Fill missing values from company master
    if "company_name_clean" in initiatives.columns:
        initiatives["company_name_fixed"] = initiatives["company_name_fixed"].fillna(initiatives["company_name_clean"])
    for c in ["industry_sector", "ownership_type", "year_of_report"]:
        cm_col = f"{c}_cm"
        if cm_col in initiatives.columns:
            if c not in initiatives.columns:
                initiatives[c] = initiatives[cm_col]
            else:
                initiatives[c] = initiatives[c].fillna(initiatives[cm_col])

    drop_cols = [c for c in ["company_name_clean", "industry_sector_cm", "ownership_type_cm", "year_of_report_cm"] if c in initiatives.columns]
    if drop_cols:
        initiatives = initiatives.drop(columns=drop_cols)


# ---------- Header ----------
st.title("China ESG Collaboration Dashboard")
st.markdown(
    """
<div class="small-note">
Evidence-backed dashboard built from structured ESG initiative records extracted from 73 Chinese company reports.
The current dashboard uses the shared <b>clean initiative subset</b> (the initiatives CSV in this repo).
</div>
""",
    unsafe_allow_html=True,
)

if missing:
    st.warning("Missing files in /data: " + ", ".join(missing))

if initiatives.empty:
    st.error("`data/initiatives.csv` is required. Upload it and reload.")
    st.stop()


# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

collab_options = sorted([x for x in initiatives["collab_type_short"].dropna().astype(str).unique() if x])
esg_options = sorted([x for x in initiatives["ESG_block_norm"].dropna().astype(str).unique() if x])
company_options = sorted([x for x in initiatives["company_name_fixed"].dropna().astype(str).unique() if x])

sector_series = initiatives["industry_sector"] if "industry_sector" in initiatives.columns else pd.Series(dtype=str)
sector_options = sorted([x for x in sector_series.dropna().astype(str).unique() if x])

theme_options = sorted([x for x in initiatives["theme_tag"].dropna().astype(str).unique() if x])

pattern_series = initiatives["pattern_label"] if "pattern_label" in initiatives.columns else pd.Series(dtype=str)
pattern_options = sorted([x for x in pattern_series.dropna().astype(str).unique() if x])

selected_collab = st.sidebar.multiselect("Collaboration type", collab_options, default=collab_options)
selected_esg = st.sidebar.multiselect("ESG block", esg_options, default=esg_options)
selected_sector = st.sidebar.multiselect("Industry sector", sector_options)
selected_company = st.sidebar.multiselect("Company", company_options)
selected_theme = st.sidebar.multiselect("Theme tag", theme_options)
selected_pattern = st.sidebar.multiselect("Pattern label", pattern_options)
kpi_filter = st.sidebar.selectbox("KPI present", ["All", "YES", "NO", "NOT STATED"], index=0)
search_text = st.sidebar.text_input("Search in title / evidence / actors", "")

f = initiatives.copy()

if selected_collab:
    f = f[f["collab_type_short"].astype(str).isin(selected_collab)]
if selected_esg:
    f = f[f["ESG_block_norm"].astype(str).isin(selected_esg)]
if selected_sector and "industry_sector" in f.columns:
    f = f[f["industry_sector"].astype(str).isin(selected_sector)]
if selected_company:
    f = f[f["company_name_fixed"].astype(str).isin(selected_company)]
if selected_theme:
    f = f[f["theme_tag"].astype(str).isin(selected_theme)]
if selected_pattern and "pattern_label" in f.columns:
    f = f[f["pattern_label"].astype(str).isin(selected_pattern)]
if kpi_filter != "All":
    f = f[f["KPI_present_norm"] == kpi_filter]

if search_text.strip():
    q = search_text.strip().lower()
    search_cols = [c for c in ["initiative_title_display", "initiative_description", "evidence_excerpt", "actors_involved"] if c in f.columns]
    mask = pd.Series(False, index=f.index)
    for c in search_cols:
        mask = mask | f[c].fillna("").astype(str).str.lower().str.contains(q, regex=False)
    f = f[mask]


# ---------- Top metrics ----------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Initiatives (filtered)", int(f.shape[0]))
m2.metric("Companies (filtered)", int(f["report_id"].nunique() if "report_id" in f.columns else f["company_name_fixed"].nunique()))
m3.metric("BG", int((f["collab_type_short"] == "BG").sum()))
m4.metric("BN", int((f["collab_type_short"] == "BN").sum()))
m5.metric("BS", int((f["collab_type_short"] == "BS").sum()))

with st.expander("Scope / interpretation note", expanded=False):
    st.markdown(
        "- Most visuals here are **initiative-level counts** from `initiatives.csv`.\n"
        "- The partner chart (if loaded) is **partner-edge/mention counts** from the precomputed file, so totals can exceed initiative counts.\n"
        "- Evidence fields (`evidence_file_name`, `page_primary`, `evidence_quote_15w`) make the rows auditable."
    )


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Patterns", "Initiative Explorer", "Method Notes"])


with tab1:
    c1, c2 = st.columns(2)

    # Collaboration distribution
    collab_ct = (
        f["collab_type_short"]
        .fillna("NOT STATED")
        .value_counts()
        .rename_axis("collab_type")
        .reset_index(name="initiative_count")
    )
    fig_collab = px.bar(collab_ct, x="collab_type", y="initiative_count", title="Initiative distribution by collaboration type")
    fig_collab.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    c1.plotly_chart(fig_collab, use_container_width=True)

    # ESG distribution
    esg_ct = (
        f["ESG_block_norm"]
        .fillna("NOT STATED")
        .value_counts()
        .rename_axis("ESG_block")
        .reset_index(name="initiative_count")
    )
    fig_esg = px.bar(esg_ct, x="ESG_block", y="initiative_count", title="Initiative distribution by ESG block")
    fig_esg.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    c2.plotly_chart(fig_esg, use_container_width=True)

    c3, c4 = st.columns(2)

    # Theme tags (filtered)
    theme_ct = (
        f["theme_tag"]
        .fillna("not_stated")
        .value_counts()
        .head(12)
        .rename_axis("theme_tag")
        .reset_index(name="initiative_count")
    )
    fig_theme = px.bar(theme_ct, x="initiative_count", y="theme_tag", orientation="h", title="Top theme tags (filtered)")
    fig_theme.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), yaxis={"categoryorder": "total ascending"})
    c3.plotly_chart(fig_theme, use_container_width=True)

    # Geography (filtered)
    if "geography" in f.columns:
        geo_ct = (
            f["geography"]
            .fillna("not stated")
            .astype(str)
            .replace("", "not stated")
            .value_counts()
            .head(10)
            .rename_axis("geography")
            .reset_index(name="initiative_count")
        )
        fig_geo = px.bar(geo_ct, x="initiative_count", y="geography", orientation="h", title="Geographic scope (filtered)")
        fig_geo.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), yaxis={"categoryorder": "total ascending"})
        c4.plotly_chart(fig_geo, use_container_width=True)
    else:
        c4.info("No geography column found in initiatives table.")

    st.markdown("### Precomputed pattern outputs (global)")

    c5, c6 = st.columns(2)

    # Mechanisms (global precomputed)
    mech = files.get("mechanisms")
    if mech is not None and {"mechanism", "initiative_count"}.issubset(mech.columns):
        mech_plot = mech.sort_values("initiative_count", ascending=True)
        fig_mech = px.bar(mech_plot, x="initiative_count", y="mechanism", orientation="h", title="Mechanism counts (precomputed)")
        fig_mech.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        c5.plotly_chart(fig_mech, use_container_width=True)
    else:
        c5.info("`patterns_mechanism_counts.csv` missing or invalid.")

    # Partner types by collab (global precomputed)
    partners = files.get("partners")
    if partners is not None and {"collab_type_short", "partner_type"}.issubset(partners.columns):
        count_col = "count" if "count" in partners.columns else ("edge_count" if "edge_count" in partners.columns else None)
        if count_col:
            fig_partner = px.bar(
                partners,
                x="collab_type_short",
                y=count_col,
                color="partner_type",
                barmode="stack",
                title="Partner types by collaboration type (partner-edge counts)",
            )
            fig_partner.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            c6.plotly_chart(fig_partner, use_container_width=True)
        else:
            c6.info("Partner file loaded, but count column not found.")
    else:
        c6.info("`patterns_partner_types_from_initiatives.csv` missing or invalid.")


with tab2:
    st.markdown("### Pattern explorer")

    top10 = files.get("top10")
    cluster_summary = files.get("cluster_summary")
    summary_df = None

    if top10 is not None and not top10.empty:
        summary_df = top10.copy()
    elif cluster_summary is not None and not cluster_summary.empty:
        summary_df = cluster_summary.copy()

    if summary_df is None or summary_df.empty:
        st.info("Pattern summary file not available.")
    else:
        # Normalize a few common columns
        if "n" not in summary_df.columns and "initiative_count" in summary_df.columns:
            summary_df["n"] = summary_df["initiative_count"]
        if "dominant_collab_type" not in summary_df.columns and "top_collab" in summary_df.columns:
            summary_df["dominant_collab_type"] = summary_df["top_collab"]
        if "dominant_esg_block" not in summary_df.columns and "top_esg" in summary_df.columns:
            summary_df["dominant_esg_block"] = summary_df["top_esg"]

        display_cols = [c for c in [
            "cluster_id",
            "pattern_label",
            "n",
            "top_theme",
            "dominant_collab_type",
            "dominant_esg_block",
            "dominant_partner_type",
            "top_mechanisms",
            "example_initiatives_with_evidence",
        ] if c in summary_df.columns]

        st.dataframe(summary_df[display_cols], use_container_width=True, hide_index=True)

        if {"cluster_id", "n"}.issubset(summary_df.columns):
            fig_cluster = px.bar(
                summary_df.sort_values("n", ascending=False),
                x="cluster_id",
                y="n",
                hover_data=[c for c in ["pattern_label", "top_theme", "dominant_collab_type"] if c in summary_df.columns],
                title="Pattern group sizes",
            )
            fig_cluster.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_cluster, use_container_width=True)


with tab3:
    st.markdown("### Initiative explorer (filtered rows)")

    compact_cols = [c for c in [
        "initiative_id",
        "company_name_fixed",
        "collab_type_short",
        "ESG_block_norm",
        "theme_tag",
        "pattern_label",
        "initiative_title_display",
        "KPI_present_norm",
        "geography",
        "evidence_file_name",
        "page_primary",
        "evidence_quote_15w",
    ] if c in f.columns]

    st.dataframe(f[compact_cols], use_container_width=True, hide_index=True)

    # Download filtered table
    st.download_button(
        "Download filtered initiatives CSV",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_initiatives.csv",
        mime="text/csv",
    )

    if not f.empty:
        st.markdown("### Initiative detail")
        choice = st.selectbox("Select initiative_id", options=f["initiative_id"].astype(str).tolist(), index=0)
        row = f.loc[f["initiative_id"].astype(str) == str(choice)].iloc[0]

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(f"**Title:** {clean_text(row.get('initiative_title_display')) or 'NOT FOUND'}")
            st.markdown(f"**Company:** {clean_text(row.get('company_name_fixed')) or 'NOT FOUND'}")
            st.markdown(
                " ".join([
                    f"<span class='badge'>{clean_text(row.get('collab_type_short'))}</span>",
                    f"<span class='badge'>{clean_text(row.get('ESG_block_norm'))}</span>",
                    f"<span class='badge'>{clean_text(row.get('theme_tag'))}</span>",
                ]),
                unsafe_allow_html=True,
            )
            st.markdown(f"**Description:** {clean_text(row.get('initiative_description')) or 'NOT FOUND'}")
            st.markdown(f"**Outputs/Outcomes:** {clean_text(row.get('outputs_or_outcomes')) or 'NOT FOUND'}")
            st.markdown(f"**Actors involved:** {clean_text(row.get('actors_involved')) or 'NOT FOUND'}")

        with right:
            st.markdown(f"**KPI present:** {clean_text(row.get('KPI_present_norm')) or 'NOT FOUND'}")
            st.markdown(f"**KPI list:** {clean_text(row.get('KPI_list')) or 'NOT FOUND'}")
            st.markdown(f"**Geography:** {clean_text(row.get('geography')) or 'NOT FOUND'}")
            st.markdown(f"**Evidence file:** {clean_text(row.get('evidence_file_name')) or 'NOT FOUND'}")
            st.markdown(f"**Page:** {clean_text(row.get('page_primary')) or 'NOT FOUND'}")
            st.markdown(f"**Evidence quote (short):** {clean_text(row.get('evidence_quote_15w')) or 'NOT FOUND'}")

            excerpt = clean_text(row.get("evidence_excerpt"))
            if excerpt:
                st.markdown("**Evidence excerpt**")
                st.code(excerpt[:1200])


with tab4:
    st.markdown("### Method notes (for supervisor / reviewers)")
    st.markdown(
        """
- **Data scope:** `company_master_clean.csv` indexes the 73 reports.  
- **Initiative table:** `initiatives.csv` is the shared clean, evidence-backed initiative dataset used here.  
- **Pattern discovery:** pattern files provide precomputed clusters and summaries (used in the “Patterns” tab).  
- **Mechanism tagging:** `patterns_mechanism_counts.csv` is the precomputed mechanism-tag summary.  
- **Partner chart meaning:** `patterns_partner_types_from_initiatives.csv` represents **partner-edge counts**, not initiative counts.  
- **Auditability:** initiative rows preserve evidence columns (file, page, short quote, excerpt).  
"""
    )

    status_df = pd.DataFrame({
        "file": list(EXPECTED_FILES.values()),
        "status": ["Loaded" if files[k] is not None else "Missing" for k in EXPECTED_FILES.keys()],
        "rows": [int(files[k].shape[0]) if files[k] is not None else 0 for k in EXPECTED_FILES.keys()],
    })
    st.dataframe(status_df, use_container_width=True, hide_index=True)
