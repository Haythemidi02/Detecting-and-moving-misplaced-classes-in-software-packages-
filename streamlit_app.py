import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import tempfile
import zipfile

import pandas as pd
import streamlit as st

# Ensure local package is importable when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# HuggingFace API token (from environment variable - for backend use only)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

from class_move_explorer.core.assistant import MoveClassAssistant
from class_move_explorer.evaluation.evaluator import Evaluator


APP_TITLE = "ClassMoveExplorer"
APP_SUBTITLE = "Detect misplaced Java classes and recommend better package locations."

RESULT_KEYS = (
    "last_analysis_df",
    "last_analysis_metrics",
    "last_eval_results",
    "last_benchmark_results",
)


def _set_page_config() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _inject_css() -> None:
    st.markdown(
        """
<style>
  /* Hide Streamlit default menu/footer for a cleaner “app” feel */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Layout polish */
  .block-container { padding-top: 1.25rem; padding-bottom: 2.0rem; }
  [data-testid="stHorizontalBlock"] { gap: 1.1rem; }

  /* Card */
  .cme-card {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 18px 18px 10px 18px;
    backdrop-filter: blur(10px);
  }
  .cme-title {
    font-size: 2.05rem;
    line-height: 1.15;
    font-weight: 750;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .cme-subtitle {
    margin-top: 0.35rem;
    color: rgba(255,255,255,0.72);
    font-size: 1.0rem;
  }
  .cme-badge {
    display: inline-block;
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(99,102,241,0.20);
    border: 1px solid rgba(99,102,241,0.35);
    color: rgba(255,255,255,0.92);
    margin-bottom: 10px;
  }

  /* KPI “pill” */
  .cme-kpi {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
    padding: 12px 14px;
  }
  .cme-kpi-label { font-size: 0.85rem; color: rgba(255,255,255,0.70); margin: 0; }
  .cme-kpi-value { font-size: 1.55rem; font-weight: 750; margin: 2px 0 0 0; }
  .cme-kpi-hint  { font-size: 0.80rem; color: rgba(255,255,255,0.55); margin: 2px 0 0 0; }

  /* Make buttons a bit more “app-like” */
  .stButton>button {
    border-radius: 12px;
    padding: 0.65rem 0.9rem;
  }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def _get_assistant(use_huggingface: bool = True, hf_model: str = "microsoft/phi-2",
                   hf_api_token: str = None) -> MoveClassAssistant:
    return MoveClassAssistant(use_huggingface=use_huggingface, hf_model=hf_model,
                               hf_api_token=hf_api_token)


def _configure_assistant(assistant: MoveClassAssistant, enable_dependencies: bool, enable_embeddings: bool,
                        use_huggingface: bool = True, hf_model: str = "microsoft/phi-2",
                        hf_api_token: str = None) -> MoveClassAssistant:
    # Mutates assistant for this run (intentional: avoids re-loading heavy models repeatedly).
    assistant.dependency_analyzer = assistant.dependency_analyzer if enable_dependencies else None
    assistant.embedding_analyzer = assistant.embedding_analyzer if enable_embeddings else None
    # Reinitialize LLM analyzer with HuggingFace settings
    from class_move_explorer.analyzers.llm_analyzer import LLMAnalyzer
    assistant.llm_analyzer = LLMAnalyzer(use_huggingface=use_huggingface, hf_model=hf_model,
                                          hf_api_token=hf_api_token)
    return assistant


def _extract_zip_to_temp(uploaded_zip: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    base_dir = Path(tempfile.mkdtemp(prefix="classmoveexplorer_"))
    zip_path = base_dir / "project.zip"
    zip_path.write_bytes(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir / "unzipped")

    unzipped = base_dir / "unzipped"
    # If the zip contains a single top-level folder, use it as project root
    top_level = [p for p in unzipped.iterdir()]
    if len(top_level) == 1 and top_level[0].is_dir():
        return top_level[0]
    return unzipped


def _count_java_files(project_root: Path) -> int:
    try:
        return sum(1 for _ in project_root.rglob("*.java"))
    except Exception:
        return 0


def _clear_results() -> None:
    for k in RESULT_KEYS:
        st.session_state.pop(k, None)


def _kpi(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
<div class="cme-kpi">
  <p class="cme-kpi-label">{label}</p>
  <p class="cme-kpi-value">{value}</p>
  <p class="cme-kpi-hint">{hint}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_metrics_cards(metrics: dict) -> None:
    if not metrics:
        st.info("No metrics available yet.")
        return

    total = int(metrics.get("total_classes", 0))
    misplaced = int(metrics.get("misplaced_count", 0))
    misplacement_rate = metrics.get("misplacement_rate", None)
    avg_conf = metrics.get("average_confidence", None)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Total classes", f"{total}", "Parsed from your project")
    with c2:
        _kpi("Misplaced found", f"{misplaced}", "Potential relocation candidates")
    with c3:
        _kpi(
            "Misplacement rate",
            f"{misplacement_rate:.1%}" if isinstance(misplacement_rate, (int, float)) else "—",
            "Misplaced / total",
        )
    with c4:
        _kpi(
            "Avg confidence",
            f"{avg_conf:.2f}" if isinstance(avg_conf, (int, float)) else "—",
            "Across misplaced classes",
        )


def _render_analysis_dashboard(df: pd.DataFrame, metrics: dict) -> None:
    _render_metrics_cards(metrics)

    st.divider()
    misplaced_df = df[df["is_misplaced"] == True].sort_values("confidence", ascending=False)  # noqa: E712

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Confidence distribution")
        # Histogram-like binning without extra deps
        series = df["confidence"].fillna(0.0).clip(0.0, 1.0)
        bins = pd.cut(series, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        hist = bins.value_counts().sort_index()
        # Altair/Streamlit can choke on IntervalIndex; convert buckets to plain strings.
        hist_df = hist.reset_index()
        hist_df.columns = ["bucket", "count"]
        hist_df["bucket"] = hist_df["bucket"].astype(str)
        st.bar_chart(hist_df.set_index("bucket")["count"], height=220)
    with c2:
        st.subheader("Top misplaced recommendations")
        if misplaced_df.empty:
            st.info("No misplaced classes detected.")
        else:
            st.dataframe(
                misplaced_df[["class_name", "current_package", "suggested_package", "confidence", "method_used"]],
                use_container_width=True,
                hide_index=True,
                height=260,
            )


def page_dashboard() -> None:
    repo_root = Path(__file__).resolve().parent

    # Top “hero” header
    st.markdown(
        f"""
<div class="cme-card">
  <div class="cme-badge">Java package placement • structural + semantic + LLM reasoning</div>
  <h1 class="cme-title">{APP_TITLE}</h1>
  <div class="cme-subtitle">{APP_SUBTITLE}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # Central “control panel” (no sidebar-centric UX)
    left, center, right = st.columns([1, 2.2, 1])
    with center:
        st.markdown('<div class="cme-card">', unsafe_allow_html=True)
        st.subheader("Run analysis")

        project_root: Optional[Path] = None

        # ZIP uploader (outside a form so changes apply immediately)
        uploaded = st.file_uploader("Upload a zipped Java project", type=["zip"], key="project_zip")
        zip_fingerprint = (uploaded.name, uploaded.size) if uploaded is not None else None

        if "zip_fingerprint" not in st.session_state:
            st.session_state.zip_fingerprint = None
        if "project_root_path" not in st.session_state:
            st.session_state.project_root_path = None

        # If user uploaded a new ZIP, clear previous results and extract new project
        if uploaded is not None and zip_fingerprint != st.session_state.zip_fingerprint:
            _clear_results()
            project_root = _extract_zip_to_temp(uploaded)
            st.session_state.project_root_path = str(project_root)
            st.session_state.zip_fingerprint = zip_fingerprint
            st.success("ZIP extracted. Ready to run.")
        elif st.session_state.project_root_path:
            project_root = Path(st.session_state.project_root_path)
            st.caption("Current project")
            st.code(str(project_root))
        else:
            st.info("Upload a `.zip` containing your Java project root (the folder that contains the full source tree).")

        st.divider()
        opt1, opt2, opt3 = st.columns([1, 1, 1])
        with opt1:
            enable_dependencies = st.toggle("Dependencies", value=True)
        with opt2:
            enable_embeddings = st.toggle("Embeddings", value=True)
        with opt3:
            run_evaluation = st.toggle("Evaluation", value=False)

        st.subheader("HuggingFace Settings")
        col1, col2 = st.columns([2, 1])
        with col1:
            hf_api_token = st.text_input("HuggingFace API Token", type="password",
                                          help="Get your free token at https://huggingface.co/settings/tokens",
                                          value=HF_API_TOKEN, label_visibility="collapsed")
        with col2:
            hf_model = st.selectbox(
                "LLM Model",
                ["microsoft/phi-2", "Qwen/Qwen2-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
                index=0,
                help="Code-specialized models available via HuggingFace Inference API"
            )

        use_huggingface = bool(hf_api_token)

        apply_threshold = st.toggle("Apply confidence filter (display only)", value=True)
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            help="Filters what you see in the dashboard. It does not change the underlying detection results or CSV export.",
        )
        misplace_ratio = st.slider(
            "Evaluation misplace ratio",
            0.01,
            0.80,
            0.25,
            0.01,
            help="Only used when Evaluation is enabled.",
        )

        c_run, c_reset = st.columns([1, 1])
        with c_run:
            run_btn = st.button("Run", type="primary", use_container_width=True)
        with c_reset:
            if st.button("Reset results", use_container_width=True):
                _clear_results()
                st.session_state.project_root_path = None
                st.session_state.zip_fingerprint = None
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if not run_btn:
        # If we already have results in session, show them immediately (keeps the app feeling “stateful”)
        existing_df = st.session_state.get("last_analysis_df", None)
        existing_metrics = st.session_state.get("last_analysis_metrics", None)
        if existing_df is None or existing_metrics is None:
            st.write("")
            st.info("Select an input above and click **Run** to generate your dashboard.")
            return

    if project_root is None:
        st.error("Please upload a ZIP first.")
        return

    if not project_root.exists() or not project_root.is_dir():
        st.error("Selected project root is not a directory.")
        return

    java_count = _count_java_files(project_root)
    if java_count == 0:
        st.warning("No `.java` files found in the selected folder. Make sure you uploaded/selected a Java project root.")
    elif run_evaluation and java_count < 5:
        st.warning(
            "Evaluation needs a reasonably-sized project. "
            f"Only **{java_count}** Java file(s) were found, so Precision/Recall may be meaningless. "
            "If you uploaded a ZIP, ensure it contains the full project root (not just a single file/folder)."
        )

    assistant = _configure_assistant(
        _get_assistant(use_huggingface=use_huggingface, hf_model=hf_model,
                       hf_api_token=hf_api_token if use_huggingface else None),
        enable_dependencies=enable_dependencies,
        enable_embeddings=enable_embeddings,
        use_huggingface=use_huggingface,
        hf_model=hf_model,
        hf_api_token=hf_api_token if use_huggingface else None
    )

    with st.spinner("Running analysis… (first run may download/load models)"):
        df = assistant.analyze_and_recommend(project_path=str(project_root), output_csv=None)

    if df is None or df.empty:
        st.warning("No results produced (no Java classes found or analysis returned empty results).")
        return

    # Never mutate model decisions for display filtering; keep df as the raw output.
    df_display = df
    if apply_threshold and confidence_threshold > 0.0:
        df_display = df[(df["is_misplaced"] != True) | (df["confidence"] >= confidence_threshold)]  # noqa: E712

    metrics = assistant.metrics.metrics_calculated
    st.session_state.last_analysis_df = df
    st.session_state.last_analysis_metrics = metrics

    eval_results = None
    if run_evaluation:
        evaluator = Evaluator(assistant)
        with st.spinner("Running evaluation…"):
            eval_results = evaluator.run_evaluation(str(project_root), misplace_ratio=float(misplace_ratio))
        st.session_state.last_eval_results = eval_results

    st.success("Done.")

    tab_overview, tab_table, tab_eval = st.tabs(["Overview", "Results table", "Evaluation"])

    with tab_overview:
        st.subheader("Analysis overview")
        _render_analysis_dashboard(df_display, metrics)

        st.divider()
        st.subheader("Export")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download analysis CSV",
            data=csv_bytes,
            file_name="class_placement_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_table:
        st.subheader("All results")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        with st.expander("Explain a class (debug why it was/wasn't flagged)"):
            try:
                class_options = df["class_name"].tolist()
                selected = st.selectbox("Class", class_options)
                if selected:
                    # Rebuild the same intermediate artifacts used for the run
                    classes_data = assistant.project_analyzer.analyze_project(str(project_root))
                    dep_graph = (
                        assistant.dependency_analyzer.analyze_dependencies(classes_data)
                        if assistant.dependency_analyzer is not None
                        else {}
                    )
                    embeddings = (
                        assistant.embedding_analyzer.compute_embeddings(classes_data)
                        if assistant.embedding_analyzer is not None
                        else {}
                    )
                    class_map = {c["class_name"]: c for c in classes_data}
                    info = assistant.llm_analyzer.explain_class(class_map[selected], dep_graph, embeddings, classes_data)
                    st.json(info)
            except Exception as e:
                st.error(f"Explain failed: {e}")

    with tab_eval:
        st.subheader("Evaluation")
        st.caption("Choose between input-dependent corruption evaluation, or a repeatable synthetic benchmark.")

        mode = st.radio(
            "Mode",
            ["Synthetic benchmark (recommended)", "Project corruption (depends on your input)"],
            horizontal=True,
        )

        if mode.startswith("Synthetic"):
            st.markdown('<div class="cme-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                bench_projects = st.number_input("Projects", min_value=5, max_value=200, value=25, step=5)
            with c2:
                bench_classes = st.number_input("Classes per type", min_value=1, max_value=10, value=3, step=1)
            with c3:
                bench_ratio = st.slider("Misplace ratio", 0.05, 0.70, 0.25, 0.05)

            bench_seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
            run_bench = st.button("Run benchmark", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if run_bench:
                assistant = _get_assistant(use_huggingface=use_huggingface, hf_model=hf_model,
                                            hf_api_token=hf_api_token if use_huggingface else None)
                evaluator = Evaluator(assistant)
                with st.spinner("Running synthetic benchmark…"):
                    bench = evaluator.run_synthetic_benchmark(
                        num_projects=int(bench_projects),
                        classes_per_type=int(bench_classes),
                        misplace_ratio=float(bench_ratio),
                        seed=int(bench_seed),
                    )
                st.session_state.last_benchmark_results = bench

            bench = st.session_state.get("last_benchmark_results", None)
            if not bench:
                st.info("Run the benchmark to get stable Precision/Recall/F1 results for the current algorithm.")
            elif "error" in bench:
                st.error(bench.get("error", "Benchmark failed."))
            else:
                cols = st.columns(4)
                cols[0].metric("Precision (mean)", f"{bench.get('precision_mean', 0.0):.2%}")
                cols[1].metric("Recall (mean)", f"{bench.get('recall_mean', 0.0):.2%}")
                cols[2].metric("F1 (mean)", f"{bench.get('f1_mean', 0.0):.2%}")
                cols[3].metric("Suggestion accuracy (mean)", f"{bench.get('suggestion_accuracy_mean', 0.0):.2%}")
                st.caption(
                    f"Std dev — P: {bench.get('precision_std', 0.0):.2%} • "
                    f"R: {bench.get('recall_std', 0.0):.2%} • "
                    f"F1: {bench.get('f1_std', 0.0):.2%} • "
                    f"Acc: {bench.get('suggestion_accuracy_std', 0.0):.2%}"
                )
                st.divider()
                st.json(bench)
        else:
            stored = eval_results if eval_results is not None else st.session_state.get("last_eval_results", None)
            if not stored:
                st.info("Enable **Evaluation** before running analysis to compute Precision / Recall / F1 on your current input.")
            elif "error" in stored:
                st.error(stored.get("error", "Evaluation failed."))
            else:
                cols = st.columns(4)
                cols[0].metric("Precision", f"{stored.get('precision', 0.0):.2%}")
                cols[1].metric("Recall", f"{stored.get('recall', 0.0):.2%}")
                cols[2].metric("F1 score", f"{stored.get('f1_score', 0.0):.2%}")
                cols[3].metric("Suggestion accuracy", f"{stored.get('suggestion_accuracy', 0.0):.2%}")
                st.divider()
                st.json(stored)


def main() -> None:
    _set_page_config()
    _inject_css()

    page_dashboard()


if __name__ == "__main__":
    main()

