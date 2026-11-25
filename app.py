"""
FreshScan AI – Fruit Freshness Detection (Streamlit App)
========================================================
A beginner-friendly, interactive Machine Learning portfolio app built with:
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit,
and an optional introductory TensorFlow/Keras learning section.

Run locally with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ml_pipeline import (
    NUMERIC_FEATURES,
    check_missing_values,
    clean_dataset,
    load_dataset,
    predict_freshness,
    run_basic_tensorflow_demo,
    train_and_evaluate,
)

# ---------------------------------------------------------------------------
# 1. Color Palette (Baby Pink, Olive Green, Light Blue, Black, Off-White)
# ---------------------------------------------------------------------------
BLACK_BG = "#0D1017"
CARD_BG = "#151A24"
OFF_WHITE = "#F5F6F2"
MUTED_TEXT = "#B8C0CC"
BABY_PINK = "#F5A9C6"
OLIVE_GREEN = "#99BF63"
LIGHT_BLUE = "#7CC4FA"

st.set_page_config(
    page_title="FreshScan AI — Fruit Freshness Detection",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_custom_theme() -> None:
    """Apply clean, modern styling in Black, Off-White, Baby Pink, Olive Green, and Light Blue."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {BLACK_BG};
                color: {OFF_WHITE};
            }}
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }}
            .hero-box {{
                background: linear-gradient(135deg, rgba(153,191,99,0.14), rgba(124,196,250,0.10), rgba(245,169,198,0.14));
                border: 1px solid rgba(245, 246, 242, 0.14);
                border-radius: 18px;
                padding: 1.75rem 2rem;
                margin-bottom: 1.4rem;
            }}
            .hero-kicker {{
                color: {LIGHT_BLUE};
                font-size: 0.85rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }}
            .hero-title {{
                color: {OFF_WHITE};
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.15;
                margin: 0 0 0.6rem 0;
            }}
            .hero-subtitle {{
                color: {MUTED_TEXT};
                font-size: 1rem;
                line-height: 1.55;
                margin: 0;
            }}
            .info-card {{
                background-color: {CARD_BG};
                border: 1px solid rgba(245, 246, 242, 0.12);
                border-radius: 14px;
                padding: 1.15rem 1.25rem;
                margin-bottom: 0.9rem;
            }}
            .card-olive {{ border-top: 3px solid {OLIVE_GREEN}; }}
            .card-blue  {{ border-top: 3px solid {LIGHT_BLUE}; }}
            .card-pink  {{ border-top: 3px solid {BABY_PINK}; }}
            .card-title {{
                color: {OFF_WHITE};
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }}
            .card-body {{
                color: {MUTED_TEXT};
                font-size: 0.92rem;
                line-height: 1.5;
                margin: 0;
            }}
            .metric-tile {{
                background-color: {CARD_BG};
                border: 1px solid rgba(245, 246, 242, 0.12);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                text-align: left;
            }}
            .metric-label {{
                color: {MUTED_TEXT};
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .metric-value {{
                font-size: 1.65rem;
                font-weight: 800;
                margin-top: 0.2rem;
            }}
            .result-fresh {{
                background: rgba(153, 191, 99, 0.15);
                border: 1.5px solid {OLIVE_GREEN};
                border-radius: 16px;
                padding: 1.3rem 1.5rem;
            }}
            .result-spoiled {{
                background: rgba(245, 169, 198, 0.15);
                border: 1.5px solid {BABY_PINK};
                border-radius: 16px;
                padding: 1.3rem 1.5rem;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                background-color: {CARD_BG};
                padding: 0.45rem;
                border-radius: 14px;
                border: 1px solid rgba(245, 246, 242, 0.1);
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 10px;
                color: {MUTED_TEXT};
                font-weight: 600;
                padding: 0.5rem 1rem;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: rgba(245, 169, 198, 0.16) !important;
                color: {OFF_WHITE} !important;
                border: 1px solid rgba(245, 169, 198, 0.45);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_matplotlib_axes(fig, ax) -> None:
    """Keep Matplotlib & Seaborn figures crisp on our dark black/off-white theme."""
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(BLACK_BG)
    ax.tick_params(colors=OFF_WHITE, labelsize=9)
    ax.xaxis.label.set_color(OFF_WHITE)
    ax.yaxis.label.set_color(OFF_WHITE)
    ax.title.set_color(OFF_WHITE)
    for spine in ax.spines.values():
        spine.set_color("#2E3748")


@st.cache_data
def get_raw_data() -> pd.DataFrame:
    """Load the CSV dataset once and cache it for fast tab switching."""
    return load_dataset()


def main() -> None:
    apply_custom_theme()

    # Load and clean dataset using ml_pipeline.py
    raw_df = get_raw_data()
    missing_df, duplicate_count = check_missing_values(raw_df)
    clean_df = clean_dataset(raw_df)

    # Ensure a real trained Scikit-learn model exists in session_state on startup
    if "ml_results" not in st.session_state:
        st.session_state["ml_results"] = train_and_evaluate(
            raw_df, model_name="Random Forest", test_size=0.20, random_state=42
        )

    # -----------------------------------------------------------------------
    # Header Banner
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-kicker">Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit</div>
            <h1 class="hero-title">FreshScan AI — Fruit Freshness Detection</h1>
            <p class="hero-subtitle">
                A clear, end-to-end beginner/intermediate Machine Learning project that explores fruit storage &amp;
                quality measurements, cleans missing data, visualizes patterns, trains a Scikit-learn Pipeline,
                and makes live predictions (<span style="color:{BABY_PINK}; font-weight:600;">educational sample dataset</span>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_eda, tab_viz, tab_ml, tab_predict, tab_tf = st.tabs(
        [
            "Overview",
            "Data Analysis",
            "Visualizations",
            "Machine Learning",
            "Prediction Demo",
            "Basic TensorFlow Lab",
        ]
    )

    # =======================================================================
    # TAB 1: OVERVIEW
    # =======================================================================
    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Sample Records</div>'
                f'<div class="metric-value" style="color:{LIGHT_BLUE};">{len(raw_df)}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Clean Unique Rows</div>'
                f'<div class="metric-value" style="color:{OLIVE_GREEN};">{len(clean_df)}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Fruit Types</div>'
                f'<div class="metric-value" style="color:{BABY_PINK};">{clean_df["fruit_type"].nunique()}</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Input Features</div>'
                f'<div class="metric-value" style="color:{OFF_WHITE};">{len(clean_df.columns) - 1}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(
                """
                <div class="info-card card-olive">
                    <div class="card-title">What Problem Does This Project Solve?</div>
                    <p class="card-body">
                        Post-harvest fruit spoilage causes food waste and quality loss. FreshScan AI demonstrates how a
                        supervised Machine Learning classifier learns patterns between storage conditions (days stored,
                        temperature, humidity) and physical inspection measurements (firmness, browning, moisture loss)
                        to classify whether a fruit sample is <strong>Fresh</strong> or <strong>Spoiled</strong>.
                    </p>
                </div>
                <div class="info-card card-blue">
                    <div class="card-title">Sample Dataset Overview</div>
                    <p class="card-body">
                        The CSV file (<code>data/fruit_freshness_dataset.csv</code>) is a <strong>clearly labeled educational sample dataset</strong>
                        created for learning Data Analysis and Scikit-learn pipelines. It includes 6 fruit types
                        (Apple, Banana, Grape, Guava, Orange, Strawberry), realistic missing values (<code>NaN</code>),
                        and duplicate rows so you can practice real data cleaning steps.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                """
                <div class="info-card card-pink">
                    <div class="card-title">10-Step Beginner ML Workflow</div>
                    <p class="card-body">
                        <strong>1.</strong> Load CSV dataset with Pandas &nbsp;→&nbsp;
                        <strong>2.</strong> Explore shape &amp; summary stats &nbsp;→&nbsp;
                        <strong>3.</strong> Check missing &amp; duplicate values &nbsp;→&nbsp;
                        <strong>4.</strong> Clean &amp; impute data &nbsp;→&nbsp;
                        <strong>5.</strong> Plot Bar, Line, Pie, Histogram &amp; Heatmap charts &nbsp;→&nbsp;
                        <strong>6.</strong> Split with <code>train_test_split</code> &nbsp;→&nbsp;
                        <strong>7.</strong> Build Scikit-learn <code>Pipeline</code> &nbsp;→&nbsp;
                        <strong>8.</strong> Train Logistic Regression / Decision Tree / Random Forest &nbsp;→&nbsp;
                        <strong>9.</strong> Evaluate Accuracy &amp; Confusion Matrix &nbsp;→&nbsp;
                        <strong>10.</strong> Run live predictions &amp; explain results.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Interview Cheat Sheet — How to Explain Every Tool Used Here"):
            st.markdown(
                """
                - **Pandas**: Loads the CSV file (`pd.read_csv`), inspects columns/shape, finds missing values (`isnull().sum()`), and drops duplicates.
                - **NumPy**: Performs fast numerical array calculations such as medians, averages, and probabilities.
                - **Matplotlib & Seaborn**: Create data visualizations (bar chart, line chart, pie chart, histogram, correlation heatmap, and confusion matrix).
                - **`train_test_split`**: Separates data into training data (to teach the model) and unseen testing data (to measure honest accuracy).
                - **Scikit-learn `Pipeline`**: Combines preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) and the ML model into one clean workflow so training and prediction apply the exact same transformations.
                - **Logistic Regression / Decision Tree / Random Forest**: Interpretable classification models for predicting `Fresh` vs. `Spoiled`.
                - **Accuracy & Confusion Matrix**: Accuracy shows the overall percentage of correct predictions; the Confusion Matrix breaks down True Fresh, False Spoiled, False Fresh, and True Spoiled counts.
                - **TensorFlow / Keras (Optional Tab)**: Deep learning library demonstrated at an introductory level using a small 3-layer Sequential Neural Network.
                """
            )

    # =======================================================================
    # TAB 2: DATA ANALYSIS
    # =======================================================================
    with tab_eda:
        st.subheader("Step 1–4: Data Exploration, Missing Values & Cleaning")
        st.caption("All tables below are computed directly from `data/fruit_freshness_dataset.csv` using Pandas.")

        view_choice = st.radio(
            "Choose dataset state to inspect:",
            ["Raw CSV Data (with missing values & duplicates)", "Cleaned Data (after removing duplicates & imputing medians)"],
            horizontal=True,
        )
        active_df = raw_df if view_choice.startswith("Raw") else clean_df

        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            fruit_options = ["All Fruits"] + sorted(active_df["fruit_type"].dropna().unique().tolist())
            selected_fruit = st.selectbox("Filter preview by fruit type:", fruit_options)
        with filter_col2:
            num_rows = st.slider("Rows to preview:", min_value=5, max_value=50, value=12)

        preview_df = active_df if selected_fruit == "All Fruits" else active_df[active_df["fruit_type"] == selected_fruit]
        st.dataframe(preview_df.head(num_rows), width="stretch")

        stat_left, stat_right = st.columns(2)
        with stat_left:
            st.markdown(f"**Dataset Shape:** `{active_df.shape[0]}` rows × `{active_df.shape[1]}` columns")
            st.markdown("**Missing Values & Data Types (`raw_df.isnull().sum()`):**")
            st.dataframe(missing_df, width="stretch", hide_index=True)
            st.info(
                f"Duplicate rows detected in raw CSV: **{duplicate_count}** rows. "
                f"Total missing cells detected: **{int(missing_df['Missing Values'].sum())}** cells. "
                f"During cleaning (`clean_dataset`), duplicates are dropped and missing numeric cells are filled with column medians."
            )

        with stat_right:
            st.markdown("**Summary Statistics (`df.describe()`):**")
            st.dataframe(active_df.describe().round(2), width="stretch")

    # =======================================================================
    # TAB 3: VISUALIZATIONS
    # =======================================================================
    with tab_viz:
        st.subheader("Step 5: Exploratory Data Visualizations (Matplotlib & Seaborn)")
        st.caption("Simple, interview-ready charts plotted live from the cleaned sample dataset.")

        row1_col1, row1_col2 = st.columns(2)

        # 1. Bar Chart
        with row1_col1:
            st.markdown(f"**1. Bar Chart — Fresh vs. Spoiled Count by Fruit**")
            fig1, ax1 = plt.subplots(figsize=(6.2, 4.0))
            sns.countplot(
                data=clean_df,
                x="fruit_type",
                hue="freshness",
                palette={"Fresh": OLIVE_GREEN, "Spoiled": BABY_PINK},
                ax=ax1,
            )
            style_matplotlib_axes(fig1, ax1)
            ax1.set_title("Fruit Count by Category & Freshness")
            ax1.set_xlabel("Fruit Type")
            ax1.set_ylabel("Number of Samples")
            ax1.legend(title="Freshness", facecolor=CARD_BG, edgecolor="#2E3748", labelcolor=OFF_WHITE, title_fontsize=9)
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
            st.caption("Shows balanced Fresh vs. Spoiled counts across all 6 fruit types.")

        # 2. Pie Chart
        with row1_col2:
            st.markdown("**2. Pie Chart — Overall Class Balance**")
            counts = clean_df["freshness"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
            fig2.patch.set_facecolor(CARD_BG)
            ax2.set_facecolor(CARD_BG)
            wedges, texts, autotexts = ax2.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[OLIVE_GREEN, BABY_PINK],
                textprops={"color": OFF_WHITE, "fontsize": 11, "weight": "bold"},
                wedgeprops={"edgecolor": BLACK_BG, "linewidth": 2},
            )
            for auto in autotexts:
                auto.set_color(BLACK_BG)
            ax2.set_title("Dataset Class Distribution (Cleaned Rows)", color=OFF_WHITE)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption("Confirms whether the dataset is balanced (50% Fresh / 50% Spoiled) before training.")

        row2_col1, row2_col2 = st.columns(2)

        # 3. Line Chart
        with row2_col1:
            st.markdown("**3. Line Chart — Browning & Moisture Loss Over Storage Days**")
            trend_df = (
                clean_df.groupby("storage_days")[["browning_pct", "moisture_loss_pct"]]
                .mean()
                .reset_index()
            )
            fig3, ax3 = plt.subplots(figsize=(6.2, 4.0))
            ax3.plot(
                trend_df["storage_days"],
                trend_df["browning_pct"],
                marker="o",
                linewidth=2.2,
                color=BABY_PINK,
                label="Avg Browning (%)",
            )
            ax3.plot(
                trend_df["storage_days"],
                trend_df["moisture_loss_pct"],
                marker="s",
                linewidth=2.2,
                color=LIGHT_BLUE,
                label="Avg Moisture Loss (%)",
            )
            style_matplotlib_axes(fig3, ax3)
            ax3.set_title("Spoilage Indicators vs. Storage Days")
            ax3.set_xlabel("Storage Days")
            ax3.set_ylabel("Average Percentage (%)")
            ax3.legend(facecolor=CARD_BG, edgecolor="#2E3748", labelcolor=OFF_WHITE)
            ax3.grid(alpha=0.18, color=OFF_WHITE)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            st.caption("Illustrates how longer storage increases average surface browning and moisture loss.")

        # 4. Histogram
        with row2_col2:
            st.markdown("**4. Histogram — Feature Distribution by Freshness**")
            hist_feature = st.selectbox(
                "Pick a numerical column to plot:",
                NUMERIC_FEATURES,
                index=3,  # firmness_score default
                key="hist_feature_select",
            )
            fig4, ax4 = plt.subplots(figsize=(6.2, 3.6))
            sns.histplot(
                data=clean_df,
                x=hist_feature,
                hue="freshness",
                bins=18,
                kde=True,
                palette={"Fresh": OLIVE_GREEN, "Spoiled": BABY_PINK},
                alpha=0.55,
                ax=ax4,
            )
            style_matplotlib_axes(fig4, ax4)
            ax4.set_title(f"Distribution of {hist_feature}")
            ax4.set_xlabel(hist_feature)
            ax4.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
            st.caption("Helps identify which measurements separate Fresh fruit from Spoiled fruit.")

        # 5. Correlation Heatmap
        st.markdown("**5. Correlation Heatmap — Relationships Between Numeric Measurements**")
        corr_matrix = clean_df[NUMERIC_FEATURES].corr().round(2)
        fig5, ax5 = plt.subplots(figsize=(8.2, 4.4))
        fig5.patch.set_facecolor(CARD_BG)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.6,
            linecolor=BLACK_BG,
            ax=ax5,
            cbar_kws={"shrink": 0.85},
        )
        ax5.set_title("Feature Correlation Matrix", color=OFF_WHITE, pad=10)
        ax5.tick_params(colors=OFF_WHITE, labelsize=9)
        cbar = ax5.collections[0].colorbar
        cbar.ax.tick_params(colors=OFF_WHITE)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)
        st.caption(
            "Positive values (near +1.0) mean features rise together (e.g., storage_days & browning_pct); "
            "negative values (near -1.0) move inversely (e.g., firmness_score falls as browning increases)."
        )

    # =======================================================================
    # TAB 4: MACHINE LEARNING
    # =======================================================================
    with tab_ml:
        st.subheader("Step 6–10: Train-Test Split, Scikit-learn Pipeline & Model Evaluation")

        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
        with ctrl1:
            selected_model = st.selectbox(
                "Select a simple Scikit-learn Classifier:",
                ["Random Forest", "Logistic Regression", "Decision Tree"],
                index=0,
            )
        with ctrl2:
            test_pct = st.slider("Test Split Size (% held out for testing):", 15, 35, 20, step=5)
        with ctrl3:
            st.markdown("<br>", unsafe_allow_html=True)
            train_clicked = st.button("Train Model", type="primary", width="stretch")

        # Auto-retrain when button clicked or if user switched controls
        current_results = st.session_state["ml_results"]
        if train_clicked or current_results["model_name"] != selected_model or current_results["test_rows"] != round(current_results["total_clean_rows"] * (test_pct / 100.0)):
            current_results = train_and_evaluate(
                raw_df,
                model_name=selected_model,
                test_size=test_pct / 100.0,
                random_state=42,
            )
            st.session_state["ml_results"] = current_results

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Selected Model</div>'
                f'<div class="metric-value" style="color:{LIGHT_BLUE}; font-size:1.25rem;">{current_results["model_name"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Test Accuracy</div>'
                f'<div class="metric-value" style="color:{OLIVE_GREEN};">{current_results["accuracy"] * 100:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Training Samples</div>'
                f'<div class="metric-value" style="color:{OFF_WHITE};">{current_results["train_rows"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-label">Testing Samples</div>'
                f'<div class="metric-value" style="color:{BABY_PINK};">{current_results["test_rows"]}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        eval_left, eval_right = st.columns(2)

        with eval_left:
            st.markdown("**Confusion Matrix (Unseen Test Set)**")
            cm = current_results["confusion_matrix"]
            labels = current_results["labels"]
            fig_cm, ax_cm = plt.subplots(figsize=(5.4, 4.1))
            fig_cm.patch.set_facecolor(CARD_BG)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=sns.light_palette(OLIVE_GREEN, as_cmap=True),
                xticklabels=labels,
                yticklabels=labels,
                linewidths=1.5,
                linecolor=BLACK_BG,
                annot_kws={"size": 14, "weight": "bold"},
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Predicted Label", color=OFF_WHITE)
            ax_cm.set_ylabel("Actual Label", color=OFF_WHITE)
            ax_cm.set_title(f"{current_results['model_name']} Confusion Matrix", color=OFF_WHITE)
            ax_cm.tick_params(colors=OFF_WHITE)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            tn, fp, fn, tp = cm.ravel()
            st.caption(
                f"True Fresh: **{tn}** | False Spoiled: **{fp}** | "
                f"False Fresh: **{fn}** | True Spoiled: **{tp}**"
            )

        with eval_right:
            st.markdown("**Classification Report (`sklearn.metrics`)**")
            st.dataframe(current_results["classification_report"], width="stretch")

            imp_df = current_results["feature_importance"]
            if not imp_df.empty:
                metric_col = imp_df.columns[1]
                st.markdown(f"**Top Features Driving {current_results['model_name']} Predictions**")
                top_imp = imp_df.head(6).sort_values(by=metric_col, ascending=True)
                fig_imp, ax_imp = plt.subplots(figsize=(5.4, 3.0))
                ax_imp.barh(top_imp["Feature"], top_imp[metric_col], color=LIGHT_BLUE)
                style_matplotlib_axes(fig_imp, ax_imp)
                ax_imp.set_xlabel(metric_col)
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)

        with st.expander("How the Scikit-learn Pipeline Works (Step-by-Step Code Explanation)"):
            st.code(
                """
# 1. Numeric columns: fill missing values with median, then standardize scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 2. Categorical column (fruit_type): one-hot encode into binary columns
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# 3. Combine preprocessing + chosen classifier into one Pipeline
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, ["fruit_type"]),
    ])),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
])
                """,
                language="python",
            )

    # =======================================================================
    # TAB 5: PREDICTION DEMO
    # =======================================================================
    with tab_predict:
        active_result = st.session_state["ml_results"]
        st.subheader("Live Prediction Demo (Connected to Trained Scikit-learn Pipeline)")
        st.info(
            f"Currently using your trained **{active_result['model_name']}** pipeline "
            f"(measured test accuracy: **{active_result['accuracy'] * 100:.2f}%**). "
            "This model predicts Fresh vs. Spoiled from physical & storage measurements in our sample dataset — "
            "it does not inspect internal bacteria or guarantee food safety."
        )

        # Quick preset buttons so users can test Fresh vs Spoiled examples cleanly
        p_col1, p_col2, _ = st.columns([1, 1, 2])
        if p_col1.button("Load Fresh Sample Values"):
            st.session_state.update(
                {
                    "pred_fruit": "Apple",
                    "pred_days": 3,
                    "pred_temp": 6.0,
                    "pred_hum": 84.0,
                    "pred_firm": 8.3,
                    "pred_brown": 5.0,
                    "pred_moist": 2.5,
                }
            )
        if p_col2.button("Load Spoiled Sample Values"):
            st.session_state.update(
                {
                    "pred_fruit": "Banana",
                    "pred_days": 14,
                    "pred_temp": 24.5,
                    "pred_hum": 91.0,
                    "pred_firm": 2.4,
                    "pred_brown": 54.0,
                    "pred_moist": 18.5,
                }
            )

        form_col, out_col = st.columns([1.3, 1.0])

        with form_col:
            fruit_list = sorted(clean_df["fruit_type"].unique().tolist())
            default_fruit = st.session_state.get("pred_fruit", fruit_list[0])
            fruit_idx = fruit_list.index(default_fruit) if default_fruit in fruit_list else 0

            fruit_type = st.selectbox("Fruit Type", fruit_list, index=fruit_idx)
            s1, s2 = st.columns(2)
            with s1:
                storage_days = st.slider("Storage Days", 1, 21, int(st.session_state.get("pred_days", 4)))
                temperature_c = st.slider("Temperature (°C)", 2.0, 30.0, float(st.session_state.get("pred_temp", 8.0)), step=0.5)
                humidity_pct = st.slider("Humidity (%)", 45.0, 97.0, float(st.session_state.get("pred_hum", 82.0)), step=0.5)
            with s2:
                firmness_score = st.slider("Firmness Score (1–10)", 1.0, 10.0, float(st.session_state.get("pred_firm", 7.8)), step=0.1)
                browning_pct = st.slider("Surface Browning (%)", 0.0, 88.0, float(st.session_state.get("pred_brown", 8.0)), step=0.5)
                moisture_loss_pct = st.slider("Moisture Loss (%)", 0.5, 28.0, float(st.session_state.get("pred_moist", 3.2)), step=0.5)

        prediction = predict_freshness(
            pipeline=active_result["pipeline"],
            fruit_type=fruit_type,
            storage_days=storage_days,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            firmness_score=firmness_score,
            browning_pct=browning_pct,
            moisture_loss_pct=moisture_loss_pct,
        )

        with out_col:
            label = prediction["predicted_label"]
            conf = prediction["confidence_pct"]
            probs = prediction["probabilities"]
            box_class = "result-fresh" if label == "Fresh" else "result-spoiled"
            accent = OLIVE_GREEN if label == "Fresh" else BABY_PINK

            st.markdown(
                f"""
                <div class="{box_class}">
                    <div class="metric-label">Model Prediction ({active_result['model_name']})</div>
                    <div class="metric-value" style="color:{accent}; font-size:2.1rem;">{label}</div>
                    <p class="card-body" style="margin-top:0.4rem;">
                        Predicted probability for <strong>{label}</strong>: <strong>{conf:.2f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Class Probability Breakdown (`predict_proba`):**")
            for cls_name, pct in probs.items():
                st.write(f"**{cls_name}**: {pct:.2f}%")
                st.progress(min(max(pct / 100.0, 0.0), 1.0))

    # =======================================================================
    # TAB 6: BASIC TENSORFLOW LEARNING LAB (OPTIONAL)
    # =======================================================================
    with tab_tf:
        st.subheader("Optional Beginner TensorFlow / Keras Learning Section")
        st.caption(
            "Scikit-learn is already sufficient for this tabular dataset, so this section is purely educational "
            "to help you understand and explain basic Neural Network concepts in interviews."
        )

        tf1, tf2, tf3 = st.columns(3)
        with tf1:
            st.markdown(
                """
                <div class="info-card card-blue">
                    <div class="card-title">Input &amp; Hidden Layers</div>
                    <p class="card-body">
                        The <strong>Input Layer</strong> receives scaled numerical &amp; one-hot fruit features.
                        <strong>Hidden Layers</strong> (e.g., 16 and 8 neurons with <code>ReLU</code> activation)
                        learn combinations of those measurements.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tf2:
            st.markdown(
                """
                <div class="info-card card-olive">
                    <div class="card-title">Output Layer</div>
                    <p class="card-body">
                        A single neuron with <code>sigmoid</code> activation outputs a probability between
                        <code>0.0</code> (Fresh) and <code>1.0</code> (Spoiled) for binary classification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tf3:
            st.markdown(
                """
                <div class="info-card card-pink">
                    <div class="card-title">Epoch, Loss &amp; Accuracy</div>
                    <p class="card-body">
                        One <strong>Epoch</strong> means one full pass through the training data.
                        <strong>Loss</strong> measures prediction error (lower is better), while
                        <strong>Accuracy</strong> measures the share of correct predictions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        epochs_choice = st.slider("Number of training Epochs for demo:", min_value=5, max_value=25, value=15)
        if st.button("Run Simple TensorFlow/Keras Demo", type="primary"):
            with st.spinner("Training small 3-layer Sequential model on the cleaned CSV dataset..."):
                tf_res = run_basic_tensorflow_demo(raw_df, epochs=epochs_choice)
                st.session_state["tf_demo_result"] = tf_res

        if "tf_demo_result" in st.session_state:
            tf_res = st.session_state["tf_demo_result"]
            st.success(
                f"TensorFlow {tf_res['tf_version']} Demo Finished — Real Held-Out Test Accuracy: "
                f"**{tf_res['test_accuracy'] * 100:.2f}%** | Test Loss: **{tf_res['test_loss']:.4f}**"
            )

            hist = tf_res["history"]
            epoch_axis = list(range(1, len(hist["loss"]) + 1))

            p1, p2 = st.columns(2)
            with p1:
                fig_loss, ax_loss = plt.subplots(figsize=(5.6, 3.6))
                ax_loss.plot(epoch_axis, hist["loss"], label="Train Loss", color=BABY_PINK, linewidth=2)
                ax_loss.plot(epoch_axis, hist["val_loss"], label="Val Loss", color=LIGHT_BLUE, linewidth=2)
                style_matplotlib_axes(fig_loss, ax_loss)
                ax_loss.set_title("Epoch vs. Loss")
                ax_loss.set_xlabel("Epoch")
                ax_loss.set_ylabel("Binary Crossentropy Loss")
                ax_loss.legend(facecolor=CARD_BG, edgecolor="#2E3748", labelcolor=OFF_WHITE)
                plt.tight_layout()
                st.pyplot(fig_loss)
                plt.close(fig_loss)

            with p2:
                fig_acc, ax_acc = plt.subplots(figsize=(5.6, 3.6))
                ax_acc.plot(epoch_axis, hist["accuracy"], label="Train Accuracy", color=OLIVE_GREEN, linewidth=2)
                ax_acc.plot(epoch_axis, hist["val_accuracy"], label="Val Accuracy", color=LIGHT_BLUE, linewidth=2)
                style_matplotlib_axes(fig_acc, ax_acc)
                ax_acc.set_title("Epoch vs. Accuracy")
                ax_acc.set_xlabel("Epoch")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.legend(facecolor=CARD_BG, edgecolor="#2E3748", labelcolor=OFF_WHITE)
                plt.tight_layout()
                st.pyplot(fig_acc)
                plt.close(fig_acc)


if __name__ == "__main__":
    main()
