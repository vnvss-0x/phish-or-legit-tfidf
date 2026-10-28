"""Lab 04 – Phish or Legit?  (Streamlit UI – redesigned dark cyber theme)
Run:  streamlit run app.py
"""
from __future__ import annotations

import re

import numpy as np
import streamlit as st
from sklearn.metrics import precision_recall_curve

from src.pipeline import (
    contributions,
    load_data,
    make_pipeline,
    predict,
    split,
    word_weights,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phish or Legit? — Message Classifier",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] label { color: #8b949e !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] p { color: #c9d1d9 !important; }

/* ── main heading ── */
h1 { color: #f0f6fc !important; letter-spacing: -0.5px; }
h2, h3, h4 { color: #e6edf3 !important; }

/* ── cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-red   { border-left: 4px solid #f85149; }
.card-green { border-left: 4px solid #3fb950; }
.card-blue  { border-left: 4px solid #388bfd; }
.card-yellow{ border-left: 4px solid #d29922; }

/* ── verdict banner ── */
.verdict-phish {
    background: linear-gradient(135deg, #3d1a1a, #2d1010);
    border: 1px solid #f85149;
    border-radius: 10px;
    padding: 16px 22px;
    text-align: center;
}
.verdict-legit {
    background: linear-gradient(135deg, #0f2d1a, #0a1f12);
    border: 1px solid #3fb950;
    border-radius: 10px;
    padding: 16px 22px;
    text-align: center;
}
.verdict-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.verdict-score {
    font-size: 0.95rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

/* ── score meter ── */
.meter-wrap {
    background: #21262d;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin: 8px 0 4px;
}
.meter-fill-phish {
    background: linear-gradient(90deg, #da3633, #f85149);
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}
.meter-fill-legit {
    background: linear-gradient(90deg, #238636, #3fb950);
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}

/* ── feature pills ── */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 3px;
    font-weight: 600;
}
.pill-phish { background: rgba(248,81,73,0.15); border: 1px solid #f85149; color: #f85149; }
.pill-legit { background: rgba(63,185,80,0.15); border: 1px solid #3fb950; color: #3fb950; }

/* ── highlighted text ── */
.msg-highlight {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    line-height: 1.8;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    word-break: break-word;
}
.tok-phish { background: rgba(248,81,73,0.22); border-radius: 3px; padding: 1px 2px; color: #f85149; font-weight: 600; }
.tok-legit { background: rgba(63,185,80,0.18);  border-radius: 3px; padding: 1px 2px; color: #3fb950; font-weight: 600; }

/* ── notice banner ── */
.notice-bar {
    background: #272115;
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #d29922;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── example buttons ── */
.stButton button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 8px 12px !important;
    text-align: left !important;
    width: 100% !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.stButton button:hover {
    background: #2d333b !important;
    border-color: #388bfd !important;
}

/* ── text area + classify button ── */
textarea {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}
textarea:focus { border-color: #388bfd !important; box-shadow: 0 0 0 2px rgba(56,139,253,0.15) !important; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
}

/* ── slider ── */
[data-testid="stSlider"] > div > div { background: #30363d !important; }

/* ── table ── */
[data-testid="stTable"] table { background: #161b22; color: #c9d1d9; }
[data-testid="stTable"] th { background: #21262d; color: #58a6ff !important; border-bottom: 1px solid #30363d; }
[data-testid="stTable"] td { border-bottom: 1px solid #21262d; }

/* ── tab strip ── */
[data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; }
[data-baseweb="tab"] { color: #8b949e; }
[aria-selected="true"][data-baseweb="tab"] { color: #58a6ff !important; border-bottom: 2px solid #388bfd !important; }

/* ── caption text ── */
small, .stCaption { color: #6e7681 !important; }

/* ── metric ── */
[data-testid="metric-container"] { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #f0f6fc !important; font-family: 'JetBrains Mono', monospace; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Example messages  (2 phish, 2 legit, 1 misclassified)
# ──────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "🎣 Phishing",
        "tag": "phish",
        "text": "URGENT: Your Northbank account is locked. Verify your identity now or lose access: http://bit.ly/nb-verify88",
    },
    {
        "label": "🎣 Phishing",
        "tag": "phish",
        "text": "ParcelGo: your package is on hold. Confirm delivery details immediately: http://tinyurl.com/pg-track99",
    },
    {
        "label": "✅ Legitimate",
        "tag": "legit",
        "text": "Hi, your monthly statement from Northbank is now available. Log in at https://www.northbank.example.com to view it.",
    },
    {
        "label": "✅ Legitimate",
        "tag": "legit",
        "text": "SwiftPost: your order #8777 has shipped and will arrive Thursday. Track it at https://www.example.com/orders/8777.",
    },
    {
        "label": "⚠️ Misclassified (legit → phish)",
        "tag": "mis",
        "text": "SwiftPost could not deliver your package today. This is your final notice. https://www.example.com/orders/8777",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training classifiers on synthetic data…")
def get_models():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    pipes = {}
    for kind in ("lr", "nb", "lr_cal"):
        pipes[kind] = make_pipeline(kind).fit(Xtr, ytr)
    # pre-compute PR curves on test set
    pr_curves = {}
    for kind in ("lr", "nb"):
        proba = pipes[kind].predict_proba(Xte)[:, 1]
        prec, rec, thr = precision_recall_curve(yte, proba)
        pr_curves[kind] = (prec, rec, thr)
    return pipes, pr_curves


pipes, pr_curves = get_models()

# ──────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────────────
if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]["text"]
if "result" not in st.session_state:
    st.session_state.result = None

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    classifier_choice = st.radio(
        "**Classifier**",
        options=["Logistic Regression", "Naive Bayes", "LR (calibrated)"],
        index=0,
        help="Choose which backend model to use for classification.",
    )
    clf_key = {"Logistic Regression": "lr", "Naive Bayes": "nb", "LR (calibrated)": "lr_cal"}[classifier_choice]
    active_pipe = pipes[clf_key]

    st.markdown("---")
    threshold = st.slider(
        "**Decision threshold**",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Score ≥ threshold → classified as PHISHING. Drag to see the verdict change live.",
    )

    st.markdown("---")
    st.markdown("### 📩 Example messages")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex["label"], key=f"ex_{i}"):
            st.session_state.text = ex["text"]
            st.session_state.result = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div class='notice-bar'>⚠️ This tool supports a human decision and "
        "does not block messages by itself. The data is synthetic.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <span style="font-size:2.6rem">🎣</span>
  <div>
    <h1 style="margin:0;font-size:2rem">Phish or Legit?</h1>
    <p style="margin:0;color:#8b949e;font-size:0.9rem">
      TF-IDF + Logistic Regression / Naive Bayes — SYNTHETIC dataset
    </p>
  </div>
</div>
<div class="notice-bar" style="margin-bottom:24px">
  ⚠️ This tool supports a human decision and does not block messages by itself. The data is synthetic.
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Main layout: input | results
# ──────────────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown("### ✉️ Message to classify")
    text_input = st.text_area(
        label="Paste or type a message:",
        value=st.session_state.text,
        height=150,
        label_visibility="collapsed",
        key="msg_input",
        placeholder="Paste a message here…",
    )
    st.session_state.text = text_input

    clicked = st.button("🔍 Classify", type="primary", use_container_width=True)

    if clicked:
        try:
            score = float(active_pipe.predict_proba([text_input])[0][1])
            contrib = contributions(active_pipe, text_input) if clf_key != "nb" else {}
            w_weights = word_weights(active_pipe, text_input) if clf_key != "nb" else {}
            st.session_state.result = {
                "score": score,
                "contrib": contrib,
                "w_weights": w_weights,
                "clf_key": clf_key,
            }
        except ValueError as err:
            st.error(str(err))
            st.session_state.result = None

# ──────────────────────────────────────────────────────────────────────────────
# Results column
# ──────────────────────────────────────────────────────────────────────────────
with col_out:
    result = st.session_state.result

    if result:
        score = result["score"]
        # Re-evaluate label based on current (live) threshold
        label = "phish" if score >= threshold else "legit"
        pct = int(score * 100)
        meter_cls = "meter-fill-phish" if label == "phish" else "meter-fill-legit"
        verdict_cls = "verdict-phish" if label == "phish" else "verdict-legit"
        verdict_icon = "⚠️ PHISHING-STYLE" if label == "phish" else "✅ LOOKS LEGIT"
        verdict_color = "#f85149" if label == "phish" else "#3fb950"

        # ── Verdict card ──────────────────────────────────────────────────────
        st.markdown(
            f"""
<div class="{verdict_cls}">
  <div class="verdict-label" style="color:{verdict_color}">{verdict_icon}</div>
  <div class="verdict-score">
    phish score = <strong>{score:.3f}</strong> &nbsp;|&nbsp; threshold = <strong>{threshold:.2f}</strong>
    &nbsp;|&nbsp; classifier = <strong>{classifier_choice}</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Score meter ───────────────────────────────────────────────────────
        st.markdown(
            f"""
<div style="margin:14px 0 4px">
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#8b949e;margin-bottom:4px">
    <span>Legit (0.0)</span><span>Threshold ({threshold:.2f})</span><span>Phish (1.0)</span>
  </div>
  <div class="meter-wrap">
    <div class="{meter_cls}" style="width:{pct}%"></div>
  </div>
  <div style="font-size:0.76rem;color:#6e7681;text-align:center">Phish probability score: {score:.3f}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Tabs: Explanation | Highlighted text ──────────────────────────────
        tab_feat, tab_hi, tab_pr = st.tabs(["📊 Feature Explanation", "🔆 Highlighted Text", "📈 PR Curve"])

        with tab_feat:
            contrib = result["contrib"]
            if contrib:
                items = list(contrib.items())
                phish_feats = [(k, v) for k, v in items if v > 0][:8]
                legit_feats = [(k, v) for k, v in items if v < 0][:8]

                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown(
                        "<p style='color:#f85149;font-weight:600;font-size:0.88rem;margin-bottom:6px'>"
                        "🔴 Pushes toward PHISH</p>",
                        unsafe_allow_html=True,
                    )
                    if phish_feats:
                        pills = " ".join(
                            f"<span class='pill pill-phish'>{k} <em style='color:#8b949e;font-style:normal'>"
                            f"{v:+.2f}</em></span>"
                            for k, v in phish_feats
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                        st.table(
                            {"feature": [k for k, _ in phish_feats],
                             "log-odds contribution": [round(v, 3) for _, v in phish_feats]}
                        )
                    else:
                        st.caption("No phish-pushing features found.")

                with fc2:
                    st.markdown(
                        "<p style='color:#3fb950;font-weight:600;font-size:0.88rem;margin-bottom:6px'>"
                        "🟢 Pushes toward LEGIT</p>",
                        unsafe_allow_html=True,
                    )
                    if legit_feats:
                        pills = " ".join(
                            f"<span class='pill pill-legit'>{k} <em style='color:#8b949e;font-style:normal'>"
                            f"{v:+.2f}</em></span>"
                            for k, v in legit_feats
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                        st.table(
                            {"feature": [k for k, _ in legit_feats],
                             "log-odds contribution": [round(v, 3) for _, v in legit_feats]}
                        )
                    else:
                        st.caption("No legit-pushing features found.")
            else:
                st.info("Feature-level contributions are only available for Logistic Regression models.")

        with tab_hi:
            ww = result["w_weights"]
            if ww:
                parts = []
                for tok in re.findall(r"\w+|\W+", st.session_state.text):
                    w = ww.get(tok.lower(), 0.0)
                    if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.15:
                        cls = "tok-phish" if w > 0 else "tok-legit"
                        parts.append(f"<span class='{cls}'>{tok}</span>")
                    else:
                        parts.append(tok.replace("<", "&lt;").replace(">", "&gt;"))
                html_msg = "<div class='msg-highlight'>" + "".join(parts) + "</div>"
                st.markdown(html_msg, unsafe_allow_html=True)
                st.caption("🔴 red = word pushes toward phishing | 🟢 green = word pushes toward legit")
            else:
                st.info("Text highlighting is only available for Logistic Regression models.")

        with tab_pr:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            pr_key = "lr" if clf_key in ("lr", "lr_cal") else "nb"
            prec, rec, thr = pr_curves[pr_key]

            fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor="#161b22")
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.xaxis.label.set_color("#8b949e")
            ax.yaxis.label.set_color("#8b949e")
            ax.set_xlabel("Recall", fontsize=9)
            ax.set_ylabel("Precision", fontsize=9)
            ax.set_title(
                f"Precision / Recall vs Threshold — {classifier_choice}",
                fontsize=9, color="#c9d1d9",
            )
            thr_ext = np.append(thr, 1.0)
            ax.plot(thr_ext, prec, color="#388bfd", lw=2, label="Precision")
            ax.plot(thr_ext, rec,  color="#3fb950", lw=2, label="Recall")
            ax.axvline(threshold, color="#d29922", lw=1.4, ls="--", label=f"threshold={threshold:.2f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("The dashed line shows the currently selected decision threshold.")
    else:
        # Placeholder when no result yet
        st.markdown(
            """
<div class="card card-blue" style="text-align:center;padding:40px 24px;color:#8b949e">
  <div style="font-size:3rem;margin-bottom:12px">🔍</div>
  <div style="font-size:1rem;font-weight:500;color:#c9d1d9">Ready to analyse</div>
  <div style="font-size:0.85rem;margin-top:6px">
    Type or paste a message on the left, then click <strong style="color:#388bfd">Classify</strong>.
    <br>Or pick one of the example messages from the sidebar.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
