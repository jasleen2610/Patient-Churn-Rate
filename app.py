import os
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.config import settings
from src.data_processor import load_and_split_data
from src.explainer import (
    get_shap_explainer, get_patient_shap,
    get_top_risk_factors, get_global_feature_importance,
)
from src.llm_advisor import generate_patient_recommendation, chat_with_data

load_dotenv()

st.set_page_config(page_title="Patient Journey Analytics", layout="wide",initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 4px;
    background: #2563eb;
}

/* hero banner */
.hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}
.hero h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 5px 0;
}
.hero p {
    color: #475569;
    font-size: 0.875rem;
    margin: 0;
}

/* metric cards */
.mcard {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out, border-color 0.25s ease-in-out;
}
.mcard:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    border-color: #cbd5e1;
}
.mcard .lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-weight: 600;
}
.mcard .val {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.mcard .sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 6px;
}

/* section pill */
.spill {
    display: inline-flex;
    align-items: center;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #1d4ed8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 22px 0 10px;
}

/* factor card */
.fcard {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    transition: all 0.25s ease-in-out;
}
.fcard:hover {
    border-color: #bfdbfe;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transform: translateY(-1px);
}

/* ai box */
.ai-wrap {
    background: #faf5ff;
    border: 1px solid #e9d5ff;
    border-left: 4px solid #8b5cf6;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 12px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
}
.ai-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}

/* chat */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
}

/* buttons */
.stButton>button {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.875rem;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 1px 2px 0 rgba(37, 99, 235, 0.1);
}
.stButton>button:hover {
    background: #1d4ed8;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    transform: translateY(-1px);
}
.stButton>button:active {
    transform: translateY(0);
}

hr {
    border-color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

def pill(label):
    st.markdown(f'<div class="spill">{label}</div>', unsafe_allow_html=True)

def _fmt(val):
    try: return f"{float(val):.0f}"
    except: return str(val)

CHART = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#0f172a", family="Inter"))
GRID  = dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")


# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading predictive models...")
def load_model_cached():
    if not os.path.exists(settings.MODEL_PATH):
        st.error("Model file not found. Run train.py first."); st.stop()
    with open(settings.MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Loading training data...")
def load_train_cached():
    X_train, *_ = load_and_split_data()
    return X_train


def make_gauge(value, title):
    c = "#ef4444" if value > .6 else ("#f59e0b" if value > .35 else "#10b981")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value*100, 1),
        number={"suffix":"%","font":{"size":36,"color":c,"family":"Inter"}},
        title={"text":title,"font":{"size":13,"color":"#64748b","family":"Inter"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#cbd5e1","tickfont":{"color":"#64748b"}},
               "bar":{"color":c,"thickness":.28},
               "bgcolor":"#ffffff","bordercolor":"#e2e8f0",
               "steps":[{"range":[0,35],"color":"#ecfdf5"},
                        {"range":[35,60],"color":"#fffbeb"},
                        {"range":[60,100],"color":"#fef2f2"}],
               "threshold":{"line":{"color":c,"width":3},"thickness":.8,"value":value*100}},
    ))
    fig.update_layout(**CHART, margin=dict(l=20,r=20,t=40,b=20), height=220)
    return fig


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;text-align:center;">
      <div style="font-size:1.2rem;font-weight:700;color:#0f172a;margin-top:4px">Patient Analytics</div>
      <div style="font-size:0.75rem;color:#64748b;margin-top:2px">Clinical Intelligence Portal</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Nav", ["Executive Overview", "Patient Profiler",
                             "Intervention Simulator", "Data Chat"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("<small style='color:#64748b'>Dataset: Diabetes 130-US Hospitals<br>"
                "Model: XGBoost · SHAP · GPT-4o-mini</small>", unsafe_allow_html=True)


# ── unpack artifacts ───────────────────────────────────────────────────────────
art          = load_model_cached()
model        = art["model"]
feature_names= art["feature_names"]
X_test       = art["X_test"]
y_test       = art["y_test"]
y_prob       = art["y_prob"]
auc          = art["auc"]

total_pts = len(y_prob)
high_risk = int((y_prob >= .6).sum())
med_risk  = int(((y_prob >= .35) & (y_prob < .6)).sum())
low_risk  = int((y_prob < .35).sum())
avg_readmission_rate = float(y_prob.mean())

# context for chatbot (built once)
imp_ctx     = get_global_feature_importance(model, feature_names)
top10_feats = imp_ctx.head(10)["Feature"].tolist()
CONTEXT = {
    "dataset": "Diabetes 130-US Hospitals (1999-2008), ~101k patient encounters",
    "target":  "30-day readmission prediction (binary)",
    "model":   "XGBoost tuned via Optuna (15 trials)",
    "auc_roc": round(auc, 4),
    "test_set_size": total_pts,
    "high_risk_count": high_risk, "high_risk_pct": round(high_risk/total_pts*100, 1),
    "medium_risk_count": med_risk,"medium_risk_pct": round(med_risk/total_pts*100, 1),
    "low_risk_count": low_risk,  "low_risk_pct": round(low_risk/total_pts*100, 1),
    "avg_readmission_probability": round(avg_readmission_rate*100, 2),
    "feature_count": len(feature_names),
    "top_10_features_by_importance": top10_feats,
    "risk_thresholds": "High ≥60%, Medium 35–60%, Low <35%",
}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Executive Overview" in page:
    hero("Executive Overview",
         "Aggregate view of patient readmission risk across the full test cohort")

    # KPI cards
    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, str(total_pts),   "Total Patients",   "#2563eb", "Test cohort"),
        (c2, f"{high_risk:,}", "High Risk",        "#ef4444", f"{high_risk/total_pts:.1%} of cohort"),
        (c3, f"{med_risk:,}",  "Medium Risk",      "#f59e0b", f"{med_risk/total_pts:.1%} of cohort"),
        (c4, f"{low_risk:,}",  "Low Risk",         "#10b981", f"{low_risk/total_pts:.1%} of cohort"),
        (c5, f"{auc:.4f}",     "AUC-ROC",          "#8b5cf6", "Model discrimination"),
    ]
    for col,val,lbl,color,sub in cards:
        with col:
            st.markdown(f"""<div class="mcard" style="border-left-color:{color}">
              <div class="lbl">{lbl}</div>
              <div class="val" style="color:{color}">{val}</div>
              <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([1.4, 1])

    with cl:
        pill("Risk Score Distribution")
        fig = go.Figure(go.Histogram(
            x=y_prob, nbinsx=50,
            marker=dict(color=y_prob,
                        colorscale=[[0,"#10b981"],[.35,"#f59e0b"],[.6,"#ef4444"],[1,"#dc2626"]],
                        line=dict(width=0)),
            opacity=.85))
        fig.add_vline(x=.35,line_dash="dash",line_color="#f59e0b",
                      annotation_text="Medium Risk",annotation_font_color="#d97706")
        fig.add_vline(x=.60,line_dash="dash",line_color="#ef4444",
                      annotation_text="High Risk",annotation_font_color="#dc2626")
        fig.update_layout(**CHART,
            xaxis=dict(title="Readmission Probability",color="#475569",**GRID),
            yaxis=dict(title="Patient Count",color="#475569",**GRID),
            margin=dict(l=10,r=10,t=10,b=10),height=280)
        st.plotly_chart(fig, key="dist_chart")

    with cr:
        pill("Risk Tier Breakdown")
        fig2 = go.Figure(go.Pie(
            labels=["High Risk","Medium Risk","Low Risk"],
            values=[high_risk,med_risk,low_risk], hole=.6,
            marker=dict(colors=["#ef4444","#f59e0b","#10b981"],
                        line=dict(color="#ffffff",width=3)),
            textfont=dict(color="#0f172a",size=11)))
        fig2.update_layout(**CHART,
            legend=dict(font=dict(color="#475569",size=11),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=10,b=10),height=280,
            annotations=[dict(text=f"<b>{avg_readmission_rate:.1%}</b><br>Avg Risk",
                              font=dict(color="#2563eb",size=14,family="Inter"),showarrow=False)])
        st.plotly_chart(fig2, key="breakdown_chart")

    pill("Primary Risk Drivers")
    imp = get_global_feature_importance(model, feature_names)
    top15 = imp.head(15)
    fig3 = go.Figure(go.Bar(
        x=top15["Importance"],
        y=top15["Feature"].str.replace("_"," ").str.title(),
        orientation="h",
        marker=dict(color=top15["Importance"],
                    colorscale=[[0,"#eff6ff"],[1,"#2563eb"]],line=dict(width=0)),
        text=[f"{v:.4f}" for v in top15["Importance"]],
        textfont=dict(color="#475569",size=10),textposition="outside"))
    fig3.update_layout(**CHART,
        xaxis=dict(title="Feature Importance",color="#475569",**GRID),
        yaxis=dict(autorange="reversed",color="#0f172a"),
        margin=dict(l=10,r=80,t=10,b=10),height=440)
    st.plotly_chart(fig3, key="importance_chart")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – PATIENT PROFILER
# ══════════════════════════════════════════════════════════════════════════════
elif "Patient Profiler" in page:
    hero("Patient Profiler",
         "Individual risk assessment with SHAP-driven explanations and AI recommendations")

    max_id = len(X_test) - 1
    idx    = st.slider("Select Patient ID", 0, min(max_id, 999), 0)
    prow   = X_test.iloc[[idx]]
    actual = int(y_test.iloc[idx])
    prob   = float(y_prob[idx])

    rlabel = "High Risk" if prob>.6 else ("Medium Risk" if prob>.35 else "Low Risk")
    rcolor = "#ef4444" if prob>.6 else ("#f59e0b" if prob>.35 else "#10b981")

    cg, ci = st.columns([1, 2])
    with cg:
        st.plotly_chart(make_gauge(prob, "Readmission Probability"), key="gauge_chart")
        st.markdown(f"""
        <div style="text-align:center;font-size:1.15rem;font-weight:700;
                    color:{rcolor};letter-spacing:.05em">{rlabel}</div>
        <div style="text-align:center;color:#475569;font-size:.85rem;margin-top:8px">
          Actual Outcome: <b style="color:#0f172a">
          {"Readmitted" if actual else "Not Readmitted"}</b></div>
        """, unsafe_allow_html=True)

    with ci:
        pill("Patient Snapshot")
        snap_cols = ["age","time_in_hospital","num_medications","num_meds_changed",
                     "number_diagnoses","num_lab_procedures","num_procedures"]
        snap_cols = [c for c in snap_cols if c in prow.columns]
        rows = "".join([
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:10px 0;border-bottom:1px solid #e2e8f0'>"
            f"<span style='color:#475569;font-size:.875rem'>{c.replace('_',' ').title()}</span>"
            f"<span style='color:#0f172a;font-weight:600'>{_fmt(prow[c].values[0])}</span></div>"
            for c in snap_cols])
        st.markdown(f"<div style='background:#ffffff;border-radius:10px;padding:20px 24px;"
                    f"border:1px solid #e2e8f0;box-shadow:0 1px 2px 0 rgba(0,0,0,0.02)'>{rows}</div>", unsafe_allow_html=True)

    # SHAP
    pill("Risk Drivers (SHAP Analysis)")
    with st.spinner("Calculating SHAP values…"):
        try:
            X_train     = load_train_cached()
            explainer   = get_shap_explainer(model, X_train)
            sv, bv, rp  = get_patient_shap(explainer, model, prow)
            top_factors = get_top_risk_factors(sv, feature_names, rp, top_n=8)

            fdf    = pd.DataFrame(top_factors)
            colors = ["#ef4444" if v>0 else "#10b981" for v in fdf["shap_value"]]
            fig_s  = go.Figure(go.Bar(
                x=fdf["shap_value"],
                y=fdf["feature"].str.replace("_"," ").str.title(),
                orientation="h",
                marker=dict(color=colors,line=dict(width=0)),
                text=[f"{v:+.3f}" for v in fdf["shap_value"]],
                textposition="outside",textfont=dict(color="#0f172a",size=11)))
            fig_s.add_vline(x=0,line_color="#cbd5e1",line_width=1)
            fig_s.update_layout(**CHART,
                xaxis=dict(title="SHAP Value (Impact on Risk)",color="#475569",**GRID),
                yaxis=dict(autorange="reversed",color="#0f172a"),
                margin=dict(l=10,r=70,t=10,b=10),height=320)
            st.plotly_chart(fig_s, key="shap_chart")

            col1,col2 = st.columns(2)
            for i,f in enumerate(top_factors[:6]):
                with (col1 if i%2==0 else col2):
                    bc = "#ef4444" if f["shap_value"]>0 else "#10b981"
                    st.markdown(f"""
                    <div class="fcard">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#0f172a;font-weight:600">
                          {f['feature'].replace('_',' ').title()}</span>
                        <span style="color:{bc};font-weight:700">{f['shap_value']:+.3f}</span>
                      </div>
                      <div style="color:#64748b;font-size:.8rem;margin-top:6px">
                        Patient value: {f['patient_value']:.1f} &nbsp;|&nbsp; {f['direction']}
                      </div></div>""", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP calculation failed: {e}")
            top_factors = []

    # AI Recommendation
    pill("Clinical Intervention Recommendations")
    patient_info = {
        "age":             prow["age"].values[0] if "age" in prow else "N/A",
        "num_medications": prow["num_medications"].values[0] if "num_medications" in prow else "N/A",
        "time_in_hospital":prow["time_in_hospital"].values[0] if "time_in_hospital" in prow else "N/A",
        "num_meds_changed":prow["num_meds_changed"].values[0] if "num_meds_changed" in prow else "N/A",
    }
    if st.button("Generate Recommendation", key="gen_rec"):
        with st.spinner("Generating recommendation…"):
            insight = generate_patient_recommendation(prob, top_factors, patient_info)
        st.markdown('<div class="ai-wrap"><div class="ai-label">AI Clinical Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(insight)   # native markdown rendering
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – INTERVENTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Intervention Simulator" in page:
    hero("Intervention Simulator",
         "Simulate clinical adjustments and observe real-time risk changes")

    idx2      = st.slider("Select Base Patient ID", 0, min(len(X_test)-1, 999), 42)
    pat_base  = X_test.iloc[[idx2]].copy()
    base_prob = float(y_prob[idx2])

    pill("Adjust Variables")
    cs, cr2 = st.columns([1.2, 1])

    with cs:
        sim = pat_base.copy()
        if "num_medications" in sim.columns:
            v = st.slider("Number of Medications", 1, 30, int(sim["num_medications"].values[0]))
            sim["num_medications"] = v
        if "num_meds_changed" in sim.columns:
            v = st.slider("Medications Changed During Visit", 0, 15, int(sim["num_meds_changed"].values[0]))
            sim["num_meds_changed"] = v
        if "time_in_hospital" in sim.columns:
            v = st.slider("Days in Hospital", 1, 14, int(sim["time_in_hospital"].values[0]))
            sim["time_in_hospital"] = v
        if "number_diagnoses" in sim.columns:
            v = st.slider("Number of Diagnoses", 1, 16, int(sim["number_diagnoses"].values[0]))
            sim["number_diagnoses"] = v

    with cr2:
        new_prob   = float(model.predict_proba(sim)[:, 1][0])
        delta      = new_prob - base_prob
        dc         = "#10b981" if delta < 0 else "#ef4444"
        dsign      = "-" if delta < 0 else "+"
        st.plotly_chart(make_gauge(new_prob, "Simulated Risk Score"), key="sim_gauge_chart")
        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px">
          <div style="color:#64748b;font-size:.875rem">
            Base Risk: <b style="color:#0f172a">{base_prob:.1%}</b></div>
          <div style="color:{dc};font-size:1.25rem;font-weight:700;margin-top:8px">
            {dsign}{abs(delta):.1%} {"reduction" if delta<0 else "increase"}</div>
        </div>""", unsafe_allow_html=True)

    pill("Risk Impact Comparison")
    fig_c = go.Figure(go.Bar(
        x=["Base Risk","After Intervention"],
        y=[base_prob*100, new_prob*100],
        marker=dict(
            color=["#ef4444" if base_prob>.6 else "#f59e0b",
                   "#10b981" if new_prob<base_prob else "#ef4444"],
            line=dict(width=0)),
        text=[f"{base_prob:.1%}", f"{new_prob:.1%}"],
        textposition="outside", textfont=dict(color="#0f172a",size=16), width=.4))
    fig_c.update_layout(**CHART,
        yaxis=dict(title="Readmission Probability (%)",color="#475569",**GRID,range=[0,100]),
        xaxis=dict(color="#0f172a"),
        margin=dict(l=20,r=20,t=10,b=10),height=280,showlegend=False)
    st.plotly_chart(fig_c, key="compare_chart")
    st.info("Adjusting variables provides a deterministic risk forecast. "
            "Reducing medication complexity or hospital stay may lower readmission probability.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – DATA CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif "Data Chat" in page:
    hero("Data Chat",
         "Ask questions about the dataset, model, and patient cohort in plain English")

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # suggested starters
    STARTERS = [
        "Which features drive readmission risk the most?",
        "How many high-risk patients are in the test set?",
        "What does an AUC-ROC of {auc:.3f} mean?".format(auc=auc),
        "Explain the risk thresholds used in this model.",
        "What dataset is this model trained on?",
        "How does XGBoost work for this prediction task?",
    ]

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:30px 0 10px;color:#475569;font-size:0.9rem">
          Ask questions about the dataset, model performance, or cohort characteristics. Try one of the suggested topics below:</div>
        """, unsafe_allow_html=True)
        cols = st.columns(3)
        for i, q in enumerate(STARTERS):
            with cols[i % 3]:
                if st.button(q, key=f"starter_{i}"):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    with st.spinner("Processing request…"):
                        reply = chat_with_data(st.session_state.chat_history, CONTEXT)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.rerun()

    # render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    if prompt := st.chat_input("Ask about the data, model, or patients…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing request…"):
                reply = chat_with_data(st.session_state.chat_history, CONTEXT)
            st.markdown(reply)
        st.session_state.chat_history.append({"role":"assistant","content":reply})

    # clear button
    if st.session_state.chat_history:
        st.divider()
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
