import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import os

from src.predict import predict_student

# ══════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="EduPredict – Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
defaults = {
    "page":         "landing",   # landing | login | main
    "dark_mode":    False,
    "history":      [],
    "username":     "",
    "logged_in":    False,
    "student_name": "John Doe",
    "study_hours":  5.0,
    "attendance":   75.0,
    "assignments":  60.0,
    "quizzes":      60.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(page):
    st.session_state.page = page
    st.rerun()

# ══════════════════════════════════════════════════
# GLOBAL CSS  (dark-mode aware)
# ══════════════════════════════════════════════════
dark = st.session_state.dark_mode

# colours
if dark:
    BG       = "#0f172a"
    CARD     = "#1e293b"
    SIDEBAR  = "#1e293b"
    TXT      = "#f1f5f9"
    SUBTXT   = "#94a3b8"
    BORDER   = "#334155"
    INPUT_BG = "#0f172a"
    BTN_TXT  = "#ffffff"
else:
    BG       = "#f8fafc"
    CARD     = "#ffffff"
    SIDEBAR  = "#f1f5f9"
    TXT      = "#1e293b"
    SUBTXT   = "#64748b"
    BORDER   = "#e2e8f0"
    INPUT_BG = "#ffffff"
    BTN_TXT  = "#ffffff"

ACCENT   = "#6366f1"
SUCCESS  = "#22c55e"
DANGER   = "#ef4444"
WARNING  = "#f97316"

st.markdown(f"""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden !important; }}
.block-container {{ padding-top: 0.5rem !important; padding-bottom: 1rem !important; }}

/* ── Global ── */
html, body, .stApp {{
    background-color: {BG} !important;
    color: {TXT} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}}

/* ── ALL text elements ── */
.stApp p, .stApp div, .stApp span, .stApp li,
.stApp label, .stApp small, .stApp strong, .stApp em,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: {TXT} !important;
}}
.stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: {TXT} !important; }}
.stCaption, [data-testid="stCaptionContainer"] p {{ color: {SUBTXT} !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {TXT} !important;
}}

/* ── Metrics ── */
[data-testid="stMetricLabel"] > div {{ color: {SUBTXT} !important; }}
[data-testid="stMetricValue"] > div {{ color: {TXT} !important; }}
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 16px !important;
}}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stPasswordInput input,
.stTextArea textarea {{
    background-color: {INPUT_BG} !important;
    color: {TXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
.stTextInput label, .stNumberInput label, .stPasswordInput label,
.stTextArea label, .stSlider label, .stFileUploader label,
.stSelectbox label, .stMultiSelect label {{
    color: {TXT} !important;
    font-weight: 500 !important;
}}
.stTextInput input::placeholder, .stPasswordInput input::placeholder {{
    color: {SUBTXT} !important;
}}

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    color: {TXT} !important;
    border-color: {BORDER} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid {BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {SUBTXT} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: {ACCENT} !important;
    color: #ffffff !important;
}}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: none !important;
    color: {BTN_TXT} !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT}, #818cf8) !important;
}}
.stButton > button[kind="secondary"] {{
    background: {CARD} !important;
    color: {TXT} !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Expander ── */
.stExpander {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
.stExpander summary, .stExpander summary p {{
    color: {TXT} !important;
    font-weight: 500 !important;
}}
.stExpander [data-testid="stExpanderDetails"] p,
.stExpander [data-testid="stExpanderDetails"] span,
.stExpander [data-testid="stExpanderDetails"] div {{
    color: {TXT} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background-color: {CARD} !important;
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Alerts / Info boxes ── */
.stAlert > div {{ border-radius: 10px !important; }}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; }}

/* ── Toggle ── */
.stToggle span {{ color: {TXT} !important; }}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background-color: {CARD} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploadDropzone"] p {{ color: {SUBTXT} !important; }}

/* ── Plotly charts bg ── */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* ── Custom card class ── */
.edu-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}}
.edu-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════
@st.cache_resource
def load_models():
    clf = joblib.load("models/model.pkl")
    exp = joblib.load("models/explainer.pkl")
    reg = joblib.load("models/grade_model.pkl")
    return clf, exp, reg

clf, explainer, reg = load_models()
df_full = pd.read_csv("data/student_data.csv")


# ══════════════════════════════════════════════════
# PAGE 1 – LANDING
# ══════════════════════════════════════════════════
def show_landing():
    # top nav
    n1, n2, n3 = st.columns([1, 6, 1])
    with n1:
        st.markdown(f"<h2 style='color:{ACCENT};margin:0;padding-top:8px;'>🎓 EduPredict</h2>", unsafe_allow_html=True)
    with n3:
        if st.button("Login →", type="primary"):
            go_to("login")

    st.markdown("<hr style='margin:0 0 0 0;'>", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 20px;
        padding: 70px 40px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 20px 60px rgba(99,102,241,0.3);
    ">
        <div style="font-size:72px; margin-bottom:12px;">🎓</div>
        <h1 style="color:#ffffff !important; font-size:3rem; font-weight:800; margin:0 0 16px 0; letter-spacing:-1px;">
            Welcome to EduPredict
        </h1>
        <p style="color:rgba(255,255,255,0.88) !important; font-size:1.25rem; margin:0 auto 32px auto; max-width:600px; line-height:1.7;">
            The AI-powered platform that predicts student performance,
            identifies academic risks, and delivers actionable insights
            — all in real time.
        </p>
        <div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.2); color:#fff !important;
                         border-radius:20px; padding:6px 18px; font-size:14px; font-weight:600;">
                ✅ 98% Accuracy
            </span>
            <span style="background:rgba(255,255,255,0.2); color:#fff !important;
                         border-radius:20px; padding:6px 18px; font-size:14px; font-weight:600;">
                🤖 Dual AI Models
            </span>
            <span style="background:rgba(255,255,255,0.2); color:#fff !important;
                         border-radius:20px; padding:6px 18px; font-size:14px; font-weight:600;">
                📊 Real-time Analytics
            </span>
            <span style="background:rgba(255,255,255,0.2); color:#fff !important;
                         border-radius:20px; padding:6px 18px; font-size:14px; font-weight:600;">
                📧 Smart Alerts
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Cards ─────────────────────────────
    st.markdown(f"<h2 style='text-align:center; color:{TXT}; margin:40px 0 24px;'>✨ Everything You Need</h2>",
                unsafe_allow_html=True)

    features = [
        ("🎯", "Smart Prediction", "Predict pass/fail outcome and expected grade with 98% accuracy using Random Forest AI."),
        ("📊", "Teacher Dashboard", "Class-wide analytics with risk distributions, correlation heatmaps, and trend charts."),
        ("📋", "PDF Report Card", "Generate and download professional report cards for individual students instantly."),
        ("📂", "Batch Analysis", "Upload a CSV of hundreds of students and get predictions for all of them in seconds."),
        ("📈", "History Tracker", "Save and monitor student progress over time to spot improvement or decline trends."),
        ("📧", "Email Alerts", "Automatically email at-risk students or parents with personalised improvement tips."),
    ]

    rows = [features[:3], features[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="edu-card" style="text-align:center; transition: transform 0.2s;">
                    <div style="font-size:40px; margin-bottom:12px;">{icon}</div>
                    <h3 style="color:{TXT} !important; margin:0 0 8px; font-size:1.1rem; font-weight:700;">{title}</h3>
                    <p style="color:{SUBTXT} !important; font-size:0.9rem; margin:0; line-height:1.6;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{ACCENT}22,#8b5cf622);
                border:1px solid {ACCENT}44; border-radius:16px; padding:36px;">
        <h2 style="text-align:center; color:{TXT} !important; margin:0 0 28px;">📈 Platform Stats</h2>
        <div style="display:flex; justify-content:space-around; flex-wrap:wrap; gap:20px;">
            <div style="text-align:center;">
                <div style="font-size:2.5rem; font-weight:800; color:{ACCENT};">500+</div>
                <div style="color:{SUBTXT} !important; font-size:0.95rem;">Students Analysed</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:2.5rem; font-weight:800; color:{ACCENT};">98%</div>
                <div style="color:{SUBTXT} !important; font-size:0.95rem;">Model Accuracy</div>
            </div>
            <div style="font-size:2.5rem; font-weight:800; color:{ACCENT}; text-align:center;">
                <div>0.94</div>
                <div style="color:{SUBTXT} !important; font-size:0.95rem;">Grade R² Score</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:2.5rem; font-weight:800; color:{ACCENT};">6</div>
                <div style="color:{SUBTXT} !important; font-size:0.95rem;">Smart Features</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:{TXT};'>Ready to get started?</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 2, 3])
    with c2:
        if st.button("🚀 Get Started Free", type="primary", use_container_width=True):
            go_to("login")

    # Footer
    st.markdown(f"""
    <div style="text-align:center; padding:40px 0 10px; color:{SUBTXT} !important; font-size:0.85rem;">
        Made with ❤️ by EduPredict Team &nbsp;•&nbsp; Powered by AI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 2 – LOGIN
# ══════════════════════════════════════════════════
def show_login():
    # back link
    if st.button("← Back to Home"):
        go_to("landing")

    st.markdown("<br>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        # Card container
        st.markdown(f"""
        <div style="
            background:{CARD};
            border:1px solid {BORDER};
            border-radius:20px;
            padding:40px 36px;
            box-shadow:0 20px 60px rgba(0,0,0,{'0.4' if dark else '0.08'});
        ">
            <div style="text-align:center; margin-bottom:28px;">
                <div style="font-size:52px;">🎓</div>
                <h1 style="color:{TXT} !important; font-size:1.8rem; font-weight:800;
                            margin:8px 0 4px; letter-spacing:-0.5px;">EduPredict</h1>
                <p style="color:{SUBTXT} !important; margin:0; font-size:0.95rem;">
                    Sign in to your dashboard
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Form inside the visual card (Streamlit form)
        with st.form("login_form"):
            st.markdown(f"<p style='color:{TXT}; font-weight:600; margin-bottom:4px;'>👤 Username</p>",
                        unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")

            st.markdown(f"<p style='color:{TXT}; font-weight:600; margin-bottom:4px; margin-top:12px;'>🔒 Password</p>",
                        unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Enter your password",
                                     label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In →", type="primary", use_container_width=True)

            if submitted:
                if username.strip() == "" or password.strip() == "":
                    st.error("⚠️ Please enter both username and password.")
                else:
                    st.session_state.username  = username
                    st.session_state.logged_in = True
                    st.success(f"✅ Welcome, {username}! Redirecting...")
                    import time; time.sleep(0.8)
                    go_to("main")

        # hint box
        st.markdown(f"""
        <div style="margin-top:20px; background:{'#1e3a5f' if dark else '#eff6ff'};
                    border:1px solid {'#3b82f6' if dark else '#bfdbfe'};
                    border-radius:10px; padding:14px 16px; text-align:center;">
            <p style="color:{'#93c5fd' if dark else '#1e40af'} !important;
                       margin:0; font-size:0.88rem;">
                💡 <strong>Demo:</strong> Enter <strong>any username & password</strong> to continue
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style="text-align:center; color:{SUBTXT} !important;
                  font-size:0.8rem; margin-top:24px;">
            By signing in you agree to our Terms of Service
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 3 – MAIN APP
# ══════════════════════════════════════════════════
def show_main():
    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:10px 0 16px;">
            <div style="font-size:40px;">🎓</div>
            <h2 style="color:{ACCENT} !important; margin:4px 0 2px; font-size:1.3rem; font-weight:800;">EduPredict</h2>
            <p style="color:{SUBTXT} !important; font-size:0.8rem; margin:0;">AI Performance System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='border-top:1px solid {BORDER}; margin:0 0 16px;'></div>",
                    unsafe_allow_html=True)

        # Greeting
        st.markdown(f"<p style='color:{SUBTXT} !important; font-size:0.85rem; margin:0;'>Logged in as</p>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color:{TXT} !important; font-weight:700; margin:0 0 16px;'>👤 {st.session_state.username}</p>",
                    unsafe_allow_html=True)

        # Dark mode toggle
        new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()

        st.markdown(f"<div style='border-top:1px solid {BORDER}; margin:16px 0;'></div>",
                    unsafe_allow_html=True)

        st.markdown(f"<p style='color:{TXT} !important; font-weight:700; font-size:0.95rem; margin-bottom:12px;'>📥 Student Input</p>",
                    unsafe_allow_html=True)

        def sync_from_sidebar():
            st.session_state.student_name = st.session_state.sb_student_name
            st.session_state.study_hours  = st.session_state.sb_study_hours
            st.session_state.attendance   = st.session_state.sb_attendance
            st.session_state.assignments  = st.session_state.sb_assignments
            st.session_state.quizzes      = st.session_state.sb_quizzes

        student_name = st.text_input("Student Name", value=st.session_state.student_name, key="sb_student_name", on_change=sync_from_sidebar)
        study_hours  = st.slider("📖 Study Hours / day", 0.0, 12.0, float(st.session_state.study_hours), 0.5, key="sb_study_hours", on_change=sync_from_sidebar)
        attendance   = st.slider("🏫 Attendance (%)",    0.0, 100.0, float(st.session_state.attendance), 1.0, key="sb_attendance", on_change=sync_from_sidebar)
        assignments  = st.slider("📝 Assignments Score", 0.0, 100.0, float(st.session_state.assignments), 1.0, key="sb_assignments", on_change=sync_from_sidebar)
        quizzes      = st.slider("📊 Quizzes Score",     0.0, 100.0, float(st.session_state.quizzes), 1.0, key="sb_quizzes", on_change=sync_from_sidebar)

        # Update session state values directly if not triggered via callback
        st.session_state.student_name = student_name
        st.session_state.study_hours  = study_hours
        st.session_state.attendance   = attendance
        st.session_state.assignments  = assignments
        st.session_state.quizzes      = quizzes

        st.markdown(f"<div style='border-top:1px solid {BORDER}; margin:16px 0;'></div>",
                    unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            go_to("landing")

    input_data = {
        "study_hours": study_hours,
        "attendance":  attendance,
        "assignments": assignments,
        "quizzes":     quizzes
    }

    # ── Top bar ───────────────────────────────────
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown(f"""
        <h1 style="color:{TXT} !important; font-size:1.8rem; font-weight:800;
                   margin:0; letter-spacing:-0.5px;">
            🎓 Student Performance Dashboard
        </h1>
        <p style="color:{SUBTXT} !important; margin:4px 0 0; font-size:0.9rem;">
            AI-powered academic risk analysis & grade prediction
        </p>
        """, unsafe_allow_html=True)
    with h2:
        now = datetime.datetime.now().strftime("%d %b %Y")
        st.markdown(f"<p style='color:{SUBTXT} !important; text-align:right; margin-top:12px;'>{now}</p>",
                    unsafe_allow_html=True)

    st.markdown(f"<div style='border-top:1px solid {BORDER}; margin:12px 0 20px;'></div>",
                unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Prediction",
        "📅 AI Study Plan",
        "📂 Batch Analysis",
        "📊 Teacher Dashboard",
        "📋 Report Card",
        "📈 History",
        "📧 Email Alerts"
    ])

    # ════════════════════════════════════════════
    # TAB 1 – PREDICTION
    # ════════════════════════════════════════════
    with tab1:
        # In-page Input Panel synced with sidebar
        with st.expander("📝 Live Student Parameters & Sliders (Real-Time Control)", expanded=True):
            st.markdown(f"<p style='color:{SUBTXT}; margin-bottom: 12px; font-size: 0.9rem;'>Use either the left sidebar or the direct inputs below. Everything syncs and recalibrates in real-time!</p>", unsafe_allow_html=True)
            ip1, ip2, ip3, ip4, ip5 = st.columns([1.4, 1, 1, 1, 1])

            def sync_from_main_name():
                st.session_state.sb_student_name = st.session_state.main_name
                st.session_state.student_name = st.session_state.main_name
            def sync_from_main_hours():
                st.session_state.sb_study_hours = st.session_state.main_hours
                st.session_state.study_hours = st.session_state.main_hours
            def sync_from_main_att():
                st.session_state.sb_attendance = st.session_state.main_att
                st.session_state.attendance = st.session_state.main_att
            def sync_from_main_asgn():
                st.session_state.sb_assignments = st.session_state.main_asgn
                st.session_state.assignments = st.session_state.main_asgn
            def sync_from_main_quiz():
                st.session_state.sb_quizzes = st.session_state.main_quiz
                st.session_state.quizzes = st.session_state.main_quiz

            with ip1:
                st.text_input("Student Name", value=st.session_state.student_name, key="main_name", on_change=sync_from_main_name)
            with ip2:
                st.number_input("Study Hours/day", 0.0, 12.0, float(st.session_state.study_hours), 0.5, key="main_hours", on_change=sync_from_main_hours)
            with ip3:
                st.number_input("Attendance (%)", 0.0, 100.0, float(st.session_state.attendance), 1.0, key="main_att", on_change=sync_from_main_att)
            with ip4:
                st.number_input("Assignments Score", 0.0, 100.0, float(st.session_state.assignments), 1.0, key="main_asgn", on_change=sync_from_main_asgn)
            with ip5:
                st.number_input("Quizzes Score", 0.0, 100.0, float(st.session_state.quizzes), 1.0, key="main_quiz", on_change=sync_from_main_quiz)

        active_student_name = st.session_state.student_name
        active_input = {
            "study_hours": float(st.session_state.study_hours),
            "attendance":  float(st.session_state.attendance),
            "assignments": float(st.session_state.assignments),
            "quizzes":     float(st.session_state.quizzes)
        }

        result     = predict_student(active_input)
        df_inp     = pd.DataFrame([active_input])
        grade_pred = reg.predict(df_inp)[0]
        grade_ltr  = "A" if grade_pred>=8 else "B" if grade_pred>=6.5 else "C" if grade_pred>=5 else "D" if grade_pred>=3.5 else "F"
        risk_color = DANGER if result["risk_level"]=="High Risk" else WARNING if result["risk_level"]=="Medium Risk" else SUCCESS

        # ── Hero Banner with Student Info & Dynamic Status ──
        outcome_color = SUCCESS if result["prediction"]=="Pass" else DANGER
        st.markdown(f"""
        <div style="
            background: {CARD};
            border: 1px solid {BORDER};
            border-left: 6px solid {outcome_color};
            border-radius: 14px;
            padding: 16px 24px;
            margin: 14px 0 20px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 15px rgba(0,0,0,{'0.25' if dark else '0.04'});
        ">
            <div>
                <span style="font-size: 0.8rem; font-weight: 700; color: {SUBTXT}; text-transform: uppercase; letter-spacing: 0.8px;">
                    LIVE PREDICTION TARGET
                </span>
                <h2 style="color: {TXT} !important; margin: 2px 0 0; font-size: 1.55rem; font-weight: 800;">
                    👤 {active_student_name}
                </h2>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span style="
                    background: {outcome_color}22;
                    color: {outcome_color} !important;
                    border: 1px solid {outcome_color}44;
                    padding: 6px 16px;
                    border-radius: 30px;
                    font-weight: 700;
                    font-size: 0.95rem;
                ">
                    {'✅ Pass Forecast' if result['prediction']=='Pass' else '⚠️ Fail Warning'}
                </span>
                <span style="
                    background: {risk_color}22;
                    color: {risk_color} !important;
                    border: 1px solid {risk_color}44;
                    padding: 6px 16px;
                    border-radius: 30px;
                    font-weight: 700;
                    font-size: 0.95rem;
                ">
                    ⚡ {result['risk_level']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key Performance Indicators (5 Metrics) ──
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🏆 Expected Outcome", result["prediction"], delta="Target Met" if result["prediction"]=="Pass" else "Action Needed", delta_color="normal" if result["prediction"]=="Pass" else "inverse")
        k2.metric("📊 Pass Probability", f"{result['probability']:.0%}")
        k3.metric("⚠️ Academic Risk", result["risk_level"])
        k4.metric("📝 Score Estimate", f"{grade_pred:.1f} / 10")
        k5.metric("🅰️ Letter Grade", grade_ltr)

        st.markdown(f"<div style='border-top: 1px solid {BORDER}; margin: 20px 0;'></div>", unsafe_allow_html=True)

        # ── 2-Column Balanced Layout for Analytics & Insights ──
        col_charts, col_insights = st.columns([1.65, 1.0])

        with col_charts:
            st.markdown(f"<h4 style='color:{TXT} !important; margin-bottom: 12px;'>📊 Live Analytical Visualizations</h4>", unsafe_allow_html=True)

            # Chart 1: What-If Dynamic Curve
            sim_hours = list(range(1, 13))
            probs = []
            for h in sim_hours:
                temp = active_input.copy(); temp["study_hours"] = h
                probs.append(predict_student(temp)["probability"])

            fig2 = px.line(x=sim_hours, y=probs, markers=True,
                           labels={"x":"Study Hours / Day","y":"Probability of Passing"},
                           color_discrete_sequence=[ACCENT])
            fig2.add_hline(y=0.5, line_dash="dash", line_color=DANGER,
                           annotation_text="50% Passing Threshold",
                           annotation_font_color=DANGER)

            current_study = active_input["study_hours"]
            current_prob = float(result["probability"])
            fig2.add_trace(go.Scatter(
                x=[current_study],
                y=[current_prob],
                mode="markers+text",
                name="Current Student",
                marker=dict(size=14, color="#ef4444", line=dict(width=2.5, color="#ffffff")),
                text=[f"📍 Current ({current_study}h → {current_prob:.0%})"],
                textposition="top center",
                textfont=dict(color=TXT, size=11, family="sans-serif")
            ))

            fig2.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                               showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color=TXT,
                               xaxis=dict(gridcolor=BORDER, color=TXT, dtick=1),
                               yaxis=dict(gridcolor=BORDER, color=TXT, tickformat=".0%"))
            st.plotly_chart(fig2, use_container_width=True)

            # Chart 2: SHAP Feature Impact
            shap_vals = explainer(df_inp)
            vals = shap_vals.values
            if len(vals.shape) == 3:  vals = vals[0, :, 1]
            elif len(vals.shape) == 2: vals = vals[0]
            vals = vals.flatten()
            shap_df = pd.DataFrame({"Feature": df_inp.columns, "Impact": vals}).sort_values("Impact")
            colors  = [DANGER if v<0 else SUCCESS for v in shap_df["Impact"]]

            fig, ax = plt.subplots(figsize=(6, 2.6), facecolor='none')
            ax.set_facecolor('none')
            ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors, edgecolor="none", height=0.55)
            ax.axvline(0, color=SUBTXT, linewidth=0.8, linestyle="--")
            ax.set_title("AI Feature Impact (SHAP Explanation)", fontsize=11, color=TXT, pad=10)
            for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
            ax.tick_params(colors=TXT, labelsize=9)
            ax.xaxis.label.set_color(TXT)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_insights:
            st.markdown(f"<h4 style='color:{TXT} !important; margin-bottom: 12px;'>💡 AI Diagnosis & Action Plan</h4>", unsafe_allow_html=True)

            # Interventions box
            if result["interventions"]:
                for tip in result["interventions"]:
                    st.warning(f"⚠️ {tip}")
            else:
                st.success("🎉 Excellent progress! All indicators are well above benchmark thresholds.")

            # Summary Card
            st.markdown(f"""
            <div class="edu-card" style="margin-top: 14px;">
                <p style="color:{SUBTXT} !important; font-size:0.78rem; margin:0 0 10px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;">
                    📋 Profile Breakdown
                </p>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid {BORDER};">
                    <span style="color:{SUBTXT} !important;">Study Routine:</span>
                    <strong style="color:{TXT} !important;">{active_input['study_hours']} hrs / day</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid {BORDER};">
                    <span style="color:{SUBTXT} !important;">Attendance:</span>
                    <strong style="color:{TXT} !important;">{active_input['attendance']}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid {BORDER};">
                    <span style="color:{SUBTXT} !important;">Assignments:</span>
                    <strong style="color:{TXT} !important;">{active_input['assignments']} / 100</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid {BORDER};">
                    <span style="color:{SUBTXT} !important;">Quizzes Score:</span>
                    <strong style="color:{TXT} !important;">{active_input['quizzes']} / 100</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0 2px 0;">
                    <span style="color:{SUBTXT} !important;">Predicted Score:</span>
                    <strong style="color:{ACCENT} !important; font-size: 1.1rem;">{grade_pred:.1f} / 10 ({grade_ltr})</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Snapshot to History", type="primary", use_container_width=True):
                st.session_state.history.append({
                    "Date":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Name":        active_student_name,
                    "Study Hours": active_input['study_hours'],
                    "Attendance":  active_input['attendance'],
                    "Assignments": active_input['assignments'],
                    "Quizzes":     active_input['quizzes'],
                    "Prediction":  result["prediction"],
                    "Probability": f"{result['probability']:.0%}",
                    "Risk":        result["risk_level"],
                    "Grade":       f"{grade_pred:.1f}/10 ({grade_ltr})"
                })
                st.success("✅ Saved to Student History Tracker!")

    # ════════════════════════════════════════════
    # TAB 2 – AI PERSONALIZED STUDY PLAN
    # ════════════════════════════════════════════
    with tab2:
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color:{TXT} !important; margin:0 0 6px;">📅 AI Personalized Weekly Study Plan</h3>
            <p style="color:{SUBTXT} !important; margin:0; font-size: 0.95rem;">
                Custom-tailored 7-day academic improvement roadmap based on current risk factors, weak areas, and workload capacity.
            </p>
        </div>
        """, unsafe_allow_html=True)

        sp_name = st.session_state.student_name
        sp_hours = float(st.session_state.study_hours)
        sp_att = float(st.session_state.attendance)
        sp_asgn = float(st.session_state.assignments)
        sp_quiz = float(st.session_state.quizzes)

        # Controls row
        spc1, spc2, spc3 = st.columns([1.5, 1.2, 1.3])
        with spc1:
            target_goal = st.selectbox("🎯 Target Grade Goal", ["Pass with Distinction (Grade A: 8.0+)", "Strong Pass (Grade B: 6.5 - 7.9)", "Safe Pass (Grade C: 5.0 - 6.4)"])
        with spc2:
            preferred_slot = st.selectbox("⏰ Best Study Time", ["Morning (7 AM - 10 AM)", "Afternoon (2 PM - 5 PM)", "Evening / Night (7 PM - 11 PM)"])
        with spc3:
            study_style = st.selectbox("🧠 Learning Style", ["Pomodoro (25m Focus + 5m Break)", "Deep Work (90m Blocks)", "Concept First, Practice Later"])

        # Determine Weaknesses & Priority
        weaknesses = []
        if sp_hours < 4.0:
            weaknesses.append(("Daily Study Hours", f"{sp_hours}h/day is critically low", "Critical Priority"))
        if sp_att < 75.0:
            weaknesses.append(("Class Attendance", f"{sp_att}% is below university cutoff", "High Priority"))
        if sp_asgn < 65.0:
            weaknesses.append(("Assignments", f"{sp_asgn}/100 indicates incomplete submissions", "Medium Priority"))
        if sp_quiz < 65.0:
            weaknesses.append(("Quiz Scores", f"{sp_quiz}/100 indicates conceptual gaps", "High Priority"))

        # Target study hours recommendation
        recommended_hours = 6.5 if "Grade A" in target_goal else 5.0 if "Grade B" in target_goal else 4.0
        extra_needed = max(0.0, round(recommended_hours - sp_hours, 1))

        # KPI Summary for Study Plan
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("👤 Student", sp_name)
        kpi2.metric("⏱️ Current Daily Study", f"{sp_hours} hrs")
        kpi3.metric("🎯 Recommended Daily Study", f"{recommended_hours} hrs", delta=f"+{extra_needed} hrs/day" if extra_needed > 0 else "Optimal Pace")
        kpi4.metric("🚨 Key Focus Areas", f"{len(weaknesses)} flagged" if weaknesses else "General Mastery")

        st.markdown(f"<div style='border-top:1px solid {BORDER}; margin: 20px 0;'></div>", unsafe_allow_html=True)

        plan_col1, plan_col2 = st.columns([1.8, 1.2])

        with plan_col1:
            st.markdown(f"<h4 style='color:{TXT} !important;'>🗓️ 7-Day Personalized Schedule ({preferred_slot.split(' ')[0]} Focus)</h4>", unsafe_allow_html=True)

            # Generate dynamic schedule based on weaknesses
            schedule_data = [
                {"Day": "Monday", "Primary Focus": "Core Theory & Concept Revision", "Key Activity": "Review lecture notes & textbook chapters for weak quiz topics.", "Target Hours": f"{max(sp_hours, 3.5):.1f} hrs", "Status": "Concept"},
                {"Day": "Tuesday", "Primary Focus": "Assignment Sprint & Homework", "Key Activity": "Solve and submit pending assignment problems with step-by-step notes.", "Target Hours": f"{max(sp_hours, 3.0):.1f} hrs", "Status": "Problem Solving"},
                {"Day": "Wednesday", "Primary Focus": "Quiz Practice & Flashcards", "Key Activity": "Take timed mock quizzes; identify formulas and concepts missed.", "Target Hours": f"{max(sp_hours, 4.0):.1f} hrs", "Status": "Testing"},
                {"Day": "Thursday", "Primary Focus": "Attendance Catch-up & Class Prep", "Key Activity": "Prepare next class syllabus questions; review previous recordings.", "Target Hours": f"{max(sp_hours, 3.5):.1f} hrs", "Status": "Class Prep"},
                {"Day": "Friday", "Primary Focus": "Doubt Clearance & Peer Discussion", "Key Activity": "Meet teaching assistants, discuss tricky assignment questions.", "Target Hours": f"{max(sp_hours, 3.0):.1f} hrs", "Status": "Collaboration"},
                {"Day": "Saturday", "Primary Focus": "Deep Work Simulation & Mock Exam", "Key Activity": "Full 2-hour uninterrupted practice test mimicking real exam conditions.", "Target Hours": f"{max(sp_hours + 1.0, 4.5):.1f} hrs", "Status": "Deep Focus"},
                {"Day": "Sunday", "Primary Focus": "Weekly Review & Schedule Calibration", "Key Activity": "Audit error log, organize notes, set realistic targets for upcoming week.", "Target Hours": "2.0 hrs", "Status": "Rest & Review"}
            ]

            # Adjust if quiz or assignment is specifically low
            if sp_quiz < 60:
                schedule_data[0]["Key Activity"] = "⚠️ Focus heavily on Quiz prep: review past mistakes and revise core definitions."
                schedule_data[2]["Key Activity"] = "⚠️ Complete 2 full mock test sets to rebuild testing speed and accuracy."
            if sp_asgn < 60:
                schedule_data[1]["Key Activity"] = "⚠️ Urgent backlog clearance: complete overdue and upcoming assignment drafts."

            df_schedule = pd.DataFrame(schedule_data)
            st.dataframe(df_schedule, use_container_width=True, hide_index=True)

            # Download Study Plan as CSV
            st.download_button("⬇️ Download 7-Day Plan (CSV)", df_schedule.to_csv(index=False), f"StudyPlan_{sp_name.replace(' ', '_')}.csv", "text/csv")

        with plan_col2:
            st.markdown(f"<h4 style='color:{TXT} !important;'>💡 High-Impact Action Tactics</h4>", unsafe_allow_html=True)

            if weaknesses:
                for area, detail, prio in weaknesses:
                    prio_color = DANGER if "Critical" in prio else WARNING
                    st.markdown(f"""
                    <div style="background:{prio_color}18; border-left:4px solid {prio_color}; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
                        <span style="font-size:0.75rem; font-weight:700; color:{prio_color}; text-transform:uppercase;">{prio}</span>
                        <p style="color:{TXT} !important; margin:2px 0 0; font-size:0.92rem; font-weight:700;">{area}</p>
                        <p style="color:{SUBTXT} !important; margin:2px 0 0; font-size:0.83rem;">{detail}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🌟 All foundational metrics look solid! Focus on advanced problem solving to maintain an A grade.")

            st.markdown(f"""
            <div class="edu-card" style="margin-top:14px;">
                <p style="color:{SUBTXT} !important; font-size:0.78rem; margin:0 0 8px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;">
                    🧠 {study_style.split(' ')[0]} Technique Protocol
                </p>
                <ul style="color:{TXT} !important; margin:0; padding-left:18px; font-size:0.86rem; line-height:1.6;">
                    <li>Keep phone outside the study room during study blocks.</li>
                    <li>Spend 15 mins every evening reviewing what you learned today.</li>
                    <li>Never skip attendance — attending classes cuts home study time in half!</li>
                    <li>Track difficult questions in an <em>'Error Notebook'</em> to revise before quizzes.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 3 – BATCH
    # ════════════════════════════════════════════
    with tab3:
        st.markdown(f"<h3 style='color:{TXT} !important;'>📂 Batch Student Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{SUBTXT} !important;'>Upload a CSV with columns: <code>study_hours, attendance, assignments, quizzes</code></p>",
                    unsafe_allow_html=True)

        sample = pd.DataFrame({
            "study_hours": [7.0, 2.0, 5.5, 1.0, 9.0],
            "attendance":  [90,  55,  70,  40,  95],
            "assignments": [85,  35,  65,  25,  90],
            "quizzes":     [80,  30,  60,  20,  88]
        })
        st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                           "sample_input.csv", "text/csv")

        uploaded = st.file_uploader("Upload Student CSV", type=["csv"])
        if uploaded:
            df_batch = pd.read_csv(uploaded)
            results = []
            for _, row in df_batch.iterrows():
                d  = row.to_dict()
                r  = predict_student(d)
                g  = reg.predict(pd.DataFrame([d]))[0]
                gl = "A" if g>=8 else "B" if g>=6.5 else "C" if g>=5 else "D" if g>=3.5 else "F"
                results.append({**r, "predicted_score": round(g,2), "grade": gl})
            res_df   = pd.DataFrame(results)
            final_df = pd.concat([df_batch, res_df], axis=1)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("👥 Students",    len(final_df))
            m2.metric("✅ Pass Rate",   f"{(final_df['prediction']=='Pass').mean():.0%}")
            m3.metric("🔴 High Risk",   (final_df["risk_level"]=="High Risk").sum())
            m4.metric("📊 Avg Score",   f"{final_df['predicted_score'].mean():.2f}/10")

            st.dataframe(final_df, use_container_width=True, height=280)

            bc1, bc2 = st.columns(2)
            with bc1:
                rc = final_df["risk_level"].value_counts().reset_index()
                rc.columns = ["Risk","Count"]
                fig_r = px.pie(rc, names="Risk", values="Count", title="Risk Distribution",
                               color_discrete_map={"High Risk":DANGER,"Medium Risk":WARNING,"Low Risk":SUCCESS})
                fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TXT)
                st.plotly_chart(fig_r, use_container_width=True)
            with bc2:
                fig_g = px.histogram(final_df, x="predicted_score", nbins=10,
                                     title="Grade Distribution", color_discrete_sequence=[ACCENT])
                fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color=TXT, xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
                st.plotly_chart(fig_g, use_container_width=True)

            st.download_button("⬇️ Download Results", final_df.to_csv(index=False),
                               "batch_results.csv", "text/csv")

    # ════════════════════════════════════════════
    # TAB 4 – TEACHER DASHBOARD
    # ════════════════════════════════════════════
    with tab4:
        st.markdown(f"<h3 style='color:{TXT} !important;'>📊 Teacher Dashboard</h3>", unsafe_allow_html=True)

        X_full  = df_full.drop(["pass","final_score"], axis=1)
        preds   = clf.predict(X_full)
        grades  = reg.predict(X_full)
        probs   = clf.predict_proba(X_full)[:,1]
        df_dash = df_full.copy()
        df_dash["prediction"] = ["Pass" if p==1 else "Fail" for p in preds]
        df_dash["grade"]      = grades
        df_dash["probability"]= probs
        df_dash["risk"]       = df_dash["probability"].apply(
            lambda p: "High Risk" if p<0.4 else "Medium Risk" if p<0.7 else "Low Risk")

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("👥 Total Students",  len(df_dash))
        k2.metric("✅ Pass Rate",        f"{(df_dash['prediction']=='Pass').mean():.0%}")
        k3.metric("🔴 High Risk Count", (df_dash["risk"]=="High Risk").sum())
        k4.metric("📊 Avg Grade",        f"{df_dash['grade'].mean():.2f}/10")

        tc1, tc2 = st.columns(2)
        layout_opts = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color=TXT,
                           xaxis=dict(gridcolor=BORDER, color=TXT),
                           yaxis=dict(gridcolor=BORDER, color=TXT))
        with tc1:
            fig1 = px.scatter(df_dash, x="study_hours", y="grade", color="prediction",
                              size="attendance", title="Study Hours vs Grade",
                              color_discrete_map={"Pass":SUCCESS,"Fail":DANGER})
            fig1.update_layout(**layout_opts)
            st.plotly_chart(fig1, use_container_width=True)
        with tc2:
            fig2 = px.box(df_dash, x="prediction", y="attendance", color="prediction",
                          title="Attendance by Outcome",
                          color_discrete_map={"Pass":SUCCESS,"Fail":DANGER})
            fig2.update_layout(**layout_opts)
            st.plotly_chart(fig2, use_container_width=True)

        tc3, tc4 = st.columns(2)
        with tc3:
            rd = df_dash["risk"].value_counts().reset_index(); rd.columns=["Risk","Count"]
            fig3 = px.bar(rd, x="Risk", y="Count", color="Risk", title="Risk Distribution",
                          color_discrete_map={"High Risk":DANGER,"Medium Risk":WARNING,"Low Risk":SUCCESS})
            fig3.update_layout(**layout_opts)
            st.plotly_chart(fig3, use_container_width=True)
        with tc4:
            corr = df_full[["study_hours","attendance","assignments","quizzes","final_score"]].corr()
            fig4 = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap",
                             color_continuous_scale="RdBu_r")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TXT)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown(f"<h4 style='color:{DANGER} !important;'>🔴 High-Risk Students</h4>", unsafe_allow_html=True)
        st.dataframe(df_dash[df_dash["risk"]=="High Risk"].head(20), use_container_width=True)

    # ════════════════════════════════════════════
    # TAB 5 – REPORT CARD
    # ════════════════════════════════════════════
    with tab5:
        st.markdown(f"<h3 style='color:{TXT} !important;'>📋 PDF Report Card Generator</h3>", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            rc_name    = st.text_input("Student Name",    value=student_name, key="rc_name")
            rc_id      = st.text_input("Student ID",      value="STU-2024-001")
            rc_class   = st.text_input("Class / Section", value="Grade 10 – A")
            rc_teacher = st.text_input("Teacher Name",    value="Mr. Smith")
        with rc2:
            rc_study = st.number_input("Study Hours/day", 0.0, 12.0, study_hours, key="rc_study")
            rc_att   = st.number_input("Attendance (%)",  0.0, 100.0, attendance,  key="rc_att")
            rc_asgn  = st.number_input("Assignments",     0.0, 100.0, assignments, key="rc_asgn")
            rc_quiz  = st.number_input("Quizzes",         0.0, 100.0, quizzes,     key="rc_quiz")

        if st.button("🖨️ Generate PDF Report Card", type="primary"):
            rc_inp   = {"study_hours":rc_study,"attendance":rc_att,"assignments":rc_asgn,"quizzes":rc_quiz}
            rc_res   = predict_student(rc_inp)
            rc_grade = reg.predict(pd.DataFrame([rc_inp]))[0]
            rc_gl    = "A" if rc_grade>=8 else "B" if rc_grade>=6.5 else "C" if rc_grade>=5 else "D" if rc_grade>=3.5 else "F"

            pdf = FPDF()
            pdf.add_page()
            # Header
            pdf.set_fill_color(99,102,241); pdf.rect(0,0,210,35,"F")
            pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",22)
            pdf.set_y(8); pdf.cell(0,10,"EduPredict – Student Report Card",align="C",ln=True)
            pdf.set_font("Helvetica","",11)
            pdf.cell(0,8,f"Generated on {datetime.datetime.now().strftime('%d %B %Y')}",align="C",ln=True)
            # Info
            pdf.set_text_color(30,30,30); pdf.set_y(45)
            pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Student Information",ln=True)
            pdf.set_draw_color(99,102,241); pdf.line(10,pdf.get_y(),200,pdf.get_y()); pdf.ln(3)
            for lbl,val in [("Name",rc_name),("ID",rc_id),("Class",rc_class),("Teacher",rc_teacher)]:
                pdf.set_font("Helvetica","B",11); pdf.cell(60,8,lbl+":"); pdf.set_font("Helvetica","",11); pdf.cell(0,8,val,ln=True)
            # Metrics
            pdf.ln(5); pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Academic Performance",ln=True)
            pdf.line(10,pdf.get_y(),200,pdf.get_y()); pdf.ln(3)
            for lbl,val in [("Study Hours",f"{rc_study}h/day"),("Attendance",f"{rc_att}%"),
                             ("Assignments",f"{rc_asgn}/100"),("Quizzes",f"{rc_quiz}/100"),
                             ("Predicted Score",f"{rc_grade:.2f}/10"),("Grade",rc_gl),
                             ("Outcome",rc_res["prediction"]),("Risk",rc_res["risk_level"]),
                             ("Pass Probability",f"{rc_res['probability']:.0%}")]:
                pdf.set_font("Helvetica","B",11); pdf.cell(70,8,lbl+":")
                pdf.set_font("Helvetica","",11); pdf.cell(0,8,val,ln=True)
            # Interventions
            pdf.ln(5); pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Recommendations",ln=True)
            pdf.line(10,pdf.get_y(),200,pdf.get_y()); pdf.ln(3); pdf.set_font("Helvetica","",11)
            for tip in (rc_res["interventions"] or ["Student is performing well. Keep it up!"]):
                pdf.cell(0,7,f"  - {tip}",ln=True)
            # Footer
            pdf.set_y(-20); pdf.set_fill_color(99,102,241); pdf.rect(0,pdf.get_y(),210,20,"F")
            pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","I",9)
            pdf.cell(0,10,"EduPredict | AI-powered Student Performance Analysis",align="C")

            st.success("✅ Report Card Generated!")
            st.download_button("⬇️ Download PDF",
                               bytes(pdf.output()),
                               f"ReportCard_{rc_name.replace(' ','_')}.pdf",
                               "application/pdf")
            p1,p2,p3 = st.columns(3)
            p1.metric("Score",f"{rc_grade:.2f}/10"); p2.metric("Grade",rc_gl); p3.metric("Outcome",rc_res["prediction"])

    # ════════════════════════════════════════════
    # TAB 6 – HISTORY
    # ════════════════════════════════════════════
    with tab6:
        st.markdown(f"<h3 style='color:{TXT} !important;'>📈 History Tracker</h3>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("No history yet. Run a prediction and click **💾 Save to History**.")
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            if len(hist_df) > 1:
                scores = hist_df["Grade"].apply(lambda g: float(g.split("/")[0]))
                fig_h  = px.line(x=hist_df["Date"], y=scores, markers=True,
                                 labels={"x":"Date","y":"Predicted Score"},
                                 title="Score Trend Over Time", color_discrete_sequence=[ACCENT])
                fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color=TXT,
                                    xaxis=dict(gridcolor=BORDER, color=TXT),
                                    yaxis=dict(gridcolor=BORDER, color=TXT))
                st.plotly_chart(fig_h, use_container_width=True)
            hc1, hc2 = st.columns(2)
            with hc1:
                st.download_button("⬇️ Export CSV", hist_df.to_csv(index=False), "history.csv","text/csv")
            with hc2:
                if st.button("🗑️ Clear History"):
                    st.session_state.history = []; st.rerun()

    # ════════════════════════════════════════════
    # TAB 7 – EMAIL ALERTS
    # ════════════════════════════════════════════
    with tab7:
        st.markdown(f"<h3 style='color:{TXT} !important;'>📧 Email Alert System</h3>", unsafe_allow_html=True)
        with st.expander("⚙️ Gmail SMTP Configuration", expanded=True):
            sender_email = st.text_input("Sender Gmail", placeholder="yourschool@gmail.com")
            app_password = st.text_input("Gmail App Password", type="password",
                                         help="Use Gmail App Password — not your login password.")
        st.markdown("---")
        ea1, ea2 = st.columns(2)
        with ea1:
            to_email = st.text_input("Recipient Email", placeholder="student@example.com")
            to_name  = st.text_input("Student Name", value=student_name, key="ea_name")
        with ea2:
            ea_study = st.number_input("Study Hours", 0.0, 12.0, study_hours, key="ea_study")
            ea_att   = st.number_input("Attendance",  0.0, 100.0, attendance,  key="ea_att")
            ea_asgn  = st.number_input("Assignments", 0.0, 100.0, assignments, key="ea_asgn")
            ea_quiz  = st.number_input("Quizzes",     0.0, 100.0, quizzes,     key="ea_quiz")

        if st.button("📤 Send Email Alert", type="primary"):
            if not sender_email or not app_password or not to_email:
                st.error("❌ Please fill in all email fields.")
            else:
                ea_inp   = {"study_hours":ea_study,"attendance":ea_att,"assignments":ea_asgn,"quizzes":ea_quiz}
                ea_res   = predict_student(ea_inp)
                ea_grade = reg.predict(pd.DataFrame([ea_inp]))[0]
                ea_gl    = "A" if ea_grade>=8 else "B" if ea_grade>=6.5 else "C" if ea_grade>=5 else "D" if ea_grade>=3.5 else "F"
                body = f"""Dear {to_name},

Performance Report from EduPredict AI System:

• Study Hours : {ea_study} hrs/day
• Attendance  : {ea_att}%
• Assignments : {ea_asgn}/100
• Quizzes     : {ea_quiz}/100

AI PREDICTION:
• Score       : {ea_grade:.1f}/10  |  Grade: {ea_gl}
• Outcome     : {ea_res['prediction']}
• Risk Level  : {ea_res['risk_level']}
• Probability : {ea_res['probability']:.0%}

RECOMMENDATIONS:
""" + "\n".join([f"  ➤ {t}" for t in (ea_res["interventions"] or ["You are performing well!"])])

                try:
                    msg = MIMEMultipart()
                    msg["From"] = sender_email; msg["To"] = to_email
                    msg["Subject"] = f"📊 EduPredict Report: {to_name}"
                    msg.attach(MIMEText(body,"plain"))
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                        s.login(sender_email, app_password); s.send_message(msg)
                    st.success(f"✅ Email sent to {to_email}!"); st.balloons()
                except Exception as e:
                    st.error(f"❌ {e}")

        st.info("💡 Use a **Gmail App Password** (Google Account → Security → App Passwords)")


# ══════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════
if st.session_state.page == "landing":
    show_landing()
elif st.session_state.page == "login":
    show_login()
elif st.session_state.page == "main":
    if not st.session_state.logged_in:
        go_to("login")
    else:
        show_main()
