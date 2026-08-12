import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# הגדרות עמוד
st.set_page_config(
    page_title="Ring Resonator Simulator",
    page_icon="⭕",
    layout="wide"
)

# עיצוב כהה ומודרני (Tidy3D / Lumerical Style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Heebo', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⭕ Ring Resonator Simulation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">מנוע סימולציה ואנליזה למהודים טבעתיים בפוטוניקת סיליקון וסיליקון ניטריד</div>', unsafe_allow_html=True)

# בחירת מצב עבודה
calc_mode = st.radio(
    "בחר מצב חישוב / Select Calculation Mode:",
    options=["1. חישוב רזוננס בודד וסריקת פרמטרים (Single Resonance & Sweep)", "2. חישוב ספקטרום רחב (Full Spectrum Analysis)"],
    horizontal=True
)

st.divider()

# ==============================================================================
# פונקציות עזר להמרת פרמטרים פיזיקליים
# ==============================================================================
def convert_loss_parameters(param_type, val, lambda0_nm, R_um, ng):
    """
    המרה בין Qi, alpha (dB/cm) ו-Loss per roundtrip (%)
    """
    L_cm = 2 * np.pi * (R_um * 1e-4) # היקף בס"מ
    lambda0_cm = lambda0_nm * 1e-7   # אורך גל בס"מ
    
    if param_type == "Qi":
        Qi = val
        alpha_cm = (2 * np.pi * ng) / (lambda0_cm * Qi)
        alpha_db_cm = alpha_cm * 4.343
        a = np.exp(-alpha_cm * L_cm / 2)
        loss_pct = (1 - a**2) * 100
    elif param_type == "alpha_db":
        alpha_db_cm = val
        alpha_cm = alpha_db_cm / 4.343
        Qi = (2 * np.pi * ng) / (lambda0_cm * alpha_cm) if alpha_cm > 0 else 1e9
        a = np.exp(-alpha_cm * L_cm / 2)
        loss_pct = (1 - a**2) * 100
    else: # loss_pct
        loss_pct = val
        a2 = max(1e-6, 1 - loss_pct / 100.0)
        a = np.sqrt(a2)
        alpha_cm = -2 * np.log(a) / L_cm
        alpha_db_cm = alpha_cm * 4.343
        Qi = (2 * np.pi * ng) / (lambda0_cm * alpha_cm) if alpha_cm > 0 else 1e9
        
    return Qi, alpha_db_cm, loss_pct, a

def kappa_to_qc(kappa, R_um, ng, lambda0_nm):
    L_cm = 2 * np.pi * (R_um * 1e-4)
    lambda0_cm = lambda0_nm * 1e-7
    kappa2 = max(1e-6, kappa**2)
    Qc = (2 * np.pi * ng * L_cm) / (kappa2 * lambda0_cm)
    t = np.sqrt(1 - kappa2)
    return Qc, t

def qc_to_kappa(Qc, R_um, ng, lambda0_nm):
    L_cm = 2 * np.pi * (R_um * 1e-4)
    lambda0_cm = lambda0_nm * 1e-7
    kappa2 = (2 * np.pi * ng * L_cm) / (Qc * lambda0_cm)
    kappa2 = min(0.99, max(1e-6, kappa2))
    kappa = np.sqrt(kappa2)
    t = np.sqrt(1 - kappa2)
    return kappa, t

# ==============================================================================
# אפשרות 1: חישוב רזוננס בודד
# ==============================================================================
if "1." in calc_mode:
    col_side, col_main = st.columns([1, 2.5], gap="large")
    
    with col_side:
        st.subheader("⚙️ פרמטרי קלט")
        lambda0 = st.number_input("אורך גל מרכזי λ₀ (nm):", value=1550.0, step=0.1)
        R_um = st.number_input("רדיוס הטבעת R (μm):", value=20.0, step=1.0)
        ng = st.number_input("מקדם שבירה קבוצתי n_g:", value=4.0, step=0.01)
        neff = st.number_input("מקדם שבירה אפקטיבי n_eff:", value=2.4, step=0.01)
        
        st.divider()
        loss_choice = st.selectbox(
            "פרמטר הפסד פנימי מועדף:",
            options=["Qi (Intrinsic Quality Factor)", "alpha (dB/cm)", "Loss per roundtrip (%)"]
        )
        
        if "Qi" in loss_choice:
            input_val = st.number_input("ערך Qi פנימי:", value=500000.0, step=50000.0, format="%.0f")
            param_key = "Qi"
        elif "alpha" in loss_choice:
            input_val = st.number_input("הפסד α (dB/cm):", value=1.0, step=0.1)
            param_key = "alpha_db"
        else:
            input_val = st.number_input("הפסד לסיבוב (%):", value=0.5, step=0.1)
            param_key = "loss_pct"
            
        Qi_fixed, alpha_db_calc, loss_pct_calc, a_fixed = convert_loss_parameters(param_key, input_val, lambda0, R_um, ng)
        
        # הצגת ההמרות למשתמש
        st.info(f"""
        **סיכום הפסדים מחושב:**
        * $Q_i = {Qi_fixed:,.0f}$
        * $\\alpha = {alpha_db_calc:.2f}\\ \\text{{dB/cm}}$
        * Loss/roundtrip = {loss_pct_calc:.3f}%
        * $a = {a_fixed:.5f}$
        """)
        
        st.divider()
        st.subheader("🔄 הגדרת סריקת פרמטרים (Sweep)")
        sweep_mode = st.radio(
            "מה לרסט/ לסרוק?",
            options=["סריקת Qc / κ (עבור Qi קבוע)", "סריקת Qi (עבור Qc / κ קבוע)"]
        )
        
        span_pm_pm = st.slider("טווח תדרים לצפייה ברזוננס (± GHz):", min_value=1.0, max_value=200.0, value=25.0)

    with col_main:
        tab_spec, tab_er_ql, tab_eq = st.tabs(["📊 ספקטרום תמסורת", "📈 אנליזת Er & QL", "📐 משוואות ופיזיקה"])
        
        # ----------------------------------------------------------------------
        # סריקת 5 ערכים
        # ----------------------------------------------------------------------
        f0 = 3e8 / (lambda0 * 1e-9) # Hz
        df = np.linspace(-span_pm_pm * 1e9, span_pm_pm * 1e9, 1000)
        freqs = f0 + df
        wl_scan = 3e8 / freqs * 1e9 # nm
        
        L_cm = 2 * np.pi * (R_um * 1e-4)
        
        fig_spec, ax_spec = plt.subplots(figsize=(8, 4.5), dpi=150)
        plt.style.use('dark_background')
        fig_spec.patch.set_facecolor('#0F172A')
        ax_spec.set_facecolor('#0F172A')
        
        colors = ['#38BDF8', '#34D399', '#FBBF24', '#F43F5E', '#A855F7']
        
        sweep_data = []
        
        if "Qc" in sweep_mode:
            # המשתמש סורק 5 ערכי Qc
            qc_center, _ = kappa_to_qc(0.1, R_um, ng, lambda0)
            qc_vals = np.linspace(Qi_fixed * 0.2, Qi_fixed * 3.0, 5)
            
            for idx, Qc_val in enumerate(qc_vals):
                kappa_val, t_val = qc_to_kappa(Qc_val, R_um, ng, lambda0)
                QL_val = (Qi_fixed * Qc_val) / (Qi_fixed + Qc_val)
                
                # תמסורת
                dphi = -2 * np.pi * ng * (L_cm * 1e-2) * (df / 3e8)
                T = (a_fixed**2 - 2*a_fixed*t_val*np.cos(dphi) + t_val**2) / (1 - 2*a_fixed*t_val*np.cos(dphi) + (a_fixed*t_val)**2)
                
                T_min = np.min(T)
                Er_db = -10 * np.log10(max(1e-6, T_min))
                
                ax_spec.plot(df * 1e-9, T, label=f"Qc = {Qc_val:,.0f} (κ={kappa_val:.3f})", color=colors[idx], linewidth=2)
                sweep_data.append({"Qc": Qc_val, "κ": kappa_val, "Qi": Qi_fixed, "QL": QL_val, "Er (dB)": Er_db, "T_min": T_min})
        else:
            # המשתמש סורק 5 ערכי Qi
            Qc_const = 500000.0
            kappa_val, t_val = qc_to_kappa(Qc_const, R_um, ng, lambda0)
            qi_vals = np.linspace(Qc_const * 0.2, Qc_const * 3.0, 5)
            
            for idx, Qi_val in enumerate(qi_vals):
                _, _, _, a_var = convert_loss_parameters("Qi", Qi_val, lambda0, R_um, ng)
                QL_val = (Qi_val * Qc_const) / (Qi_val + Qc_const)
                
                dphi = -2 * np.pi * ng * (L_cm * 1e-2) * (df / 3e8)
                T = (a_var**2 - 2*a_var*t_val*np.cos(dphi) + t_val**2) / (1 - 2*a_var*t_val*np.cos(dphi) + (a_var*t_val)**2)
                
                T_min = np.min(T)
                Er_db = -10 * np.log10(max(1e-6, T_min))
                
                ax_spec.plot(df * 1e-9, T, label=f"Qi = {Qi_val:,.0f}", color=colors[idx], linewidth=2)
                sweep_data.append({"Qi": Qi_val, "Qc": Qc_const, "κ": kappa_val, "QL": QL_val, "Er (dB)": Er_db, "T_min": T_min})
                
        ax_spec.set_xlabel("Frequency Offset Δf (GHz)", color='#94A3B8', fontweight='bold')
        ax_spec.set_ylabel("Power Transmission T", color='#94A3B8', fontweight='bold')
        ax_spec.grid(True, color='#334155', linestyle=':', alpha=0.6)
        ax_spec.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        
        with tab_spec:
            st.pyplot(fig_spec)
            st.dataframe(pd.DataFrame(sweep_data), use_container_width=True)

        with tab_er_ql:
            st.markdown("#### 📈 סריקת יחס דעיכה ($E_r$) וגורם איכות טעון ($Q_L$)")
            
            # סריקה רציפה רחבה
            qc_ratio = np.logspace(-1, 1, 200) # Qc/Qi ratio
            er_list = []
            ql_list = []
            
            for r in qc_ratio:
                Qc_curr = Qi_fixed * r
                QL_curr = (Qi_fixed * Qc_curr) / (Qi_fixed + Qc_curr)
                _, t_curr = qc_to_kappa(Qc_curr, R_um, ng, lambda0)
                T_min_curr = ((a_fixed - t_curr) / (1 - a_fixed * t_curr))**2
                Er_curr = -10 * np.log10(max(1e-6, T_min_curr))
                
                er_list.append(Er_curr)
                ql_list.append(QL_curr)
                
            fig_er, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
            fig_er.patch.set_facecolor('#0F172A')
            ax1.set_facecolor('#0F172A')
            ax2.set_facecolor('#0F172A')
            
            ax1.semilogx(qc_ratio, er_list, color='#38BDF8', linewidth=2.5)
            ax1.axvline(1.0, color='#F43F5E', linestyle='--', label='Critical Coupling (Qc=Qi)')
            ax1.set_xlabel("Qc / Qi Ratio", color='#94A3B8', fontweight='bold')
            ax1.set_ylabel("Extinction Ratio Er (dB)", color='#94A3B8', fontweight='bold')
            ax1.grid(True, color='#334155', linestyle=':', alpha=0.6)
            ax1.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
            
            ax2.semilogx(qc_ratio, ql_list, color='#34D399', linewidth=2.5)
            ax2.axvline(1.0, color='#F43F5E', linestyle='--', label='Critical Coupling')
            ax2.set_xlabel("Qc / Qi Ratio", color='#94A3B8', fontweight='bold')
            ax2.set_ylabel("Loaded Quality Factor QL", color='#94A3B8', fontweight='bold')
            ax2.grid(True, color='#334155', linestyle=':', alpha=0.6)
            
            st.pyplot(fig_er)

        with tab_eq:
            st.markdown("""
            ### 📐 משוואות יסוד של Ring Resonator (All-Pass Configuration)
            
            1. **היקף הטבעת ($L$):**
               $$L = 2\\pi R$$
            
            2. **Free Spectral Range (FSR):**
               $$\\Delta\\lambda_{\\text{FSR}} = \\frac{\\lambda_0^2}{n_g L}$$
            
            3. **מקדם העברת אמפליטודה בסיבוב ($a$):**
               $$a = \\exp\\left(-\\frac{\\alpha L}{2}\\right)$$
            
            4. **קשר בין הפסדים ל-$Q_i$ הפנימי:**
               $$Q_i = \\frac{2\\pi n_g}{\\alpha \\lambda_0}$$
            
            5. **גורם איכות טעון ($Q_L$):**
               $$\\frac{1}{Q_L} = \\frac{1}{Q_i} + \\frac{1}{Q_c}$$
            
            6. **ספקטרום התמסורת ($T$):**
               $$T(\\phi) = \\frac{a^2 - 2at\\cos\\phi + t^2}{1 - 2at\\cos\\phi + (at)^2}$$
               כאשר $\\phi = \\frac{2\\pi n_{\\text{eff}} L}{\\lambda}$ ו-$t = \\sqrt{1 - \\kappa^2}$.
            
            7. **תנאי צימוד קריטי (Critical Coupling):**
               $$a = t \\iff Q_i = Q_c$$
               במצב זה $T(\\lambda_0) = 0$ והיחס $E_r \\to \\infty\\ \\text{dB}$.
            """)

# ==============================================================================
# אפשרות 2: חישוב ספקטרום רחב
# ==============================================================================
else:
    col_side2, col_main2 = st.columns([1, 2.5], gap="large")
    
    with col_side2:
        st.subheader("🌐 טווח ספקטרלי וגיאומטריה")
        wl_min = st.number_input("אורך גל התחלתי λ_min (nm):", value=1500.0, step=5.0)
        wl_max = st.number_input("אורך גל סופי λ_max (nm):", value=1600.0, step=5.0)
        
        R_um2 = st.number_input("רדיוס R (μm):", value=10.0, step=1.0, key="r2")
        ng2 = st.number_input("מקדם קבוצתי n_g:", value=4.0, step=0.01, key="ng2")
        neff2 = st.number_input("מקדם אפקטיבי n_eff (ב-1550nm):", value=2.4, step=0.01, key="neff2")
        
        st.divider()
        st.subheader("📉 הפסדים וצימוד")
        alpha_db2 = st.number_input("הפסד ליניארי α (dB/cm):", value=2.0, step=0.5)
        
        st.markdown("**תלות מקדם הצימוד בגל $\\kappa(\\lambda)$:**")
        kappa0 = st.number_input("מקדם צימוד מרכזי κ₀ (ב-1550nm):", value=0.15, min_value=0.01, max_value=0.9, step=0.01)
        dkappa_dwl = st.number_input("שיפוע דיספרסיה dκ/dλ (1/μm):", value=0.0, step=0.01, help="השאר 0 לקאפה קבוע")
        
        st.divider()
        add_noise = st.checkbox("כלול 0.5% רעש קיטוב אורתוגונלי (Unresonant Background)", value=True)

    with col_main2:
        st.subheader("📊 ספקטרום תמסורת רחב")
        
        wls = np.linspace(wl_min, wl_max, 5000) # nm
        L_cm2 = 2 * np.pi * (R_um2 * 1e-4)
        
        # חישוב אוקסיד/הפסד
        alpha_cm2 = alpha_db2 / 4.343
        a2 = np.exp(-alpha_cm2 * L_cm2 / 2)
        
        # תלות קאפה
        kappa_lambda = kappa0 + dkappa_dwl * (wls - 1550.0) * 1e-3
        kappa_lambda = np.clip(kappa_lambda, 0.01, 0.99)
        t_lambda = np.sqrt(1 - kappa_lambda**2)
        
        # פאזה
        phi = 2 * np.pi * neff2 * (L_cm2 * 1e4) / wls
        
        # תמסורת
        T_broad = (a2**2 - 2*a2*t_lambda*np.cos(phi) + t_lambda**2) / (1 - 2*a2*t_lambda*np.cos(phi) + (a2*t_lambda)**2)
        
        if add_noise:
            T_broad = 0.995 * T_broad + 0.005
            
        T_db = 10 * np.log10(np.maximum(1e-5, T_broad))
        
        fig_broad, ax_b = plt.subplots(figsize=(9, 4.5), dpi=150)
        plt.style.use('dark_background')
        fig_broad.patch.set_facecolor('#0F172A')
        ax_b.set_facecolor('#0F172A')
        
        ax_b.plot(wls, T_db, color='#38BDF8', linewidth=1.5)
        ax_b.set_xlabel("Wavelength λ (nm)", color='#94A3B8', fontweight='bold')
        ax_b.set_ylabel("Transmission (dB)", color='#94A3B8', fontweight='bold')
        ax_b.grid(True, color='#334155', linestyle=':', alpha=0.6)
        
        st.pyplot(fig_broad)
        
        # חישוב FSR תיאורטי
        lambda_mid = (wl_min + wl_max) / 2.0
        fsr_nm = (lambda_mid**2) / (ng2 * (L_cm2 * 1e7))
        
        st.success(f"💡 **FSR מחושב סביב {lambda_mid:.0f}nm:** {fsr_nm:.3f} nm ({3e8 / (lambda_mid*1e-9)**2 * (fsr_nm*1e-9) * 1e-9:.2f} GHz)")
