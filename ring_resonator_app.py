import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d

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

    .stDownloadButton > button {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #38BDF8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
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
# פונקציות עזר להמרת פרמטרים והורדת תמונות
# ==============================================================================
def convert_loss_parameters(param_type, val, lambda0_nm, R_um, ng):
    """ המרה בין Qi, alpha (dB/cm) ו-Loss per roundtrip (%) """
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

def figure_to_bytes(fig):
    """ המרת איור Matplotlib ל-Buffer של PNG להורדה """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

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
        
        f0 = 3e8 / (lambda0 * 1e-9) # Hz
        df = np.linspace(-span_pm_pm * 1e9, span_pm_pm * 1e9, 1000)
        freqs = f0 + df
        
        L_cm = 2 * np.pi * (R_um * 1e-4)
        
        fig_spec, ax_spec = plt.subplots(figsize=(8, 4.5), dpi=150)
        plt.style.use('dark_background')
        fig_spec.patch.set_facecolor('#0F172A')
        ax_spec.set_facecolor('#0F172A')
        
        colors = ['#38BDF8', '#34D399', '#FBBF24', '#F43F5E', '#A855F7']
        sweep_data = []
        
        if "Qc" in sweep_mode:
            qc_vals = np.linspace(Qi_fixed * 0.2, Qi_fixed * 3.0, 5)
            for idx, Qc_val in enumerate(qc_vals):
                kappa_val, t_val = qc_to_kappa(Qc_val, R_um, ng, lambda0)
                QL_val = (Qi_fixed * Qc_val) / (Qi_fixed + Qc_val)
                
                dphi = -2 * np.pi * ng * (L_cm * 1e-2) * (df / 3e8)
                T = (a_fixed**2 - 2*a_fixed*t_val*np.cos(dphi) + t_val**2) / (1 - 2*a_fixed*t_val*np.cos(dphi) + (a_fixed*t_val)**2)
                T_min = np.min(T)
                Er_db = -10 * np.log10(max(1e-6, T_min))
                
                ax_spec.plot(df * 1e-9, T, label=f"Qc = {Qc_val:,.0f} (κ={kappa_val:.3f})", color=colors[idx], linewidth=2)
                sweep_data.append({"Qc": Qc_val, "κ": kappa_val, "Qi": Qi_fixed, "QL": QL_val, "Er (dB)": Er_db, "T_min": T_min})
        else:
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
            st.download_button(
                label="📥 הורד איור ספקטרום (PNG)",
                data=figure_to_bytes(fig_spec),
                file_name="single_resonance_spectrum.png",
                mime="image/png"
            )
            st.dataframe(pd.DataFrame(sweep_data), use_container_width=True)

        with tab_er_ql:
            st.markdown("#### 📈 סריקת יחס דעיכה ($E_r$) וגורם איכות טעון ($Q_L$)")
            
            qc_ratio = np.logspace(-1, 1, 200)
            er_list, ql_list = [], []
            
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
            st.download_button(
                label="📥 הורד איור Er & QL (PNG)",
                data=figure_to_bytes(fig_er),
                file_name="er_ql_analysis.png",
                mime="image/png"
            )

        with tab_eq:
            st.markdown("""
            ### 📐 משוואות יסוד של Ring Resonator (All-Pass Configuration)
            
            1. **היקף הטבעת ($L$):** $L = 2\\pi R$
            2. **Free Spectral Range (FSR):** $\\Delta\\lambda_{\\text{FSR}} = \\frac{\\lambda_0^2}{n_g L}$
            3. **מקדם העברת אמפליטודה בסיבוב ($a$):** $a = \\exp\\left(-\\frac{\\alpha L}{2}\\right)$
            4. **קשר בין הפסדים ל-$Q_i$ הפנימי:** $Q_i = \\frac{2\\pi n_g}{\\alpha \\lambda_0}$
            5. **גורם איכות טעון ($Q_L$):** $\\frac{1}{Q_L} = \\frac{1}{Q_i} + \\frac{1}{Q_c}$
            6. **ספקטרום התמסורת ($T$):**
               $$T(\\phi) = \\frac{a^2 - 2at\\cos\\phi + t^2}{1 - 2at\\cos\\phi + (at)^2}$$
            """)

# ==============================================================================
# אפשרות 2: חישוב ספקטרום רחב (כולל העלאת קובץ kappa vs wl)
# ==============================================================================
else:
    col_side2, col_main2 = st.columns([1.2, 2.5], gap="large")
    
    with col_side2:
        st.subheader("🌐 טווח וגיאומטריה")
        arch_type = st.radio("ארכיטקטורת רזונטור:", ["All-Pass (Bus-Ring)", "Add-Drop (Two Buses)"])
        
        col_w1, col_w2 = st.columns(2)
        wl_min = col_w1.number_input("λ_min (nm):", value=1500.0, step=5.0)
        wl_max = col_w2.number_input("λ_max (nm):", value=1600.0, step=5.0)
        
        R_um2 = st.number_input("רדיוס R (μm):", value=10.0, step=1.0, key="r2")
        ng2 = st.number_input("מקדם קבוצתי n_g:", value=4.0, step=0.01, key="ng2")
        neff2 = st.number_input("מקדם אפקטיבי n_eff (ב-1550):", value=2.4, step=0.01, key="neff2")
        
        st.divider()
        st.subheader("📉 הפסדים וצימוד")
        alpha_db2 = st.number_input("הפסד ליניארי α (dB/cm):", value=2.0, step=0.5)
        
        # בחירת מקור הנתונים של קאפה
        kappa_source = st.radio(
            "מקור נתוני מקדם הצימוד κ(λ):",
            ["ערך קבוע / שיפוע ליניארי", "📂 העלאת קובץ נתונים (CSV / Excel / TXT)"]
        )
        
        uploaded_file = None
        if "העלאת קובץ" in kappa_source:
            uploaded_file = st.file_file = st.file_uploader(
                "העלה קובץ עם עמודות אורך גל ו-κ:",
                type=["csv", "xlsx", "xls", "txt"]
            )
            interp_kind = st.selectbox("שיטת אינטרפולציה והחלקה:", ["cubic (Spline חלקה)", "linear (ליניארית)"])
        else:
            col_k1, col_k2 = st.columns(2)
            kappa1 = col_k1.number_input("צימוד Input (κ₁):", value=0.15, min_value=0.01, max_value=0.99, step=0.01)
            if "Add-Drop" in arch_type:
                kappa2 = col_k2.number_input("צימוד Drop (κ₂):", value=0.15, min_value=0.01, max_value=0.99, step=0.01)
            else:
                kappa2 = 0.0
            dkappa_dwl = st.number_input("שיפוע דיספרסיה dκ/dλ (1/μm):", value=0.0, step=0.01)

        st.divider()
        add_noise = st.checkbox("כלול 0.5% רעש קיטוב אורתוגונלי", value=True)

    with col_main2:
        st.subheader("📊 ספקטרום תמסורת רחב")
        
        wls = np.linspace(wl_min, wl_max, 5000) # nm
        c = 299792458.0 # m/s
        f = c / (wls * 1e-9)
        f0 = c / (1550e-9)
        
        L_m = 2 * np.pi * (R_um2 * 1e-6)
        phi0 = 2 * np.pi * neff2 * L_m / 1550e-9
        phi = phi0 + (2 * np.pi * ng2 * L_m / c) * (f - f0)
        
        alpha_m = (alpha_db2 / 4.343) * 100
        a = np.exp(-alpha_m * L_m / 2)
        
        # חישוב k1_lambda לפי הבחירה
        k1_lambda = None
        fig_kappa = None
        
        if "העלאת קובץ" in kappa_source and uploaded_file is not None:
            try:
                # טעינת קובץ נתונים
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    df_k = pd.read_csv(uploaded_file)
                else:
                    df_k = pd.read_excel(uploaded_file)
                
                # מציאת עמודות
                cols = df_k.columns
                st.sidebar.success(f"הקובץ נטען בהצלחה! העמודות שנמצאו: {list(cols)}")
                
                wl_raw = df_k.iloc[:, 0].values # עמודה ראשונה: אורך גל
                k_raw = df_k.iloc[:, 1].values  # עמודה שנייה: קאפה
                
                # המרת יחידות אורך גל במידת הצורך (אם הוזן במיקרונים)
                if np.max(wl_raw) < 10.0:
                    wl_raw = wl_raw * 1000.0 # um -> nm
                
                # בניית פונקציית אינטרפולציה והחלקה
                f_interp = interp1d(
                    wl_raw, k_raw, 
                    kind=interp_kind.split()[0], 
                    bounds_error=False, 
                    fill_value=(k_raw[0], k_raw[-1])
                )
                
                k1_lambda = np.clip(f_interp(wls), 0.001, 0.999)
                k2_lambda = k1_lambda if "Add-Drop" in arch_type else np.zeros_like(k1_lambda)
                
                # גרף הצגת הצימוד והאינטרפולציה
                fig_kappa, ax_k = plt.subplots(figsize=(8, 2.5), dpi=150)
                plt.style.use('dark_background')
                fig_kappa.patch.set_facecolor('#0F172A')
                ax_k.set_facecolor('#0F172A')
                
                ax_k.scatter(wl_raw, k_raw, color='#F43F5E', label='נקודות דיסקרטיות מהקובץ', zorder=5)
                ax_k.plot(wls, k1_lambda, color='#38BDF8', label='עקומת החלקה (Interpolated)', linewidth=2)
                ax_k.set_xlabel("Wavelength λ (nm)", color='#94A3B8')
                ax_k.set_ylabel("Coupling κ(λ)", color='#94A3B8')
                ax_k.grid(True, color='#334155', linestyle=':', alpha=0.6)
                ax_k.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
                
            except Exception as e:
                st.error(f"שגיאה בקריאת הקובץ: {e}. מפעיל חישוב לפי ערכי ברירת מחדל.")
                k1_lambda = np.clip(0.15 + 0.0 * (wls - 1550), 0.001, 0.999)
                k2_lambda = k1_lambda if "Add-Drop" in arch_type else np.zeros_like(k1_lambda)
        else:
            k1_lambda = np.clip(kappa1 + dkappa_dwl * (wls - 1550) * 1e-3, 0.001, 0.999)
            k2_lambda = np.clip(kappa2 + dkappa_dwl * (wls - 1550) * 1e-3, 0.001, 0.999) if "Add-Drop" in arch_type else np.zeros_like(k1_lambda)

        t1 = np.sqrt(1 - k1_lambda**2)
        
        if "Add-Drop" in arch_type:
            t2 = np.sqrt(1 - k2_lambda**2)
            T_thru = (t1**2 - 2*a*t1*t2*np.cos(phi) + (a*t2)**2) / (1 - 2*a*t1*t2*np.cos(phi) + (a*t1*t2)**2)
            T_drop = (k1_lambda**2 * k2_lambda**2 * a) / (1 - 2*a*t1*t2*np.cos(phi) + (a*t1*t2)**2)
        else:
            t2 = 1.0
            T_thru = (a**2 - 2*a*t1*np.cos(phi) + t1**2) / (1 - 2*a*t1*np.cos(phi) + (a*t1)**2)
            T_drop = np.zeros_like(T_thru)

        if add_noise:
            T_thru = 0.995 * T_thru + 0.005
            if "Add-Drop" in arch_type:
                T_drop = 0.995 * T_drop + 0.0001
                
        T_thru_db = 10 * np.log10(np.maximum(1e-7, T_thru))
        
        # אם הועלה קובץ, מציגים קודם את הגרף של אינטרפולציית הקאפה
        if fig_kappa is not None:
            st.markdown("##### 📈 עקומת החלקה של מקדם הצימוד κ(λ)")
            st.pyplot(fig_kappa)
            st.write("")

        # גרף הספקטרום הרחב
        fig_broad, ax_b = plt.subplots(figsize=(10, 4.5), dpi=150)
        plt.style.use('dark_background')
        fig_broad.patch.set_facecolor('#0F172A')
        ax_b.set_facecolor('#0F172A')
        
        ax_b.plot(wls, T_thru_db, color='#38BDF8', linewidth=1.5, label='Through Port')
        
        if "Add-Drop" in arch_type:
            T_drop_db = 10 * np.log10(np.maximum(1e-7, T_drop))
            ax_b.plot(wls, T_drop_db, color='#F43F5E', linewidth=1.5, label='Drop Port')
            
        ax_b.set_xlabel("Wavelength λ (nm)", color='#94A3B8', fontweight='bold')
        ax_b.set_ylabel("Transmission (dB)", color='#94A3B8', fontweight='bold')
        ax_b.grid(True, color='#334155', linestyle=':', alpha=0.6)
        ax_b.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC')
        
        st.pyplot(fig_broad)
        
        st.download_button(
            label="📥 הורד איור ספקטרום רחב (PNG)",
            data=figure_to_bytes(fig_broad),
            file_name="broadband_spectrum.png",
            mime="image/png"
        )
        
        lambda_mid = (wl_min + wl_max) / 2.0
        fsr_nm = (lambda_mid**2) / (ng2 * (L_m * 1e9))
        fsr_ghz = c / (ng2 * L_m) * 1e-9
        
        st.success(f"💡 **נתונים תיאורטיים למבנה זה:** \n"
                   f"**FSR (Free Spectral Range):** {fsr_nm:.3f} nm  |  {fsr_ghz:.2f} GHz  \n"
                   f"**L (היקף הטבעת):** {L_m*1e6:.2f} μm")
