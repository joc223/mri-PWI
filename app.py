"""
MRI 灌流影像互動教學系統
PWI Interactive Teaching System
================================
知識來源（按優先級）：
  [PRIMARY]  08_PWI.pdf  - 蔡炳煇副教授，血流灌注權重影像
  [PRIMARY]  09_ASL.pdf  - 蔡炳煇副教授，動脈自旋標記法
  [PRIMARY]  03_SWI.pdf  - 磁化率權重影像
  [PRIMARY]  04_DWI.pdf  - 水分子擴散權重影像
  [PRIMARY]  01_TOF_MRA.pdf  - 磁振血管造影 (TOF)
  [PRIMARY]  02_Phase_contrast_MRA.pdf - 相位對比磁振血管攝影
  [SUPPLEMENT] 一般 MRI 物理知識（已標示）

部署：Streamlit Community Cloud
需求：streamlit, numpy, matplotlib
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Circle, Rectangle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ① 頁面基本設定
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MRI 灌流影像教學系統",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ② 全域 CSS 樣式
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* 側欄深色背景 */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a1628 0%,#1a3a5c 100%);
  }
  [data-testid="stSidebar"] * { color:#d6eaf8 !important; }

  /* 標題卡 */
  .main-header {
    background: linear-gradient(135deg,#0d2137,#1a5276);
    color:white; padding:22px 30px; border-radius:14px;
    margin-bottom:18px; text-align:center;
  }
  .main-header h1 { font-size:1.75rem; margin:0; }
  .main-header p  { font-size:.88rem; opacity:.8; margin:6px 0 0; }

  /* 概念卡片 */
  .concept-card {
    background:#eaf4fb; border-left:5px solid #2196F3;
    border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0;
  }
  .concept-card h4 { color:#0d47a1; margin:0 0 7px; font-size:.95rem; }
  .concept-card p  { margin:0; font-size:.87rem; line-height:1.65; }

  /* 強調卡 */
  .key-card {
    background:#fff8e1; border-left:5px solid #FF9800;
    border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0;
  }
  .key-card h4 { color:#e65100; margin:0 0 7px; font-size:.95rem; }
  .key-card p  { margin:0; font-size:.87rem; line-height:1.65; }

  /* 成功卡 */
  .ok-card {
    background:#e8f5e9; border-left:5px solid #4CAF50;
    border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0;
  }
  .ok-card h4 { color:#1b5e20; margin:0 0 7px; font-size:.95rem; }
  .ok-card p  { margin:0; font-size:.87rem; line-height:1.65; }

  /* 分隔線 */
  .divider {
    height:3px;
    background:linear-gradient(to right,#2196F3,transparent);
    border-radius:3px; margin:18px 0;
  }

  /* 信號亮度指示盒 */
  .sig-box {
    border-radius:12px; padding:16px;
    text-align:center; font-weight:bold; margin:8px 0;
  }

  /* 小提示文字 */
  .hint { font-size:.75rem; color:#888; font-style:italic; margin-top:3px; }

  /* 隱藏 Streamlit 預設 footer */
  footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ③ 物理模擬核心函數
#    Physics Simulation Core Functions
# ═══════════════════════════════════════════════════════════════

def spgr_signal(M0: float, T1_ms: float, TR_ms: float, flip_deg: float) -> float:
    """
    穩態 SPGR 信號公式 [SUPPLEMENT - MRI物理基礎]
    S = M0 · sin(α) · (1 - E1) / (1 - cos(α)·E1)
    E1 = exp(-TR/T1)

    參數：
        M0       : 平衡態磁化量（歸一化為 1）
        T1_ms    : 縱向弛豫時間（ms）
        TR_ms    : 重複時間（ms）
        flip_deg : RF 翻轉角（度）
    """
    alpha = np.radians(flip_deg)
    E1 = np.exp(-TR_ms / max(T1_ms, 1e-6))
    return M0 * np.sin(alpha) * (1.0 - E1) / (1.0 - np.cos(alpha) * E1 + 1e-12)


def t1_with_gd(T1_0_ms: float, r1_per_mM_per_s: float, Gd_mM: float) -> float:
    """
    Gd 顯影劑縮短 T1 [PRIMARY - PWI講義 p.3]
    1/T1_eff = 1/T1_0 + r1 × [Gd]

    r1 ≈ 4.5 L/(mmol·s) for Gd-DTPA at 1.5 T
    """
    r1_ms = r1_per_mM_per_s / 1000.0          # 轉換至 ms 單位
    inv_T1 = 1.0 / max(T1_0_ms, 1.0) + r1_ms * max(Gd_mM, 0.0)
    return 1.0 / max(inv_T1, 1e-9)


def t2star_with_gd(T2s_0_ms: float, r2s_per_mM_per_s: float, Gd_mM: float) -> float:
    """
    Gd 磁化率效應縮短 T2* [PRIMARY - PWI講義 p.7-8]
    1/T2*_eff = 1/T2*_0 + r2* × [Gd]

    r2* (tissue susceptibility effect) ≈ 12 L/(mmol·s) for brain tissue
    (注意：血管內 r2* 更高達 60-80，此為組織等效值)
    """
    r2s_ms = r2s_per_mM_per_s / 1000.0
    inv_T2s = 1.0 / max(T2s_0_ms, 1.0) + r2s_ms * max(Gd_mM, 0.0)
    return 1.0 / max(inv_T2s, 1e-9)


def gamma_variate(t: np.ndarray, t0: float = 15, alpha: float = 3.0, beta: float = 2.5) -> np.ndarray:
    """
    Gamma-variate 函數：模擬動脈輸入函數 (AIF) [PRIMARY - PWI講義 p.9]
    C(t) = ((t-t0)^alpha) × exp(-(t-t0)/beta)  for t > t0

    用於計算 CBF（需要對 AIF 去卷積）
    """
    result = np.zeros_like(t, dtype=float)
    mask = t > t0
    dt = t[mask] - t0
    result[mask] = (dt ** alpha) * np.exp(-dt / beta)
    mx = result.max()
    if mx > 0:
        result /= mx
    return result


def dce_signal_curve(t: np.ndarray, Gd_peak: float = 2.0,
                     curve_type: str = "washout") -> np.ndarray:
    """
    DCE 時間-信號曲線 [PRIMARY - PWI講義 p.3-4，乳房PWI三種曲線]

    T1WI 信號因 Gd 縮短 T1 而增強（wash-in）
    三種典型 Breast PWI 曲線：
      persistent (I)  → 良性腺體
      plateau    (II) → 中度可疑
      washout   (III) → 惡性腫瘤

    回傳值：相對信號強度（baseline = 1.0）
    """
    baseline = 1.0
    sig = np.ones_like(t, dtype=float)
    inj_t = 5.0    # 注射時間點 (s)
    mask = t > inj_t
    dt = t[mask] - inj_t

    if curve_type == "persistent":       # Type I
        amp = Gd_peak * 0.18
        sig[mask] = baseline + amp * (1.0 - np.exp(-0.12 * dt))
    elif curve_type == "plateau":        # Type II
        amp = Gd_peak * 0.26
        sig[mask] = baseline + amp * (1.0 - np.exp(-0.20 * dt)) * np.exp(-0.012 * dt)
    else:                                # Type III (washout)
        amp = Gd_peak * 0.38
        sig[mask] = baseline + amp * (1.0 - np.exp(-0.28 * dt)) * np.exp(-0.055 * dt)

    return sig


def dsc_signal_curve(t: np.ndarray, Gd_peak: float = 3.0,
                     tissue: str = "normal",
                     bolus_t: float = 15.0) -> np.ndarray:
    """
    DSC 時間-信號曲線 [PRIMARY - PWI講義 p.8，T2*WI perfusion curve]

    Gd 進入腦微血管 → 磁場不均勻 → T2* ↓ → 信號下降（變暗）
    信號變化幅度：30~60%（來源：PWI講義 p.9）

    tissue 類型：
      "normal"   - 正常腦組織
      "ischemic" - 缺血組織（延遲且幅度小）
      "tumor"    - 腦腫瘤（prolonged drop，部分 BBB 破壞）
    """
    baseline = 1.0
    sig = np.ones_like(t, dtype=float)

    if tissue == "normal":
        drop_amp = Gd_peak * 0.14      # 30-60% signal change
        rise_rate, recovery = 1.8, 0.09
    elif tissue == "ischemic":
        drop_amp = Gd_peak * 0.04      # 嚴重減少
        rise_rate, recovery = 0.6, 0.12
        bolus_t += 8                    # 到達時間延遲
    else:                               # tumor
        drop_amp = Gd_peak * 0.18
        rise_rate, recovery = 2.0, 0.03

    # 建立 V 型信號曲線
    for i, ti in enumerate(t):
        if ti < bolus_t:
            sig[i] = baseline
        elif ti < bolus_t + 8:
            r = (ti - bolus_t) / 8.0
            # 下降
            sig[i] = baseline - drop_amp * np.sin(r * np.pi / 2) ** rise_rate
        else:
            dt_r = ti - bolus_t - 8
            nadir = baseline - drop_amp
            sig[i] = baseline - drop_amp * np.exp(-recovery * dt_r) + 0.0

    return np.maximum(sig, 0.25)


def asl_delta_M(CBF_mL_100g_min: float,
                T1_blood_ms: float = 1650,
                alpha: float = 0.85,
                tau_s: float = 1.8,
                w_s: float = 1.5) -> float:
    """
    ASL 信號差值 ΔM/M0 [PRIMARY - ASL講義 p.6，General Kinetic Model]

    ΔM/M0 = 2·α·f·T1_blood·exp(-w/T1_blood)·(1-exp(-τ/T1_blood))

    根據講義：
      - 90° RF → 信號變化 < 3%
      - 180° RF (反轉) → 信號變化 < 5%
      - 需要多次平均（幾十次）以提升 SNR

    參數：
        CBF_mL_100g_min : 腦血流 (mL/100g/min)
        T1_blood_ms     : 血液 T1 (ms)；1.5T≈1350ms，3T≈1650ms
        alpha           : 標記效率 pCASL≈0.85
        tau_s           : 標記持續時間 (s)
        w_s             : PLD 等待時間 (s)
    """
    f = CBF_mL_100g_min / 6000.0           # 轉為 mL/g/s
    T1b = T1_blood_ms / 1000.0             # 轉為 s
    delta_M = (2.0 * alpha * f * T1b
               * np.exp(-w_s / T1b)
               * (1.0 - np.exp(-tau_s / T1b)))
    return delta_M                          # 回傳為小數（乘 100 得 %）


def dwi_signal(b_val: float, ADC_mm2_s: float, S0: float = 1.0) -> float:
    """
    DWI 信號強度 [PRIMARY - DWI講義 p.5]
    S(b) = S0 × exp(-b × ADC)

    正常腦組織  ADC ≈ 0.75–0.80 × 10⁻³ mm²/s  → 中等亮度
    急性缺血    ADC ≈ 0.30–0.45 × 10⁻³ mm²/s  → DWI 亮 (restricted diffusion)
    CSF         ADC ≈ 3.0    × 10⁻³ mm²/s  → DWI 暗 (free diffusion)
    """
    return S0 * np.exp(-b_val * ADC_mm2_s)


def tof_fre(flow_cm_s: float, TR_ms: float = 30.0,
            flip_deg: float = 50.0,
            slice_mm: float = 3.0,
            T1_tissue_ms: float = 900.0,
            T1_blood_ms: float = 1200.0):
    """
    TOF-MRA 血流增強效應 (FRE) [PRIMARY - TOF講義 p.3，血流增強效應]

    靜止組織：短 TR → 飽和 → 信號弱
    流入血液：新鮮（未飽和）→ 信號強

    返回：(S_tissue, S_blood, FRE_percent)
    """
    alpha = np.radians(flip_deg)
    E1t = np.exp(-TR_ms / max(T1_tissue_ms, 1.0))
    E1b = np.exp(-TR_ms / max(T1_blood_ms, 1.0))

    # 靜止組織（完全飽和）
    S_tissue = np.sin(alpha) * (1.0 - E1t) / (1.0 - np.cos(alpha) * E1t + 1e-12)

    # 新鮮血液流入比例
    if flow_cm_s > 0:
        transit_ms = (slice_mm / 10.0) / flow_cm_s * 1000.0
        fresh = min(1.0, transit_ms / max(TR_ms, 1.0))
        S_blood_sat = np.sin(alpha) * (1.0 - E1b) / (1.0 - np.cos(alpha) * E1b + 1e-12)
        S_blood = fresh * np.sin(alpha) + (1.0 - fresh) * S_blood_sat
    else:
        S_blood = np.sin(alpha) * (1.0 - E1b) / (1.0 - np.cos(alpha) * E1b + 1e-12)

    fre_pct = (S_blood - S_tissue) / (S_tissue + 1e-12) * 100.0
    return S_tissue, S_blood, fre_pct


def pc_phase(velocity_cm_s: float, venc_cm_s: float = 80.0):
    """
    PC-MRA 相位角計算 [PRIMARY - PC-MRA講義 p.3-5]
    Φ = π × v / Venc

    相角與流速成正比（講義重點）
    超過 Venc → Phase aliasing（相位混疊）

    返回：(phase_rad, phase_wrapped_deg, is_aliased)
    """
    phi = np.pi * velocity_cm_s / max(abs(venc_cm_s), 1e-6)
    # 包裹相位至 [-π, π]
    phi_wrapped = (phi + np.pi) % (2 * np.pi) - np.pi
    aliased = abs(velocity_cm_s) > abs(venc_cm_s)
    return phi, np.degrees(phi_wrapped), aliased


# ═══════════════════════════════════════════════════════════════
# ④ 腦部模型生成（簡化橢圓幻象）
# ═══════════════════════════════════════════════════════════════

def make_brain_phantom(size: int = 80, lesion: bool = True) -> np.ndarray:
    """
    生成簡化圓形腦部模型
    組織標籤：0=背景, 1=腦脊液, 2=灰質, 3=白質, 4=病變/缺血區
    """
    yy, xx = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(xx ** 2 + yy ** 2)

    ph = np.zeros((size, size), dtype=int)
    ph[r < size * 0.46] = 2           # 灰質環
    ph[r < size * 0.36] = 3           # 白質
    ph[r < size * 0.08] = 1           # 腦室（腦脊液）

    if lesion:
        # 右半球病變（偏離中心）
        lx, ly = size // 5, -size // 10
        lr = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
        ph[lr < size * 0.10] = 4

    return ph


def phantom_dce(ph: np.ndarray, Gd_mM: float) -> np.ndarray:
    """
    DCE 模式：T1 效應 → 信號增強（變亮）[PRIMARY - PWI講義 p.3]
    使用 SPGR 信號，TR=5ms，α=30°（Fast T1WI 參數）
    """
    # T1 參照值（ms）: 灰質 1300, 白質 800, CSF 3500, 腫瘤 1200
    T1_ref = {0: 9999, 1: 3500, 2: 1300, 3: 800, 4: 1200}
    r1 = 4.5   # Gd-DTPA r1 (L/mmol/s)
    TR, flip = 5.0, 30.0

    sig = np.zeros(ph.shape, dtype=float)
    for label, T1_0 in T1_ref.items():
        mask = ph == label
        if not mask.any() or label == 0:
            continue
        # 病變區有更多 Gd 滲入（BBB 破壞）
        gd_local = Gd_mM * 1.8 if label == 4 else Gd_mM * 0.04
        T1_eff = t1_with_gd(T1_0, r1, gd_local)
        sig[mask] = spgr_signal(1.0, T1_eff, TR, flip)

    return sig


def phantom_dsc(ph: np.ndarray, Gd_mM: float) -> np.ndarray:
    """
    DSC 模式：T2* 效應 → 信號下降（變暗）[PRIMARY - PWI講義 p.7-8]
    使用 GRE，TE=30ms（EPI for DSC）
    BBB 完整 → Gd 侷限於微血管 → T2* 磁化率效應擴及周邊組織
    """
    T2s_ref = {0: 9999, 1: 250, 2: 55, 3: 60, 4: 50}
    r2s = 14.0   # 等效組織 r2* (L/mmol/s)
    TE = 30.0    # ms

    sig = np.zeros(ph.shape, dtype=float)
    for label, T2s_0 in T2s_ref.items():
        mask = ph == label
        if not mask.any() or label == 0:
            continue
        # 正常腦組織：Gd 侷限血管內，磁化率效應影響周邊
        gd_local = Gd_mM * 1.0
        T2s_eff = t2star_with_gd(T2s_0, r2s, gd_local)
        sig[mask] = np.exp(-TE / T2s_eff) * 0.85

    return sig


def phantom_dwi(ph: np.ndarray, b_val: float = 1000.0,
                lesion_adc: float = 0.38e-3) -> np.ndarray:
    """
    DWI 模式：擴散限制區域信號較亮 [PRIMARY - DWI講義 p.5]
    急性缺血 → ADC 降低 → DWI 亮（restricted diffusion）
    """
    ADC_ref = {0: 0, 1: 3.0e-3, 2: 0.78e-3, 3: 0.72e-3, 4: lesion_adc}
    sig = np.zeros(ph.shape, dtype=float)
    for label, adc in ADC_ref.items():
        mask = ph == label
        if not mask.any():
            continue
        sig[mask] = dwi_signal(b_val, adc)
    return sig


# ═══════════════════════════════════════════════════════════════
# ⑤ 灌流參數地圖生成
# ═══════════════════════════════════════════════════════════════

def make_perfusion_maps(cbf_normal: float = 60.0,
                        cbf_lesion: float = 20.0,
                        mtt_normal: float = 4.0,
                        mtt_lesion: float = 9.0,
                        size: int = 80):
    """
    模擬 rCBF / rCBV / rMTT 參數地圖 [PRIMARY - PWI講義 p.9]
    根據講義：相關參數 maps → rCBV, rMTT, rCBF
    關係式：CBF = CBV / MTT

    返回：(cbf_map, cbv_map, mtt_map, phantom)
    """
    rng = np.random.default_rng(42)
    ph = make_brain_phantom(size, lesion=True)

    cbf = np.zeros((size, size))
    cbv = np.zeros((size, size))
    mtt = np.zeros((size, size))

    table = {
        1: (5,   0.8, 3.5),    # CSF
        2: (cbf_normal * 1.2, 4.2, mtt_normal),   # 灰質 CBF 較高
        3: (cbf_normal * 0.55, 1.9, mtt_normal),  # 白質
        4: (cbf_lesion,        1.1, mtt_lesion),  # 病變/缺血
    }

    for label, (c, v, m) in table.items():
        mask = ph == label
        cbf[mask] = c + rng.normal(0, c * 0.06, mask.sum())
        cbv[mask] = v + rng.normal(0, v * 0.05, mask.sum())
        mtt[mask] = m + rng.normal(0, m * 0.04, mask.sum())

    cbf = np.maximum(cbf, 0)
    cbv = np.maximum(cbv, 0)
    mtt = np.maximum(mtt, 0.5)
    return cbf, cbv, mtt, ph


# ═══════════════════════════════════════════════════════════════
# ⑥ 共用圖表樣式工具
# ═══════════════════════════════════════════════════════════════

DARK_BG   = "#0d1117"
SPINE_CLR = "#333a45"

def dark_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "",
            title_color: str = "white") -> None:
    """統一深色背景軸設定"""
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=title_color, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color="#aaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE_CLR)


def dark_fig(nrows: int = 1, ncols: int = 1, figsize=(8, 4)) -> tuple:
    """建立深色背景 Figure"""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    return fig, axes


# ═══════════════════════════════════════════════════════════════
# ⑦ 各頁面函數
# ═══════════════════════════════════════════════════════════════

# ────────────────────────── 首頁 ──────────────────────────────
def page_home():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MRI 灌流影像互動教學系統</h1>
        <p>蔡炳煇副教授課程 ｜ 中山醫學大學醫學影像暨放射科學系</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("### 🔬 什麼是灌流（Perfusion）？")
        st.markdown("""
        <div class="concept-card">
          <h4>微血管層級的血液交換 [PRIMARY - PWI講義 p.1]</h4>
          <p>
          灌流 = 組織層級的微血管血流，反映「多少血液在一定時間內流過組織」。<br>
          微血管直徑僅 <b>0.005~0.01 mm</b>，MRI 空間解析度無法直接看到，<br>
          因此使用<b>追蹤劑（Tracer）</b>觀察整體血流交換表現。
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="key-card">
          <h4>⚗️ MRI 主要對比劑 [PRIMARY - PWI講義 p.1]</h4>
          <p>
          <b>Gd-DTPA</b>（釓螯合物）：Magnevist、Dotarem<br>
          &nbsp;→ 順磁性（paramagnetic），同時縮短 T1 <i>與</i> T2*<br>
          &nbsp;→ 分子較大，正常腦組織 <b>無法穿越 BBB</b><br><br>
          <b>SPIO</b>（超順磁氧化鐵）：主要縮短 T2/T2*，磁化率效應強
          </p>
        </div>
        """, unsafe_allow_html=True)

        # 三種方法橫排說明
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="concept-card">
              <h4>💉 DCE</h4>
              <p>Gd → T1 縮短<br>T1WI 信號 <b>↑ 亮</b><br>適合乳房/BBB破壞</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="concept-card">
              <h4>🌀 DSC</h4>
              <p>Gd → T2* 縮短<br>T2*WI 信號 <b>↓ 暗</b><br>適合腦部灌流</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="ok-card">
              <h4>🔄 ASL</h4>
              <p>水分子 = 追蹤劑<br>信號差值 <b>< 5%</b><br><b>不需注射顯影劑</b></p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # ── 追蹤劑動態曲線示意圖 ──
        t = np.linspace(0, 70, 500)

        fig, axes = dark_fig(2, 1, figsize=(4.5, 5.5))
        ax1, ax2 = axes

        # DCE 曲線（T1WI：信號上升）
        curves = {
            "Washout (III)": ("#e74c3c", dce_signal_curve(t, 3.0, "washout")),
            "Plateau  (II)": ("#f39c12", dce_signal_curve(t, 3.0, "plateau")),
            "Persistent (I)": ("#2ecc71", dce_signal_curve(t, 3.0, "persistent")),
        }
        for lbl, (clr, sig) in curves.items():
            ax1.plot(t, sig, color=clr, lw=2, label=lbl)
        ax1.axvline(5, color="#aaa", lw=1, ls="--", alpha=0.6)
        ax1.text(5.5, 1.0, "Gd注射", color="#aaa", fontsize=7)
        ax1.set_ylim(0.9, 1.62)
        dark_ax(ax1, "DCE (T1WI)：信號 ↑ 增強",
                "Time (s)", "Signal (a.u.)", title_color="#f39c12")
        ax1.legend(fontsize=7, framealpha=0.2, labelcolor="white", loc="upper right")

        # DSC 曲線（T2*WI：信號下降）
        ax2.plot(t, dsc_signal_curve(t, 3.0, "normal"), color="#3498db", lw=2.5, label="Normal")
        ax2.plot(t, dsc_signal_curve(t, 3.0, "ischemic"), color="#e74c3c", lw=2,
                 ls="--", label="Ischemic")
        ax2.axvline(15, color="#aaa", lw=1, ls="--", alpha=0.6)
        ax2.text(15.5, 0.97, "Bolus到達", color="#aaa", fontsize=7)
        ax2.set_ylim(0.25, 1.08)
        dark_ax(ax2, "DSC (T2*WI)：信號 ↓ 下降",
                "Time (s)", "Signal (a.u.)", title_color="#3498db")
        ax2.legend(fontsize=7, framealpha=0.2, labelcolor="white")

        fig.suptitle("Tracer 動態曲線", color="white", fontsize=11, y=1.0)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig)
        plt.close(fig)

    # 灌流參數說明卡
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 主要灌流參數 [PRIMARY - PWI講義 p.2]")

    pc = st.columns(4)
    params = [
        ("CBV", "腦血容積", "每百克組織內的血液量", "mL/100g", "#3498db"),
        ("CBF", "腦血流量", "單位時間內流過組織的血液量", "mL/100g/min", "#2ecc71"),
        ("MTT", "平均通過時間", "血液通過微血管的平均時間", "seconds", "#9b59b6"),
        ("TTP", "達峰時間", "從注射到信號達最大變化的時間", "seconds", "#f39c12"),
    ]
    for col, (abbr, zh, desc, unit, clr) in zip(pc, params):
        with col:
            st.markdown(f"""
            <div style="background:#1a2233; border:2px solid {clr}; border-radius:10px;
                        padding:14px; text-align:center;">
              <div style="font-size:1.5rem; font-weight:bold; color:{clr};">{abbr}</div>
              <div style="color:white; font-size:.9rem; font-weight:bold;">{zh}</div>
              <div style="color:#aaa; font-size:.78rem; margin:5px 0;">{desc}</div>
              <div style="color:{clr}; font-size:.75rem; font-style:italic;">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="key-card" style="margin-top:12px;">
      <h4>⚡ 核心關係式 [PRIMARY - PWI講義 p.2]</h4>
      <p><b>CBF = CBV ÷ MTT</b> &nbsp;（CBF 由 CBV 與 MTT 共同決定）<br>
      缺血核心：CBV↓↓ CBF↓↓ MTT↑ &nbsp;｜&nbsp; 缺血半暗帶：CBV≈正常, CBF↓, MTT↑</p>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────── DCE 頁 ──────────────────────────
def page_dce():
    st.markdown("""
    <div class="main-header">
        <h1>💉 DCE 動態對比增強灌流影像</h1>
        <p>Dynamic Contrast Enhancement ｜ T1 效應 ｜ 信號增強（變亮）</p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_ctrl = st.columns([1.3, 1])

    with col_info:
        st.markdown("""
        <div class="concept-card">
          <h4>🔬 DCE 原理 [PRIMARY - PWI講義 p.3-4]</h4>
          <p>
          <b>Gd-DTPA 的 T1 縮短效應</b>：<br>
          顯影劑進入組織 → <code>1/T1_eff = 1/T1₀ + r₁×[Gd]</code><br>
          T1 縮短 → SPGR信號增強 → <b>T1WI 影像變亮</b><br><br>
          <b>微血管通透性（BBB）的重要性</b>：<br>
          &bull; 正常腦：Gd 無法滲出微血管，T1 增強效應極小<br>
          &bull; 腫瘤/BBB破壞：Gd 滲入 EES（血管外細胞外空間）→ 明顯增強<br>
          &bull; 乳房腫瘤：血管生成旺盛 → 快速 wash-in
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="key-card">
          <h4>⚗️ Tofts Model [PRIMARY - PWI講義 p.4]</h4>
          <p>
          假設微血管與組織間交換達平衡：<br>
          <code>C_t(t) = v_p·C_p(t) + K<sup>trans</sup>·∫C_p(τ)·e<sup>-kep(t-τ)</sup>dτ</code><br>
          → 連續 T1WI 追蹤 <b>10 分鐘以上</b><br>
          → 臨床應用：BBB破壞、乳房、骨頭、關節
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_ctrl:
        st.markdown("### 🎮 互動控制")
        Gd = st.slider("💉 Gd-DTPA 濃度 [Gd] (mM)",
                       0.0, 5.0, 0.0, 0.1,
                       help="滑動模擬注射顯影劑後組織中濃度")
        tissue_sel = st.selectbox("🧬 病變/組織類型",
            ["正常腦組織（BBB 完整）",
             "腦腫瘤（BBB 破壞）",
             "乳房惡性腫瘤（Washout）",
             "乳房良性病變（Persistent）"])
        TR_val  = st.slider("TR (ms) — Fast SPGR", 3, 15, 5, 1)
        flip_val = st.slider("Flip Angle (°)", 10, 50, 30, 5)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_img, col_curve, col_stat = st.columns([1, 1.6, 0.9])

    # ── 模擬影像 ──
    with col_img:
        st.markdown("#### 🖼️ DCE 模擬影像")
        ph = make_brain_phantom(80, lesion=True)
        pre  = phantom_dce(ph, 0.0)
        post = phantom_dce(ph, Gd)

        fig, axes = dark_fig(1, 2, figsize=(5, 2.8))
        for ax, sig, ttl in zip(axes, [pre, post],
                                 ["Pre-contrast\n(Gd=0)", f"Post-contrast\n({Gd:.1f}mM)"]):
            ax.imshow(sig, cmap="gray", vmin=0, vmax=0.55, interpolation="bilinear")
            ax.set_title(ttl, color="white", fontsize=8)
            ax.axis("off")
        fig.suptitle("T1W DCE 模擬", color="white", fontsize=9)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)
        plt.close(fig)

        # 信號增強指示條
        base_s = pre[ph == 4].mean() if (ph == 4).any() else 0.1
        post_s = post[ph == 4].mean() if (ph == 4).any() else 0.1
        enhance = (post_s - base_s) / max(base_s, 1e-6) * 100
        bright = min(255, int(160 + enhance * 2))
        st.markdown(f"""
        <div class="sig-box" style="background:rgb({bright},{max(60,bright-80)},30);
             color:{'white' if bright<200 else 'black'}; border:2px solid #f39c12;">
          病變區增強<br>
          <span style="font-size:1.5rem; font-weight:bold;">+{enhance:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 時間-信號曲線 ──
    with col_curve:
        st.markdown("#### 📈 三種典型乳房 DCE 曲線 [PRIMARY - PWI講義 p.4]")
        t = np.linspace(0, 70, 500)

        fig2, ax2 = dark_fig(1, 1, figsize=(6.5, 4))
        colors = {"washout": "#e74c3c", "plateau": "#f39c12", "persistent": "#2ecc71"}
        labels = {
            "washout":    "Type III (Washout) ← 惡性↑",
            "plateau":    "Type II  (Plateau) ← 中度可疑",
            "persistent": "Type I   (Persistent) ← 良性可能",
        }

        if "惡性" in tissue_sel:
            shown = ["washout", "plateau", "persistent"]
        elif "良性" in tissue_sel:
            shown = ["persistent", "plateau", "washout"]
        else:
            shown = ["washout", "plateau", "persistent"]

        for ctype in shown:
            sig = dce_signal_curve(t, Gd, ctype)
            lw  = 3.0 if ctype == shown[0] else 1.5
            ls  = "-" if ctype == shown[0] else "--"
            ax2.plot(t, sig, color=colors[ctype], lw=lw, ls=ls, label=labels[ctype])

        ax2.axvline(5, color="#aaa", lw=1.2, ls=":", label="Gd 注射時間 (t=5s)")
        ax2.axhline(1.0, color="#555", lw=0.8, ls=":")

        # 填入 wash-in 面積（代表 AUC）
        sig_main = dce_signal_curve(t, Gd, shown[0])
        ax2.fill_between(t, 1.0, sig_main, where=sig_main > 1.0,
                         alpha=0.12, color=colors[shown[0]])

        ax2.set_ylim(0.88, 1.0 + Gd * 0.16 + 0.1)
        dark_ax(ax2, f"DCE Time-Signal Curves ｜ Peak [Gd] ≈ {Gd:.1f} mM",
                "Time (s)", "Relative Signal Intensity")
        ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")

        # Wash-in / Wash-out 標示
        if Gd > 0.3:
            ax2.annotate("Wash-in ↑", xy=(12, 1 + Gd * 0.05),
                         color="#f39c12", fontsize=9, ha="center")
            ax2.annotate("Wash-out ↓\n(malignant indicator)",
                         xy=(45, 1 + Gd * 0.02),
                         color="#e74c3c", fontsize=8, ha="center")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── 物理參數計算 ──
    with col_stat:
        st.markdown("#### 📊 即時計算")
        r1 = 4.5
        T1_0_tumor = 1200
        T1_eff = t1_with_gd(T1_0_tumor, r1, Gd * 1.8)
        S_pre  = spgr_signal(1.0, T1_0_tumor, TR_val, flip_val)
        S_post = spgr_signal(1.0, T1_eff,     TR_val, flip_val)
        enh = (S_post - S_pre) / max(S_pre, 1e-6) * 100

        st.metric("腫瘤 T1₀", f"{T1_0_tumor} ms")
        st.metric("T1 (Gd後)", f"{T1_eff:.0f} ms",
                  delta=f"{T1_eff-T1_0_tumor:.0f} ms")
        st.metric("信號增強率", f"{enh:.1f}%",
                  delta=f"+{enh:.1f}%" if enh > 0 else "—")
        Ktrans = 0.012 * Gd if Gd > 0 else 0
        st.metric("Ktrans 估算", f"{Ktrans:.3f} min⁻¹",
                  help="體積轉移係數，反映血管通透性")

        st.markdown("---")
        st.markdown("""
        <div class="concept-card" style="padding:10px">
          <h4>💡 臨床判讀</h4>
          <p style="font-size:.8rem">
          🔴 <b>Washout</b>：高惡性可能<br>
          🟡 <b>Plateau</b>：中度可疑<br>
          🟢 <b>Persistent</b>：傾向良性<br><br>
          使用 <b>Tofts Model</b> 可定量計算 K<sup>trans</sup>、v_e
          </p>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────── DSC 頁 ──────────────────────────
def page_dsc():
    st.markdown("""
    <div class="main-header">
        <h1>🌀 DSC 動態磁化率對比灌流影像</h1>
        <p>Dynamic Susceptibility Contrast ｜ T2* 效應 ｜ 信號下降（變暗）｜ 腦灌流首選</p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_ctrl = st.columns([1.3, 1])

    with col_info:
        st.markdown("""
        <div class="concept-card">
          <h4>🔬 DSC 原理 [PRIMARY - PWI講義 p.7-8]</h4>
          <p>
          <b>顯影劑干擾磁場均勻度</b>：<br>
          Gd 在微血管內 → 順磁性使周邊磁場不均勻<br>
          → T2* 大幅縮短 → T2*WI 信號<b>明顯下降</b><br><br>
          <b>為何選 T2* 而非 T1？</b><br>
          &bull; T1 效應侷限在血管腔內（效應範圍小）<br>
          &bull; <b>T2* 磁化率效應擴及微血管周邊組織</b>（範圍大）<br>
          &bull; 信號變化高達 <b>30~60%</b>（講義原文）<br>
          &bull; First-pass 即可看到 → 掃描時間短（1-2 分鐘）<br>
          &bull; 使用 EPI：1 秒涵蓋 3~6 個切面
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="key-card">
          <h4>⚗️ 物理公式 [PRIMARY - PWI講義 p.9]</h4>
          <p>
          <code>1/T2*_eff = 1/T2*₀ + r₂* × [Gd]</code><br>
          信號：<code>S = S₀ × exp(-TE / T2*_eff)</code><br><br>
          Gamma-variate AIF → 去卷積 → CBF, CBV, MTT<br>
          關係式：<code>CBF = CBV / MTT</code>
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_ctrl:
        st.markdown("### 🎮 互動控制")
        Gd_peak = st.slider("💉 Gd 峰值濃度 (mM)", 0.0, 8.0, 3.0, 0.2,
                             help="First-pass 時 Gd 在腦微血管中的峰值濃度")
        tissue_type = st.radio("🧠 組織狀態",
                               ["正常腦組織", "缺血組織（Ischemic）", "腦腫瘤（Tumor）"])
        TE_val = st.slider("TE (ms) — GRE-EPI", 20, 60, 30, 5,
                           help="DSC 常用 TE ≈ 30 ms (1.5T)")
        show_aif = st.checkbox("顯示 AIF 曲線", value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_img, col_curve, col_maps = st.columns([1, 1.5, 1])

    # ── 模擬影像（5-23s 序列） ──
    with col_img:
        st.markdown("#### 🖼️ DSC 腦部時序影像")

        # 模擬 5-23 秒多個時間點（仿 PWI講義 p.8 的腦影像序列）
        ph = make_brain_phantom(80, lesion=True)
        tissue_str = ("ischemic" if "缺血" in tissue_type
                      else "tumor" if "腫瘤" in tissue_type
                      else "normal")

        t_pts = np.array([5.0, 10.0, 15.0, 20.0, 23.0])
        t_full = np.linspace(0, 70, 1000)
        full_curve = dsc_signal_curve(t_full, Gd_peak, tissue_str, bolus_t=15)

        fig_seq, ax_seq_arr = dark_fig(1, 5, figsize=(10, 2.4))

        for ax_s, tp in zip(ax_seq_arr, t_pts):
            idx = np.argmin(np.abs(t_full - tp))
            sig_frac = full_curve[idx]
            img = phantom_dsc(ph, Gd_peak * (1 - sig_frac) * 8.0)
            ax_s.imshow(img, cmap="gray", vmin=0, vmax=0.85,
                        interpolation="bilinear")
            ax_s.set_title(f"{tp:.0f}s", color="white", fontsize=8)
            ax_s.axis("off")

        fig_seq.suptitle("DSC 時序影像（Gd 通過時信號下降）",
                         color="white", fontsize=9)
        plt.tight_layout(pad=0.1)
        st.pyplot(fig_seq)
        plt.close(fig_seq)

        # 即時信號下降指示
        T2s_0 = 55.0
        T2s_eff = t2star_with_gd(T2s_0, 14.0, Gd_peak)
        S_pre  = np.exp(-TE_val / T2s_0)
        S_peak = np.exp(-TE_val / T2s_eff)
        drop_pct = (S_pre - S_peak) / max(S_pre, 1e-6) * 100

        dark_val = max(20, int(200 - drop_pct * 2.2))
        st.markdown(f"""
        <div class="sig-box"
             style="background:rgb({dark_val},{dark_val},{min(255,dark_val+40)});
                    color:{'white' if dark_val<130 else '#333'};
                    border:2px solid #3498db;">
          腦組織信號下降<br>
          <span style="font-size:1.5rem;">−{drop_pct:.0f}%</span><br>
          <span style="font-size:.8rem">(T2*: {T2s_0:.0f}→{T2s_eff:.0f} ms)</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 時間-信號曲線（V 型） ──
    with col_curve:
        st.markdown("#### 📈 T2*WI 灌流曲線 [PRIMARY - PWI講義 p.8]")

        t = np.linspace(0, 70, 500)
        sig_main  = dsc_signal_curve(t, Gd_peak, tissue_str)
        sig_norm  = dsc_signal_curve(t, Gd_peak, "normal")

        fig_c, ax_c = dark_fig(1, 1, figsize=(6.5, 4))

        # 正常曲線（灰色參考）
        ax_c.plot(t, sig_norm, color="#95a5a6", lw=1.5, ls="--", label="Normal (ref)")

        # 主曲線
        clr_map = {"normal": "#3498db", "ischemic": "#e74c3c", "tumor": "#9b59b6"}
        main_clr = clr_map[tissue_str]
        ax_c.plot(t, sig_main, color=main_clr, lw=2.8, label=tissue_type)

        # AIF
        if show_aif:
            aif = gamma_variate(t, t0=15, alpha=3, beta=2.5)
            aif_scaled = 1.0 - aif * Gd_peak * 0.08
            ax_c.plot(t, aif_scaled, color="#f39c12", lw=1.5, ls=":",
                      label="AIF（動脈輸入）", alpha=0.9)

        # 填入 CBV 積分面積
        ax_c.fill_between(t, 1.0, sig_norm, where=sig_norm < 1.0,
                          alpha=0.15, color="#3498db",
                          label="∫ ΔS dt ∝ CBV")

        # 最低點標示
        min_i = np.argmin(sig_main)
        ax_c.scatter(t[min_i], sig_main[min_i], color="#f39c12", s=120, zorder=5)
        ax_c.annotate(f"Nadir ΔS={( 1-sig_main[min_i])*100:.0f}%\n(T2* susceptibility)",
                      xy=(t[min_i], sig_main[min_i]),
                      xytext=(t[min_i] + 7, sig_main[min_i] + 0.15),
                      arrowprops=dict(arrowstyle="->", color="#f39c12"),
                      color="#f39c12", fontsize=8)

        ax_c.axvline(15, color="#aaa", lw=1, ls=":", alpha=0.7)
        ax_c.text(15.5, 0.99, "Bolus", color="#aaa", fontsize=7)

        dark_ax(ax_c, f"DSC Perfusion Curve ｜ TE={TE_val}ms ｜ Peak Gd={Gd_peak:.1f}mM",
                "Time (s)", "Relative Signal S/S₀")
        ax_c.legend(fontsize=8, framealpha=0.2, labelcolor="white")
        ax_c.set_ylim(0.2, 1.08)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

    # ── 灌流參數地圖 ──
    with col_maps:
        st.markdown("#### 🗺️ 參數地圖 [PRIMARY - PWI講義 p.9]")

        if "缺血" in tissue_type:
            cl, ml = 18, 10
        elif "腫瘤" in tissue_type:
            cl, ml = 120, 4
        else:
            cl, ml = 60, 4

        cbf_m, cbv_m, mtt_m, _ = make_perfusion_maps(
            cbf_normal=60, cbf_lesion=cl, mtt_normal=4, mtt_lesion=ml)

        maps_info = [
            (cbf_m, "rCBF",  "hot",   0, 90),
            (cbv_m, "rCBV",  "hot",   0, 6),
            (mtt_m, "rMTT",  "cool_r", 0, 12),
        ]

        fig_m, ax_m_arr = dark_fig(1, 3, figsize=(7, 2.6))
        for ax_m, (data, name, cmp, vmn, vmx) in zip(ax_m_arr, maps_info):
            im = ax_m.imshow(data, cmap=cmp, vmin=vmn, vmax=vmx,
                              interpolation="bilinear")
            ax_m.set_title(name, color="white", fontsize=9)
            ax_m.axis("off")
            cb = fig_m.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors="white", labelsize=7)

        fig_m.suptitle("Perfusion Maps", color="white", fontsize=9)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_m)
        plt.close(fig_m)

        # 缺血分類表格（來自 PWI講義 p.10）
        st.markdown("""
        <div class="concept-card" style="padding:10px; margin-top:8px">
          <h4>📋 缺血分類 [PRIMARY - PWI p.10]</h4>
          <p style="font-size:.78rem">
          <table style="width:100%; border-collapse:collapse;">
          <tr><th style="color:#3498db">組織狀態</th><th>TTP</th><th>CBV</th><th>CBF</th></tr>
          <tr><td>正常</td><td>—</td><td>—</td><td>—</td></tr>
          <tr><td>半暗帶</td><td>↑</td><td>—↑</td><td>↓</td></tr>
          <tr><td>梗塞核心</td><td>↑</td><td>↓</td><td>↓↓</td></tr>
          </table>
          </p>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────── ASL 頁 ──────────────────────────
def page_asl():
    st.markdown("""
    <div class="main-header">
        <h1>🔄 ASL 動脈自旋標記灌流影像</h1>
        <p>Arterial Spin Labeling ｜ 水分子作追蹤劑 ｜ 不需注射顯影劑</p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_ctrl = st.columns([1.3, 1])

    with col_info:
        st.markdown("""
        <div class="ok-card">
          <h4>💡 ASL 核心概念 [PRIMARY - ASL講義 p.1-2]</h4>
          <p>
          <b>天然追蹤劑 = 水分子！</b><br>
          &bull; 水分子可<b>自由通過 BBB</b>（不受血腦障壁限制）<br>
          &bull; 用 RF 脈衝標記（飽和/反轉）上游動脈血中的水<br>
          &bull; 等待 PLD 後水流入腦組織<br>
          &bull; <b>有標記影像</b> − <b>無標記影像</b> = 灌流信號<br><br>
          ⚠️ <b>信號差異非常微小</b>（講義強調）：<br>
          &nbsp;&nbsp;90° RF → 信號變化 <b>&lt; 3%</b><br>
          &nbsp;&nbsp;180° RF → 信號變化 <b>&lt; 5%</b><br>
          &nbsp;&nbsp;→ 需要多次平均（幾十次掃描）
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="key-card">
          <h4>⚗️ 動力學模型 [PRIMARY - ASL講義 p.6]</h4>
          <p>
          <code>ΔM/M₀ = 2α·f·T₁blood·e<sup>-w/T1b</sup>·(1-e<sup>-τ/T1b</sup>)</code><br><br>
          f = CBF（mL/g/s）｜α = 標記效率<br>
          τ = 標記持續時間（s）｜w = PLD（s）
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_ctrl:
        st.markdown("### 🎮 互動控制")
        CBF_val = st.slider("🩸 腦血流 CBF (mL/100g/min)",
                             0, 120, 60, 5,
                             help="正常灰質≈60-80 ｜ 缺血<20 ｜ 腫瘤>100")
        asl_type = st.selectbox("⚙️ ASL 技術類型",
            ["pCASL（偽連續，臨床主流）",
             "pASL（脈衝式，EPISTAR/FAIR）",
             "cASL（連續式）"])
        PLD_val = st.slider("⏱️ PLD 等待時間 (s)", 0.5, 3.0, 1.5, 0.1,
                             help="標記後等待血液流入組織的時間")
        tau_val = st.slider("📏 標記持續時間 τ (s)", 0.5, 3.0, 1.8, 0.1)
        alpha_val = {"pCASL": 0.85, "pASL": 0.95, "cASL": 0.70}
        asl_key = "pCASL" if "pCASL" in asl_type else ("pASL" if "pASL" in asl_type else "cASL")
        alpha_use = alpha_val[asl_key]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_concept, col_cbf, col_sub = st.columns([1, 1.5, 1])

    # ── 標記概念示意圖 ──
    with col_concept:
        st.markdown("#### 🖼️ 標記流程示意 [PRIMARY - ASL講義 p.3]")

        fig_asl, ax_asl = dark_fig(1, 1, figsize=(3.8, 5))
        ax_asl.set_xlim(0, 10)
        ax_asl.set_ylim(0, 10)
        ax_asl.axis("off")

        # 頸動脈
        neck = plt.Rectangle((3.8, 0.3), 1.5, 3.8,
                              facecolor="#7b241c", edgecolor="#e74c3c", lw=2)
        ax_asl.add_patch(neck)

        # 標記平面
        ax_asl.axhline(3.5, xmin=0.1, xmax=0.9,
                       color="#f39c12", lw=3, alpha=0.9)
        ax_asl.text(8.5, 3.7, "Labeling\nPlane", color="#f39c12",
                    fontsize=8, va="bottom")

        # RF 脈衝圖示
        ax_asl.annotate("RF Pulse\n(Inversion)", xy=(4.55, 3.5),
                        xytext=(1.2, 4.8),
                        arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.8),
                        color="#f39c12", fontsize=8)

        # 標記的紅血球（紫色 = 反轉磁矩）
        for yi in [3.1, 2.5, 1.9]:
            circ = Circle((4.55, yi), 0.28,
                          facecolor="#8e44ad", edgecolor="#9b59b6", alpha=0.9)
            ax_asl.add_patch(circ)

        # 流向箭頭
        ax_asl.annotate("", xy=(4.55, 7.0), xytext=(4.55, 4.2),
                        arrowprops=dict(arrowstyle="->", color="#3498db", lw=2.5))

        # PLD 等待標示
        ax_asl.annotate(f"PLD = {PLD_val:.1f}s",
                        xy=(4.55, 5.5), xytext=(6.5, 5.5),
                        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
                        color="#2ecc71", fontsize=9)

        # 腦部
        brain = Circle((4.55, 8.1), 1.6,
                        facecolor="#154360", edgecolor="#2e86c1", lw=2, alpha=0.75)
        ax_asl.add_patch(brain)
        ax_asl.text(4.55, 8.1, "Brain\nTissue", color="white",
                    fontsize=8, ha="center", va="center")

        ax_asl.text(4.55, 1.0, "Labeled Blood", color="#9b59b6",
                    fontsize=8, ha="center")
        ax_asl.set_title("ASL 標記流程概念圖", color="white", fontsize=9, pad=8)

        plt.tight_layout()
        st.pyplot(fig_asl)
        plt.close(fig_asl)

    # ── CBF vs ΔM 曲線 ──
    with col_cbf:
        st.markdown("#### 📊 CBF vs ASL 信號差值")

        cbf_range = np.arange(0, 121, 2, dtype=float)
        dM_arr = np.array([asl_delta_M(c, 1650, alpha_use, tau_val, PLD_val) * 100
                           for c in cbf_range])

        current_dM = asl_delta_M(CBF_val, 1650, alpha_use, tau_val, PLD_val) * 100

        fig_cbf, ax_cbf = dark_fig(1, 1, figsize=(6.5, 4))

        ax_cbf.plot(cbf_range, dM_arr, color="#2ecc71", lw=2.5)
        ax_cbf.fill_between(cbf_range, 0, dM_arr, alpha=0.15, color="#2ecc71")

        # 當前值
        ax_cbf.scatter([CBF_val], [current_dM], color="#f39c12", s=150, zorder=5)
        ax_cbf.axvline(CBF_val, color="#f39c12", ls="--", alpha=0.6, lw=1.5)

        # 臨床參考線
        ax_cbf.axvline(20, color="#e74c3c", ls=":", lw=1, alpha=0.8)
        ax_cbf.text(21, dM_arr.max() * 0.8, "缺血\n閾值", color="#e74c3c", fontsize=8)
        ax_cbf.axvline(60, color="#3498db", ls=":", lw=1, alpha=0.8)
        ax_cbf.text(61, dM_arr.max() * 0.55, "正常\n灰質", color="#3498db", fontsize=8)

        # 3% 與 5% 上限線
        ax_cbf.axhline(3.0, color="#f39c12", ls=":", lw=1, alpha=0.5)
        ax_cbf.text(100, 3.1, "~3% (90°RF)", color="#f39c12", fontsize=7)
        ax_cbf.axhline(5.0, color="#f39c12", ls=":", lw=1, alpha=0.5)
        ax_cbf.text(100, 5.1, "~5% (180°RF)", color="#f39c12", fontsize=7)

        dark_ax(ax_cbf,
                f"ASL Signal vs CBF ｜ PLD={PLD_val:.1f}s, τ={tau_val:.1f}s, α={alpha_use}",
                "CBF (mL/100g/min)", "ΔM/M₀ (%)")
        ax_cbf.text(CBF_val + 2, current_dM,
                    f"CBF={CBF_val}\nΔM={current_dM:.3f}%",
                    color="#f39c12", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cbf)
        plt.close(fig_cbf)

        # 臨床狀態判斷
        if CBF_val < 10:
            st.error("🔴 嚴重缺血 (CBF < 10)：梗塞核心")
        elif CBF_val < 20:
            st.warning("🟡 缺血 (CBF 10-20)：缺血半暗帶")
        elif CBF_val > 100:
            st.info("🔵 高灌流 (CBF > 100)：腫瘤/充血")
        else:
            st.success("🟢 正常範圍 (CBF 20-100)")

    # ── 相減原理示意 ──
    with col_sub:
        st.markdown("#### 🔄 相減成像原理 [PRIMARY - ASL講義 p.4]")

        rng = np.random.default_rng(17)
        noise = rng.normal(0, 0.008, (50, 50))
        base = np.ones((50, 50)) * 0.55 + noise

        dM_frac = current_dM / 100.0
        labeled = base * (1.0 - dM_frac)
        control = base.copy()
        diff    = (control - labeled) * 30     # 放大 30 倍顯示

        fig_sub, ax_s_arr = dark_fig(1, 3, figsize=(6, 2.8))

        titles_s = ["Label 影像\n(↓ 微小)", "Control 影像\n(不標記)", f"相減 ×30\n(CBF map)"]
        sigs_s   = [labeled, control, diff]
        cmaps_s  = ["gray",  "gray",  "hot"]
        vmins_s  = [0.45,    0.45,    0.0]
        vmaxs_s  = [0.65,    0.65,    0.08]

        for ax_s, sig, ttl, cmp, vmn, vmx in zip(
                ax_s_arr, sigs_s, titles_s, cmaps_s, vmins_s, vmaxs_s):
            ax_s.imshow(sig, cmap=cmp, vmin=vmn, vmax=vmx,
                        interpolation="bilinear")
            ax_s.set_title(ttl, color="white", fontsize=8)
            ax_s.axis("off")

        fig_sub.suptitle("有標記 − 無標記 = 灌流信號", color="white", fontsize=9)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig_sub)
        plt.close(fig_sub)

        st.metric("CBF", f"{CBF_val} mL/100g/min")
        st.metric("ΔM/M₀", f"{current_dM:.4f}%",
                  help="非常微小的信號差！需多次平均")

        st.markdown("""
        <div class="concept-card" style="padding:10px; margin-top:8px">
          <h4>🔧 ASL vs PWI [PRIMARY - ASL p.9]</h4>
          <p style="font-size:.78rem">
          <b>ASL 優勢</b>：不需顯影劑、可重複、兒童安全<br>
          <b>ASL 缺點</b>：SNR 低、需多次平均<br>
          <b>AT 敏感性</b>：極敏感（ATT 影響大）
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── MT 效應說明 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 😱 磁轉移效應（MT Effect）與校正 [PRIMARY - ASL講義 p.5]")

    col_mt1, col_mt2, col_mt3 = st.columns(3)
    with col_mt1:
        st.markdown("""
        <div class="key-card">
          <h4>❓ 意外發現</h4>
          <p>
          理論預測信號差 &lt;5%<br>
          實際測量卻達 <b>20~30%</b>？！<br><br>
          原因：<b>磁轉移效應 (MT)</b>
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col_mt2:
        st.markdown("""
        <div class="concept-card">
          <h4>🔬 MT 機制</h4>
          <p>
          組織中同時存在：<br>
          &bull; 自由水質子（觀察目標）<br>
          &bull; 束縛巨分子質子（蛋白質）<br><br>
          Off-resonance RF → 束縛質子 → 偶極交換 → 影響自由水信號
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col_mt3:
        st.markdown("""
        <div class="ok-card">
          <h4>✅ 解決方案</h4>
          <p>
          Label 和 Control 都給相同頻率的 off-resonance RF<br><br>
          但<b>位置不同</b>：<br>
          Label → 頸部上游<br>
          Control → 相同距離下游<br><br>
          → MT 效應相同 → 相減後消除
          </p>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────── 比較系統頁 ──────────────────────────
def page_compare():
    st.markdown("""
    <div class="main-header">
        <h1>📊 比較系統</h1>
        <p>DCE vs DSC vs ASL ｜ SWI vs DSC ｜ PWI vs DWI（急性中風）｜ TOF vs PC-MRA</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🔵 DCE vs DSC vs ASL",
        "⚫ SWI vs DSC",
        "🧠 PWI vs DWI（急性中風）",
        "🩸 TOF vs PC-MRA",
    ])

    # ═══════════════════ Tab 1：DCE vs DSC vs ASL ═══════════════════
    with tabs[0]:
        st.markdown("### 三種灌流方法全面比較")

        cs = st.columns(3)
        with cs[0]:
            Gd_c = st.slider("Gd 濃度 (mM)", 0.0, 5.0, 2.0, 0.5, key="gd_cmp")
        with cs[1]:
            CBF_c = st.slider("CBF (mL/100g/min)", 0, 100, 60, 10, key="cbf_cmp")
        with cs[2]:
            SNR_c = st.slider("雜訊強度（低=低雜訊）", 1, 8, 4, 1, key="snr_cmp")

        # 方法特性對照表
        st.markdown("#### 📋 方法特性對照表 [PRIMARY - ASL講義 p.9 + PWI講義 p.10]")
        st.markdown("""
        | 特性 | 💉 DCE | 🌀 DSC | 🔄 ASL |
        |------|--------|--------|--------|
        | 造影劑 | Gd（必要） | Gd（必要） | **無需** |
        | 主要效應 | T1 縮短 | T2* 縮短 | 自旋標記 |
        | 信號方向 | ↑ 變亮 | ↓ 變暗 | 差值 <5% |
        | 信號變化量 | 5-20% | **30-60%** | <5% |
        | 掃描時間 | 10+ 分鐘 | 1-2 分鐘 | 3-5 分鐘 |
        | BBB 影響 | **需 BBB 破壞** | BBB 完整最佳 | 不受影響 |
        | 腎功能限制 | **有（Gd禁忌）** | **有** | **無** |
        | 可重複性 | 中 | 低 | **高** |
        | SNR | 中 | 高 | **低（需多次平均）** |
        """)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 三列曲線視覺化
        st.markdown("#### 🖼️ 三種方法信號曲線對比")
        t = np.linspace(0, 80, 500)

        fig_3, axes_3 = dark_fig(1, 3, figsize=(14, 4))
        clrs_3 = ["#f39c12", "#3498db", "#2ecc71"]
        titles_3 = ["DCE (T1WI)\nSignal ↑ 增強",
                    "DSC (T2*WI)\nSignal ↓ 下降",
                    "ASL\nΔM/M₀（極微小差值）"]

        # DCE
        for ctype, clr in zip(["washout", "plateau", "persistent"],
                               ["#e74c3c", "#f39c12", "#2ecc71"]):
            axes_3[0].plot(t, dce_signal_curve(t, Gd_c, ctype),
                           color=clr, lw=2 if ctype == "washout" else 1.5,
                           ls="-" if ctype == "washout" else "--",
                           label=ctype.capitalize())
        axes_3[0].axhline(1.0, color="#555", ls=":", lw=0.8)
        axes_3[0].fill_between(t, 1.0, dce_signal_curve(t, Gd_c, "washout"),
                               where=dce_signal_curve(t, Gd_c, "washout") > 1.0,
                               alpha=0.1, color="#f39c12")

        # DSC
        for ts, clr, lb in zip(["normal", "ischemic"],
                                 ["#3498db", "#e74c3c"], ["Normal", "Ischemic"]):
            axes_3[1].plot(t, dsc_signal_curve(t, Gd_c, ts),
                           color=clr, lw=2.5 if ts == "normal" else 1.8,
                           ls="-" if ts == "normal" else "--",
                           label=lb)
        axes_3[1].fill_between(t, 1.0, dsc_signal_curve(t, Gd_c, "normal"),
                               where=dsc_signal_curve(t, Gd_c, "normal") < 1.0,
                               alpha=0.1, color="#3498db")

        # ASL
        rng_asl = np.random.default_rng(99)
        noise_asl = rng_asl.normal(0, 0.005 / max(SNR_c, 1), len(t))
        dM_c = asl_delta_M(CBF_c) * 100
        ctrl = np.ones_like(t) * 0.7 + noise_asl
        lbl  = ctrl - dM_c / 100.0 * 0.7
        axes_3[2].plot(t, ctrl, color="#aaa", lw=1.2, label="Control")
        axes_3[2].plot(t, lbl,  color="#2ecc71", lw=1.5, label="Labeled")
        axes_3[2].fill_between(t, lbl, ctrl, alpha=0.4, color="#2ecc71",
                               label=f"ΔM={dM_c:.3f}%")
        axes_3[2].set_ylim(0.62, 0.78)

        for ax_i, (ax, ttl, clr) in enumerate(zip(axes_3, titles_3, clrs_3)):
            dark_ax(ax, ttl, "Time (s)", "Signal (a.u.)", title_color=clr)
            ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

        plt.tight_layout(pad=0.8)
        st.pyplot(fig_3)
        plt.close(fig_3)

        # 臨床選擇指南
        st.markdown("#### 🏥 臨床選擇指南 [PRIMARY - PWI講義 p.10]")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="concept-card">
              <h4>💉 選 DCE 的情境</h4>
              <p>
              ✅ 乳房腫瘤評估（Ktrans）<br>
              ✅ 骨骼/關節灌流<br>
              ✅ BBB 破壞評估<br>
              ✅ 腫瘤分期/治療追蹤<br>
              ⚠️ 需較長掃描時間
              </p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="concept-card">
              <h4>🌀 選 DSC 的情境</h4>
              <p>
              ✅ <b>急性腦中風（首選）</b><br>
              ✅ 腦腫瘤 angiogenesis<br>
              ✅ 信號敏感度最高<br>
              ✅ 掃描時間短<br>
              ⚠️ BBB 破壞時有 T1/T2* 混淆
              </p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="ok-card">
              <h4>🔄 選 ASL 的情境</h4>
              <p>
              ✅ <b>腎功能不佳患者</b><br>
              ✅ 兒童/重複檢查<br>
              ✅ 縱向追蹤研究<br>
              ✅ BBB 評估干擾少<br>
              ⚠️ SNR 低、需多次平均
              </p>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════ Tab 2：SWI vs DSC ═══════════════════
    with tabs[1]:
        st.markdown("### SWI vs DSC：磁化率效應的兩種應用")

        col_s1, col_s2 = st.columns([1, 1])

        with col_s1:
            st.markdown("""
            <div class="concept-card">
              <h4>⚫ SWI 原理 [PRIMARY - SWI講義 p.3-4]</h4>
              <p>
              <b>內源性（Endogenous）對比劑</b> = 去氧血紅素 (Deoxy-Hb)<br>
              &bull; Oxy-Hb：逆磁性（diamagnetic），不影響磁場<br>
              &bull; Deoxy-Hb：<b>順磁性（paramagnetic）</b>，干擾磁場均勻度<br>
              &bull; 靜脈血（含 Deoxy-Hb）→ 局部 T2* 縮短 → 信號損失<br><br>
              <b>SWI 後處理</b>（講義 p.5-7）：<br>
              ① GRE 長 TE 掃描 → ② 高通濾波相位影像 (HPF)<br>
              → ③ 建立相位遮罩（負相位）→ ④ 乘以振幅影像（×4次）<br>
              → ⑤ <b>mIP（最小強度投影）</b>展示靜脈
              </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="key-card">
              <h4>🌀 DSC 的磁化率效應</h4>
              <p>
              <b>外源性（Exogenous）</b>：注射 Gd-DTPA<br>
              &bull; 動態追蹤 First-pass 通過<br>
              &bull; 主要反映<b>微血管灌流</b>（CBV/CBF/MTT）<br>
              &bull; 信號變化 30-60%（遠大於 SWI 的靜態效應）
              </p>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("#### 🩸 氧合狀態與 T2* 的關係")

            SO2 = st.slider("💧 靜脈血氧飽和度 SO₂ (%)", 40, 100, 65, 5,
                            help="正常靜脈血 SO₂ ≈ 60-70%")

            deoxy_frac = (100 - SO2) / 100.0
            T2s_vein = 60.0 / (1.0 + deoxy_frac * 7.5)
            TE_range = np.linspace(0, 100, 200)

            fig_swi, ax_swi = dark_fig(1, 1, figsize=(6, 3.8))

            ax_swi.plot(TE_range, np.exp(-TE_range / 250),
                        color="#e74c3c", lw=2.2, label="Artery (Oxy-Hb, long T2*)")
            ax_swi.plot(TE_range, np.exp(-TE_range / 60),
                        color="#95a5a6", lw=1.8, ls="--", label="Tissue (T2*≈60ms)")
            ax_swi.plot(TE_range, np.exp(-TE_range / T2s_vein),
                        color="#3498db", lw=2.5,
                        label=f"Vein (SO₂={SO2}%, T2*≈{T2s_vein:.0f}ms)")

            te_swi = 25
            ax_swi.axvline(te_swi, color="#f39c12", ls=":", lw=2, alpha=0.9)
            s_t = np.exp(-te_swi / 60)
            s_v = np.exp(-te_swi / T2s_vein)
            contrast_pct = (s_t - s_v) / (s_t + 1e-6) * 100
            ax_swi.annotate(f"TE={te_swi}ms\nContrast={contrast_pct:.0f}%",
                            xy=(te_swi, s_v), xytext=(te_swi + 12, s_v + 0.18),
                            arrowprops=dict(arrowstyle="->", color="#f39c12"),
                            color="#f39c12", fontsize=8)

            dark_ax(ax_swi,
                    f"T2* Decay ｜ SWI 靜脈對比（SO₂={SO2}%）",
                    "TE (ms)", "Relative Signal")
            ax_swi.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig_swi)
            plt.close(fig_swi)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("靜脈 T2*", f"{T2s_vein:.0f} ms",
                          delta=f"{T2s_vein-60:.0f} ms vs tissue")
            with col_m2:
                st.metric("SWI 靜脈對比 (TE=25ms)", f"{contrast_pct:.1f}%")

        # SWI 後處理流程圖（動畫式卡片）
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🔄 SWI 後處理步驟 [PRIMARY - SWI講義 p.5-7]")
        steps_swi = [
            ("①", "GRE 長TE 掃描", "取得振幅 + 相位影像", "#3498db"),
            ("②", "HPF 高通濾波", "去除背景低頻相位", "#9b59b6"),
            ("③", "相位遮罩", "負相位→mask (0~1)", "#e74c3c"),
            ("④", "mask × 4 次", "乘以振幅影像強化靜脈", "#f39c12"),
            ("⑤", "mIP 投影", "5-10 切面最小強度投影", "#2ecc71"),
        ]
        step_cols = st.columns(5)
        for col_st, (num, ttl, desc, clr) in zip(step_cols, steps_swi):
            with col_st:
                st.markdown(f"""
                <div style="background:#1a2233; border:2px solid {clr};
                     border-radius:10px; padding:12px; text-align:center;">
                  <div style="font-size:1.4rem; color:{clr}; font-weight:bold;">{num}</div>
                  <div style="color:white; font-size:.85rem; font-weight:bold;">{ttl}</div>
                  <div style="color:#aaa; font-size:.72rem; margin-top:4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════ Tab 3：PWI vs DWI ═══════════════════
    with tabs[2]:
        st.markdown("### PWI vs DWI：急性腦中風黃金組合 [PRIMARY - PWI講義 p.9-10]")

        col_stroke_ctrl, col_stroke_img = st.columns([1, 2])

        with col_stroke_ctrl:
            hours = st.slider("⏰ 發病後時間 (小時)", 0.5, 24.0, 3.0, 0.5)

            # 根據時間設定模擬參數
            if hours < 1.5:
                adc_core, cbf_core = 0.50e-3, 12
            elif hours < 6:
                adc_core, cbf_core = 0.38e-3, 10
            elif hours < 12:
                adc_core, cbf_core = 0.42e-3, 18
            else:
                adc_core, cbf_core = 0.62e-3, 28

            mismatch = (hours < 9) and (cbf_core < 25)

            st.markdown("""
            <div class="concept-card">
              <h4>🧠 DWI/PWI 不吻合 (Mismatch)</h4>
              <p>
              <b>DWI</b>：偵測核心梗塞（細胞毒性水腫）<br>
              &nbsp;→ 急性缺血 ADC ↓ → DWI 亮<br><br>
              <b>PWI</b>：偵測低灌流整體區域<br>
              &nbsp;→ rMTT ↑ rCBF ↓<br><br>
              <b>PWI 異常 &gt; DWI 異常</b><br>
              &nbsp;= 可救的<b>缺血半暗帶</b>！<br>
              &nbsp;→ 溶栓治療決策依據
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric("核心區 ADC", f"{adc_core*1000:.2f} ×10⁻³ mm²/s",
                      delta=f"{(adc_core-0.75e-3)*1000:.2f} vs normal")
            st.metric("核心區 rCBF", f"{cbf_core} mL/100g/min",
                      delta=f"{cbf_core-60}")

            if mismatch:
                st.success(f"✅ {hours:.1f}h：DWI/PWI 不吻合存在\n→ 可能有可救半暗帶")
            else:
                st.warning(f"⚠️ {hours:.1f}h：不吻合縮小或消失")

        with col_stroke_img:
            # 建立 4 張地圖
            ph = make_brain_phantom(80, lesion=True)
            size = 80

            # DWI map
            dwi_map = phantom_dwi(ph, 1000, adc_core)

            # ADC map
            adc_map_img = np.zeros((size, size))
            for lab, adc_v in {0: 0, 1: 3.0e-3, 2: 0.78e-3, 3: 0.72e-3, 4: adc_core}.items():
                adc_map_img[ph == lab] = adc_v * 1000  # ×10⁻³ mm²/s

            # CBF map（半暗帶比 DWI 核心更大）
            cbf_stroke = np.zeros((size, size))
            yy, xx = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
            lx, ly = size // 5, -size // 10
            dist_lesion = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)

            cbf_stroke[ph == 1] = 5
            cbf_stroke[ph == 2] = 72
            cbf_stroke[ph == 3] = 40
            cbf_stroke[ph == 4] = cbf_core   # Core
            # 半暗帶（略大於核心）
            penumbra_mask = (dist_lesion < size // 4) & (ph != 0) & (ph != 4)
            cbf_stroke[penumbra_mask] = 35

            # Mismatch 視覺圖
            mismatch_vis = np.zeros((size, size))
            mismatch_vis[ph == 4] = 0.3
            if mismatch:
                mismatch_vis[penumbra_mask] = 0.8
            mismatch_vis[ph == 2] = 0.05
            mismatch_vis[ph == 3] = 0.05

            fig_stroke, ax_st = dark_fig(1, 4, figsize=(13, 3.5))
            maps_st = [
                (dwi_map, "DWI (b=1000)\n缺血核心↑亮", "gray", 0, 0.7),
                (adc_map_img, "ADC Map (×10⁻³)\n核心↓暗",  "gray", 0, 1.5),
                (cbf_stroke,  "rCBF (mL/100g/min)\n缺血→低灌流", "hot",  0, 80),
                (mismatch_vis, "Mismatch 圖\n橙=可救半暗帶", "hot", 0, 1),
            ]
            for ax_i_s, (data_s, ttl_s, cmp_s, mn_s, mx_s) in zip(ax_st, maps_st):
                im_s = ax_i_s.imshow(data_s, cmap=cmp_s, vmin=mn_s, vmax=mx_s,
                                     interpolation="bilinear")
                ax_i_s.set_title(ttl_s, color="white", fontsize=8)
                ax_i_s.axis("off")
                fig_stroke.colorbar(im_s, ax=ax_i_s, fraction=0.046).ax.tick_params(
                    colors="white", labelsize=7)

            fig_stroke.suptitle(f"急性中風模擬 ｜ 發病後 {hours:.1f} 小時",
                                color="white", fontsize=10)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_stroke)
            plt.close(fig_stroke)

        # ADC vs 時間演變曲線
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📈 缺血核心 ADC 隨時間演變 [PRIMARY - DWI講義 p.8]")

        t_hr = np.linspace(0.5, 24, 200)
        # ADC 先降（急性期）後假正常化再升（慢性期）[DWI講義 知識]
        adc_time = np.where(
            t_hr < 8,
            0.75 - 0.35 * (1 - np.exp(-t_hr / 3)),
            0.40 + 0.30 * (1 - np.exp(-0.08 * (t_hr - 8)))
        )
        cbf_time = 12 + 28 * (1 - np.exp(-0.12 * t_hr))  # 側枝循環逐漸建立

        fig_time, (ax_t1, ax_t2) = dark_fig(1, 2, figsize=(12, 3.5))

        ax_t1.plot(t_hr, adc_time, color="#3498db", lw=2.5)
        ax_t1.axhline(0.75, color="#aaa", lw=1, ls=":", label="Normal ADC ~0.75")
        ax_t1.axhline(0.45, color="#e74c3c", lw=1, ls=":", label="Ischemic threshold")
        ax_t1.scatter([hours], [np.interp(hours, t_hr, adc_time)],
                     color="#f39c12", s=150, zorder=5, label=f"Now ({hours:.1f}h)")
        ax_t1.axvline(hours, color="#f39c12", ls="--", alpha=0.5)
        dark_ax(ax_t1, "缺血核心 ADC 演變",
                "Time (hours)", "ADC (×10⁻³ mm²/s)")
        ax_t1.legend(fontsize=8, framealpha=0.2, labelcolor="white")

        ax_t2.plot(t_hr, cbf_time, color="#e74c3c", lw=2.5)
        ax_t2.axhline(20, color="#f39c12", lw=1, ls=":", label="Ischemic threshold (20)")
        ax_t2.scatter([hours], [np.interp(hours, t_hr, cbf_time)],
                     color="#f39c12", s=150, zorder=5, label=f"Now ({hours:.1f}h)")
        ax_t2.axvline(hours, color="#f39c12", ls="--", alpha=0.5)
        ax_t2.axvline(4.5, color="#2ecc71", ls=":", lw=1.5)
        ax_t2.text(4.6, 35, "溶栓\n時窗\n~4.5h", color="#2ecc71", fontsize=7)
        dark_ax(ax_t2, "缺血核心 rCBF 演變",
                "Time (hours)", "rCBF (mL/100g/min)")
        ax_t2.legend(fontsize=8, framealpha=0.2, labelcolor="white")

        plt.tight_layout()
        st.pyplot(fig_time)
        plt.close(fig_time)

    # ═══════════════════ Tab 4：TOF vs PC-MRA ═══════════════════
    with tabs[3]:
        st.markdown("### TOF-MRA vs PC-MRA：非侵入性血管造影比較")

        col_tof_ctrl, col_pc_ctrl = st.columns(2)

        with col_tof_ctrl:
            st.markdown("#### ⏱️ TOF-MRA 互動")
            flow_v = st.slider("🩸 血流速度 (cm/s)", 0, 100, 40, 5, key="tof_v")
            TR_tof = st.slider("TR (ms)", 10, 60, 30, 5, key="tof_tr")
            fa_tof = st.slider("Flip Angle (°)", 20, 70, 50, 5, key="tof_fa")

            S_tis, S_bld, fre_pct = tof_fre(flow_v, TR_tof, fa_tof)

            st.metric("靜止組織信號", f"{S_tis:.3f}", help="飽和→低")
            st.metric("流入血液信號", f"{S_bld:.3f}",
                      delta=f"FRE: +{fre_pct:.0f}%",
                      delta_color="normal")
            st.markdown("""
            <div class="concept-card" style="padding:10px">
              <h4>⏱️ TOF 原理 [PRIMARY - TOF講義 p.3]</h4>
              <p style="font-size:.8rem">
              短 TR → 靜止組織飽和（信號低）<br>
              <b>流入血液未激發 → 信號強（FRE）</b><br>
              影響 FRE：TR、翻轉角、流速、切面厚度<br>
              MIP 重建顯示血管結構
              </p>
            </div>
            """, unsafe_allow_html=True)

        with col_pc_ctrl:
            st.markdown("#### 📐 PC-MRA 互動")
            vel_pc = st.slider("🩸 流速 (cm/s)", -120, 120, 50, 5, key="pc_v")
            venc_pc = st.slider("Venc (cm/s)", 20, 150, 80, 10, key="pc_venc",
                                 help="調整 Venc 應稍大於預期最大流速")

            phi, phi_deg, aliased = pc_phase(vel_pc, venc_pc)

            st.metric("相位角", f"{phi_deg:.1f}°",
                      delta="⚠️ Phase Aliasing!" if aliased else "✅ 正常",
                      delta_color="inverse" if aliased else "normal")
            if aliased:
                st.error(f"⚠️ 流速({vel_pc}) > Venc({venc_pc}) cm/s → Phase Aliasing！")
            st.markdown("""
            <div class="key-card" style="padding:10px">
              <h4>📐 PC 原理 [PRIMARY - PC講義 p.3]</h4>
              <p style="font-size:.8rem">
              <b>相角與流速成正比</b>：Φ = γ·Gx·Vx·t²<br>
              Bipolar gradient → 只有運動的質子累積相角<br>
              流速 > Venc → 相位混疊（Phase Aliasing）
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_tof_fig, col_pc_fig = st.columns(2)

        # ── TOF FRE 曲線 ──
        with col_tof_fig:
            st.markdown("#### 📈 TOF：FRE vs 流速")

            v_range = np.linspace(0, 100, 200)
            S_t_arr = np.zeros(200)
            S_b_arr = np.zeros(200)
            fre_arr = np.zeros(200)
            for i, v in enumerate(v_range):
                S_t_arr[i], S_b_arr[i], fre_arr[i] = tof_fre(v, TR_tof, fa_tof)

            fig_tof, ax_tof = dark_fig(1, 1, figsize=(6, 4))

            ax_tof.plot(v_range, S_b_arr, color="#e74c3c", lw=2.5, label="Blood Signal")
            ax_tof.axhline(S_t_arr[0], color="#aaa", lw=1.5, ls="--",
                           label=f"Tissue (static) = {S_t_arr[0]:.3f}")
            ax_tof.fill_between(v_range, S_t_arr[0], S_b_arr,
                                where=S_b_arr > S_t_arr[0],
                                alpha=0.2, color="#e74c3c", label="FRE Contrast")

            _, S_b_now, fre_now = tof_fre(flow_v, TR_tof, fa_tof)
            ax_tof.scatter([flow_v], [S_b_now], color="#f39c12", s=150, zorder=5)
            ax_tof.text(flow_v + 1, S_b_now,
                        f"v={flow_v}\nFRE={fre_now:.0f}%",
                        color="#f39c12", fontsize=8)

            dark_ax(ax_tof, f"TOF FRE ｜ TR={TR_tof}ms, α={fa_tof}°",
                    "Flow Velocity (cm/s)", "Signal (normalized)")
            ax_tof.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            plt.tight_layout()
            st.pyplot(fig_tof)
            plt.close(fig_tof)

        # ── PC 相位曲線 ──
        with col_pc_fig:
            st.markdown("#### 📐 PC-MRA：相位 vs 流速")

            v_pc_range = np.linspace(-120, 120, 400)
            phi_arr_deg = np.array([pc_phase(v, venc_pc)[1] for v in v_pc_range])

            fig_pc, ax_pc = dark_fig(1, 1, figsize=(6, 4))

            ax_pc.plot(v_pc_range, phi_arr_deg, color="#2ecc71", lw=2.5,
                       label="Measured Phase (wrapped)")

            # 理想線性（在 Venc 範圍內）
            ideal = np.clip(np.degrees(np.pi * v_pc_range / venc_pc), -180, 180)
            ax_pc.plot(v_pc_range, ideal, color="#3498db", lw=1.5, ls="--",
                       label="Ideal linear")

            ax_pc.axvline(venc_pc, color="#e74c3c", ls=":", lw=2, alpha=0.9)
            ax_pc.axvline(-venc_pc, color="#e74c3c", ls=":", lw=2, alpha=0.9)
            ax_pc.axhline(180, color="#e74c3c", ls=":", alpha=0.4)
            ax_pc.axhline(-180, color="#e74c3c", ls=":", alpha=0.4)

            # Aliasing 紅色區
            ax_pc.fill_betweenx([-200, 200], venc_pc, 120,
                                alpha=0.1, color="#e74c3c")
            ax_pc.fill_betweenx([-200, 200], -120, -venc_pc,
                                alpha=0.1, color="#e74c3c")
            ax_pc.text(venc_pc + 3, -170, f"Venc={venc_pc}",
                       color="#e74c3c", fontsize=8)

            # 當前點
            ax_pc.scatter([vel_pc], [phi_deg], color="#f39c12", s=150, zorder=5)
            ax_pc.text(vel_pc + 3, phi_deg, f"v={vel_pc}\nφ={phi_deg:.0f}°",
                       color="#f39c12", fontsize=8)

            dark_ax(ax_pc, f"PC-MRA Phase vs Velocity ｜ Venc={venc_pc} cm/s",
                    "Velocity (cm/s)", "Phase Angle (degrees)")
            ax_pc.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            ax_pc.set_xlim(-120, 120)
            ax_pc.set_ylim(-200, 200)
            plt.tight_layout()
            st.pyplot(fig_pc)
            plt.close(fig_pc)

        # 比較總表
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📋 TOF vs PC-MRA 比較 [PRIMARY - TOF p.10 + PC p.10]")

        col_tof_sum, col_pc_sum = st.columns(2)
        with col_tof_sum:
            st.markdown("""
            <div class="concept-card">
              <h4>⏱️ TOF-MRA</h4>
              <p>
              ✅ 不需顯影劑<br>
              ✅ 不需 subtraction<br>
              ✅ 快速掃描<br>
              ✅ 2D / 3D 都可行<br><br>
              ⚠️ Intra-voxel dephasing（彎曲血管）<br>
              ⚠️ 飽和效應（長血管下游）<br>
              ⚠️ 血管走向影響對比
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_pc_sum:
            st.markdown("""
            <div class="concept-card">
              <h4>📐 PC-MRA</h4>
              <p>
              ✅ 可<b>定量</b>流速與血流方向<br>
              ✅ 配合 ECG gating → 心動週期<br>
              ✅ CSF 流動測量<br>
              ✅ 主動脈剝離評估<br><br>
              ⚠️ 必須設定 Venc<br>
              ⚠️ 流速 > Venc → Phase Aliasing<br>
              ⚠️ 掃描時間較長（三方向梯度）
              </p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ⑧ 主程式導覽
# ═══════════════════════════════════════════════════════════════

def main():
    """主程式：側欄導覽 + 頁面路由"""

    with st.sidebar:
        # 系統標誌
        st.markdown("""
        <div style="text-align:center; padding:16px 0 10px;">
          <div style="font-size:2.8rem;">🧠</div>
          <div style="font-size:1.05rem; font-weight:bold; color:#5dade2;">
            MRI 灌流影像<br>互動教學系統
          </div>
          <div style="font-size:.72rem; color:#85c1e9; margin-top:6px;">
            蔡炳煇副教授 課程<br>中山醫學大學 醫放系
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "📚 選擇教學章節",
            [
                "🏠 首頁：灌流基礎",
                "💉 DCE 動態對比增強",
                "🌀 DSC 動態磁化率對比",
                "🔄 ASL 動脈自旋標記",
                "📊 比較系統",
            ],
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:.72rem; color:#85c1e9; padding:6px 0;">
          📖 <b>知識來源</b><br>
          <span style="color:#aaa;">
          [PRIMARY] 08_PWI.pdf<br>
          [PRIMARY] 09_ASL.pdf<br>
          [PRIMARY] 03_SWI.pdf<br>
          [PRIMARY] 04_DWI.pdf<br>
          [PRIMARY] 01_TOF_MRA.pdf<br>
          [PRIMARY] 02_PC_MRA.pdf
          </span>
        </div>
        <div style="font-size:.68rem; color:#7fb3d3; margin-top:8px; text-align:center;">
          ⚠️ 物理模型為教學用簡化近似
        </div>
        """, unsafe_allow_html=True)

    # 頁面路由
    if "首頁" in page:
        page_home()
    elif "DCE" in page:
        page_dce()
    elif "DSC" in page:
        page_dsc()
    elif "ASL" in page:
        page_asl()
    elif "比較" in page:
        page_compare()


if __name__ == "__main__":
    main()
