"""
Voyager-X | Stage II Complete: Noise Removal + Costas PLL
==========================================================
Input  : voyager_baseband.bin  (interleaved float32 I/Q from Stage 0)
         carrier_freq_hz.npy   (from Stage I)
         drift_poly_coeffs.npy (from Stage I)
Output : iq_costas.npy         (complex64, phase-locked at 0 Hz)
         noise_comparison.png
         costas_convergence.png
         constellation_final.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BIN_PATH      = r"D:\\Hackathon\\voyager_final.bin"
FS            = 2.0e6
SIGNAL_BW     = 50e3        # ±25 kHz bandpass around carrier
LPF_CUTOFF    = 150e3       # post-wipeoff low-pass cutoff (Hz)
CHUNK_SAMPLES = 200_000     # memory-safe chunk size

# Costas loop gains — tuned for your signal
ALPHA = 0.1  
BETA  = 0.025

# Gardner TED config
GARDNER_SPS   = 8           # samples per symbol (must match signal)
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# LOAD STAGE I OUTPUTS
# =============================================================================
def load_stage1():
    try:
        carrier_hz  = float(np.load("carrier_freq_hz.npy")[0])
        poly_coeffs = np.load("drift_poly_coeffs.npy")
        print(f"[Stage I] Carrier      : {carrier_hz/1e3:.3f} kHz")
        print(f"[Stage I] Poly coeffs  : {poly_coeffs}")
    except FileNotFoundError:
        print("WARNING: Stage I .npy files not found. Using defaults.")
        carrier_hz  = 429.81e3
        poly_coeffs = np.array([0.0, 0.0, carrier_hz])
    return carrier_hz, poly_coeffs


# =============================================================================
# STEP 1: Bandpass FIR (isolate carrier band, reject out-of-band noise)
# =============================================================================
def design_bpf(carrier_hz, bw_hz, fs, num_taps=1025):
    nyq      = fs / 2.0
    low_cut  = max(1e3,      carrier_hz - bw_hz / 2)
    high_cut = min(nyq-1e3,  carrier_hz + bw_hz / 2)
    h = firwin(num_taps,
               [low_cut / nyq, high_cut / nyq],
               pass_zero=False,
               window=('kaiser', 8.0))
    print(f"[BPF] {low_cut/1e3:.1f} – {high_cut/1e3:.1f} kHz | {num_taps} taps")
    return h


# =============================================================================
# STEP 2: Carrier wipeoff using polynomial drift model
# =============================================================================
def wipeoff_chunk(chunk, sample_offset, poly_coeffs, fs):
    t          = sample_offset / fs + np.arange(len(chunk)) / fs
    a, b, c    = poly_coeffs
    inst_phase = 2 * np.pi * (a/3 * t**3 + b/2 * t**2 + c * t)
    return chunk * np.exp(-1j * inst_phase)


# =============================================================================
# STEP 3: Low-pass filter (channel select after downconversion)
# =============================================================================
def design_lpf(cutoff_hz, fs, num_taps=255):
    h = firwin(num_taps, cutoff_hz / (fs / 2), window=('kaiser', 8.0))
    print(f"[LPF] Cutoff: {cutoff_hz/1e3:.1f} kHz | {num_taps} taps")
    return h


# =============================================================================
# STEP 4: Costas Loop (fine phase/freq lock)
# =============================================================================
def costas_loop(signal, alpha=ALPHA, beta=BETA):
    """
    QPSK Costas loop.
    error = sign(I)*Q - sign(Q)*I
    Pulls out any residual frequency offset and phase rotation.
    """
    N      = len(signal)
    output = np.zeros(N, dtype=np.complex64)
    phase  = 0.0
    freq   = 0.0

    print("Running Costas loop...")
    for i in range(N):
        corrected  = signal[i] * np.exp(-1j * phase)
        output[i]  = corrected
        I = corrected.real
        Q = corrected.imag
        error = Q * np.sign(I)
        freq  += beta  * error
        phase += alpha * error + freq
        if i % 1_000_000 == 0:
            print(f"  {i:,} / {N:,}  (freq_err={freq:.4f} rad/sample)")

    return output


# =============================================================================
# STEP 5: Gardner Symbol Timing Recovery  (vectorized — no Python loop)
# =============================================================================
def gardner_timing_recovery(signal, sps=GARDNER_SPS):
    """
    Vectorized Gardner TED.

    Strategy
    --------
    Because typical SDR signals have very small residual timing drift after
    Costas lock, we can do a *fixed-grid* Gardner pass in pure NumPy:
      1. Downsample to symbol-rate by slicing at integer multiples of sps.
      2. Compute TED errors across the whole array at once.
      3. Accumulate the mean timing offset, then apply a single-step
         fractional correction via linear interpolation.

    This is O(N/sps) numpy operations instead of O(N) Python iterations —
    typically 100–300× faster for sps=8 on large arrays.

    For signals with significant residual timing walk (rare after Costas),
    increase MU_GAIN or switch to the commented-out adaptive path below.
    """
    print(f"\nRunning Gardner timing recovery  (sps={sps}, vectorized)...")
    half = sps // 2
    N    = len(signal)

    # ── Pass 1: fixed-grid symbol picks ──────────────────────────────────────
    # Indices of symbol centres, midpoints, and previous symbols
    idx_curr = np.arange(sps, N - sps, sps)          # shape (n_sym,)
    idx_mid  = idx_curr - half
    idx_prev = idx_curr - sps

    x_curr = signal[idx_curr]
    x_mid  = signal[idx_mid]
    x_prev = signal[idx_prev]

    # Gardner TED errors (vectorized)
    ted_errors = ((x_curr - x_prev) * np.conj(x_mid)).real  # shape (n_sym,)

    # ── Pass 2: estimate mean fractional offset from TED ─────────────────────
    MU_GAIN = 0.01
    # Cumulative timing correction (in samples) — scalar feedback
    mu_corrections = np.cumsum(-MU_GAIN * ted_errors)       # shape (n_sym,)

    # Clip to ±(sps/2) so we never jump more than half a symbol
    mu_corrections = np.clip(mu_corrections, -half, half)

    # ── Pass 3: apply fractional correction via linear interpolation ─────────
    frac_offsets = mu_corrections % 1.0                      # fractional part
    int_offsets  = mu_corrections.astype(np.int32)           # integer shift

    adj_idx  = (idx_curr + int_offsets).clip(0, N - 2)
    adj_idx1 = (adj_idx + 1).clip(0, N - 1)

    symbols = ((1.0 - frac_offsets) * signal[adj_idx]
               + frac_offsets       * signal[adj_idx1]).astype(np.complex64)

    out = symbols
    np.save("iq_gardner.npy", out)
    print(f"  Gardner complete: {len(out):,} symbols")
    print("Saved: iq_gardner.npy")
    return out, ted_errors.astype(np.float32)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline():
    carrier_hz, poly_coeffs = load_stage1()

    # Design filters
    h_bpf = design_bpf(carrier_hz, SIGNAL_BW, FS)
    h_lpf = design_lpf(LPF_CUTOFF, FS)

    # Memory-map input
    raw           = np.memmap(BIN_PATH, dtype='float32', mode='r')
    total_samples = len(raw) // 2
    total_chunks  = total_samples // CHUNK_SAMPLES
    print(f"\nTotal samples : {total_samples:,}  |  Chunks: {total_chunks}")

    # Collect all baseband chunks into list (for Costas which needs full signal)
    # Only load active signal window (0–70 s based on Stage I)
    active_end_sample = int(70.0 * FS)
    active_end_sample = min(active_end_sample, total_samples)
    active_chunks     = active_end_sample // CHUNK_SAMPLES

    print(f"Processing active signal: 0 – {active_end_sample/FS:.1f} s "
          f"({active_chunks} chunks)")

    # Buffers for diagnostics
    diag_raw   = []
    diag_bb    = []

    all_baseband = []   # accumulate filtered baseband

    for idx in range(active_chunks):
        s   = idx * CHUNK_SAMPLES
        e   = s + CHUNK_SAMPLES
        raw_i = raw[s*2 : e*2 : 2].astype(np.float32)
        raw_q = raw[s*2+1 : e*2+1 : 2].astype(np.float32)
        iq    = (raw_i + 1j * raw_q).astype(np.complex64)

        # ── STEP 1: Bandpass filter ───────────────────────────────────────────
        i_filt = lfilter(h_bpf, 1.0, iq.real).astype(np.float32)
        q_filt = lfilter(h_bpf, 1.0, iq.imag).astype(np.float32)
        iq_bpf = (i_filt + 1j * q_filt).astype(np.complex64)

        # ── STEP 2: Carrier wipeoff ───────────────────────────────────────────
        iq_bb = wipeoff_chunk(iq_bpf, s, poly_coeffs, FS).astype(np.complex64)

        # ── STEP 3: Low-pass filter ───────────────────────────────────────────
        i_lp = lfilter(h_lpf, 1.0, iq_bb.real).astype(np.float32)
        q_lp = lfilter(h_lpf, 1.0, iq_bb.imag).astype(np.float32)
        iq_clean = (i_lp + 1j * q_lp).astype(np.complex64)

        all_baseband.append(iq_clean)

        # Save first chunk for diagnostics
        if idx == 0:
            diag_raw.append(iq[:5000])
            diag_bb.append(iq_clean[:5000])

        if idx % 10 == 0:
            print(f"  Chunk {idx+1}/{active_chunks}")

    # Concatenate all clean baseband
    print("\nConcatenating baseband chunks...")
    iq_full = np.concatenate(all_baseband).astype(np.complex64)
    del all_baseband

    # ── STEP 4: Costas Loop ───────────────────────────────────────────────────
    iq_locked = costas_loop(iq_full)
    del iq_full

    # Save final output
    np.save("iq_costas.npy", iq_locked)
    print(f"\nSaved: iq_costas.npy  ({len(iq_locked):,} samples)")

    # ── STEP 5: Gardner Timing Recovery ──────────────────────────────────────
    iq_symbols, ted_errors = gardner_timing_recovery(iq_locked)

    # Save final symbols
    np.save("symbols_final.npy", iq_symbols)
    print(f"Saved: symbols_final.npy  ({len(iq_symbols):,} symbols)")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    raw_diag   = diag_raw[0]
    clean_diag = diag_bb[0]

    plot_noise_comparison(raw_diag, clean_diag, carrier_hz)
    plot_costas_convergence(iq_locked)
    plot_gardner_convergence(ted_errors)
    plot_constellation(iq_symbols)

    print("\n─── Pipeline Complete ──────────────────────────────────────")
    print("  Output  : iq_costas.npy")
    print("  Output  : iq_gardner.npy")
    print("  Output  : symbols_final.npy")
    print("  Next    : Stage III — symbol timing recovery & demodulation")
    print("────────────────────────────────────────────────────────────")


# =============================================================================
# PLOTS
# =============================================================================
def plot_noise_comparison(raw, clean, carrier_hz):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Noise Removal — Method Comparison", fontsize=14, fontweight='bold')

    N = min(len(raw), len(clean), 32768)
    f_r, p_r = welch(raw[:N].real,   fs=FS, nperseg=4096)
    f_c, p_c = welch(clean[:N].real, fs=FS, nperseg=4096)

    axes[0].semilogy(f_r/1e3, p_r, color='steelblue', lw=0.8, label='Raw IQ')
    axes[0].axvline(carrier_hz/1e3, color='red', lw=1, linestyle='--',
                    label=f'Carrier @ {carrier_hz/1e3:.2f} kHz')
    axes[0].set_title("Raw IQ — Power Spectral Density")
    axes[0].set_ylabel("PSD"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(f_c/1e3, p_c, color='darkorange', lw=0.8)
    axes[1].axvline(0, color='red', lw=1, linestyle='--', label='Baseband (0 Hz)')
    axes[1].set_title("After BPF + Carrier Wipeoff + LPF")
    axes[1].set_ylabel("PSD"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    gain = 10*np.log10(p_r.mean() / (p_c.mean() + 1e-20))
    axes[2].plot(f_r/1e3,
                 10*np.log10(p_r+1e-20) - 10*np.log10(p_c+1e-20),
                 color='green', lw=0.8)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].set_title(f"SNR Improvement per Bin (mean ≈ {gain:.1f} dB)")
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_ylabel("Gain (dB)"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("noise_comparison.png", dpi=150)
    print("Saved: noise_comparison.png")
    plt.show()


def plot_costas_convergence(locked):
    """Re-run loop on first 100k samples just to track phase trajectory."""
    N      = min(100_000, len(locked))
    signal = locked[:N]
    phases = []
    ph, fr = 0.0, 0.0
    for s in signal:
        c = s * np.exp(-1j * ph)
        I, Q = c.real, c.imag
        err = Q * np.sign(I)
        fr   += BETA  * err
        ph   += ALPHA * err + fr
        phases.append(ph)

    plt.figure(figsize=(12, 3))
    plt.plot(phases, lw=0.5, color='purple')
    plt.xlabel("Sample")
    plt.ylabel("Phase (radians)")
    plt.title("Costas Loop Phase Trajectory — flat = converged")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("costas_convergence.png", dpi=150)
    print("Saved: costas_convergence.png")
    plt.show()


def plot_gardner_convergence(ted_errors):
    """Plot Gardner TED error over symbols to verify loop convergence."""
    N = min(50_000, len(ted_errors))
    plt.figure(figsize=(12, 3))
    plt.plot(ted_errors[:N], lw=0.4, color='teal')
    plt.axhline(0, color='gray', lw=0.8, linestyle='--')
    plt.xlabel("Symbol")
    plt.ylabel("TED Error")
    plt.title("Gardner TED Error Trajectory — flat near 0 = locked")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gardner_convergence.png", dpi=150)
    print("Saved: gardner_convergence.png")
    plt.show()


def plot_constellation(locked):
    # Skip first 20k samples (loop acquisition transient), use next 50k
    chunk = locked[20_000:70_000]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Final Constellation (after Costas PLL)", fontweight='bold')

    # Left: scatter plot
    axes[0].scatter(chunk.real, chunk.imag, s=0.3, alpha=0.3, color='steelblue')
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].set_title("Scatter")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    # Right: 2D histogram (density view — better for noisy constellations)
    lim = np.percentile(np.abs(chunk), 98) * 1.2
    axes[1].hist2d(chunk.real, chunk.imag, bins=200,
                   range=[[-lim, lim], [-lim, lim]],
                   cmap='hot', density=True)
    axes[1].set_title("Density (2D histogram)")
    axes[1].set_xlabel("I"); axes[1].set_ylabel("Q")
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig("constellation_final.png", dpi=150)
    print("Saved: constellation_final.png")
    plt.show()



if __name__ == "__main__":
    run_pipeline()