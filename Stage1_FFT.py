import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d



def generate_voyager_waterfall(bin_path=r"D:\Hackathon\voyager_baseband.bin", fs=2.0e6, fft_size=16384):
    """
    Analyzes the binary IQ file to locate the drifting carrier frequency.
    Includes carrier detection, drift tracking, and all analysis plots.
    """
    print(f"Opening binary file: {bin_path}")

    # ── STAGE 0: Load IQ data ─────────────────────────────────────────────────
    try:
        raw_data = np.memmap(bin_path, dtype='float32', mode='r')
    except FileNotFoundError:
        print("Error: Binary file not found. Run your Phase 1 parser first.")
        return

    # Reconstruct complex IQ from interleaved floats (I, Q, I, Q, ...)
    iq_samples    = raw_data[0::2] + 1j * raw_data[1::2]
    total_samples = len(iq_samples)
    duration      = total_samples / fs
    print(f"Total duration of capture: {duration:.2f} seconds")
    print(f"Total IQ samples         : {total_samples:,}")

    # ── STAGE I-A: Build Waterfall Matrix ─────────────────────────────────────
    num_rows  = 1000
    step_size = total_samples // num_rows
    waterfall_matrix = []

    print("\nComputing FFTs for Waterfall Plot...")
    for i in range(num_rows):
        start_idx = i * step_size
        if start_idx + fft_size > total_samples:
            break

        chunk          = iq_samples[start_idx : start_idx + fft_size]
        windowed_chunk = chunk * np.blackman(fft_size)
        spectrum       = np.fft.fftshift(np.fft.fft(windowed_chunk))
        psd            = 10 * np.log10(np.abs(spectrum)**2 + 1e-12)
        waterfall_matrix.append(psd)

    waterfall_matrix = np.array(waterfall_matrix)   # shape: (num_rows, fft_size)
    print(f"Waterfall matrix shape: {waterfall_matrix.shape}")

    # ── STAGE I-B: Detect Active Signal Rows ─────────────────────────────────
    row_max_power = waterfall_matrix.max(axis=1)
    noise_floor   = np.percentile(row_max_power, 30)   # 30th pct = quiet rows
    signal_thresh = noise_floor + 6.0                   # 6 dB above noise

    active_rows = np.where(row_max_power > signal_thresh)[0]
    if len(active_rows) > 0:
        signal_start_s = active_rows[0]  * (duration / num_rows)
        signal_end_s   = active_rows[-1] * (duration / num_rows)
        print(f"\nSignal active from {signal_start_s:.1f}s to {signal_end_s:.1f}s")
        print(f"Active rows: {len(active_rows)} out of {num_rows}")
    else:
        print("WARNING: No active signal rows detected. Using first 100 rows.")
        active_rows    = np.arange(100)
        signal_end_s   = 100 * (duration / num_rows)
        signal_start_s = 0.0

    # ── STAGE I-C: Plot 1 — Waterfall (tight colormap, zoomed to carrier band) 
    freqs_hz  = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0/fs))   # Hz axis
    freqs_khz = freqs_hz / 1e3                                          # kHz axis

    vmax = np.percentile(waterfall_matrix, 99.5)
    vmin = vmax - 25.0     # show only top 25 dB — makes weak carrier pop

    plt.figure(figsize=(14, 8))
    img = plt.imshow(
        waterfall_matrix,
        aspect='auto',
        extent=[freqs_khz[0], freqs_khz[-1], duration, 0],
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    plt.title("Voyager-X: Spectral Waterfall (Carrier Drift Analysis)")
    plt.xlabel("Frequency Offset (kHz)")
    plt.ylabel("Time (Seconds)")
    plt.colorbar(img, label='Relative Power (dB)')
    plt.xlim(300, 600)     # zoom to 300–600 kHz where carrier lives
    plt.axhline(y=signal_end_s, color='cyan', lw=1,
                linestyle='--', label=f'Signal end (~{signal_end_s:.0f}s)')
    plt.legend(loc='upper right')
    plt.grid(color='white', linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig("waterfall_zoomed.png", dpi=150)
    print("\nSaved: waterfall_zoomed.png")
    plt.show()

    # ── STAGE I-D: Averaged Power Spectrum over active rows only ──────────────
    print("\nComputing averaged power spectrum (active rows only)...")

    # Average in linear scale — more accurate than averaging dB values
    avg_power_linear = np.mean(10 ** (waterfall_matrix[active_rows] / 10.0), axis=0)
    avg_psd_db       = 10 * np.log10(avg_power_linear + 1e-12)

    # ── STAGE I-E: Carrier Detection in 350–550 kHz band ─────────────────────
    search_mask    = (freqs_hz >= 350e3) & (freqs_hz <= 550e3)
    freqs_search   = freqs_hz[search_mask]
    avg_psd_search = avg_psd_db[search_mask]

    smoothed   = uniform_filter1d(avg_psd_search, size=7)
    peaks, props = find_peaks(smoothed, prominence=4.0, distance=30)

    if len(peaks) == 0:
        print("No peak in 350–550 kHz band. Falling back to full band search...")
        smoothed_full    = uniform_filter1d(avg_psd_db, size=7)
        peaks_fb, props_fb = find_peaks(smoothed_full, prominence=3.0, distance=30)
        if len(peaks_fb) == 0:
            print("ERROR: Could not detect carrier. Check your IQ data.")
            return
        best_idx        = peaks_fb[np.argmax(props_fb['prominences'])]
        carrier_freq_hz = freqs_hz[best_idx]
        best_prominence = props_fb['prominences'].max()
    else:
        best_idx        = peaks[np.argmax(props['prominences'])]
        carrier_freq_hz = freqs_search[best_idx]
        best_prominence = props['prominences'].max()

    print(f"Carrier frequency : {carrier_freq_hz/1e3:.3f} kHz")
    print(f"Peak prominence   : {best_prominence:.2f} dB")

    # ── STAGE I-F: Plot 2 — Averaged Power Spectrum ───────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(freqs_hz / 1e3, avg_psd_db, lw=0.7,
             color='steelblue', label='Avg PSD (active rows)')
    plt.axvline(carrier_freq_hz / 1e3, color='red', lw=1.5,
                linestyle='--',
                label=f'Carrier @ {carrier_freq_hz/1e3:.2f} kHz')
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power (dB)")
    plt.title("Averaged Power Spectrum — carrier detection (active signal rows)")
    plt.xlim(300, 600)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("carrier_detected.png", dpi=150)
    print("Saved: carrier_detected.png")
    plt.show()

    # ── STAGE I-G: Carrier Drift Tracking ─────────────────────────────────────
    print("\nTracking carrier drift over time...")

    bin_res   = fs / fft_size
    half_bins = int(80e3 / 2 / bin_res)    # ±80 kHz search window around carrier

    center_bin = np.argmin(np.abs(freqs_hz - carrier_freq_hz))
    lo = max(0, center_bin - half_bins)
    hi = min(fft_size, center_bin + half_bins)

    drift_freqs_hz  = []
    time_axis_drift = []

    for idx in active_rows:
        row_psd    = waterfall_matrix[idx]
        local_peak = np.argmax(row_psd[lo:hi])
        drift_freqs_hz.append(freqs_hz[lo + local_peak])
        time_axis_drift.append(idx * (duration / num_rows))

    drift_freqs_hz  = np.array(drift_freqs_hz)
    time_axis_drift = np.array(time_axis_drift)

    if len(drift_freqs_hz) < 3:
        print("Not enough points to fit drift polynomial. Skipping drift plot.")
        return

    # Fit degree-2 polynomial: constant Doppler + linear acceleration component
    poly_coeffs = np.polyfit(time_axis_drift, drift_freqs_hz, deg=2)
    drift_fit   = np.polyval(poly_coeffs, time_axis_drift)
    total_drift = drift_freqs_hz.max() - drift_freqs_hz.min()

    print(f"Total carrier drift : {total_drift:.1f} Hz over active window")
    print(f"Polynomial coeffs   : a={poly_coeffs[0]:.4f}, "
          f"b={poly_coeffs[1]:.4f}, c={poly_coeffs[2]:.2f}")

    # ── STAGE I-H: Plot 3 — Carrier Drift Curve ───────────────────────────────
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis_drift, drift_freqs_hz / 1e3,
             lw=0.8, color='darkorange', alpha=0.7, label='Measured drift')
    plt.plot(time_axis_drift, drift_fit / 1e3,
             lw=1.8, color='red', linestyle='--',
             label=f'Poly fit (deg 2) — drift={total_drift:.0f} Hz')
    plt.xlabel("Time (s)")
    plt.ylabel("Carrier frequency (kHz)")
    plt.title("Carrier Frequency Drift — Doppler + oscillator degradation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("carrier_drift.png", dpi=150)
    print("Saved: carrier_drift.png")
    plt.show()

    # ── STAGE I-I: Save outputs for Stage II ──────────────────────────────────
    np.save("carrier_freq_hz.npy",   np.array([carrier_freq_hz]))
    np.save("carrier_drift_hz.npy",  drift_freqs_hz)
    np.save("time_axis_s.npy",       time_axis_drift)
    np.save("drift_poly_coeffs.npy", poly_coeffs)

    print("\n─── Stage I Summary ───────────────────────────────")
    print(f"  Carrier frequency  : {carrier_freq_hz/1e3:.3f} kHz")
    print(f"  Signal active      : {signal_start_s:.1f}s → {signal_end_s:.1f}s")
    print(f"  Total drift        : {total_drift:.1f} Hz")
    print(f"  Poly fit           : {poly_coeffs[0]:.4f}t² + "
          f"{poly_coeffs[1]:.4f}t + {poly_coeffs[2]:.2f}")
    print(f"  Outputs saved      : carrier_freq_hz.npy, carrier_drift_hz.npy,")
    print(f"                       time_axis_s.npy, drift_poly_coeffs.npy")
    print("────────────────────────────────────────────────────")
    print("Pass these .npy files to your Stage II carrier wipeoff script.")


if __name__ == "__main__":
    generate_voyager_waterfall()