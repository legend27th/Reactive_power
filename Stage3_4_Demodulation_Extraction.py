"""
Voyager-X | Stage III & IV: BPSK Demodulation + Packet Recovery
================================================================
WHERE THIS FITS IN THE PIPELINE:
---------------------------------
Stage 0  →  parse hex files      → voyager_baseband.bin
Stage I  →  FFT + carrier detect → carrier_freq_hz.npy, drift_poly_coeffs.npy
Stage II →  BPF + wipeoff + PLL  → symbols_final.npy        ← WE START HERE
Stage III→  THIS FILE            → raw_bits.npy, packets_raw.bin

Inputs  : symbols_final.npy   (saved at end of your Stage II run_pipeline())
Outputs : raw_bits.npy
          bits_descrambled.npy
          packets_raw.bin
          sync_search_result.txt
          bpsk_constellation.png
          phase_histogram.png
          eye_diagram.png
          rotation_score.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG — must match your Stage II settings ───────────────────────────────
SYMBOL_RATE = 65000.0   # Hz  (from your Stage II symbol rate estimate)
GARDNER_SPS = 8         # must match GARDNER_SPS in your Stage II config

# =============================================================================
# LOAD symbols_final.npy  — output of gardner_timing_recovery() in Stage II
# =============================================================================
print("=" * 62)
print("  Voyager-X | Stage III: BPSK Demodulation & Packet Recovery")
print("=" * 62)

print("\n[Load] Reading symbols_final.npy ...")
symbols = np.load("symbols_final.npy")
print(f"       Total symbols : {len(symbols):,}")
print(f"       dtype         : {symbols.dtype}")

# =============================================================================
# STEP 1 — AMPLITUDE NORMALIZATION
# Purpose : bring constellation to unit circle so decision threshold = 0
# =============================================================================
print("\n[Step 1] Normalizing amplitude...")
rms          = np.sqrt(np.mean(np.abs(symbols) ** 2))
symbols_norm = symbols / rms
print(f"         RMS before  : {rms:.5f}")
print(f"         RMS after   : {np.sqrt(np.mean(np.abs(symbols_norm)**2)):.5f}")

# =============================================================================
# STEP 2 — ROTATION SEARCH  (fix polarization mismatch)
# Purpose : the backup antenna orientation is unknown → constellation may be
#           rotated by any angle.  BPSK has 180° ambiguity so we only need
#           to search 0–179°.
# Method  : brute force every integer degree; score = mean|I| - 0.5*std(Q)
#           This rewards constellations where all energy is on the I axis.
# =============================================================================
print("\n[Step 2] Brute-force rotation search (0–179°)...")
chunk_for_search = symbols_norm[500:50000]

best_angle = 0.0
best_score = -np.inf
all_scores = np.zeros(180)

for deg in range(0, 180):
    rad     = np.deg2rad(deg)
    rotated = chunk_for_search * np.exp(-1j * rad)
    score   = np.mean(np.abs(rotated.real)) - 0.5 * np.std(rotated.imag)
    all_scores[deg] = score
    if score > best_score:
        best_score = score
        best_angle = deg

symbols_rot = symbols_norm * np.exp(-1j * np.deg2rad(best_angle))
print(f"         Best angle  : {best_angle}°")
print(f"         Best score  : {best_score:.5f}")

# Plot rotation score curve
plt.figure(figsize=(10, 3))
plt.plot(all_scores, color='teal', lw=1.0)
plt.axvline(best_angle, color='red', lw=1.5, linestyle='--',
            label=f'Best = {best_angle}°')
plt.xlabel("Rotation angle (degrees)")
plt.ylabel("Score")
plt.title("Rotation Search — Score vs Angle")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rotation_score.png", dpi=150)
print("         Saved: rotation_score.png")
plt.show()

# =============================================================================
# STEP 3 — CONSTELLATION PLOT  (before & after rotation + density view)
# Purpose : visual confirmation of BPSK — should see 2 blobs on I axis
# =============================================================================
print("\n[Step 3] Plotting constellation...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BPSK Constellation — Stage III", fontsize=13, fontweight='bold')

# ── Left: before rotation ────────────────────────────────────────────────────
axes[0].scatter(symbols_norm[500:20000].real,
                symbols_norm[500:20000].imag,
                s=0.4, alpha=0.2, color='steelblue')
axes[0].set_title("Before Rotation Fix")
axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
axes[0].axis('equal'); axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='k', lw=0.5)
axes[0].axvline(0, color='k', lw=0.5)

# ── Middle: after rotation — BPSK should show 2 clean blobs ──────────────────
axes[1].scatter(symbols_rot[500:20000].real,
                symbols_rot[500:20000].imag,
                s=0.4, alpha=0.2, color='green')
axes[1].set_title(f"After {best_angle}° Rotation\n(BPSK: 2 blobs on I axis)")
axes[1].set_xlabel("I"); axes[1].set_ylabel("Q")
axes[1].axis('equal'); axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].axvline(0, color='k', lw=0.5)

# ── Right: 2D density histogram — better for noisy constellations ────────────
lim = np.percentile(np.abs(symbols_rot[500:]), 98) * 1.2
axes[2].hist2d(symbols_rot[500:50000].real,
               symbols_rot[500:50000].imag,
               bins=200,
               range=[[-lim, lim], [-lim, lim]],
               cmap='hot', density=True)
axes[2].set_title("Density View (2D histogram)")
axes[2].set_xlabel("I"); axes[2].set_ylabel("Q")
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig("bpsk_constellation.png", dpi=150)
print("         Saved: bpsk_constellation.png")
plt.show()

# =============================================================================
# STEP 4 — PHASE HISTOGRAM
# Purpose : confirm BPSK — should see exactly 2 peaks at 0° and ±180°
#           QPSK would show 4 peaks at ±45°, ±135°
# =============================================================================
print("\n[Step 4] Phase histogram...")
angles_deg = np.angle(symbols_rot[500:], deg=True)

plt.figure(figsize=(10, 4))
plt.hist(angles_deg, bins=360, color='purple', alpha=0.8)
plt.axvline(   0, color='red', lw=1.5, linestyle='--', label='0°')
plt.axvline( 180, color='red', lw=1.5, linestyle='--', label='±180°')
plt.axvline(-180, color='red', lw=1.5, linestyle='--')
plt.xlabel("Phase (degrees)"); plt.ylabel("Count")
plt.title("Phase Histogram\nBPSK = 2 peaks at 0° & ±180°")
plt.xticks(np.arange(-180, 181, 45))
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phase_histogram.png", dpi=150)
print("         Saved: phase_histogram.png")
plt.show()

# =============================================================================
# STEP 5 — EYE DIAGRAM
# Purpose : verify timing quality from Gardner TED in Stage II
#           Open eye = good timing | Closed/blurry = timing error
# =============================================================================
print("\n[Step 5] Eye diagram...")
eye_len  = GARDNER_SPS * 2

plt.figure(figsize=(8, 5))
for i in range(0, min(600 * GARDNER_SPS, len(symbols_rot) - eye_len),
               GARDNER_SPS):
    plt.plot(symbols_rot[i:i+eye_len].real,
             color='steelblue', alpha=0.04, lw=0.8)
plt.axhline( 1.0, color='red',   lw=1.2, linestyle='--', label='+1 level')
plt.axhline(-1.0, color='green', lw=1.2, linestyle='--', label='-1 level')
plt.axhline( 0.0, color='gray',  lw=0.5)
plt.title("Eye Diagram — BPSK I-channel\n"
          "Open eye = good timing | Closed = timing problem")
plt.xlabel("Sample offset"); plt.ylabel("Amplitude")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eye_diagram.png", dpi=150)
print("         Saved: eye_diagram.png")
plt.show()

# =============================================================================
# STEP 6 — HARD BIT DECISIONS
# BPSK rule : I > 0 → bit 0
#             I < 0 → bit 1
# =============================================================================
print("\n[Step 6] Making hard bit decisions (BPSK)...")
bits    = (symbols_rot.real < 0).astype(np.uint8)
balance = np.mean(bits)

print(f"         Total bits  : {len(bits):,}")
print(f"         Bit 0       : {np.sum(bits==0):,}  ({100*(1-balance):.1f}%)")
print(f"         Bit 1       : {np.sum(bits==1):,}  ({100*balance:.1f}%)")
print(f"         Balance     : {'GOOD ✓' if 0.4 < balance < 0.6 else 'SKEWED — check rotation angle'}")

np.save("raw_bits.npy", bits)
print("         Saved: raw_bits.npy")

# =============================================================================
# STEP 7 — CCSDS SYNC MARKER SEARCH
# Marker : 0x1ACFFC1D  (32 bits)
# Tries  : normal → inverted → 1-bit offset → 1-bit offset + inverted
# =============================================================================
print("\n[Step 7] Searching CCSDS sync marker 0x1ACFFC1D ...")

SYNC_HEX  = 0x1ACFFC1D
SYNC_BITS = np.array([(SYNC_HEX >> (31-i)) & 1
                       for i in range(32)], dtype=np.uint8)
SYNC_INV  = 1 - SYNC_BITS

def fast_sync_search(bitstream, pattern, max_hits=50):
    """Sliding 32-bit window search. Returns list of hit positions."""
    hits = []
    plen = len(pattern)
    for i in range(len(bitstream) - plen):
        if np.array_equal(bitstream[i:i+plen], pattern):
            hits.append(i)
            if len(hits) >= max_hits:
                break
    return hits

hits     = []
inverted = False
offset   = 0

# Try 1 — normal bits
hits = fast_sync_search(bits, SYNC_BITS)
if hits:
    print(f"         Found (normal)          : {len(hits)} hits ✓")

# Try 2 — inverted bits  (180° phase ambiguity)
if not hits:
    hits = fast_sync_search(bits, SYNC_INV)
    if hits:
        print(f"         Found (inverted)        : {len(hits)} hits → flipping bits")
        bits     = 1 - bits
        inverted = True
        np.save("raw_bits.npy", bits)

# Try 3 — 1-bit offset  (timing slip by 1)
if not hits:
    hits = fast_sync_search(bits[1:], SYNC_BITS)
    if hits:
        hits   = [h+1 for h in hits]
        offset = 1
        print(f"         Found (1-bit offset)    : {len(hits)} hits ✓")

# Try 4 — 1-bit offset + inverted
if not hits:
    hits = fast_sync_search((1 - bits)[1:], SYNC_INV)
    if hits:
        hits     = [h+1 for h in hits]
        bits     = 1 - bits
        inverted = True
        offset   = 1
        np.save("raw_bits.npy", bits)
        print(f"         Found (offset+inverted) : {len(hits)} hits ✓")

# Print hit positions
if hits:
    print(f"\n         First 5 sync positions:")
    for h in hits[:5]:
        print(f"           bit {h:>10,}  →  time {h/SYMBOL_RATE:.4f}s")

    # Estimate packet size from spacing between consecutive sync markers
    if len(hits) >= 2:
        spacings  = np.diff(hits)
        pkt_bits  = int(np.median(spacings))
        pkt_bytes = pkt_bits // 8
        print(f"\n         Packet size : {pkt_bits} bits = {pkt_bytes} bytes")
    else:
        pkt_bits  = 892 * 8    # CCSDS TM default
        pkt_bytes = 892
        print(f"         Only 1 sync — using CCSDS default: {pkt_bytes} bytes")
else:
    print("         Sync NOT found in raw bits — trying descrambler...")

# =============================================================================
# STEP 8 — CCSDS PN DESCRAMBLER
# Standard CCSDS scrambling polynomial: x^8 + x^7 + x^5 + x^3 + 1
# Shift register initialized to 0xFF
# =============================================================================
print("\n[Step 8] Applying CCSDS PN descrambler...")

def ccsds_pn_sequence(length):
    """
    Generates CCSDS standard PN sequence.
    Poly: x^8 + x^7 + x^5 + x^3 + 1
    Init: 0xFF
    Feedback taps at bit positions 8,7,5,3 (1-indexed)
    """
    reg = 0xFF
    seq = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        # XOR taps: positions 7,6,4,2 (0-indexed from MSB)
        feedback = ((reg >> 7) ^ (reg >> 6) ^
                    (reg >> 4) ^ (reg >> 2)) & 1
        seq[i]   = (reg >> 7) & 1      # output MSB
        reg      = ((reg << 1) | feedback) & 0xFF
    return seq

pn_seq   = ccsds_pn_sequence(len(bits))
bits_ds  = (bits ^ pn_seq).astype(np.uint8)
np.save("bits_descrambled.npy", bits_ds)
print("         Saved: bits_descrambled.npy")

# Search sync in descrambled bits if not found yet
if not hits:
    print("         Searching sync in descrambled bits...")
    hits_ds = fast_sync_search(bits_ds, SYNC_BITS)
    if not hits_ds:
        hits_ds = fast_sync_search(bits_ds, SYNC_INV)
        if hits_ds:
            bits_ds = 1 - bits_ds
            np.save("bits_descrambled.npy", bits_ds)

    if hits_ds:
        print(f"         SYNC FOUND IN DESCRAMBLED BITS: {len(hits_ds)} hits ✓")
        hits      = hits_ds
        bits      = bits_ds
        pkt_bits  = (int(np.median(np.diff(hits_ds)))
                     if len(hits_ds) >= 2 else 892 * 8)
        pkt_bytes = pkt_bits // 8
        print(f"         Packet size: {pkt_bits} bits = {pkt_bytes} bytes")
    else:
        print("         Sync NOT found after descrambling.")

# =============================================================================
# STEP 9 — PACKET EXTRACTION + CCSDS PRIMARY HEADER PARSE
# CCSDS Space Packet Primary Header (6 bytes):
#   Bits 0-2   : Version (always 000)
#   Bit  3     : Packet Type (0=TM, 1=TC)
#   Bit  4     : Secondary Header Flag
#   Bits 5-15  : APID (Application Process Identifier)
#   Bits 16-17 : Sequence Flags
#   Bits 18-31 : Sequence Count
#   Bits 32-47 : Data Length (actual length - 1)
# =============================================================================
if hits:
    print(f"\n[Step 9] Extracting {len(hits)} packets...")

    SEQ_FLAG_MAP = {0: 'Continuation', 1: 'First segment',
                    2: 'Last segment',  3: 'Standalone'}
    packets = []

    for pos in hits:
        start         = pos + 32           # skip the 32-bit sync marker itself
        end           = start + pkt_bits - 32
        if end > len(bits):
            continue
        n_bytes       = (end - start) // 8
        pkt_bytes_arr = np.packbits(bits[start : start + n_bytes * 8])
        packets.append(pkt_bytes_arr)

    # Save all packets as flat binary
    with open("packets_raw.bin", "wb") as f:
        for pkt in packets:
            f.write(pkt.tobytes())
    print(f"         Saved: packets_raw.bin  ({len(packets)} packets)")

    # ── Parse CCSDS primary header of first 5 packets ─────────────────────
    print("\n─── CCSDS Packet Header Parse ──────────────────────────────")
    for idx, pkt in enumerate(packets[:5]):
        if len(pkt) < 6:
            print(f"  Packet {idx+1}: too short ({len(pkt)} bytes), skipping")
            continue
        h          = pkt[:6]
        version    = (h[0] >> 5) & 0x07
        pkt_type   = (h[0] >> 4) & 0x01
        has_shdr   = (h[0] >> 3) & 0x01
        apid       = ((h[0] & 0x07) << 8) | h[1]
        seq_flags  = (h[2] >> 6) & 0x03
        seq_count  = ((h[2] & 0x3F) << 8) | h[3]
        data_len   = ((h[4] << 8) | h[5]) + 1   # CCSDS: field value + 1

        print(f"\n  Packet {idx+1}:")
        print(f"    Version       : {version}")
        print(f"    Type          : {'Telecommand' if pkt_type else 'Telemetry'}")
        print(f"    Sec Header    : {'Present' if has_shdr else 'Absent'}")
        print(f"    APID          : 0x{apid:03X}  ({apid})")
        print(f"    Seq flags     : {SEQ_FLAG_MAP.get(seq_flags,'?')}")
        print(f"    Seq count     : {seq_count}")
        print(f"    Data length   : {data_len} bytes")
        print(f"    Payload (hex) : "
              f"{' '.join(f'{b:02X}' for b in pkt[6:38])}...")

    print("\n────────────────────────────────────────────────────────────")

    # Save summary
    with open("sync_search_result.txt", "w") as f:
        f.write(f"Sync marker hits    : {len(hits)}\n")
        f.write(f"Bits inverted       : {inverted}\n")
        f.write(f"Bit offset used     : {offset}\n")
        f.write(f"Packet size (bits)  : {pkt_bits}\n")
        f.write(f"Packet size (bytes) : {pkt_bytes}\n")
        f.write(f"Packets extracted   : {len(packets)}\n")
        f.write(f"Sync positions      : {hits[:10]}\n")
    print("  Saved: sync_search_result.txt")

else:
    print("\n[!] No packets recovered.")
    print("    → Share bpsk_constellation.png + phase_histogram.png")
    print("    → Likely cause: constellation not fully locked in Stage II")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("  Stage III Complete")
print("=" * 62)
print("  Plots   : bpsk_constellation.png")
print("            phase_histogram.png")
print("            eye_diagram.png")
print("            rotation_score.png")
print("  Data    : raw_bits.npy")
print("            bits_descrambled.npy")
print("            packets_raw.bin")
print("            sync_search_result.txt")
print("=" * 62)