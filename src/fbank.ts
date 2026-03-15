/**
 * Log Mel Filterbank (Fbank) feature extractor.
 *
 * Matches lhotse FbankConfig defaults used by the zipa models:
 *   num_filters=80, sample_rate=16000, frame_length=25ms, frame_shift=10ms,
 *   dither=0, snip_edges=False, preemphasis=0.97, window=Povey,
 *   low_freq=20, high_freq=8000 (Nyquist), remove_dc_offset=True
 */

const SAMPLE_RATE   = 16000;
const FRAME_LEN_S   = 0.025;   // 25 ms → 400 samples
const FRAME_SHIFT_S = 0.010;   // 10 ms → 160 samples
const N_MELS        = 80;
const N_FFT         = 512;     // next power-of-2 >= 400
const LOW_FREQ      = 20.0;
const HIGH_FREQ     = 8000.0;  // Nyquist at 16 kHz
const PREEMPHASIS   = 0.97;

const FRAME_LEN   = Math.round(FRAME_LEN_S   * SAMPLE_RATE); // 400
const FRAME_SHIFT = Math.round(FRAME_SHIFT_S * SAMPLE_RATE); // 160

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------
function buildPoveyWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)), 0.85);
  }
  return w;
}

// ---------------------------------------------------------------------------
// FFT (in-place radix-2 Cooley-Tukey, power-of-2 size)
// ---------------------------------------------------------------------------
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let uRe = 1, uIm = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const a = i + j;
        const b = a + half;
        const vRe = re[b] * uRe - im[b] * uIm;
        const vIm = re[b] * uIm + im[b] * uRe;
        re[b] = re[a] - vRe;  im[b] = im[a] - vIm;
        re[a] += vRe;         im[a] += vIm;
        const nu = uRe * wRe - uIm * wIm;
        uIm = uRe * wIm + uIm * wRe;
        uRe = nu;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Mel filterbank matrix  (N_MELS × numBins)
// ---------------------------------------------------------------------------
function hzToMel(hz: number): number { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel: number): number { return 700 * (Math.pow(10, mel / 2595) - 1); }

function buildMelFilters(): Float32Array[] {
  const numBins = N_FFT / 2 + 1;
  const melLow  = hzToMel(LOW_FREQ);
  const melHigh = hzToMel(HIGH_FREQ);
  const melPts  = Array.from({ length: N_MELS + 2 }, (_, i) =>
    melLow + (i / (N_MELS + 1)) * (melHigh - melLow));
  const hzPts = melPts.map(melToHz);

  return Array.from({ length: N_MELS }, (_, m) => {
    const f = new Float32Array(numBins);
    const lo = hzPts[m], mid = hzPts[m + 1], hi = hzPts[m + 2];
    for (let k = 0; k < numBins; k++) {
      const freq = (k * SAMPLE_RATE) / N_FFT;
      if (freq >= lo && freq <= mid) {
        f[k] = (freq - lo) / (mid - lo);
      } else if (freq > mid && freq <= hi) {
        f[k] = (hi - freq) / (hi - mid);
      }
    }
    return f;
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
const WINDOW  = buildPoveyWindow(FRAME_LEN);
const FILTERS = buildMelFilters();

/**
 * Extract log Mel filterbank features from a 16 kHz mono Float32Array.
 * Returns a flat Float32Array of shape [numFrames × N_MELS] and numFrames.
 */
export function extractFbank(audio: Float32Array): { features: Float32Array; numFrames: number } {
  // With snip_edges=False the signal is centre-padded.
  const padLeft  = Math.floor(FRAME_LEN / 2);
  const numFrames = Math.round(audio.length / FRAME_SHIFT);
  const paddedLen = padLeft + audio.length + FRAME_LEN; // generous right pad
  const padded    = new Float32Array(paddedLen);
  padded.set(audio, padLeft);

  const numBins = N_FFT / 2 + 1;
  const features = new Float32Array(numFrames * N_MELS);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);

  for (let f = 0; f < numFrames; f++) {
    const start = f * FRAME_SHIFT;
    const end   = Math.min(start + FRAME_LEN, paddedLen);

    // Copy frame
    re.fill(0); im.fill(0);
    for (let i = 0; i < end - start; i++) re[i] = padded[start + i];

    // Remove DC offset
    let mean = 0;
    for (let i = 0; i < FRAME_LEN; i++) mean += re[i];
    mean /= FRAME_LEN;
    for (let i = 0; i < FRAME_LEN; i++) re[i] -= mean;

    // Preemphasis (applied in reverse to avoid a temp copy)
    for (let i = FRAME_LEN - 1; i > 0; i--) re[i] -= PREEMPHASIS * re[i - 1];

    // Apply Povey window
    for (let i = 0; i < FRAME_LEN; i++) re[i] *= WINDOW[i];

    // FFT
    fftInPlace(re, im);

    // Power spectrum
    const base = f * N_MELS;
    for (let m = 0; m < N_MELS; m++) {
      let energy = 0;
      const filt = FILTERS[m];
      for (let k = 0; k < numBins; k++) {
        energy += filt[k] * (re[k] * re[k] + im[k] * im[k]);
      }
      features[base + m] = Math.log(Math.max(energy, 1e-10));
    }
  }

  return { features, numFrames };
}
