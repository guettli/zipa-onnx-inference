/**
 * Whisper-specific utilities: log-mel spectrogram, language tokens, tokenizer.
 *
 * Mel spec matches openai/whisper:
 *   n_fft=400, hop_length=160, n_mels=80, sr=16000
 *   Hann window, power spectrum (|FFT|²), slaney-normalised mel filterbank,
 *   log10 + dynamic-range clamp + linear normalisation.
 *   Audio is zero-padded to 30 s (480 000 samples) and centre-padded by
 *   n_fft/2 on each side (mimicking torch.stft(center=True)), yielding
 *   exactly 3000 mel frames.
 */

// ── Constants ─────────────────────────────────────────────────────────────────

const SAMPLE_RATE = 16000;
const N_FFT       = 400;       // 25 ms frame
const HOP_LENGTH  = 160;       // 10 ms shift
const N_MELS      = 80;
const N_SAMPLES   = 480_000;   // 30 s × 16 kHz
export const N_FRAMES = 3000;  // output mel frames

// ── Hann window ───────────────────────────────────────────────────────────────

const HANN_WINDOW = (() => {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N_FFT - 1)));
  }
  return w;
})();

// ── Radix-2 Cooley-Tukey FFT (in-place) ──────────────────────────────────────

function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let uRe = 1, uIm = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = a + half;
        const vRe = re[b] * uRe - im[b] * uIm;
        const vIm = re[b] * uIm + im[b] * uRe;
        re[b] = re[a] - vRe; im[b] = im[a] - vIm;
        re[a] += vRe; im[a] += vIm;
        const nu = uRe * wRe - uIm * wIm;
        uIm = uRe * wIm + uIm * wRe;
        uRe = nu;
      }
    }
  }
}

// ── Mel filterbank (librosa slaney-normalised, matches whisper's mel_filters.npz) ──

const N_FREQ = N_FFT / 2 + 1; // 201 one-sided FFT bins

const MEL_FILTERS = (() => {
  function hzToMel(hz: number): number { return 2595 * Math.log10(1 + hz / 700); }
  function melToHz(mel: number): number { return 700 * (Math.pow(10, mel / 2595) - 1); }

  const melMin = hzToMel(0);
  const melMax = hzToMel(SAMPLE_RATE / 2); // 0 .. 8000 Hz

  // N_MELS+2 uniformly-spaced points in mel space → Hz
  const hzPts = Array.from({ length: N_MELS + 2 }, (_, i) =>
    melToHz(melMin + (i / (N_MELS + 1)) * (melMax - melMin)),
  );

  return Array.from({ length: N_MELS }, (_, m) => {
    const lo = hzPts[m], mid = hzPts[m + 1], hi = hzPts[m + 2];
    const f = new Float32Array(N_FREQ);
    for (let k = 0; k < N_FREQ; k++) {
      const freq = (k * SAMPLE_RATE) / N_FFT;
      if (freq >= lo && freq <= mid) {
        f[k] = (freq - lo) / (mid - lo);
      } else if (freq > mid && freq <= hi) {
        f[k] = (hi - freq) / (hi - mid);
      }
    }
    // Slaney area normalisation: 2 / (upper_hz − lower_hz)
    const norm = 2 / (hi - lo);
    for (let k = 0; k < N_FREQ; k++) f[k] *= norm;
    return f;
  });
})();

// ── Public: mel spectrogram ───────────────────────────────────────────────────

/**
 * Compute the Whisper log-mel spectrogram from 16 kHz mono PCM audio.
 * Returns a Float32Array stored as [N_MELS × N_FRAMES] row-major,
 * suitable for wrapping in `new ort.Tensor('float32', mel, [1, 80, 3000])`.
 */
export function computeWhisperMel(audio: Float32Array): Float32Array {
  // 1. Zero-pad / truncate to exactly N_SAMPLES (30 s)
  const padded = new Float32Array(N_SAMPLES);
  padded.set(audio.subarray(0, Math.min(audio.length, N_SAMPLES)));

  // 2. Centre-pad by N_FFT/2 on each side (mirrors torch.stft center=True)
  const centerPad = N_FFT >> 1; // 200
  const working = new Float32Array(N_SAMPLES + 2 * centerPad);
  working.set(padded, centerPad);

  // 3. STFT → power spectrum → mel filterbank (N_FRAMES rows, N_MELS cols)
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);

  // mel[t * N_MELS + m] = energy of mel bin m at frame t
  const melRaw = new Float32Array(N_FRAMES * N_MELS);

  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP_LENGTH;
    re.fill(0); im.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      re[i] = working[start + i] * HANN_WINDOW[i];
    }
    fftInPlace(re, im);

    const base = t * N_MELS;
    for (let m = 0; m < N_MELS; m++) {
      const filt = MEL_FILTERS[m];
      let energy = 0;
      for (let k = 0; k < N_FREQ; k++) {
        energy += filt[k] * (re[k] * re[k] + im[k] * im[k]);
      }
      melRaw[base + m] = energy;
    }
  }

  // 4. log10, then dynamic-range clamp + linear normalisation
  //    log_spec = max(log10(max(mel, 1e-10)), max_val - 8)
  //    output   = (log_spec + 4) / 4
  let maxLog = -Infinity;
  for (let i = 0; i < melRaw.length; i++) {
    const v = Math.log10(Math.max(melRaw[i], 1e-10));
    melRaw[i] = v;
    if (v > maxLog) maxLog = v;
  }
  const floor = maxLog - 8;
  for (let i = 0; i < melRaw.length; i++) {
    melRaw[i] = (Math.max(melRaw[i], floor) + 4) / 4;
  }

  // 5. Transpose from [N_FRAMES, N_MELS] to [N_MELS, N_FRAMES]
  //    (ONNX encoder expects [1, N_MELS, N_FRAMES])
  const out = new Float32Array(N_MELS * N_FRAMES);
  for (let t = 0; t < N_FRAMES; t++) {
    for (let m = 0; m < N_MELS; m++) {
      out[m * N_FRAMES + t] = melRaw[t * N_MELS + m];
    }
  }
  return out;
}

// ── Language tokens (Whisper multilingual) ────────────────────────────────────

export const SUPPORTED_LANGUAGES = ['de', 'en', 'fr', 'it', 'es'] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

/** Token ID of the <|lang|> prefix token for each supported language. */
export const LANG_TOKENS: Record<SupportedLanguage, number> = {
  en: 50259,
  de: 50261,
  es: 50262,
  fr: 50265,
  it: 50274,
};

export const LANG_LABELS: Record<SupportedLanguage, string> = {
  en: 'English',
  de: 'German',
  fr: 'French',
  it: 'Italian',
  es: 'Spanish',
};

// Standard Whisper special-token IDs (multilingual, non-.en variant)
export const SOT           = 50258; // <|startoftranscript|>
export const EOT           = 50257; // <|endoftext|>
export const TRANSCRIBE    = 50359; // <|transcribe|>
export const NO_TIMESTAMPS = 50363; // <|notimestamps|>

/** Return the best supported language from the browser's navigator.language. */
export function detectBrowserLanguage(): SupportedLanguage {
  if (typeof navigator === 'undefined') return 'en';
  const code = (navigator.language ?? 'en').split('-')[0].toLowerCase();
  return (SUPPORTED_LANGUAGES as readonly string[]).includes(code)
    ? (code as SupportedLanguage)
    : 'en';
}

// ── GPT-2 byte-level BPE tokenizer decoder ────────────────────────────────────

/**
 * Build the inverse mapping used by GPT-2's bytes_to_unicode():
 * Unicode codepoint → original byte value.
 *
 * GPT-2 maps the 256 byte values to 256 Unicode codepoints so that every
 * byte has a printable representation.  Bytes 33-126, 161-172, 174-255 are
 * left as-is; the remaining 68 bytes are mapped to codepoints 256-323.
 */
function buildUnicodeToByteMap(): Map<number, number> {
  const m = new Map<number, number>();
  // Pass-through ranges (byte == codepoint)
  for (let b = 33; b <= 126; b++) m.set(b, b);
  for (let b = 161; b <= 172; b++) m.set(b, b);
  for (let b = 174; b <= 255; b++) m.set(b, b);
  // Non-pass-through bytes → codepoints starting at 256
  let extra = 256;
  for (let b = 0; b < 256; b++) {
    if (!m.has(b)) m.set(extra++, b);
  }
  return m;
}

const UNICODE_TO_BYTE = buildUnicodeToByteMap();

/**
 * Build a reverse-vocab map (token ID → token string) from the vocab object
 * found in vocab.json  (format: { "token_string": id, … }).
 */
export function buildTokenDecoder(vocab: Record<string, number>): Map<number, string> {
  const dec = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) {
    dec.set(id, token);
  }
  return dec;
}

/**
 * Decode a list of Whisper token IDs to a UTF-8 string.
 * Special tokens (ID ≥ 50257) are silently skipped.
 */
export function decodeTokens(ids: number[], tokenDecoder: Map<number, string>): string {
  const bytes: number[] = [];
  for (const id of ids) {
    if (id >= 50257) continue; // skip all special tokens
    const token = tokenDecoder.get(id);
    if (!token) continue;
    for (const ch of token) {
      const cp = ch.codePointAt(0)!;
      const byte = UNICODE_TO_BYTE.get(cp);
      if (byte !== undefined) bytes.push(byte);
    }
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
}
