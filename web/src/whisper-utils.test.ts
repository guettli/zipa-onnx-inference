// Lightweight tests — run with: tsx src/whisper-utils.test.ts
import assert from 'node:assert/strict';
import {
  computeWhisperMel, N_FRAMES,
  SUPPORTED_LANGUAGES, LANG_TOKENS, LANG_LABELS,
  SOT, EOT, TRANSCRIBE, NO_TIMESTAMPS,
  detectBrowserLanguage,
  buildTokenDecoder, decodeTokens,
} from './whisper-utils.js';

let failed = 0;

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`  ✗ ${name}: ${msg}`);
    failed++;
  }
}

// ── Language tokens ───────────────────────────────────────────────────────────

test('all supported languages have a LANG_TOKENS entry', () => {
  for (const lang of SUPPORTED_LANGUAGES) {
    assert.ok(LANG_TOKENS[lang] > 50000, `${lang} token ID should be > 50000`);
    assert.ok(LANG_TOKENS[lang] < 51000, `${lang} token ID should be < 51000`);
  }
});

test('all supported languages have a LANG_LABELS entry', () => {
  for (const lang of SUPPORTED_LANGUAGES) {
    assert.ok(typeof LANG_LABELS[lang] === 'string' && LANG_LABELS[lang].length > 0);
  }
});

test('well-known token IDs are correct', () => {
  assert.equal(LANG_TOKENS.en, 50259);
  assert.equal(LANG_TOKENS.de, 50261);
  assert.equal(LANG_TOKENS.es, 50262);
  assert.equal(LANG_TOKENS.fr, 50265);
  assert.equal(LANG_TOKENS.it, 50274);
  assert.equal(SOT,           50258);
  assert.equal(EOT,           50257);
  assert.equal(TRANSCRIBE,    50359);
  assert.equal(NO_TIMESTAMPS, 50363);
});

test('LANG_TOKENS entries are unique (no collisions)', () => {
  const ids = Object.values(LANG_TOKENS);
  assert.equal(new Set(ids).size, ids.length);
});

// ── detectBrowserLanguage ─────────────────────────────────────────────────────

test('detectBrowserLanguage returns a supported language or en fallback', () => {
  // navigator is not available in Node; the function returns 'en' in that case
  const lang = detectBrowserLanguage();
  assert.ok((SUPPORTED_LANGUAGES as readonly string[]).includes(lang),
    `Expected one of ${SUPPORTED_LANGUAGES.join(',')} but got ${lang}`);
});

// ── Mel spectrogram shape & normalisation ─────────────────────────────────────

test('computeWhisperMel: output has correct length for silence', () => {
  const silence = new Float32Array(16000); // 1 second of silence
  const mel = computeWhisperMel(silence);
  assert.equal(mel.length, 80 * N_FRAMES, `Expected 80*${N_FRAMES}=${80 * N_FRAMES} elements`);
});

test('computeWhisperMel: output values are finite', () => {
  const audio = new Float32Array(16000);
  // A simple sine wave at 440 Hz
  for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;
  const mel = computeWhisperMel(audio);
  for (let i = 0; i < mel.length; i++) {
    assert.ok(isFinite(mel[i]), `mel[${i}] is not finite: ${mel[i]}`);
  }
});

test('computeWhisperMel: normalised values are in [-1, 2] range (Whisper normalisation)', () => {
  // After Whisper normalisation: (log10(mel) + 4) / 4, clamped to [max-8, max]
  // Typical range is roughly [0, 1] but can go slightly negative or above 1.
  const audio = new Float32Array(32000);
  for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(2 * Math.PI * 1000 * i / 16000) * 0.8;
  const mel = computeWhisperMel(audio);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < mel.length; i++) { if (mel[i] < min) min = mel[i]; if (mel[i] > max) max = mel[i]; }
  assert.ok(min >= -1, `min value ${min.toFixed(4)} < -1`);
  assert.ok(max <= 2,  `max value ${max.toFixed(4)} > 2`);
});

test('computeWhisperMel: longer audio is trimmed to 30 s', () => {
  const longAudio = new Float32Array(480001); // slightly over 30 s
  longAudio.fill(0.1);
  const mel = computeWhisperMel(longAudio);
  assert.equal(mel.length, 80 * N_FRAMES);
});

test('computeWhisperMel: shorter audio is zero-padded to 30 s', () => {
  const shortAudio = new Float32Array(1000); // 62.5 ms
  const mel = computeWhisperMel(shortAudio);
  assert.equal(mel.length, 80 * N_FRAMES);
});

// ── GPT-2 tokenizer decoder ───────────────────────────────────────────────────

test('buildTokenDecoder: basic round-trip', () => {
  const vocab: Record<string, number> = { Hello: 15496, Ġworld: 995, '!': 0 };
  const dec = buildTokenDecoder(vocab);
  assert.equal(dec.get(15496), 'Hello');
  assert.equal(dec.get(995), 'Ġworld');
  assert.equal(dec.get(0), '!');
});

test('decodeTokens: skips special tokens (>= 50257)', () => {
  // Vocab with a few printable-ASCII tokens
  const vocab: Record<string, number> = { 'H': 36, 'i': 72 };
  const dec = buildTokenDecoder(vocab);
  // SOT (50258) should be ignored; 36 → 'H', 72 → 'i'
  const result = decodeTokens([SOT, 36, 72, EOT], dec);
  assert.equal(result, 'Hi');
});

test('decodeTokens: Ġ prefix decodes as leading space', () => {
  // 'Ġ' is Unicode U+0120 which maps back to byte 0x20 (space)
  const vocab: Record<string, number> = { 'ĠHello': 9999 };
  const dec = buildTokenDecoder(vocab);
  const result = decodeTokens([9999], dec);
  assert.equal(result, ' Hello');
});

test('decodeTokens: unknown token ID is skipped gracefully', () => {
  const dec = buildTokenDecoder({});
  const result = decodeTokens([12345], dec);
  assert.equal(result, '');
});

// ── Summary ───────────────────────────────────────────────────────────────────

if (failed > 0) {
  console.error(`\n${failed} test(s) failed`);
  process.exit(1);
} else {
  console.log('\nAll tests passed.');
}
