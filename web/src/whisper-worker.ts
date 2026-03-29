/**
 * Whisper Web Worker: download, cache, and run onnx-community/whisper-tiny (int8).
 *
 * Uses the same Cache Storage cache ("zipa-models-v1") and Service Worker proxy
 * pattern as the zipa worker so ort.InferenceSession.create(url) can load the
 * model without materialising the full ArrayBuffer in the JS heap.
 *
 * Model files (both served under the same cache name):
 *   encoder  10 MB  onnx/encoder_model_int8.onnx
 *   decoder  31 MB  onnx/decoder_model_merged_int8.onnx   (merged: first-pass + KV-cache)
 *   vocab    ~1 MB  vocab.json  (GPT-2 BPE vocabulary)
 *
 * Inference pipeline:
 *   1. computeWhisperMel(audio) → [1, 80, 3000] mel features
 *   2. encoder.run({ input_features }) → last_hidden_state [1, 1500, 384]
 *   3. greedy decode with KV cache until <|endoftext|> or MAX_NEW_TOKENS
 *   4. decodeTokens(ids, tokenDecoder) → UTF-8 transcript string
 */
import * as ort from 'onnxruntime-web';
import {
  computeWhisperMel,
  LANG_TOKENS,
  SOT, EOT, TRANSCRIBE, NO_TIMESTAMPS,
  buildTokenDecoder, decodeTokens,
} from './whisper-utils.js';
import type { WhisperWorkerInMsg, WhisperWorkerOutMsg } from './types.js';

// ── ONNX Runtime config (matches main worker) ─────────────────────────────────
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  wasm: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd-threaded.wasm',
  mjs:  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd-threaded.mjs',
};

// ── HuggingFace URLs ──────────────────────────────────────────────────────────
const HF_BASE    = 'https://huggingface.co/onnx-community/whisper-tiny/resolve/main';
const ENCODER_HF = `${HF_BASE}/onnx/encoder_model_int8.onnx`;
const DECODER_HF = `${HF_BASE}/onnx/decoder_model_merged_int8.onnx`;
const VOCAB_HF   = `${HF_BASE}/vocab.json`;

// ── Same-origin Cache Storage proxy URLs (served by sw.js) ───────────────────
const PROXY_BASE    = '/zipa-onnx-inference/model-proxy/';
const ENCODER_PROXY = `${PROXY_BASE}whisper-tiny%2Fencoder-int8`;
const DECODER_PROXY = `${PROXY_BASE}whisper-tiny%2Fdecoder-merged-int8`;
const VOCAB_PROXY   = `${PROXY_BASE}whisper-tiny%2Fvocab`;

const MODEL_CACHE_NAME = 'zipa-models-v1';

// ── Whisper-tiny architecture constants ───────────────────────────────────────
const N_HEADS   = 6;
const HEAD_DIM  = 64;
const MAX_NEW_TOKENS = 224; // Whisper default

// ── Worker state ─────────────────────────────────────────────────────────────
let encoderSession: ort.InferenceSession | null = null;
let decoderSession: ort.InferenceSession | null = null;
let tokenDecoder:   Map<number, string>  | null = null;
let cancelled = false;

// ── Cache Storage helpers ─────────────────────────────────────────────────────

async function isInCache(proxyUrl: string): Promise<boolean> {
  if (!('caches' in self)) return false;
  const cache = await caches.open(MODEL_CACHE_NAME);
  return (await cache.match(proxyUrl)) != null;
}

async function streamDownloadToCache(
  hfUrl: string,
  proxyUrl: string,
  sizeMb: number,
  onProgress: (pct: number, mbReceived: number, mbTotal: number) => void,
): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(hfUrl);
  } catch {
    throw new Error('Network error — check your internet connection.');
  }
  if (resp.status === 404) throw new Error(`Model file not found (404): ${hfUrl}`);
  if (resp.status === 429) throw new Error('Rate-limited by Hugging Face (429). Wait a moment and retry.');
  if (!resp.ok) throw new Error(`Download failed HTTP ${resp.status}.`);

  const total = parseInt(resp.headers.get('content-length') ?? '0', 10) || sizeMb * 1e6;
  let received = 0;
  const cache = await caches.open(MODEL_CACHE_NAME);

  if (typeof TransformStream !== 'undefined' && resp.body) {
    const ts = new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        received += chunk.byteLength;
        onProgress(received / total, received / 1e6, total / 1e6);
        controller.enqueue(chunk);
      },
    });
    await cache.put(proxyUrl, new Response(resp.body.pipeThrough(ts), {
      headers: {
        'Content-Type': 'application/octet-stream',
        ...(total > 0 ? { 'Content-Length': String(total) } : {}),
      },
    }));
  } else {
    // Fallback: accumulate then store
    const reader = resp.body!.getReader();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress(received / total, received / 1e6, total / 1e6);
    }
    const merged = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
    await cache.put(proxyUrl, new Response(merged.buffer, {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(received) },
    }));
  }
}

// ── ONNX session creation ─────────────────────────────────────────────────────

async function createSession(urlOrBuf: string | ArrayBuffer): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(urlOrBuf as string, { executionProviders: ['wasm'] });
}

async function loadSession(proxyUrl: string): Promise<ort.InferenceSession> {
  try {
    return await createSession(proxyUrl);
  } catch {
    // SW not yet controlling this page — fall back to reading the buffer
    const cache = await caches.open(MODEL_CACHE_NAME);
    const resp  = await cache.match(proxyUrl);
    if (!resp) throw new Error('Model not in cache — reload the page.');
    return createSession(await resp.arrayBuffer());
  }
}

// ── Greedy decoder with KV cache ──────────────────────────────────────────────

function argmaxSlice(data: Float32Array, offset: number, length: number): number {
  let best = 0, bestVal = -Infinity;
  for (let i = 0; i < length; i++) {
    if (data[offset + i] > bestVal) { bestVal = data[offset + i]; best = i; }
  }
  return best;
}

async function greedyDecode(
  session: ort.InferenceSession,
  encoderHiddenStates: ort.Tensor,
  langToken: number,
  dec: Map<number, string>,
): Promise<string> {
  const kvInputNames  = session.inputNames.filter(n => n.startsWith('past_key_values.'));
  const kvOutputNames = session.outputNames.filter(n => n.startsWith('present.'));

  // Empty KV cache — zero-length sequence dimension
  let kvCache: Record<string, ort.Tensor> = {};
  for (const name of kvInputNames) {
    kvCache[name] = new ort.Tensor('float32', new Float32Array(0), [1, N_HEADS, 0, HEAD_DIM]);
  }

  // ── Step 0: process initial prompt tokens together ────────────────────────
  const prompt = [SOT, langToken, TRANSCRIBE, NO_TIMESTAMPS];
  const promptIds = new BigInt64Array(prompt.map(BigInt));

  let outputs = await session.run({
    input_ids:             new ort.Tensor('int64',   promptIds, [1, prompt.length]),
    encoder_hidden_states: encoderHiddenStates,
    use_cache_branch:      new ort.Tensor('bool',    new Uint8Array([0]), [1]),
    ...kvCache,
  });

  // Logits shape: [1, prompt.length, vocab_size] — take the last position
  const vocabSize  = outputs.logits.dims[2] as number;
  const logits0    = outputs.logits.data as Float32Array;
  let nextToken = argmaxSlice(logits0, (prompt.length - 1) * vocabSize, vocabSize);

  // Rename present.* → past_key_values.*
  kvCache = {};
  for (const outName of kvOutputNames) {
    kvCache['past_key_values.' + outName.slice('present.'.length)] = outputs[outName];
  }

  // ── Autoregressive loop ───────────────────────────────────────────────────
  const generated: number[] = [];

  for (let step = 0; step < MAX_NEW_TOKENS; step++) {
    if (nextToken === EOT || cancelled) break;
    generated.push(nextToken);

    outputs = await session.run({
      input_ids:             new ort.Tensor('int64',  new BigInt64Array([BigInt(nextToken)]), [1, 1]),
      encoder_hidden_states: encoderHiddenStates,
      use_cache_branch:      new ort.Tensor('bool',   new Uint8Array([1]), [1]),
      ...kvCache,
    });

    const stepLogits = outputs.logits.data as Float32Array;
    nextToken = argmaxSlice(stepLogits, 0, outputs.logits.dims[2] as number);

    kvCache = {};
    for (const outName of kvOutputNames) {
      kvCache['past_key_values.' + outName.slice('present.'.length)] = outputs[outName];
    }
  }

  return decodeTokens(generated, dec);
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (ev: MessageEvent<WhisperWorkerInMsg>) => {
  const msg = ev.data;
  const post = (m: WhisperWorkerOutMsg) => self.postMessage(m);

  // ── Load ────────────────────────────────────────────────────────────────────
  if (msg.type === 'load') {
    if (encoderSession && decoderSession && tokenDecoder) {
      post({ type: 'model-ready' });
      return;
    }
    try {
      // Encoder
      if (!(await isInCache(ENCODER_PROXY))) {
        await streamDownloadToCache(ENCODER_HF, ENCODER_PROXY, 10.1, (pct, mbR, mbT) => {
          post({ type: 'download-progress', file: 'encoder', pct, mbReceived: mbR, mbTotal: mbT });
        });
      }

      // Decoder
      if (!(await isInCache(DECODER_PROXY))) {
        await streamDownloadToCache(DECODER_HF, DECODER_PROXY, 30.7, (pct, mbR, mbT) => {
          post({ type: 'download-progress', file: 'decoder', pct, mbReceived: mbR, mbTotal: mbT });
        });
      }

      // Vocab JSON (small — fetch as text, cache for offline use)
      if (!(await isInCache(VOCAB_PROXY))) {
        post({ type: 'download-progress', file: 'vocab', pct: 0, mbReceived: 0, mbTotal: 1 });
        let vocabResp: Response;
        try { vocabResp = await fetch(VOCAB_HF); }
        catch { throw new Error('Network error downloading vocab.json'); }
        if (!vocabResp.ok) throw new Error(`vocab.json download failed HTTP ${vocabResp.status}`);
        const text  = await vocabResp.text();
        const cache = await caches.open(MODEL_CACHE_NAME);
        await cache.put(VOCAB_PROXY, new Response(text, {
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
        }));
        post({ type: 'download-progress', file: 'vocab', pct: 1, mbReceived: text.length / 1e6, mbTotal: text.length / 1e6 });
      }

      // Parse vocab
      const cache     = await caches.open(MODEL_CACHE_NAME);
      const vocabResp = await cache.match(VOCAB_PROXY);
      if (!vocabResp) throw new Error('vocab.json not in cache after download');
      const vocab = await vocabResp.json() as Record<string, number>;
      tokenDecoder = buildTokenDecoder(vocab);

      // Create ONNX sessions
      encoderSession = await loadSession(ENCODER_PROXY);
      decoderSession = await loadSession(DECODER_PROXY);

      post({ type: 'model-ready' });
    } catch (e) {
      post({ type: 'error', message: String(e) });
    }
    return;
  }

  if (msg.type === 'cancel') {
    cancelled = true;
    return;
  }

  // ── Infer ───────────────────────────────────────────────────────────────────
  if (msg.type === 'infer') {
    cancelled = false;
    if (!encoderSession || !decoderSession || !tokenDecoder) {
      post({ type: 'error', message: 'Whisper models not loaded yet' });
      return;
    }

    const TIMEOUT_MS = 120_000;
    const timeoutId = setTimeout(() => {
      cancelled = true;
      post({ type: 'error', message: 'Whisper inference timed out after 120 s.' });
    }, TIMEOUT_MS);

    try {
      const langToken = LANG_TOKENS[msg.language as keyof typeof LANG_TOKENS] ?? LANG_TOKENS.en;

      // Mel spectrogram
      const melData = computeWhisperMel(msg.audio);
      const inputFeatures = new ort.Tensor('float32', melData, [1, 80, 3000]);

      // Encoder
      const encOutputs  = await encoderSession.run({ input_features: inputFeatures });
      inputFeatures.dispose();
      if (cancelled) return;

      const encoderHiddenStates = encOutputs[encoderSession.outputNames[0]];

      // Greedy decode
      const transcript = await greedyDecode(decoderSession, encoderHiddenStates, langToken, tokenDecoder);
      encoderHiddenStates.dispose();

      if (!cancelled) post({ type: 'result', transcript: transcript.trim() });
    } catch (e) {
      if (!cancelled) post({ type: 'error', message: String(e) });
    } finally {
      clearTimeout(timeoutId);
    }
  }
};
