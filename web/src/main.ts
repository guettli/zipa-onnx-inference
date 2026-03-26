import type { WorkerInMsg, WorkerOutMsg, FrameOut, BeamOut } from './types.js';
import { MODELS, modelUrl, modelProxyUrl } from './types.js';
import { isModelCached, getPartialDownload, pruneStalePartials,
         saveHistoryEntry, loadHistory, loadHistoryAudio,
         appendCrashLog, loadCrashLog, clearCrashLog } from './model-cache.js';
import type { HistoryEntry, CrashLogEntry } from './model-cache.js';
import { isActiveFrame, findActiveRange } from './frame-utils.js';

// ipa-descriptions.json is loaded lazily (dynamic import) to keep the initial
// bundle small and avoid 60 KB of JSON being parsed at startup.
type IpaEntry = { name: string; desc: string; url: string } | null;
let _ipaDesc: Record<string, IpaEntry> | null = null;
async function getIpaDesc(): Promise<Record<string, IpaEntry>> {
  if (!_ipaDesc) {
    const mod = await import('./ipa-descriptions.json');
    _ipaDesc = mod.default as Record<string, IpaEntry>;
  }
  return _ipaDesc;
}

// ipa-neighbours is also deferred — only needed when rendering the IPA overview.
let _computeAllNeighbours: ((desc: Record<string, IpaEntry>) => Record<string, Array<{sym:string;score:number}>>) | null = null;
let _symAnchorId: ((sym: string) => string) | null = null;

(document.getElementById('build-time') as HTMLElement).textContent =
  new Date(__BUILD_TIME__).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

// ── Service Worker ────────────────────────────────────────────────────────────
// Must be registered as early as possible so it is active when the worker
// calls ort.InferenceSession.create(proxyUrl).
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(() => {/* non-fatal */});
}

// ── Crash log ─────────────────────────────────────────────────────────────────
appendCrashLog('page-load').catch(() => {});

(async () => {
  try {
    const entries = await loadCrashLog();
    renderCrashLog(entries);
  } catch { /* non-fatal */ }
})();

// ── DOM refs ──────────────────────────────────────────────────────────────────

const errorLog          = document.getElementById('error-log')          as HTMLDivElement;
const modelSelect       = document.getElementById('model-select')       as HTMLSelectElement;
const modelStatus       = document.getElementById('model-status')       as HTMLDivElement;
const progressBar       = document.getElementById('progress-bar')       as HTMLDivElement;
const progressFill      = document.getElementById('progress-fill')      as HTMLDivElement;
const recordBtn         = document.getElementById('record-btn')         as HTMLButtonElement;
const uploadBtn         = document.getElementById('upload-btn')         as HTMLButtonElement;
const fileInput         = document.getElementById('file-input')         as HTMLInputElement;
const playbackSection   = document.getElementById('playback-section')   as HTMLElement;
const playbackAudio     = document.getElementById('playback')           as HTMLAudioElement;
const rerunBtn          = document.getElementById('rerun-btn')          as HTMLButtonElement;
const statusLine        = document.getElementById('status-line')        as HTMLDivElement;
const beamsSection      = document.getElementById('beams-section')      as HTMLElement;
const beamsEl           = document.getElementById('beams')              as HTMLDivElement;
const framesSection     = document.getElementById('frames-section')     as HTMLElement;
const framesLabel       = framesSection.querySelector('label')          as HTMLElement;
const framesBody        = document.getElementById('frames-body')        as HTMLTableSectionElement;
const resultModelEl     = document.getElementById('result-model')       as HTMLDivElement;
const transcriptSection = document.getElementById('transcript-section') as HTMLDetailsElement;
const transcriptEl      = document.getElementById('transcript')         as HTMLDivElement;
const historySection    = document.getElementById('history-section')    as HTMLDetailsElement;
const historyList       = document.getElementById('history-list')       as HTMLDivElement;
const crashLogSection   = document.getElementById('crash-log-section')  as HTMLElement | null;
const crashLogList      = document.getElementById('crash-log-list')     as HTMLElement | null;

// ── Crash log rendering ───────────────────────────────────────────────────────

function renderCrashLog(entries: CrashLogEntry[]) {
  if (!crashLogSection || !crashLogList || entries.length === 0) return;

  // Detect potential crash: last event is 'infer-start' with no 'infer-done' after it
  const last = entries[entries.length - 1];
  const mayHaveCrashed = last?.event === 'infer-start';

  // Show only if there are meaningful entries
  crashLogSection.style.display = '';
  if (mayHaveCrashed) {
    const warn = document.createElement('div');
    warn.className = 'crash-warning';
    warn.textContent = '⚠ Previous session may have crashed during inference. Consider using the small int8 model and shorter recordings.';
    crashLogSection.insertBefore(warn, crashLogList);
  }

  crashLogList.innerHTML = '';
  for (const e of entries.slice(-20)) {  // show last 20 entries
    const row = document.createElement('div');
    row.className = 'crash-log-row';
    const ts = new Date(e.timestamp).toLocaleTimeString();
    row.textContent = `${ts}  ${e.event}${e.detail ? '  ' + e.detail : ''}`;
    crashLogList.appendChild(row);
  }

  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear log';
  clearBtn.className = 'clear-log-btn';
  clearBtn.addEventListener('click', () => {
    clearCrashLog().catch(() => {});
    crashLogSection.style.display = 'none';
  });
  crashLogSection.appendChild(clearBtn);
}

// ── Populate model selector ───────────────────────────────────────────────────

const optionByModelId = new Map<string, HTMLOptionElement>();

for (const m of MODELS) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label;
  optionByModelId.set(m.id, opt);
  modelSelect.appendChild(opt);
}

/** Check Cache Storage API for a model (new path) or IndexedDB (legacy). */
async function isModelAvailable(m: typeof MODELS[0]): Promise<boolean> {
  if ('caches' in window) {
    try {
      const cache = await caches.open('zipa-models-v1');
      if (await cache.match(modelProxyUrl(m))) return true;
    } catch { /* ignore */ }
  }
  return isModelCached(modelUrl(m)); // legacy IndexedDB fallback
}

async function refreshDropdownCacheStatus() {
  for (const m of MODELS) {
    const opt = optionByModelId.get(m.id)!;
    try {
      if (await isModelAvailable(m)) {
        opt.textContent = `${m.label} (downloaded)`;
      } else {
        const partial = await getPartialDownload(modelUrl(m));
        if (partial && partial.total > 0) {
          const pct = ((partial.data.byteLength / partial.total) * 100).toFixed(0);
          opt.textContent = `↓${pct}% ${m.label}`;
        } else {
          opt.textContent = m.label;
        }
      }
    } catch {
      opt.textContent = m.label;
    }
  }
}

refreshDropdownCacheStatus();
pruneStalePartials().catch(() => {});

// ── Worker ────────────────────────────────────────────────────────────────────

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

function send(msg: WorkerInMsg) { worker.postMessage(msg); }

let activeDownloadModelId: string | null = null;
let pendingRerun = false;

function setInputsDisabled(disabled: boolean) {
  recordBtn.disabled  = disabled;
  uploadBtn.disabled  = disabled;
  rerunBtn.disabled   = disabled || !hasSessionAudio;
}

worker.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
  const msg = ev.data;

  if (msg.type === 'download-progress') {
    progressBar.style.display = 'block';
    progressFill.style.width  = `${(msg.pct * 100).toFixed(0)}%`;
    const pct = (msg.pct * 100).toFixed(0);
    const mb  = msg.mbReceived.toFixed(1);
    const tot = msg.mbTotal > 0 ? ` / ${msg.mbTotal.toFixed(0)} MB` : '';
    setStatus(`Downloading… ${pct}%  (${mb}${tot} MB)`);
    if (activeDownloadModelId) {
      const m   = MODELS.find(x => x.id === activeDownloadModelId)!;
      const opt = optionByModelId.get(activeDownloadModelId)!;
      opt.textContent = `↓${pct}% ${m.label}`;
    }
    return;
  }

  if (msg.type === 'model-ready') {
    progressBar.style.display = 'none';
    modelStatus.textContent   = `${msg.modelId} (downloaded)`;
    modelStatus.className     = 'ready';
    recordBtn.disabled        = false;
    uploadBtn.disabled        = false;
    const m   = MODELS.find(x => x.id === msg.modelId);
    const opt = optionByModelId.get(msg.modelId);
    if (m && opt) opt.textContent = `${m.label} (downloaded)`;
    activeDownloadModelId = null;
    appendCrashLog('model-ready', msg.modelId).catch(() => {});
    if (browserInfoRows) {
      const provEl = browserInfoRows.querySelector<HTMLElement>('.bi-prov');
      if (provEl) provEl.textContent = msg.executionProvider;
    }
    if (pendingRerun && hasSessionAudio) {
      pendingRerun = false;
      rerunBtn.disabled = true;
      setStatus('Processing audio…');
      getSessionAudioBlob()
        .then(blob => blob ? processAudio(blob) : Promise.reject('no audio'))
        .catch(e => { setStatus(''); showError(String(e)); });
    } else {
      rerunBtn.disabled = !hasSessionAudio;
      setStatus('');
    }
    return;
  }

  if (msg.type === 'result') {
    const inferenceDurationMs = inferStartMs > 0 ? performance.now() - inferStartMs : undefined;
    inferStartMs = 0;
    appendCrashLog('infer-done', inferenceDurationMs ? `${(inferenceDurationMs/1000).toFixed(1)}s` : undefined).catch(() => {});
    recordBtn.textContent = '🎤 Record';
    recordBtn.classList.remove('recording');
    setInputsDisabled(false);
    const inferSec = inferenceDurationMs !== undefined ? ` | inference: ${(inferenceDurationMs / 1000).toFixed(1)} s` : '';
    resultModelEl.textContent = `Result from: ${modelSelect.value}${inferSec}`;
    resultModelEl.style.display = '';
    // Render transcript with lazy IPA data
    getIpaDesc().then(desc => {
      transcriptEl.innerHTML = msg.transcript ? ipaStringHtml(msg.transcript, desc) : '(no speech detected)';
    }).catch(() => {
      transcriptEl.textContent = msg.transcript || '(no speech detected)';
    });
    transcriptSection.style.display = '';
    transcriptSection.open = true;
    const durationMs = lastAudioDurationSec * 1000;
    renderResult(msg.frames, msg.beams, durationMs, msg.transcript);
    setStatus('');
    // Save to history — fetch audio from Cache API (off-heap) only when needed
    getSessionAudioBlob()
      .then(blob => blob?.arrayBuffer() ?? Promise.resolve(undefined))
      .then(audioBuf => saveHistoryEntry({
        datetime:            new Date().toISOString(),
        durationSec:         lastAudioDurationSec,
        transcript:          msg.transcript,
        modelId:             modelSelect.value,
        audioBuf,
        inferenceDurationMs,
      }))
      .then(() => loadAndRenderHistory())
      .catch(() => {});
    return;
  }

  if (msg.type === 'error') {
    modelStatus.className = 'error';
    setStatus(`Error: ${msg.message}`);
    recordBtn.disabled    = false;
    uploadBtn.disabled    = false;
    rerunBtn.disabled     = !hasSessionAudio;
    activeDownloadModelId = null;
    return;
  }
};

// ── Model loading ─────────────────────────────────────────────────────────────

function loadModel(autoRerun = false) {
  const id = modelSelect.value;
  modelStatus.textContent = 'Loading…';
  modelStatus.className   = '';
  recordBtn.disabled      = true;
  uploadBtn.disabled      = true;
  rerunBtn.disabled       = true;
  pendingRerun            = autoRerun;
  activeDownloadModelId   = id;
  setStatus('');
  // Clear any result from the previous model.
  resultModelEl.style.display     = 'none';
  transcriptSection.style.display = 'none';
  transcriptSection.open          = false;
  beamsSection.style.display      = 'none';
  framesSection.style.display     = 'none';
  send({ type: 'load', modelId: id });
}

modelSelect.addEventListener('change', () => loadModel(hasSessionAudio));
loadModel();

// ── Mobile detection ──────────────────────────────────────────────────────────

function isMobileDevice(): boolean {
  return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
}

/** Max recording duration in seconds. Cap on mobile to avoid WASM OOM. */
const MAX_AUDIO_SEC = isMobileDevice() ? 20 : 120;

// ── Session audio — stored in Cache API to keep PCM out of the JS heap ────────
//
// After decoding, PCM lives only as a local variable inside processAudio().
// The encoded WebM blob is stored in Cache Storage under a fixed session URL
// so the SW can serve it (keeping it off-heap) and re-run can fetch it.

const AUDIO_CACHE = 'zipa-session-audio';
const AUDIO_SESSION_URL = '/zipa-onnx-inference/session/current-audio';

async function storeSessionAudio(blob: Blob): Promise<void> {
  if (!('caches' in window)) return;
  const cache = await caches.open(AUDIO_CACHE);
  await cache.put(AUDIO_SESSION_URL, new Response(blob, {
    headers: { 'Content-Type': blob.type || 'audio/webm' },
  }));
}

async function getSessionAudioBlob(): Promise<Blob | null> {
  if (!('caches' in window)) return null;
  const cache = await caches.open(AUDIO_CACHE);
  const resp = await cache.match(AUDIO_SESSION_URL);
  return resp ? resp.blob() : null;
}

// ── Audio state ───────────────────────────────────────────────────────────────

let mediaRecorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let stream: MediaStream | null = null;
let hasSessionAudio = false;   // true when Cache API holds current-audio
let lastBlobUrl: string | null = null;
let lastAudioDurationSec = 0;
let inferStartMs = 0;

// ── Recording ─────────────────────────────────────────────────────────────────

async function startRecording() {
  recordBtn.textContent = '⏳ …';
  recordBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    recordBtn.textContent = '🎤 Record';
    recordBtn.disabled = false;
    showError(`Microphone access denied: ${err}`);
    return;
  }
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.start();
  recordBtn.textContent = '⏹ Stop';
  recordBtn.disabled = false;
}

async function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  return new Promise<void>(resolve => {
    mediaRecorder!.onstop = async () => {
      stream?.getTracks().forEach(t => t.stop());
      stream = null;
      const blob = new Blob(chunks, { type: 'audio/webm' });
      chunks = [];
      if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
      lastBlobUrl = URL.createObjectURL(blob);
      playbackAudio.src = lastBlobUrl;
      playbackSection.style.display = '';
      await storeSessionAudio(blob);
      hasSessionAudio = true;
      await processAudio(blob);
      resolve();
    };
    mediaRecorder!.stop();
  });
}

// Simple click toggle: first click = start, second click = stop.
recordBtn.addEventListener('click', () => {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    startRecording();
  } else {
    stopRecording();
  }
});

// ── File upload ───────────────────────────────────────────────────────────────

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  fileInput.value = '';
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
  lastBlobUrl = URL.createObjectURL(file);
  playbackAudio.src = lastBlobUrl;
  playbackSection.style.display = '';
  await storeSessionAudio(file);
  hasSessionAudio = true;
  await processAudio(file);
});

// ── Audio processing ──────────────────────────────────────────────────────────

async function processAudio(blob: Blob) {
  setStatus('Processing audio…');
  const arrayBuffer  = await blob.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 16000 });
  // Pass a copy so arrayBuffer is not detached by decodeAudioData.
  const decoded      = await audioContext.decodeAudioData(arrayBuffer.slice(0));
  audioContext.close();

  // Cap duration on mobile to prevent WASM OOM.
  const duration       = Math.min(decoded.duration, MAX_AUDIO_SEC);
  lastAudioDurationSec = duration;

  const numSamples = Math.ceil(duration * 16000);
  const offline    = new OfflineAudioContext(1, numSamples, 16000);
  const source     = offline.createBufferSource();
  source.buffer    = decoded;
  source.connect(offline.destination);
  source.start();
  const resampled = await offline.startRendering();
  // audio is a local variable — PCM stays off the module heap, GC'd after send
  const audio = resampled.getChannelData(0);

  if (decoded.duration > MAX_AUDIO_SEC) {
    setStatus(`Audio trimmed to ${MAX_AUDIO_SEC} s on mobile. Running inference…`);
  } else {
    setStatus('Running inference…');
  }
  inferStartMs = performance.now();
  appendCrashLog('infer-start', `model=${modelSelect.value} dur=${duration.toFixed(1)}s`).catch(() => {});
  send({ type: 'infer', audio });
  // audio goes out of scope here — PCM Float32Array can be GC'd
}

rerunBtn.addEventListener('click', async () => {
  if (!hasSessionAudio) return;
  rerunBtn.disabled = true;
  const blob = await getSessionAudioBlob();
  if (!blob) { rerunBtn.disabled = false; return; }
  await processAudio(blob);
});

// ── History ───────────────────────────────────────────────────────────────────

function formatDatetime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' });
  } catch { return iso; }
}

function renderHistory(entries: HistoryEntry[]) {
  if (entries.length === 0) return;
  historySection.style.display = '';
  const table = document.createElement('table');
  table.className = 'history-table';
  table.innerHTML = `<thead><tr>
    <th>Date &amp; time</th><th>Audio</th><th>Inference</th><th>Model</th><th>IPA transcript</th><th></th>
  </tr></thead>`;
  const tbody = document.createElement('tbody');
  for (const e of entries) {
    const tr = document.createElement('tr');
    const inferStr = e.inferenceDurationMs !== undefined
      ? `${(e.inferenceDurationMs / 1000).toFixed(1)} s`
      : '—';
    tr.innerHTML = `
      <td class="hist-time">${escHtml(formatDatetime(e.datetime))}</td>
      <td class="hist-dur">${e.durationSec.toFixed(1)} s</td>
      <td class="hist-infer">${inferStr}</td>
      <td class="hist-model">${escHtml(e.modelId)}</td>
      <td class="hist-ipa">${escHtml(e.transcript || '—')}</td>
      <td class="hist-load"></td>`;
    const btn = document.createElement('button');
    btn.textContent = 'Load';
    btn.className = 'load-hist-btn';
    btn.addEventListener('click', () => loadHistoryEntry(e));
    tr.querySelector('.hist-load')!.appendChild(btn);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  historyList.innerHTML = '';
  historyList.appendChild(table);
}

async function loadHistoryEntry(e: HistoryEntry) {
  const audioBuf = e.audioBuf
    ?? (e.id !== undefined ? await loadHistoryAudio(e.id).catch(() => null) : null);
  if (!audioBuf) { setStatus('No audio stored for this entry.'); return; }
  const blob = new Blob([audioBuf], { type: 'audio/webm' });
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
  lastBlobUrl = URL.createObjectURL(blob);
  playbackAudio.src = lastBlobUrl;
  playbackSection.style.display = '';
  await storeSessionAudio(blob);
  hasSessionAudio = true;
  if (modelSelect.value !== e.modelId && optionByModelId.has(e.modelId)) {
    modelSelect.value = e.modelId;
    loadModel(true); // pendingRerun path: processAudio called when model-ready fires
  } else {
    await processAudio(blob);
  }
}

async function loadAndRenderHistory() {
  const entries = await loadHistory().catch(() => [] as HistoryEntry[]);
  renderHistory(entries);
}

loadAndRenderHistory();

// ── Rendering ─────────────────────────────────────────────────────────────────

function probClass(p: number): string {
  return p >= 0.7 ? 'prob-high' : p >= 0.3 ? 'prob-mid' : 'prob-low';
}

function addSpacesFromTranscript(beamText: string, transcript: string): string {
  if (beamText.includes(' ') || !transcript.includes(' ')) return beamText;
  const transcriptNoSpaces = transcript.replace(/ /g, '');
  if (transcriptNoSpaces.length === 0 || beamText.length === 0) return beamText;
  const spaceFractions: number[] = [];
  let charsSeen = 0;
  for (const ch of transcript) {
    if (ch === ' ') spaceFractions.push(charsSeen / transcriptNoSpaces.length);
    else charsSeen++;
  }
  let result = '';
  let nextFracIdx = 0;
  for (let i = 0; i < beamText.length; i++) {
    const frac = i / beamText.length;
    while (nextFracIdx < spaceFractions.length && frac >= spaceFractions[nextFracIdx]) {
      result += ' ';
      nextFracIdx++;
    }
    result += beamText[i];
  }
  return result;
}

function renderResult(frames: FrameOut[], beams: BeamOut[], audioDurationMs = 0, transcript = '') {
  // Render beam rows and frame table without IPA tooltips first (fast path),
  // then patch in IPA HTML once the lazy desc is loaded.
  beamsSection.style.display = '';
  beamsEl.innerHTML = '';

  const beamSpans: HTMLElement[] = [];
  for (const b of beams) {
    if (b.prob < 0.01) continue;
    const pct  = (b.prob * 100).toFixed(1);
    const text = addSpacesFromTranscript(b.text, transcript);
    const row  = document.createElement('div');
    row.className = 'beam-row';
    const textSpan = document.createElement('span');
    textSpan.className = 'beam-text';
    textSpan.textContent = text || '(empty)';
    row.innerHTML = `<span class="beam-prob">${pct}%</span><div class="beam-bar-bg"><div class="beam-bar-fill" style="width:${pct}%"></div></div>`;
    row.appendChild(textSpan);
    beamsEl.appendChild(row);
    if (text) beamSpans.push(textSpan);
  }

  framesSection.style.display = '';
  framesBody.innerHTML = '';

  const range       = findActiveRange(frames);
  const firstActive = range?.first ?? -1;
  const lastActive  = range?.last  ?? -1;
  const total       = frames.length;
  const shown       = range ? lastActive - firstActive + 1 : 0;
  const skipped     = total - shown;
  framesLabel.textContent =
    `Frame-level predictions — showing ${shown} of ${total} frames` +
    (skipped > 0 ? ` (${skipped} blank frames trimmed from edges)` : '');

  if (firstActive === -1) return;

  const msPerFrame = frames.length > 0 && audioDurationMs > 0
    ? audioDurationMs / frames.length
    : 40;

  const symCells: Array<{ el: HTMLElement; sym: string }> = [];
  for (let i = firstActive; i <= lastActive; i++) {
    const { top5 } = frames[i];
    const ms = Math.round(i * msPerFrame);
    const tr = document.createElement('tr');
    if (!isActiveFrame(frames[i])) {
      tr.className = 'blank-frame';
      tr.innerHTML = `<td class="f-idx">${i}</td><td class="f-ms">${ms}</td><td class="tok blank-sym" colspan="3">–</td>`;
      framesBody.appendChild(tr);
      continue;
    }
    const topProb = top5[0].prob;
    const topCell = document.createElement('td');
    topCell.className = 'tok';
    topCell.textContent = top5[0].sym;
    symCells.push({ el: topCell, sym: top5[0].sym });

    const altCell = document.createElement('td');
    const altSpans: Array<{ el: HTMLSpanElement; sym: string }> = [];
    for (const t of top5.slice(1)) {
      if (t.prob <= 0.01) continue;
      const sp = document.createElement('span');
      sp.className = 'alt';
      sp.textContent = `${t.sym}:${(t.prob * 100).toFixed(0)}%`;
      altCell.appendChild(sp);
      altSpans.push({ el: sp, sym: t.sym });
    }

    tr.innerHTML = `<td class="f-idx">${i}</td><td class="f-ms">${ms}</td>`;
    tr.appendChild(topCell);
    tr.innerHTML += `<td class="${probClass(topProb)}">${(topProb * 100).toFixed(0)}%</td>`;
    tr.appendChild(altCell);
    framesBody.appendChild(tr);

    // Patch alt spans with IPA HTML asynchronously
    if (altSpans.length > 0) {
      getIpaDesc().then(desc => {
        for (const { el, sym } of altSpans) {
          el.innerHTML = `${ipaSymHtml(sym, desc)}:${el.textContent!.split(':')[1]}`;
        }
      }).catch(() => {});
    }
  }

  // Patch beam text spans and top sym cells with IPA HTML asynchronously
  getIpaDesc().then(desc => {
    for (const sp of beamSpans) {
      sp.innerHTML = ipaStringHtml(sp.textContent ?? '', desc);
    }
    for (const { el, sym } of symCells) {
      el.innerHTML = ipaSymHtml(sym, desc);
    }
  }).catch(() => {});
}

function setStatus(msg: string) { statusLine.textContent = msg; }

function showError(msg: string) {
  errorLog.style.display = 'block';
  errorLog.textContent += msg + '\n';
}

window.addEventListener('pagehide', () => {
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
});


function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── IPA symbol tooltips (lazy — desc loaded on first use) ─────────────────────

/** Wrap a single IPA symbol in a linked span with tooltip data. */
function ipaSymHtml(sym: string, desc: Record<string, IpaEntry>): string {
  const info = desc[sym];
  if (!info) return escHtml(sym);
  const tip = info.desc ? `${info.name} — ${info.desc.split('. ')[0]}.` : info.name;
  return `<a class="ipa-sym" href="${escHtml(info.url)}" target="_blank" rel="noopener" ` +
         `data-sym="${escHtml(sym)}" data-name="${escHtml(info.name)}" ` +
         `data-desc="${escHtml(info.desc.split('. ')[0] + '.')}" ` +
         `data-tip="${escHtml(tip)}">${escHtml(sym)}</a>`;
}

/** Render an IPA string with each symbol wrapped for tooltip support. */
function ipaStringHtml(text: string, desc: Record<string, IpaEntry>): string {
  let html = '';
  for (const ch of text) {
    if (ch === ' ') {
      html += ' ';
    } else if (/\p{M}/u.test(ch)) {
      // Combining diacritics: emit as plain text so they visually attach to prev char
      html += escHtml(ch);
    } else {
      html += ipaSymHtml(ch, desc);
    }
  }
  return html;
}

// ── IPA overview table — loaded on demand when user expands the <details> ──────

const ipaOverviewSection = document.getElementById('ipa-overview-section') as HTMLDetailsElement | null;
let ipaOverviewRendered = false;

async function renderIpaOverview() {
  if (ipaOverviewRendered) return;
  ipaOverviewRendered = true;

  const tbody = document.getElementById('ipa-table-body') as HTMLTableSectionElement;
  if (!tbody) return;

  const desc = await getIpaDesc();

  // Lazy-load neighbours module too
  if (!_computeAllNeighbours || !_symAnchorId) {
    const mod = await import('./ipa-neighbours.js');
    _computeAllNeighbours = mod.computeAllNeighbours;
    _symAnchorId = mod.symAnchorId;
  }
  const neighbours = _computeAllNeighbours!(desc);

  for (const [sym, entry] of Object.entries(desc)) {
    if (!entry) continue;
    const id  = _symAnchorId!(sym);
    const shortDesc = entry.desc ? entry.desc.split('. ')[0] + '.' : '';
    const nbrs = neighbours[sym] ?? [];
    const nbrsHtml = nbrs
      .map(n => {
        const nEntry = desc[n.sym];
        const nTip = nEntry ? escHtml(nEntry.name) : '';
        return `<a class="nbr-link" href="#${_symAnchorId!(n.sym)}" title="${nTip}">${escHtml(n.sym)}<span class="nbr-score">${(n.score * 100).toFixed(0)}%</span></a>`;
      })
      .join('');

    const tr = document.createElement('tr');
    tr.id = id;
    tr.innerHTML = `
      <td class="ov-sym"><a class="ipa-sym" href="${escHtml(entry.url)}" target="_blank" rel="noopener"
          data-sym="${escHtml(sym)}" data-name="${escHtml(entry.name)}"
          data-desc="${escHtml(shortDesc)}"
          data-tip="${escHtml(entry.name)}">${escHtml(sym)}</a></td>
      <td class="ov-name">${escHtml(entry.name)}</td>
      <td class="ov-desc">${escHtml(shortDesc)}</td>
      <td class="ov-nbrs">${nbrsHtml}</td>`;
    tbody.appendChild(tr);
  }
}

// Load IPA data only when the user opens the <details> box.
if (ipaOverviewSection) {
  ipaOverviewSection.addEventListener('toggle', () => {
    if (ipaOverviewSection.open) renderIpaOverview().catch(() => {});
  });
}

// ── Browser / package info ────────────────────────────────────────────────────

const browserInfoRows = document.getElementById('browser-info-rows') as HTMLDivElement | null;

function biRow(label: string, value: string, cls = 'bi-val'): string {
  return `<div><span class="bi-label">${label}</span><span class="${cls}">${value}</span></div>`;
}

function detectBrowser(): string {
  const ua = navigator.userAgent;
  // Order matters — check most specific first
  if (/EdgA?\/[\d.]+/.test(ua))  return 'Edge '  + (ua.match(/Edg[A]?\/([.\d]+)/)?.[1] ?? '');
  if (/OPR\/[\d.]+/.test(ua))    return 'Opera '  + (ua.match(/OPR\/([.\d]+)/)?.[1] ?? '');
  if (/Chrom(?:e|ium)\/[\d.]+/.test(ua)) return 'Chrome ' + (ua.match(/Chrom(?:e|ium)\/([.\d]+)/)?.[1] ?? '');
  if (/Firefox\/[\d.]+/.test(ua)) return 'Firefox ' + (ua.match(/Firefox\/([.\d]+)/)?.[1] ?? '');
  if (/Safari\/[\d.]+/.test(ua)) return 'Safari '  + (ua.match(/Version\/([.\d]+)/)?.[1] ?? '');
  return navigator.userAgent.slice(0, 40) + '…';
}

function detectOS(): string {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod/.test(ua))  return 'iOS '  + (ua.match(/OS ([\d_]+)/)?.[1]?.replace(/_/g,'.') ?? '');
  if (/Android/.test(ua))           return 'Android ' + (ua.match(/Android ([.\d]+)/)?.[1] ?? '');
  if (/Mac OS X/.test(ua))          return 'macOS ' + (ua.match(/Mac OS X ([\d_]+)/)?.[1]?.replace(/_/g,'.') ?? '');
  if (/Windows NT/.test(ua))        return 'Windows ' + (ua.match(/Windows NT ([\d.]+)/)?.[1] ?? '');
  if (/Linux/.test(ua))             return 'Linux';
  return navigator.platform || 'unknown';
}

/** Detect WASM SIMD support by trying to validate a minimal SIMD module. */
async function detectSimd(): Promise<boolean> {
  try {
    // Minimal WASM module with a SIMD v128.const instruction
    const bytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
      0x03, 0x02, 0x01, 0x00,
      0x0a, 0x0f, 0x01, 0x0d, 0x00,
        0xfd, 0x0c,  // v128.const
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x0b,
    ]);
    return WebAssembly.validate(bytes);
  } catch { return false; }
}

async function populateBrowserInfo() {
  if (!browserInfoRows) return;
  const simd = await detectSimd();
  let html = '';
  html += biRow('Browser',          detectBrowser());
  html += biRow('OS',               detectOS());
  html += biRow('ORT version',      '1.18.0');
  html += `<div><span class="bi-label">Execution prov.</span><span class="bi-val bi-prov">wasm</span></div>`;
  html += biRow('WASM SIMD',        simd ? 'supported' : 'not supported', simd ? 'bi-ok' : 'bi-no');
  browserInfoRows.innerHTML = html;
}

populateBrowserInfo().catch(() => {});

// Mobile popup for IPA symbols
const ipaPopup = document.getElementById('ipa-popup') as HTMLDivElement;

document.addEventListener('click', (e: PointerEvent) => {
  const target = e.target as HTMLElement;
  const sym = target.closest<HTMLAnchorElement>('.ipa-sym');
  if (!sym) {
    ipaPopup.style.display = 'none';
    return;
  }
  // On touch devices: show popup instead of following link
  if (e.pointerType === 'touch' || ('ontouchstart' in window && navigator.maxTouchPoints > 0)) {
    e.preventDefault();
    const ch   = sym.dataset.sym  ?? sym.textContent ?? '';
    const name = sym.dataset.name ?? '';
    const desc = sym.dataset.desc ?? '';
    const url  = sym.href;
    ipaPopup.innerHTML =
      `<span class="pop-sym">${escHtml(ch)}</span><span class="pop-name">${escHtml(name)}</span>` +
      `<span class="pop-desc">${escHtml(desc)}</span>` +
      `<a href="${escHtml(url)}" target="_blank" rel="noopener">→ Wikipedia</a>`;
    ipaPopup.style.display = 'block';
  }
}, true);
