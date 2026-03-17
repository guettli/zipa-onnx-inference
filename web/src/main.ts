import type { WorkerInMsg, WorkerOutMsg, FrameOut, BeamOut } from './types.js';
import { MODELS, modelUrl } from './types.js';
import { isModelCached, getPartialDownload, saveRecording, getRecording,
         pruneStalePartials, saveHistoryEntry, loadHistory } from './model-cache.js';
import type { HistoryEntry } from './model-cache.js';
import { isActiveFrame, findActiveRange } from './frame-utils.js';

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

// ── Populate model selector ───────────────────────────────────────────────────

const optionByModelId = new Map<string, HTMLOptionElement>();

for (const m of MODELS) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label;
  optionByModelId.set(m.id, opt);
  modelSelect.appendChild(opt);
}

async function refreshDropdownCacheStatus() {
  for (const m of MODELS) {
    const url = modelUrl(m);
    const opt = optionByModelId.get(m.id)!;
    try {
      if (await isModelCached(url)) {
        opt.textContent = `${m.label} (downloaded)`;
      } else {
        const partial = await getPartialDownload(url);
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
  rerunBtn.disabled   = disabled || lastAudio === null;
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
    if (pendingRerun && lastAudio) {
      pendingRerun = false;
      rerunBtn.disabled = true;
      setStatus('Running inference…');
      send({ type: 'infer', audio: lastAudio });
    } else {
      rerunBtn.disabled = lastAudio === null;
      setStatus('');
    }
    return;
  }

  if (msg.type === 'result') {
    recordBtn.textContent = '▶ Record';
    recordBtn.classList.remove('recording');
    setInputsDisabled(false);
    resultModelEl.textContent = `Result from: ${modelSelect.value}`;
    resultModelEl.style.display = '';
    transcriptEl.textContent = msg.transcript || '(no speech detected)';
    transcriptSection.style.display = '';
    transcriptSection.open = true;
    const durationMs = lastAudio ? (lastAudio.length / 16000) * 1000 : 0;
    renderResult(msg.frames, msg.beams, durationMs, msg.transcript);
    setStatus('');
    // Save to history
    saveHistoryEntry({
      datetime:    new Date().toISOString(),
      durationSec: lastAudioDurationSec,
      transcript:  msg.transcript,
      modelId:     modelSelect.value,
    }).then(() => loadAndRenderHistory()).catch(() => {});
    return;
  }

  if (msg.type === 'error') {
    modelStatus.className = 'error';
    setStatus(`Error: ${msg.message}`);
    recordBtn.disabled    = false;
    uploadBtn.disabled    = false;
    rerunBtn.disabled     = lastAudio === null;
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
  send({ type: 'load', modelId: id });
}

modelSelect.addEventListener('change', () => loadModel(lastAudio !== null));
loadModel();

// ── Audio state ───────────────────────────────────────────────────────────────

let mediaRecorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let stream: MediaStream | null = null;
let lastAudio: Float32Array | null = null;
let lastBlobUrl: string | null = null;
let lastAudioDurationSec = 0;

// Restore last recording from IndexedDB if available.
(async () => {
  const stored = await getRecording().catch(() => null);
  if (!stored) return;
  const blob = new Blob([stored], { type: 'audio/webm' });
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
  lastBlobUrl = URL.createObjectURL(blob);
  playbackAudio.src = lastBlobUrl;
  playbackSection.style.display = '';
  try {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const decoded      = await audioContext.decodeAudioData(stored.slice(0));
    lastAudioDurationSec = decoded.duration;
    const numSamples     = Math.ceil(decoded.duration * 16000);
    const offline        = new OfflineAudioContext(1, numSamples, 16000);
    const source         = offline.createBufferSource();
    source.buffer        = decoded;
    source.connect(offline.destination);
    source.start();
    const resampled = await offline.startRendering();
    lastAudio = resampled.getChannelData(0);
    audioContext.close();
    if (!recordBtn.disabled) rerunBtn.disabled = false;
  } catch { /* non-fatal */ }
})();

// ── Recording ─────────────────────────────────────────────────────────────────

async function startRecording() {
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  }
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.start();
  recordBtn.textContent = '● Recording…';
  recordBtn.classList.add('recording');
}

async function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  return new Promise<void>(resolve => {
    mediaRecorder!.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
      lastBlobUrl = URL.createObjectURL(blob);
      playbackAudio.src = lastBlobUrl;
      playbackSection.style.display = '';
      saveRecording(await blob.arrayBuffer()).catch(() => {});
      await processAudio(blob);
      resolve();
    };
    mediaRecorder!.stop();
  });
}

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
  fileInput.value = ''; // reset so same file can be picked again
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
  lastBlobUrl = URL.createObjectURL(file);
  playbackAudio.src = lastBlobUrl;
  playbackSection.style.display = '';
  // Persist as the "last recording" for re-evaluate across reloads
  file.arrayBuffer().then(buf => saveRecording(buf).catch(() => {}));
  await processAudio(file);
});

// ── Audio processing ──────────────────────────────────────────────────────────

async function processAudio(blob: Blob) {
  setStatus('Processing audio…');
  const arrayBuffer  = await blob.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const decoded      = await audioContext.decodeAudioData(arrayBuffer);
  audioContext.close();
  lastAudioDurationSec = decoded.duration;

  const numSamples = Math.ceil(decoded.duration * 16000);
  const offline    = new OfflineAudioContext(1, numSamples, 16000);
  const source     = offline.createBufferSource();
  source.buffer    = decoded;
  source.connect(offline.destination);
  source.start();
  const resampled = await offline.startRendering();
  lastAudio       = resampled.getChannelData(0);

  setStatus('Running inference…');
  send({ type: 'infer', audio: lastAudio });
}

rerunBtn.addEventListener('click', () => {
  if (!lastAudio) return;
  rerunBtn.disabled = true;
  setStatus('Running inference…');
  send({ type: 'infer', audio: lastAudio });
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
    <th>Date &amp; time</th><th>Duration</th><th>Model</th><th>IPA transcript</th>
  </tr></thead>`;
  const tbody = document.createElement('tbody');
  for (const e of entries) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="hist-time">${escHtml(formatDatetime(e.datetime))}</td>
      <td class="hist-dur">${e.durationSec.toFixed(1)} s</td>
      <td class="hist-model">${escHtml(e.modelId)}</td>
      <td class="hist-ipa">${escHtml(e.transcript || '—')}</td>`;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  historyList.innerHTML = '';
  historyList.appendChild(table);
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
  beamsSection.style.display = '';
  beamsEl.innerHTML = '';
  for (const b of beams) {
    if (b.prob < 0.01) continue;
    const pct  = (b.prob * 100).toFixed(1);
    const text = addSpacesFromTranscript(b.text, transcript);
    const row  = document.createElement('div');
    row.className = 'beam-row';
    row.innerHTML = `
      <span class="beam-prob">${pct}%</span>
      <div class="beam-bar-bg"><div class="beam-bar-fill" style="width:${pct}%"></div></div>
      <span class="beam-text">${escHtml(text || '(empty)')}</span>`;
    beamsEl.appendChild(row);
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
    const alts = top5.slice(1)
      .filter(t => t.prob > 0.01)
      .map(t => `<span class="alt">${escHtml(t.sym)}:${(t.prob * 100).toFixed(0)}%</span>`)
      .join('');
    tr.innerHTML = `
      <td class="f-idx">${i}</td>
      <td class="f-ms">${ms}</td>
      <td class="tok">${escHtml(top5[0].sym)}</td>
      <td class="${probClass(topProb)}">${(topProb * 100).toFixed(0)}%</td>
      <td>${alts}</td>`;
    framesBody.appendChild(tr);
  }
}

function setStatus(msg: string) { statusLine.textContent = msg; }

function showError(msg: string) {
  errorLog.style.display = 'block';
  errorLog.textContent += msg + '\n';
}

window.addEventListener('pagehide', () => {
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
});

window.addEventListener('error', e => {
  showError(`${e.message}  (${e.filename}:${e.lineno}:${e.colno})`);
});

window.addEventListener('unhandledrejection', e => {
  const msg = e.reason instanceof Error ? e.reason.stack ?? e.reason.message : String(e.reason);
  showError(`Unhandled promise rejection: ${msg}`);
});

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
