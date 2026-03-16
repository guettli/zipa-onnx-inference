import type { WorkerInMsg, WorkerOutMsg, FrameOut, BeamOut } from './types.js';
import { MODELS } from './types.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const modelSelect   = document.getElementById('model-select')   as HTMLSelectElement;
const modelStatus   = document.getElementById('model-status')   as HTMLDivElement;
const progressBar   = document.getElementById('progress-bar')   as HTMLDivElement;
const progressFill  = document.getElementById('progress-fill')  as HTMLDivElement;
const recordBtn     = document.getElementById('record-btn')     as HTMLButtonElement;
const statusLine    = document.getElementById('status-line')    as HTMLDivElement;
const beamsSection  = document.getElementById('beams-section')  as HTMLElement;
const beamsEl       = document.getElementById('beams')          as HTMLDivElement;
const framesSection = document.getElementById('frames-section') as HTMLElement;
const framesBody    = document.getElementById('frames-body')    as HTMLTableSectionElement;

// ── Populate model selector ───────────────────────────────────────────────────

for (const m of MODELS) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label;
  modelSelect.appendChild(opt);
}

// ── Worker ────────────────────────────────────────────────────────────────────

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

function send(msg: WorkerInMsg) {
  worker.postMessage(msg);
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
    return;
  }

  if (msg.type === 'model-ready') {
    progressBar.style.display = 'none';
    modelStatus.textContent   = `✓ ${msg.modelId}`;
    modelStatus.className     = 'ready';
    recordBtn.disabled        = false;
    setStatus('');
    return;
  }

  if (msg.type === 'result') {
    recordBtn.textContent = '▶ Record';
    recordBtn.classList.remove('recording');
    renderResult(msg.frames, msg.beams);
    setStatus('');
    return;
  }

  if (msg.type === 'error') {
    modelStatus.className = 'error';
    setStatus(`Error: ${msg.message}`);
    recordBtn.disabled = false;
    return;
  }
};

// ── Model loading ─────────────────────────────────────────────────────────────

function loadModel() {
  const id = modelSelect.value;
  modelStatus.textContent = 'Loading…';
  modelStatus.className   = '';
  recordBtn.disabled      = true;
  setStatus('');
  send({ type: 'load', modelId: id });
}

modelSelect.addEventListener('change', loadModel);
loadModel();

// ── Recording ─────────────────────────────────────────────────────────────────

let mediaRecorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let stream: MediaStream | null = null;

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
      await processAudio(blob);
      resolve();
    };
    mediaRecorder!.stop();
  });
}

async function processAudio(blob: Blob) {
  setStatus('Processing audio…');
  const arrayBuffer  = await blob.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const decoded      = await audioContext.decodeAudioData(arrayBuffer);

  // Resample to 16 kHz mono via OfflineAudioContext
  const targetRate   = 16000;
  const duration     = decoded.duration;
  const numSamples   = Math.ceil(duration * targetRate);
  const offline      = new OfflineAudioContext(1, numSamples, targetRate);
  const source       = offline.createBufferSource();
  source.buffer      = decoded;
  source.connect(offline.destination);
  source.start();
  const resampled    = await offline.startRendering();
  const audio        = resampled.getChannelData(0);

  setStatus('Running inference…');
  send({ type: 'infer', audio });
}

// Hold-to-record on both mouse and touch
recordBtn.addEventListener('mousedown', () => startRecording());
recordBtn.addEventListener('mouseup',   () => stopRecording());
recordBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener('touchend',   e => { e.preventDefault(); stopRecording(); });

// ── Rendering ─────────────────────────────────────────────────────────────────

function probClass(p: number): string {
  return p >= 0.7 ? 'prob-high' : p >= 0.3 ? 'prob-mid' : 'prob-low';
}

function renderResult(frames: FrameOut[], beams: BeamOut[]) {
  // Beams
  beamsSection.style.display = '';
  beamsEl.innerHTML = '';
  for (const b of beams) {
    const pct = (b.prob * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'beam-row';
    row.innerHTML = `
      <span class="beam-prob">${pct}%</span>
      <div class="beam-bar-bg"><div class="beam-bar-fill" style="width:${pct}%"></div></div>
      <span class="beam-text">${escHtml(b.text || '(empty)')}</span>`;
    beamsEl.appendChild(row);
  }

  // Frames — show only non-blank frames (where blank is not top token)
  framesSection.style.display = '';
  framesBody.innerHTML = '';
  for (let i = 0; i < frames.length; i++) {
    const { top5 } = frames[i];
    if (!top5.length) continue;
    const topProb = top5[0].prob;
    if (topProb < 0.05) continue;          // skip low-signal frames
    if (top5[0].sym === '<blk>') continue; // skip blank-dominant frames

    const alts = top5.slice(1)
      .filter(t => t.prob > 0.01)
      .map(t => `<span class="alt">${escHtml(t.sym)}:${(t.prob * 100).toFixed(0)}%</span>`)
      .join('');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="f-idx">${i}</td>
      <td class="tok">${escHtml(top5[0].sym)}</td>
      <td class="${probClass(topProb)}">${(topProb * 100).toFixed(0)}%</td>
      <td>${alts}</td>`;
    framesBody.appendChild(tr);
  }
}

function setStatus(msg: string) { statusLine.textContent = msg; }

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
