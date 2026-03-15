import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import os from 'os';

export type ModelType = 'ctc' | 'transducer';
export type Precision = 'fp32' | 'fp16' | 'int8';

export interface ModelDef {
  id: string;
  hfRepo: string;
  type: ModelType;
  params: string;
  steps: string;
  desc: string;
}

export const MODELS: ModelDef[] = [
  // CTC models
  { id: 'zipa-small-crctc-300k',                 hfRepo: 'anyspeech/zipa-small-crctc-300k',                 type: 'ctc',         params: '64M',  steps: '300k', desc: 'Small CRCTC' },
  { id: 'zipa-small-crctc-500k',                 hfRepo: 'anyspeech/zipa-small-crctc-500k',                 type: 'ctc',         params: '64M',  steps: '500k', desc: 'Small CRCTC' },
  { id: 'zipa-large-crctc-300k',                 hfRepo: 'anyspeech/zipa-large-crctc-300k',                 type: 'ctc',         params: '300M', steps: '300k', desc: 'Large CRCTC' },
  { id: 'zipa-large-crctc-500k',                 hfRepo: 'anyspeech/zipa-large-crctc-500k',                 type: 'ctc',         params: '300M', steps: '500k', desc: 'Large CRCTC' },
  { id: 'zipa-small-crctc-ns-700k',              hfRepo: 'anyspeech/zipa-small-crctc-ns-700k',              type: 'ctc',         params: '64M',  steps: '700k', desc: 'Small CRCTC noise-robust' },
  { id: 'zipa-large-crctc-ns-800k',              hfRepo: 'anyspeech/zipa-large-crctc-ns-800k',              type: 'ctc',         params: '300M', steps: '800k', desc: 'Large CRCTC noise-robust' },
  { id: 'zipa-small-crctc-ns-no-diacritics-700k',hfRepo: 'anyspeech/zipa-small-crctc-ns-no-diacritics-700k',type: 'ctc',        params: '64M',  steps: '700k', desc: 'Small CRCTC noise-robust, no diacritics' },
  { id: 'zipa-large-crctc-ns-no-diacritics-780k',hfRepo: 'anyspeech/zipa-large-crctc-ns-no-diacritics-780k',type: 'ctc',        params: '300M', steps: '780k', desc: 'Large CRCTC noise-robust, no diacritics' },
  // Transducer models
  { id: 'zipa-small-noncausal-300k',             hfRepo: 'anyspeech/zipa-small-noncausal-300k',             type: 'transducer',  params: '65M',  steps: '300k', desc: 'Small Transducer' },
  { id: 'zipa-large-noncausal-300k',             hfRepo: 'anyspeech/zipa-large-noncausal-300k',             type: 'transducer',  params: '302M', steps: '300k', desc: 'Large Transducer' },
  { id: 'zipa-small-noncausal-500k',             hfRepo: 'anyspeech/zipa-small-noncausal-500k',             type: 'transducer',  params: '65M',  steps: '500k', desc: 'Small Transducer' },
  { id: 'zipa-large-noncausal-500k',             hfRepo: 'anyspeech/zipa-large-noncausal-500k',             type: 'transducer',  params: '302M', steps: '500k', desc: 'Large Transducer' },
];

export const CACHE_DIR = path.join(os.homedir(), '.cache', 'zipa');

function modelCacheDir(modelId: string): string {
  return path.join(CACHE_DIR, 'models', modelId);
}

function ctcModelPath(modelId: string, precision: Precision): string {
  const suffix = precision === 'fp32' ? '' : `.${precision}`;
  return path.join(modelCacheDir(modelId), 'exp', `model${suffix}.onnx`);
}

function transducerModelPaths(modelId: string, precision: Precision): { encoder: string; decoder: string; joiner: string } {
  const suffix = precision === 'fp32' ? '.onnx' : `.${precision}.onnx`;
  const dir = path.join(modelCacheDir(modelId), 'exp');
  return {
    encoder: path.join(dir, `encoder${suffix}`),
    decoder: path.join(dir, `decoder${suffix}`),
    joiner:  path.join(dir, `joiner${suffix}`),
  };
}

export function isModelDownloaded(modelId: string, precision: Precision = 'fp32'): boolean {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) return false;
  if (model.type === 'ctc') {
    return fs.existsSync(ctcModelPath(modelId, precision));
  }
  const { encoder, decoder, joiner } = transducerModelPaths(modelId, precision);
  return fs.existsSync(encoder) && fs.existsSync(decoder) && fs.existsSync(joiner);
}

export function getModelOnnxPath(modelId: string, precision: Precision = 'fp32'): string {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) throw new Error(`Unknown model: ${modelId}`);
  if (model.type !== 'ctc') throw new Error(`Use getTransducerPaths() for transducer models`);
  return ctcModelPath(modelId, precision);
}

export function getTransducerPaths(modelId: string, precision: Precision = 'fp32') {
  return transducerModelPaths(modelId, precision);
}

function hfDownloadUrl(hfRepo: string, filePath: string): string {
  return `https://huggingface.co/${hfRepo}/resolve/main/${filePath}`;
}

async function downloadFile(url: string, destPath: string): Promise<void> {
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const tmpPath = destPath + '.tmp';

  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(tmpPath);
    let redirectCount = 0;

    function request(url: string) {
      const mod = url.startsWith('https') ? https : http;
      mod.get(url, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307 || res.statusCode === 308) {
          if (redirectCount++ > 5) { reject(new Error('Too many redirects')); return; }
          request(res.headers.location!);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          return;
        }
        const total = parseInt(res.headers['content-length'] ?? '0', 10);
        let received = 0;
        res.on('data', (chunk: Buffer) => {
          received += chunk.length;
          if (total > 0) {
            const pct = Math.floor((received / total) * 100);
            process.stderr.write(`\r  ${pct}% (${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB)`);
          }
        });
        res.pipe(file);
        res.on('end', () => {
          process.stderr.write('\n');
          file.close();
          fs.renameSync(tmpPath, destPath);
          resolve();
        });
        res.on('error', reject);
      }).on('error', reject);
    }

    request(url);
  });
}

// Discover transducer ONNX filenames via HF API
async function discoverTransducerFiles(hfRepo: string, precision: Precision): Promise<{ encoder: string; decoder: string; joiner: string } | null> {
  const apiUrl = `https://huggingface.co/api/models/${hfRepo}`;
  return new Promise((resolve) => {
    https.get(apiUrl, { headers: { 'User-Agent': 'zipa-onnx-inference' } }, (res) => {
      let data = '';
      res.on('data', (c: Buffer) => (data += c.toString()));
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          const siblings: Array<{ rfilename: string }> = json.siblings ?? [];
          const suffix = precision === 'fp32' ? '.onnx' : `.${precision}.onnx`;
          const enc = siblings.find(f => f.rfilename.startsWith('exp/encoder') && f.rfilename.endsWith(suffix) && !f.rfilename.includes('fp16') && !f.rfilename.includes('int8') || f.rfilename.endsWith(suffix));
          const dec = siblings.find(f => f.rfilename.startsWith('exp/decoder') && f.rfilename.endsWith(suffix));
          const joi = siblings.find(f => f.rfilename.startsWith('exp/joiner') && f.rfilename.endsWith(suffix));
          if (enc && dec && joi) {
            resolve({ encoder: enc.rfilename, decoder: dec.rfilename, joiner: joi.rfilename });
          } else {
            resolve(null);
          }
        } catch {
          resolve(null);
        }
      });
      res.on('error', () => resolve(null));
    }).on('error', () => resolve(null));
  });
}

export async function downloadModel(modelId: string, precision: Precision = 'fp32'): Promise<void> {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) throw new Error(`Unknown model: ${modelId}`);

  const cacheDir = modelCacheDir(modelId);
  fs.mkdirSync(path.join(cacheDir, 'exp'), { recursive: true });

  if (model.type === 'ctc') {
    const suffix = precision === 'fp32' ? '' : `.${precision}`;
    const remoteFile = `exp/model${suffix}.onnx`;
    const localPath = ctcModelPath(modelId, precision);
    console.error(`Downloading ${remoteFile} ...`);
    await downloadFile(hfDownloadUrl(model.hfRepo, remoteFile), localPath);
  } else {
    // Transducer: discover filenames from HF API
    console.error(`Discovering transducer model files from HF API...`);
    const files = await discoverTransducerFiles(model.hfRepo, precision);
    if (!files) {
      throw new Error(`Could not discover transducer ONNX files for ${modelId} (${precision}). Check ${hfDownloadUrl(model.hfRepo, 'exp/')} manually.`);
    }
    const paths = transducerModelPaths(modelId, precision);
    for (const [part, remotePath] of [['encoder', files.encoder], ['decoder', files.decoder], ['joiner', files.joiner]] as const) {
      const localPath = paths[part as keyof typeof paths];
      console.error(`Downloading ${remotePath} ...`);
      await downloadFile(hfDownloadUrl(model.hfRepo, remotePath), localPath);
    }
  }

  console.error(`Model ${modelId} (${precision}) cached at ${cacheDir}`);
}

export function listModels(): void {
  const COL = {
    id:     28,
    type:   12,
    params:  6,
    steps:   6,
    desc:   40,
  };

  const header =
    'ID'.padEnd(COL.id) +
    'TYPE'.padEnd(COL.type) +
    'PARAMS'.padEnd(COL.params) +
    'STEPS'.padEnd(COL.steps) +
    'DESC';
  console.log(header);
  console.log('-'.repeat(header.length + 10));

  for (const m of MODELS) {
    const downloaded = (['fp32', 'fp16', 'int8'] as Precision[]).filter(p => isModelDownloaded(m.id, p));
    const badge = downloaded.length > 0 ? ` [${downloaded.join(',')}]` : '';
    const line =
      m.id.padEnd(COL.id) +
      m.type.padEnd(COL.type) +
      m.params.padEnd(COL.params) +
      m.steps.padEnd(COL.steps) +
      m.desc + badge;
    if (downloaded.length > 0) {
      // Bold/green via ANSI
      console.log(`\x1b[32m${line}\x1b[0m`);
    } else {
      console.log(line);
    }
  }
  console.log('\nDownloaded models shown in green with precision tags.');
}
