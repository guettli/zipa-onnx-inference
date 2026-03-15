#!/usr/bin/env tsx
import fs from 'fs';
import readline from 'readline';
import {
  MODELS,
  listModels,
  isModelDownloaded,
  downloadModel,
  getModelOnnxPath,
  type Precision,
} from './models.js';
import { loadAudio } from './audio.js';
import { runCtcInference } from './inference.js';
import { printDebugOutput } from './output.js';

function printHelp() {
  console.log(`
Usage:
  tsx src/cli.ts <audio-file> [options]
  tsx src/cli.ts --list-models
  tsx src/cli.ts --download <model-id> [precision]

Options:
  --model <id>           Model to use (default: zipa-small-crctc-300k)
  --precision <p>        fp32 | fp16 | int8  (default: fp32)
  --list-models          Show all available models; downloaded ones are green
  --download <id> [p]    Download a model to ~/.cache/zipa/

Examples:
  tsx src/cli.ts audio.wav
  tsx src/cli.ts audio.wav --model zipa-large-crctc-500k --precision fp16
  tsx src/cli.ts --list-models
  tsx src/cli.ts --download zipa-small-crctc-300k fp32
`);
}

function parseArgs(argv: string[]) {
  const args = argv.slice(2);
  const out: {
    audioFile?: string;
    modelId: string;
    precision: Precision;
    listModels: boolean;
    download?: string;
    downloadPrecision: Precision;
    help: boolean;
  } = {
    modelId: 'zipa-small-crctc-300k',
    precision: 'fp32',
    listModels: false,
    downloadPrecision: 'fp32',
    help: false,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--help' || a === '-h') { out.help = true; }
    else if (a === '--list-models') { out.listModels = true; }
    else if (a === '--download') { out.download = args[++i]; out.downloadPrecision = (args[i + 1] as Precision) ?? 'fp32'; if (['fp32','fp16','int8'].includes(args[i+1])) i++; }
    else if (a === '--model') { out.modelId = args[++i]; }
    else if (a === '--precision') { out.precision = args[++i] as Precision; }
    else if (!a.startsWith('--')) { out.audioFile = a; }
  }
  return out;
}

async function confirm(question: string): Promise<boolean> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise(resolve => {
    rl.question(`${question} [y/N] `, answer => {
      rl.close();
      resolve(answer.trim().toLowerCase() === 'y');
    });
  });
}

async function main() {
  const opts = parseArgs(process.argv);

  if (opts.help) { printHelp(); process.exit(0); }

  if (opts.listModels) { listModels(); process.exit(0); }

  if (opts.download) {
    const model = MODELS.find(m => m.id === opts.download);
    if (!model) {
      console.error(`Unknown model: ${opts.download}`);
      console.error('Run --list-models to see available models.');
      process.exit(1);
    }
    if (model.type === 'transducer') {
      console.error('Note: transducer inference not yet implemented. Downloading files only.');
    }
    await downloadModel(opts.download, opts.downloadPrecision);
    process.exit(0);
  }

  if (!opts.audioFile) {
    printHelp();
    process.exit(1);
  }

  if (!fs.existsSync(opts.audioFile)) {
    console.error(`Audio file not found: ${opts.audioFile}`);
    process.exit(1);
  }

  const model = MODELS.find(m => m.id === opts.modelId);
  if (!model) {
    console.error(`Unknown model: ${opts.modelId}`);
    console.error('Run --list-models to see available models.');
    process.exit(1);
  }

  if (model.type === 'transducer') {
    console.error('Transducer inference is not yet implemented. Please choose a CRCTC model.');
    process.exit(1);
  }

  // Check / download model
  if (!isModelDownloaded(opts.modelId, opts.precision)) {
    console.error(`Model ${opts.modelId} (${opts.precision}) is not downloaded.`);
    const yes = await confirm(`Download it now? (~hundreds of MB)`);
    if (!yes) { console.error('Aborted.'); process.exit(1); }
    await downloadModel(opts.modelId, opts.precision);
  }

  const modelPath = getModelOnnxPath(opts.modelId, opts.precision);

  console.error(`Loading audio: ${opts.audioFile}`);
  const audio = loadAudio(opts.audioFile);
  console.error(`Audio: ${audio.length} samples (${(audio.length / 16000).toFixed(2)}s)`);

  console.error(`Running inference with ${opts.modelId} (${opts.precision})...`);
  const result = await runCtcInference(modelPath, audio);
  console.error(`Output frames: ${result.frames.length}  Fbank frames: ${result.numFbankFrames}`);
  console.error('');

  printDebugOutput(result.frames, audio, result.ipaText);
}

main().catch(err => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
