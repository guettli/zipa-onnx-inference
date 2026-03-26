/**
 * Service Worker for ZIPA ONNX Inference.
 *
 * Serves ONNX model files from the Cache Storage API.
 * Models are stored under /zipa-onnx-inference/model-proxy/<encoded-id>
 * by the web worker (which has access to `caches` directly).
 *
 * Using Cache API instead of IndexedDB means:
 *  - ort.InferenceSession.create(url) can fetch the model via this SW
 *  - The model never needs to be materialised as a full JS ArrayBuffer on our side
 *  - The browser manages the disk↔memory boundary of the cached response
 */

const CACHE_NAME = 'zipa-models-v1';
const PROXY_PATH = '/zipa-onnx-inference/model-proxy/';
const AUDIO_CACHE = 'zipa-session-audio';
const AUDIO_PATH  = '/zipa-onnx-inference/session/';

self.addEventListener('install', () => {
  // Take control immediately — don't wait for old tabs to close.
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  // Claim all clients so the SW intercepts fetches on the first page load.
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', event => {
  const { pathname } = new URL(event.request.url);
  if (pathname.startsWith(PROXY_PATH)) {
    event.respondWith(
      caches.open(CACHE_NAME)
        .then(c => c.match(event.request))
        .then(r => r ?? new Response('Model not in cache — download it first.', { status: 404 }))
    );
  } else if (pathname.startsWith(AUDIO_PATH)) {
    event.respondWith(
      caches.open(AUDIO_CACHE)
        .then(c => c.match(event.request))
        .then(r => r ?? new Response('No session audio', { status: 404 }))
    );
  }
});
