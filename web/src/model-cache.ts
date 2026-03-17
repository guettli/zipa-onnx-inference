// src/speech/model-cache.ts
// Simple IndexedDB cache for ONNX model files

const DB_NAME = "onnx-model-cache";
const STORE_NAME = "models";
const PARTIAL_STORE_NAME = "partial-downloads";
const HISTORY_STORE = "history";
const DB_VERSION = 3;

export interface HistoryEntry {
  id?: number;
  datetime: string;   // ISO 8601
  durationSec: number;
  transcript: string;
  modelId: string;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
      if (!db.objectStoreNames.contains(PARTIAL_STORE_NAME)) {
        db.createObjectStore(PARTIAL_STORE_NAME);
      }
      if (!db.objectStoreNames.contains(HISTORY_STORE)) {
        db.createObjectStore(HISTORY_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getModelFromCache(key: string): Promise<ArrayBuffer | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function saveModelToCache(key: string, arrayBuffer: ArrayBuffer): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(arrayBuffer, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export interface PartialDownload {
  data: ArrayBuffer;
  total: number; // expected total bytes; 0 if unknown
}

export async function getPartialDownload(key: string): Promise<PartialDownload | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PARTIAL_STORE_NAME, "readonly");
    const store = tx.objectStore(PARTIAL_STORE_NAME);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function savePartialDownload(key: string, partial: PartialDownload): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PARTIAL_STORE_NAME, "readwrite");
    const store = tx.objectStore(PARTIAL_STORE_NAME);
    const req = store.put(partial, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function clearPartialDownload(key: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PARTIAL_STORE_NAME, "readwrite");
    const store = tx.objectStore(PARTIAL_STORE_NAME);
    const req = store.delete(key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function deleteModelFromCache(key: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.delete(key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

const RECORDING_KEY = '__last_recording__';

export async function saveRecording(data: ArrayBuffer): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(data, RECORDING_KEY);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function getRecording(): Promise<ArrayBuffer | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(RECORDING_KEY);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function saveHistoryEntry(entry: Omit<HistoryEntry, 'id'>): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    const req = tx.objectStore(HISTORY_STORE).add(entry);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function loadHistory(): Promise<HistoryEntry[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readonly");
    const req = tx.objectStore(HISTORY_STORE).getAll();
    req.onsuccess = () => resolve((req.result as HistoryEntry[]).reverse()); // newest first
    req.onerror = () => reject(req.error);
  });
}

/** Delete partial downloads whose URL already exists in the completed models store. */
export async function pruneStalePartials(): Promise<void> {
  const db = await openDB();
  const keys = await new Promise<IDBValidKey[]>((resolve, reject) => {
    const tx = db.transaction(PARTIAL_STORE_NAME, "readonly");
    const req = tx.objectStore(PARTIAL_STORE_NAME).getAllKeys();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  for (const key of keys) {
    const isComplete = await new Promise<boolean>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).getKey(key);
      req.onsuccess = () => resolve(req.result !== undefined);
      req.onerror = () => reject(req.error);
    });
    if (isComplete) {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(PARTIAL_STORE_NAME, "readwrite");
        const req = tx.objectStore(PARTIAL_STORE_NAME).delete(key);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    }
  }
}

export async function isModelCached(key: string): Promise<boolean> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.getKey(key);
    req.onsuccess = () => resolve(req.result !== undefined);
    req.onerror = () => reject(req.error);
  });
}

export async function getModelChecksum(key: string): Promise<string | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(key + ":sha256");
    req.onsuccess = () => resolve((req.result as string) || null);
    req.onerror = () => reject(req.error);
  });
}

export async function saveModelChecksum(key: string, checksum: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(checksum, key + ":sha256");
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}
