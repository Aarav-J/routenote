const DB_NAME = "routenote";
const DB_VERSION = 1;
const STORE_NAME = "images";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function putImage(dataUrl: string): Promise<string> {
  const db = await openDb();
  const key = `img_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.oncomplete = () => {
      db.close();
      resolve(key);
    };
    tx.onerror = () => {
      const err = tx.error || new Error("Failed to write image");
      db.close();
      reject(err);
    };
    tx.objectStore(STORE_NAME).put(dataUrl, key);
  });
}

export async function getImage(key: string): Promise<string | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => {
      db.close();
      resolve((req.result as string | undefined) ?? null);
    };
    req.onerror = () => {
      const err = req.error || new Error("Failed to read image");
      db.close();
      reject(err);
    };
  });
}

export async function deleteImage(key: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      const err = tx.error || new Error("Failed to delete image");
      db.close();
      reject(err);
    };
    tx.objectStore(STORE_NAME).delete(key);
  });
}

