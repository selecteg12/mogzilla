export async function openStore() {
  const request = indexedDB.open('facelab-personal-labeler-v1', 1);
  request.onupgradeneeded = () => {
    request.result.createObjectStore('photos', { keyPath: 'id' });
    request.result.createObjectStore('meta');
  };
  const db = await new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Закройте другую вкладку просмотрщика и обновите страницу.'));
  });
  db.onversionchange = () => db.close();
  function write(rows, meta = null) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['photos', 'meta'], 'readwrite');
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error('Не удалось сохранить подборку.'));
      tx.onabort = () => reject(tx.error || new Error('Сохранение отменено.'));
      for (const row of rows) tx.objectStore('photos').put(row);
      if (meta) for (const [key, value] of Object.entries(meta)) tx.objectStore('meta').put(value, key);
    });
  }
  return {
    persistent: true, write,
    async all() {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('photos');
        const get = tx.objectStore('photos').getAll();
        get.onsuccess = () => resolve(get.result);
        get.onerror = () => reject(get.error);
      });
    },
  };
}
