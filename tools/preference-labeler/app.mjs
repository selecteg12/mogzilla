import { LEGACY_KEY, MAX_PHOTO_BYTES, MAX_DATASET_BYTES, MAX_RECORDS, IMAGE_EXT,
  validScore, safeFilename, sha256, mergeLegacy, decision, isLabeled,
  exportEntry, labelsCsv, buildArchive, readArchive, shouldRestoreLabel } from './core.mjs';
import { openStore } from './storage.mjs';

const $ = id => document.getElementById(id);
const records = new Map();
let store, loaded = false, busy = false, currentId = null, history = [], referenceMode = false;
let currentUrl = null, renderToken = 0, referenceUrls = [], referenceToken = 0;
const controls = ['export-zip','export-csv','add-photos','add-folder','restore-zip','filter','references'];

function status(message = '', error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function userError(error) {
  return error?.name === 'QuotaExceededError'
    ? 'В браузере закончилось место. Текущая разметка сохранена; скачайте ZIP перед добавлением новых фото.'
    : error?.message || 'Не удалось выполнить действие.';
}
function all() { return [...records.values()].sort((a,b) => a.order - b.order || a.id.localeCompare(b.id)); }
function visible() {
  return all().filter(r => !r.reference).filter(r => {
    const filter = $('filter').value;
    return filter === 'all' || (filter === 'pending' && ['pending','needs_review'].includes(r.status))
      || (filter === 'rated' && isLabeled(r)) || (filter === 'excluded' && r.status === 'excluded');
  });
}
function pending(row) { return ['pending', 'needs_review'].includes(row.status); }
function mime(name) {
  return /\.png$/i.test(name) ? 'image/png' : /\.webp$/i.test(name) ? 'image/webp'
    : /\.avif$/i.test(name) ? 'image/avif' : 'image/jpeg';
}
function assetUrl(row) {
  const url = new URL(row.assetPath, location.href);
  if (url.origin !== location.origin) throw new Error('Фотография ссылается на другой сайт.');
  return url.href;
}
function sourceUrl(value) {
  try { const url = new URL(value); return /^https?:$/.test(url.protocol) ? url.href : null; }
  catch { return null; }
}
async function imageSource(row) {
  if (row.blob) return URL.createObjectURL(row.blob);
  return assetUrl(row);
}
function releasePortrait() {
  if (currentUrl?.startsWith('blob:')) URL.revokeObjectURL(currentUrl);
  currentUrl = null;
}
function render() {
  const list = visible();
  if (!list.some(r => r.id === currentId)) currentId = (list.find(pending) || list[0])?.id || null;
  const row = records.get(currentId), position = list.findIndex(r => r.id === currentId);
  const ordinary = all().filter(r => !r.reference);
  const labeled = ordinary.filter(isLabeled).length, excluded = ordinary.filter(r => r.status === 'excluded').length;
  const remaining = ordinary.filter(pending).length;
  $('intro').textContent = `${ordinary.length} фото для разметки · ${all().filter(r => r.reference).length} исходных примеров`;
  $('progress').textContent = `Оценено: ${labeled} · Исключено: ${excluded} · Без оценки: ${remaining}`;
  for (const id of controls) $(id).disabled = busy || !loaded;
  $('export-zip').disabled ||= !records.size;
  $('export-csv').disabled ||= !records.size;
  $('previous').disabled = busy || referenceMode || position <= 0;
  $('next').disabled = busy || referenceMode || position < 0 || position === list.length - 1;
  $('undo').disabled = busy || !history.length;
  $('jump').disabled = busy || !remaining || referenceMode;
  $('references').textContent = referenceMode ? 'Вернуться к разметке' : 'Мои исходные примеры';
  $('review').classList.toggle('hidden', !row || referenceMode);
  $('references-grid').classList.toggle('hidden', !referenceMode);
  $('empty').classList.toggle('hidden', Boolean(row) || referenceMode);
  $('empty-text').textContent = ordinary.length ? 'В этом фильтре нет снимков. Измените фильтр или добавьте фотографии.' : 'Добавьте фотографии пачкой и начните с первого снимка.';
  if (!row || referenceMode) return;
  $('position').textContent = `${position + 1} из ${list.length} · ${row.id}`;
  $('score').textContent = isLabeled(row) ? `${row.score} / 10` : row.status === 'excluded' ? 'Исключено' : 'Ваша оценка';
  $('filename').textContent = row.filename;
  $('review-note').textContent = row.reviewNote || '';
  $('review-note').classList.toggle('hidden', !row.reviewNote);
  $('suggestion').textContent = validScore(row.suggestedScore) ? `Предварительное предложение: ${row.suggestedScore}/10. Его можно изменить.` : 'Поставьте вашу оценку от 1 до 10.';
  $('accept').classList.toggle('hidden', !validScore(row.suggestedScore));
  $('accept').disabled = busy;
  for (const button of $('keys').children) {
    button.classList.toggle('chosen', isLabeled(row) && row.score === Number(button.dataset.score));
    button.disabled = busy;
  }
  $('skip').disabled = busy;
  $('exclude').disabled = busy;
  $('clear').disabled = busy || pending(row);
  const source = sourceUrl(row.sourcePage);
  $('source').classList.toggle('hidden', !source);
  if (source) $('source').href = source;
  else $('source').removeAttribute('href');
  // Keep the current photo visible during rating writes, avoiding flashes.
  if ($('portrait').dataset.id !== row.id) {
    const token = ++renderToken;
    releasePortrait();
    $('portrait').dataset.id = row.id;
    $('portrait').alt = `Портрет ${row.id}`;
    imageSource(row).then(url => {
      if (token !== renderToken) { if (url.startsWith('blob:')) URL.revokeObjectURL(url); return; }
      currentUrl = url; $('portrait').src = url;
    }).catch(error => status(userError(error), true));
  }
}

function legacyBackup(row) {
  if (!/^C\d+$/.test(row.id)) return;
  try {
    const data = JSON.parse(localStorage.getItem(LEGACY_KEY) || '{"version":1,"answers":{}}');
    data.version = 1; data.answers ||= {};
    if (['rated','excluded'].includes(row.status)) data.answers[row.id] = { status:row.status, score:row.score, reviewed_at:row.updatedAt };
    else delete data.answers[row.id];
    localStorage.setItem(LEGACY_KEY, JSON.stringify(data));
  } catch { /* IndexedDB remains the authoritative store. */ }
}
async function change(statusValue, score = null) {
  if (busy || !currentId || referenceMode) return;
  const previous = records.get(currentId), list = visible();
  const position = list.findIndex(r => r.id === currentId);
  const nextId = list[position + 1]?.id;
  const updated = decision(previous, statusValue, score);
  busy = true; render();
  try {
    await store.write([updated]);
    records.set(updated.id, updated);
    history.push(previous); if (history.length > 100) history.shift();
    legacyBackup(updated);
    currentId = statusValue === 'pending' ? updated.id : nextId || updated.id;
    status('Сохранено в браузере.');
  } catch (error) { status(userError(error), true); }
  finally { busy = false; render(); }
}
async function undo() {
  if (busy || !history.length) return;
  const previous = { ...history[history.length - 1], updatedAt:new Date().toISOString() };
  busy = true; render();
  try {
    await store.write([previous]);
    history.pop(); records.set(previous.id, previous); currentId = previous.id;
    if ($('filter').value !== 'all') $('filter').value = 'all';
    legacyBackup(previous); status('Последнее действие отменено.');
  } catch (error) { status(userError(error), true); }
  finally { busy = false; render(); }
}
function move(delta) {
  if (busy || referenceMode) return;
  const list = visible(), index = list.findIndex(r => r.id === currentId);
  currentId = list[Math.max(0, Math.min(list.length - 1, index + delta))]?.id || null;
  render();
}

async function dimensions(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > 40_000_000) {
      throw new Error('Фотография слишком большая: максимум 40 мегапикселей.');
    }
    return { width:image.naturalWidth, height:image.naturalHeight };
  } catch (error) {
    if (error.message?.includes('мегапикселей')) throw error;
    throw new Error('Не удалось прочитать фотографию. Попробуйте JPG, PNG, WebP или AVIF.');
  } finally { URL.revokeObjectURL(url); }
}
function errorsList(errors) {
  $('error-list').replaceChildren();
  for (const error of errors) { const item = document.createElement('li'); item.textContent = error; $('error-list').append(item); }
  $('errors').classList.toggle('hidden', !errors.length);
}
async function importPhotos(files) {
  if (busy || !loaded || !files.length) return;
  busy = true; render(); errorsList([]);
  const added = [], errors = [], hashes = new Set(all().map(r => r.sha256));
  const known = new Map(all().map(r => [r.sha256,r])), refreshed = new Map();
  let duplicates = 0, bytes = all().reduce((n,r) => n + (r.size || 0), 0);
  let order = Math.max(-1, ...all().map(r => r.order)) + 1;
  try {
    for (let i = 0; i < files.length; i++) {
      const file = files[i]; status(`Добавление фотографий: ${i + 1} из ${files.length}…`);
      try {
        if (!IMAGE_EXT.test(file.name)) throw new Error('Поддерживаются JPG, PNG, WebP и AVIF.');
        if (!file.size || file.size > MAX_PHOTO_BYTES) throw new Error('Нужен файл размером до 20 МБ.');
        // Own the bytes: a Blob composed from a File can retain a temporary
        // filesystem backing that becomes unavailable after the picker closes.
        const raw = await file.arrayBuffer();
        const hash = await sha256(raw);
        const blob = new Blob([raw],{type:mime(file.name)});
        if (hashes.has(hash)) {
          duplicates++;
          const existing = known.get(hash);
          if (existing) refreshed.set(existing.id,{...existing,blob,size:raw.byteLength});
          continue;
        }
        if (records.size + added.length >= MAX_RECORDS) throw new Error('Максимум 2000 фотографий в подборке.');
        if (bytes + file.size > MAX_DATASET_BYTES) throw new Error('В подборке больше 300 МБ.');
        const size = await dimensions(blob);
        added.push({ id:`U_${hash.slice(0,20)}`, sha256:hash, filename:safeFilename(file.name),
          blob, size:raw.byteLength, ...size, order:order++,
          status:'pending', score:null, scoreSource:null, suggestedScore:null, reference:false,
          origin:'user_upload', sourcePage:null, usageRights:null, updatedAt:null,
          addedAt:new Date().toISOString(), originalRelativePath:file.webkitRelativePath || null });
        hashes.add(hash); bytes += file.size;
      } catch (error) { errors.push(`${file.name}: ${userError(error)}`); }
    }
    if (added.length || refreshed.size) {
      await store.write([...added,...refreshed.values()]);
      for (const row of [...added,...refreshed.values()]) records.set(row.id, row);
    }
    if (added.length) {
      referenceMode = false; $('filter').value = 'all'; currentId = added[0].id;
    }
    status(`Добавлено: ${added.length}. Точных повторов пропущено: ${duplicates}. Ошибок: ${errors.length}.` +
      (refreshed.size ? ` Файлов восстановлено/обновлено: ${refreshed.size}; оценки сохранены.` : ''), Boolean(errors.length && !added.length));
    errorsList(errors);
  } catch (error) { status(userError(error), true); }
  finally { busy = false; $('photos-input').value = ''; $('folder-input').value = ''; render(); }
}

async function restoreArchive(file) {
  if (busy || !loaded || !file) return;
  busy = true; render(); errorsList([]); status('Чтение архива…');
  try {
    if (file.size > MAX_DATASET_BYTES) throw new Error('Архив больше 300 МБ.');
    const incoming = await readArchive(new Uint8Array(await file.arrayBuffer()));
    const byHash = new Map(all().map(r => [r.sha256,r]));
    const changed = []; let added = 0, restored = 0;
    let order = Math.max(-1,...all().map(r => r.order)) + 1;
    let bytes = all().reduce((n,r) => n + (r.size || 0),0);
    for (let i=0;i<incoming.length;i++) {
      const row = incoming[i], existing = byHash.get(row.sha256);
      status(`Восстановление: ${i+1} из ${incoming.length}…`);
      if (existing) {
        const apply = shouldRestoreLabel(existing,row);
        const updated = { ...existing, blob:existing.blob || new Blob([row.bytes],{type:mime(row.filename)}) };
        if (apply) {
          Object.assign(updated,{status:row.status,score:row.score,scoreSource:row.scoreSource,updatedAt:row.updatedAt,reviewNote:null});
          restored++;
        }
        changed.push(updated);
      } else {
        if (records.size + added >= MAX_RECORDS || bytes + row.bytes.length > MAX_DATASET_BYTES) throw new Error('Подборка превышает 2000 фото или 300 МБ.');
        const blob = new Blob([row.bytes],{type:mime(row.filename)}), size = await dimensions(blob);
        const { bytes:raw, ...metadata } = row;
        changed.push({...metadata,blob,size:raw.length,...size,order:order++,addedAt:new Date().toISOString()});
        added++; bytes += raw.length;
      }
    }
    await store.write(changed);
    for (const row of changed) { records.set(row.id,row); legacyBackup(row); }
    referenceMode = false; $('filter').value = 'all';
    currentId = all().find(r => !r.reference && pending(r))?.id || visible()[0]?.id || null;
    status(`Архив восстановлен. Новых фото: ${added}. Обновлено оценок: ${restored}. Более свежая разметка сохранена.`);
  } catch (error) { status(userError(error),true); }
  finally { busy = false; $('zip-input').value = ''; render(); }
}

async function photoBytes(row) {
  if (row.blob) return new Uint8Array(await row.blob.arrayBuffer());
  const response = await fetch(assetUrl(row),{cache:'no-store'});
  if (!response.ok) throw new Error(`Не найдена фотография ${row.filename}. Восстановите её из исходной папки или архива.`);
  return new Uint8Array(await response.arrayBuffer());
}
function download(blob, filename) {
  const url = URL.createObjectURL(blob), anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
async function exportZip() {
  if (busy || !loaded) return;
  busy = true; render();
  try {
    const result = await buildArchive(all(),photoBytes,(done,total) => status(`Подготовка ZIP: ${done} из ${total} фото…`));
    const date = new Intl.DateTimeFormat('sv-SE',{timeZone:'Europe/Moscow',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date()).replace(/[ :]/g,'-');
    download(new Blob([result.bytes],{type:'application/zip'}),`FaceLab_dataset_${date}.zip`);
    status(`ZIP подготовлен: ${result.manifest.counts.total} фото, с вашими оценками — ${result.manifest.counts.labeled}. Передайте этот архив для подготовки обучения.`);
  } catch (error) { status(userError(error),true); }
  finally { busy = false; render(); }
}
async function showReferences() {
  referenceMode = !referenceMode; render();
  if (!referenceMode) return;
  const token = ++referenceToken;
  for (const url of referenceUrls) URL.revokeObjectURL(url);
  referenceUrls = []; $('references-grid').replaceChildren();
  for (const row of all().filter(r => r.reference)) {
    const box = document.createElement('div'); box.className = 'ref';
    const img = document.createElement('img'); img.alt = `Ваш пример ${row.id}`; img.loading = 'lazy';
    const p = document.createElement('p'); p.textContent = `${row.id} · ваша оценка ${row.score}/10`;
    box.append(img,p); $('references-grid').append(box);
    const url = await imageSource(row);
    if (token !== referenceToken || !referenceMode) { if (url.startsWith('blob:')) URL.revokeObjectURL(url); return; }
    if (url.startsWith('blob:')) referenceUrls.push(url);
    img.src = url;
  }
}

for (let score=1;score<=10;score++) {
  const button = document.createElement('button'); button.textContent = score; button.dataset.score = score;
  button.title = `Клавиша ${score===10?0:score}`; button.onclick = () => change('rated',score); $('keys').append(button);
}
$('add-photos').onclick = () => $('photos-input').click();
$('add-folder').onclick = () => $('folder-input').click();
$('restore-zip').onclick = () => $('zip-input').click();
$('photos-input').onchange = event => importPhotos([...event.target.files]);
$('folder-input').onchange = event => importPhotos([...event.target.files]);
$('zip-input').onchange = event => restoreArchive(event.target.files[0]);
$('previous').onclick = () => move(-1); $('next').onclick = () => move(1);
$('skip').onclick = () => move(1); $('undo').onclick = undo;
$('exclude').onclick = () => change('excluded'); $('clear').onclick = () => change('pending');
$('accept').onclick = () => { const row=records.get(currentId); if(validScore(row?.suggestedScore)) change('rated',row.suggestedScore); };
$('filter').onchange = () => { currentId=null; render(); };
$('jump').onclick = () => { $('filter').value='pending'; currentId=null; render(); };
$('references').onclick = () => showReferences().catch(error=>status(userError(error),true));
$('export-zip').onclick = exportZip;
$('export-csv').onclick = () => { download(new Blob([labelsCsv(all().map(exportEntry))],{type:'text/csv;charset=utf-8'}),'FaceLab_ratings.csv'); status('CSV скачан. Для передачи самих фотографий используйте ZIP.'); };
document.addEventListener('dragover',event=>event.preventDefault());
document.addEventListener('drop',event=>event.preventDefault());
$('dropzone').addEventListener('dragenter',()=> $('dropzone').classList.add('dragging'));
$('dropzone').addEventListener('dragleave',()=> $('dropzone').classList.remove('dragging'));
$('dropzone').addEventListener('drop',event=>{ $('dropzone').classList.remove('dragging'); importPhotos([...event.dataTransfer.files]); });
document.addEventListener('keydown',event=>{
  if (event.repeat || busy || !loaded || referenceMode || event.ctrlKey || event.altKey || event.metaKey ||
    ['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
  if (/^\d$/.test(event.key)) { event.preventDefault(); change('rated',Number(event.key)||10); }
  else if (event.key===' ') { event.preventDefault(); move(1); }
  else if (event.key==='ArrowLeft') { event.preventDefault(); move(-1); }
  else if (event.key==='ArrowRight') { event.preventDefault(); move(1); }
  else if (event.key==='Backspace') { event.preventDefault(); undo(); }
  else if (['x','X','ч','Ч'].includes(event.key)) { event.preventDefault(); change('excluded'); }
  else if (event.key==='Enter' && !['BUTTON','A'].includes(event.target.tagName)) { event.preventDefault(); $('accept').click(); }
});
window.addEventListener('beforeunload',event=>{ if(busy){event.preventDefault();event.returnValue='';} });
window.addEventListener('storage',event=>{ if(event.key===LEGACY_KEY && loaded) status('Разметка изменилась в другой вкладке. Обновите страницу, чтобы объединить изменения.'); });
$('portrait').onerror = () => status('Не удалось открыть фотографию. Оценки сохранены; файл можно восстановить из ZIP.',true);

async function start() {
  try {
    store = await openStore();
    const response = await fetch('./review-data.json',{cache:'no-store'});
    if (!response.ok) throw new Error('Не удалось открыть подборку. Проверьте, что просмотрщик запущен через локальный сервер.');
    const config = await response.json();
    if (!Array.isArray(config.records)) throw new Error('Некорректная конфигурация подборки.');
    const stored = new Map((await store.all()).map(r=>[r.id,r]));
    let legacy = {};
    try { const data=JSON.parse(localStorage.getItem(LEGACY_KEY)||'{}'); if(data.version===1) legacy=data.answers||{}; } catch { /* Old progress remains untouched. */ }
    const changed = [];
    for (const base of config.records) {
      const existing = stored.get(base.id);
      let row = {...base,...existing};
      row = mergeLegacy(row,legacy[row.id],config.ignoredLegacyRatings||[]);
      changed.push(row); records.set(row.id,row); stored.delete(row.id);
    }
    for (const row of stored.values()) records.set(row.id,row);
    await store.write(changed);
    loaded=true;
    currentId=all().find(r=>!r.reference&&pending(r))?.id || all().find(r=>!r.reference)?.id || null;
    render();
    const needs=all().filter(r=>r.status==='needs_review');
    if(needs.length) status(`Требуют перепроверки после теста: ${needs.map(r=>r.id).join(', ')}. Их прежние баллы не считаются вашей разметкой.`);
  } catch (error) { status(userError(error),true); $('intro').textContent='Подборка не загружена. Оценки в браузере не удалялись.'; }
}
start();
