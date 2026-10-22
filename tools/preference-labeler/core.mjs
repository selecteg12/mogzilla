import { zipSync, unzipSync, strToU8, strFromU8 } from './vendor/fflate.mjs';

export const LEGACY_KEY = 'facelab-review-20261020-batch1';
export const MAX_PHOTO_BYTES = 20 * 1024 * 1024;
export const MAX_DATASET_BYTES = 300 * 1024 * 1024;
export const MAX_RECORDS = 2000;
export const IMAGE_EXT = /\.(jpe?g|png|webp|avif)$/i;
const HUMAN_SOURCES = new Set(['user', 'user_archive', 'imported_user']);

export function validScore(value) {
  return Number.isInteger(value) && value >= 1 && value <= 10;
}

export function safeFilename(value) {
  const name = String(value || 'photo.jpg').split(/[\\/]/).pop()
    .replace(/[\x00-\x1f<>:"|?*]/g, '_').replace(/[. ]+$/g, '');
  const extension = name.match(/\.[^.]{1,8}$/)?.[0] || '';
  const stem = name.slice(0, name.length - extension.length).slice(0, 100) || 'photo';
  return stem + extension;
}

export async function sha256(bytes) {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(v => v.toString(16).padStart(2, '0')).join('');
}

function timestamp(value) {
  return typeof value === 'string' && Number.isFinite(Date.parse(value)) ? value : null;
}

export function mergeLegacy(record, answer, ignored = []) {
  if (!answer || !['rated', 'excluded'].includes(answer.status)) return record;
  if (answer.status === 'rated' && !validScore(answer.score)) return record;
  const at = timestamp(answer.reviewed_at);
  if (record.updatedAt && (!at || Date.parse(record.updatedAt) > Date.parse(at))) return record;
  const agentTest = ignored.some(row => row.id === record.id && row.reviewed_at_to_ignore === at);
  if (agentTest) {
    return { ...record, score: null, status: 'needs_review', scoreSource: null, updatedAt: at,
      reviewNote: 'Перепроверьте этот снимок: прежняя оценка была тестовой.' };
  }
  return { ...record, status: answer.status, score: answer.status === 'rated' ? answer.score : null,
    scoreSource: answer.status === 'rated' ? 'user' : null, updatedAt: at, reviewNote: null };
}

export function decision(record, status, score = null, at = new Date().toISOString()) {
  if (!['rated', 'excluded', 'pending'].includes(status)) throw new Error('Некорректный статус.');
  if (status === 'rated' && !validScore(score)) throw new Error('Нужен целый балл от 1 до 10.');
  return { ...record, status, score: status === 'rated' ? score : null,
    scoreSource: status === 'rated' ? 'user' : null, updatedAt: at, reviewNote: null };
}

export function isLabeled(record) {
  return record.status === 'rated' && validScore(record.score) && HUMAN_SOURCES.has(record.scoreSource);
}

export function shouldRestoreLabel(existing, incoming) {
  const currentTime = timestamp(existing.updatedAt);
  const incomingTime = timestamp(incoming.updatedAt);
  if (currentTime) return Boolean(incomingTime && Date.parse(incomingTime) > Date.parse(currentTime));
  if (incomingTime) return true;
  return ['pending', 'needs_review'].includes(existing.status) && ['rated', 'excluded'].includes(incoming.status);
}

export function csvText(rows) {
  const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
  return '\ufeff' + rows.map(row => row.map(quote).join(',')).join('\r\n');
}

export function exportEntry(record) {
  const labeled = isLabeled(record);
  const status = labeled ? 'rated' : record.status === 'excluded' ? 'excluded'
    : record.status === 'needs_review' ? 'needs_review' : 'pending';
  const folder = labeled ? String(record.score) : status === 'excluded' ? 'excluded'
    : status === 'needs_review' ? 'needs-review' : 'unrated';
  return {
    id: record.id,
    path: `photos/${folder}/${record.id.replace(/[^a-zA-Z0-9_-]/g, '_')}__${safeFilename(record.filename)}`,
    filename: safeFilename(record.filename), sha256: record.sha256,
    status, score: labeled ? record.score : null,
    score_source: labeled ? record.scoreSource : null,
    suggested_score: validScore(record.suggestedScore) ? record.suggestedScore : null,
    reviewed_at: record.updatedAt || null, reference: Boolean(record.reference),
    source_url: record.sourcePage || null, usage_rights: record.usageRights || null,
    person_id: record.personId || null, training_eligible: labeled,
  };
}

export function labelsCsv(entries) {
  return csvText([
    ['id', 'filename', 'score', 'status', 'score_source', 'sha256', 'source_url', 'reference', 'reviewed_at', 'person_id'],
    ...entries.map(r => [r.id, r.path, r.score, r.status, r.score_source,
      r.sha256, r.source_url, r.reference, r.reviewed_at, r.person_id]),
  ]);
}

export async function buildArchive(records, readBytes, progress = () => {}) {
  if (records.length > MAX_RECORDS) throw new Error('В подборке больше 2000 фотографий.');
  const files = Object.create(null);
  const entries = [];
  let total = 0;
  for (let i = 0; i < records.length; i++) {
    const row = records[i];
    let data;
    try { data = await readBytes(row); }
    catch (error) {
      throw new Error(`Не удалось прочитать фото ${row.filename} (${row.id}). Оценки сохранены — их можно скачать кнопкой «Скачать оценки CSV». Для восстановления фото повторно выберите исходные файлы или папку: баллы останутся на месте.`, { cause:error });
    }
    total += data.length;
    if (total > MAX_DATASET_BYTES) throw new Error('Подборка превышает 300 МБ. Экспортируйте её частями.');
    const hash = await sha256(data);
    if (row.sha256 && hash !== row.sha256) throw new Error(`Файл ${row.filename} изменился. Экспорт остановлен, оценки сохранены.`);
    const entry = exportEntry({ ...row, sha256: hash });
    if (Object.hasOwn(files, entry.path)) throw new Error('Повторяющийся путь в архиве.');
    files[entry.path] = [data, { level: 0 }];
    entries.push(entry);
    progress(i + 1, records.length);
  }
  const manifest = {
    schema: 'facelab.personal-dataset', version: 1, created_at: new Date().toISOString(),
    score_range: [1, 10], records: entries,
    counts: { total: entries.length, labeled: entries.filter(r => r.training_eligible).length,
      pending: entries.filter(r => ['pending', 'needs_review'].includes(r.status)).length,
      excluded: entries.filter(r => r.status === 'excluded').length },
    split: null,
    note: 'Only human-confirmed ratings are training eligible. Identity groups and train/test splits still need review.',
  };
  files['manifest.json'] = strToU8(JSON.stringify(manifest, null, 2));
  files['ratings.csv'] = strToU8(labelsCsv(entries));
  files['README.txt'] = strToU8(
    'FaceLab — фотографии и личные оценки\r\n\r\n' +
    'photos/1–10: фотографии с вашими оценками.\r\n' +
    'photos/unrated: ещё не оценённые фотографии.\r\n' +
    'photos/needs-review: фотографии, требующие повторной оценки.\r\n' +
    'photos/excluded: исключённые фотографии.\r\n' +
    'ratings.csv и manifest.json связывают файлы и баллы. Предварительные предложения не являются вашей разметкой.\r\n' +
    'Для обучения подходят только training_eligible=true; повторы людей и разделение обучения/проверки нужно проверить отдельно.\r\n' +
    'Передайте этот ZIP для подготовки обучения. Сам экспорт не обучает модель.\r\n' +
    'Этот же ZIP можно загрузить в просмотрщик, чтобы восстановить фотографии и оценки.\r\n');
  return { bytes: zipSync(files, { level: 0 }), manifest };
}

function safeArchivePath(path) {
  return typeof path === 'string' && !/[\\:\x00-\x1f]/.test(path) && !path.startsWith('/') &&
    path.split('/').every(part => part && part !== '.' && part !== '..' && !['__proto__', 'constructor', 'prototype'].includes(part));
}

export async function readArchive(bytes) {
  if (bytes.length > MAX_DATASET_BYTES) throw new Error('Архив больше 300 МБ.');
  let total = 0, count = 0;
  const files = unzipSync(bytes, { filter(entry) {
    if (++count > MAX_RECORDS + 10) throw new Error('Слишком много файлов в архиве.');
    if (!safeArchivePath(entry.name)) throw new Error('В архиве некорректное имя файла.');
    total += entry.originalSize;
    if (!Number.isSafeInteger(total) || total > MAX_DATASET_BYTES + 4 * 1024 * 1024) throw new Error('Распакованный архив больше 300 МБ.');
    if (entry.name === 'manifest.json' && entry.originalSize > 4 * 1024 * 1024) throw new Error('Слишком большой manifest.json.');
    return entry.name === 'manifest.json' || entry.name.startsWith('photos/');
  } });
  if (!Object.hasOwn(files, 'manifest.json')) throw new Error('Это не экспорт FaceLab: отсутствует manifest.json.');
  const manifest = JSON.parse(strFromU8(files['manifest.json']));
  if (manifest.schema !== 'facelab.personal-dataset' || manifest.version !== 1 || !Array.isArray(manifest.records)) {
    throw new Error('Неизвестный формат архива. Выберите ZIP, скачанный из этого просмотрщика.');
  }
  if (manifest.records.length > MAX_RECORDS) throw new Error('В архиве больше 2000 фотографий.');
  const result = [], hashes = new Set();
  for (const entry of manifest.records) {
    if (!safeArchivePath(entry.path) || !entry.path.startsWith('photos/') || !IMAGE_EXT.test(entry.path) || !Object.hasOwn(files, entry.path)) {
      throw new Error('Не найдена фотография, указанная в разметке.');
    }
    const data = files[entry.path];
    if (data.length > MAX_PHOTO_BYTES) throw new Error('В архиве есть фотография больше 20 МБ.');
    const hash = await sha256(data);
    if (hash !== entry.sha256 || hashes.has(hash)) throw new Error('Фотографии в архиве повреждены либо повторяются.');
    hashes.add(hash);
    const labeled = entry.status === 'rated' && validScore(entry.score) && HUMAN_SOURCES.has(entry.score_source) && entry.training_eligible === true;
    result.push({
      id: `U_${hash.slice(0, 20)}`, sha256: hash, filename: safeFilename(entry.filename),
      status: labeled ? 'rated' : entry.status === 'excluded' ? 'excluded' : entry.status === 'needs_review' ? 'needs_review' : 'pending',
      score: labeled ? entry.score : null, scoreSource: labeled ? 'imported_user' : null,
      suggestedScore: validScore(entry.suggested_score) ? entry.suggested_score : null,
      reference: Boolean(entry.reference), sourcePage: entry.source_url || null,
      usageRights: entry.usage_rights || null, personId: typeof entry.person_id === 'string' ? entry.person_id : null,
      origin: 'restored_archive', updatedAt: timestamp(entry.reviewed_at), bytes: data,
    });
  }
  return result;
}
