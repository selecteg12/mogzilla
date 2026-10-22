import test from 'node:test';
import assert from 'node:assert/strict';
import { zipSync, strToU8, unzipSync, strFromU8 } from './vendor/fflate.mjs';
import { sha256, mergeLegacy, decision, buildArchive, readArchive, exportEntry, shouldRestoreLabel } from './core.mjs';

const pixels = new Uint8Array([137,80,78,71,13,10,26,10,0,1,2,3]);
const row = { id:'U_sample', filename:'портрет, тест.png', status:'pending', score:null,
  scoreSource:null, suggestedScore:9, reference:false, updatedAt:null };

test('drafts and exclusions never become human training labels', () => {
  assert.equal(exportEntry(row).training_eligible,false);
  assert.equal(exportEntry(row).score,null);
  assert.equal(exportEntry({...row,status:'excluded',score:9}).score,null);
  assert.throws(()=>decision(row,'rated',0));
  assert.throws(()=>decision(row,'rated',7.5));
  assert.equal(exportEntry(decision(row,'rated',10)).score,10);
});

test('migration preserves personal scores and quarantines only the exact agent test', () => {
  const at='2026-10-20T16:29:59.813Z';
  const ignores=[{id:'C012',reviewed_at_to_ignore:at}];
  assert.equal(mergeLegacy({...row,id:'C012'},{status:'rated',score:8,reviewed_at:at},ignores).status,'needs_review');
  assert.equal(mergeLegacy({...row,id:'C012'},{status:'rated',score:7,reviewed_at:'2026-10-20T17:00:00Z'},ignores).score,7);
  const newer=decision({...row,id:'C012'},'rated',6,'2026-10-20T18:00:00Z');
  assert.equal(mergeLegacy(newer,{status:'rated',score:8,reviewed_at:at},ignores).score,6);
  assert.equal(mergeLegacy(row,{status:'excluded',reviewed_at:at}).status,'excluded');
});

test('archive roundtrip retains exact bytes, Unicode names, confirmed labels and pending rows', async () => {
  const a=decision({...row,sha256:await sha256(pixels)},'rated',4);
  const b={...row,id:'pending',filename:'ещё.png',suggestedScore:10,sha256:await sha256(new Uint8Array([1,2,3]))};
  const result=await buildArchive([a,b],async r=>r.id===a.id?pixels:new Uint8Array([1,2,3]));
  const restored=await readArchive(result.bytes);
  assert.equal(restored.length,2);
  assert.deepEqual(restored[0].bytes,pixels);
  assert.equal(restored[0].score,4);
  assert.equal(restored[1].score,null);
  const zip=unzipSync(result.bytes);
  assert.ok(Object.keys(zip).some(k=>k.includes('photos/4/')&&k.includes('портрет, тест.png')));
  assert.ok(Object.keys(zip).some(k=>k.startsWith('photos/unrated/')));
  const manifest=JSON.parse(strFromU8(zip['manifest.json']));
  assert.equal(manifest.counts.labeled,1);
  assert.match(strFromU8(zip['ratings.csv']),/"score","status"/);
});

test('export stops on a changed image instead of attaching a score to different bytes', async () => {
  const source={...row,sha256:await sha256(pixels)};
  await assert.rejects(()=>buildArchive([source],async()=>new Uint8Array([4,5,6])),/изменился/);
});

test('missing photo errors identify the file and leave CSV labels independently available', async () => {
  await assert.rejects(()=>buildArchive([row],async()=>{throw new DOMException('missing','NotFoundError');}),/портрет, тест\.png.*Скачать оценки CSV/);
  assert.equal(exportEntry(decision(row,'rated',8)).score,8);
});

test('restoring an old archive cannot undo a more recent score removal or exclusion', () => {
  const older=decision(row,'rated',8,'2026-10-20T17:00:00Z');
  assert.equal(shouldRestoreLabel(decision(row,'pending',null,'2026-10-20T18:00:00Z'),older),false);
  assert.equal(shouldRestoreLabel(decision(row,'excluded',null,'2026-10-20T18:00:00Z'),older),false);
  assert.equal(shouldRestoreLabel(row,older),true);
});

test('restore rejects path traversal and damaged images', async () => {
  const unsafe=zipSync({'../outside.png':pixels});
  await assert.rejects(()=>readArchive(unsafe),/имя файла/);
  const bad=zipSync({'manifest.json':strToU8(JSON.stringify({schema:'facelab.personal-dataset',version:1,
    records:[{path:'photos/4/a.png',filename:'a.png',sha256:'bad',status:'rated',score:4}]})),
    'photos/4/a.png':pixels});
  await assert.rejects(()=>readArchive(bad),/повреждены/);
});
