import test from 'node:test'
import assert from 'node:assert/strict'
import { regionDrawing, regionViewport } from '../src/regionGeometry.mjs'
import { mascotMood } from '../src/mascotMood.mjs'

const landmarks = Array.from({length:478},(_,id)=>({id,x:id%30*10,y:Math.floor(id/30)*10}))
const metric = (id,point_ids,substitution={})=>({id,title:id,point_ids,substitution})
const result = metrics=>({landmarks,geometry:{metrics,paths:[],axis:{origin:[100,0],direction:[0,1]}}})

test('both eyes and brows are outlined, and both measured inclinations are drawn',()=>{
  const eyes=[metric('eye_tilt_0',[33,133],{inner:133,outer:33}),metric('eye_tilt_1',[362,263],{inner:362,outer:263})]
  const eyeDrawing=regionDrawing(result(eyes),{id:'eyes',metrics:eyes})
  assert.equal(eyeDrawing.points.length,32)
  assert.deepEqual(eyeDrawing.lines.map(line=>[line.a.id,line.b.id]),[[133,33],[362,263]])
  const brows=[metric('brow_angle_0',[70,107],{inner:107,outer:70}),metric('brow_angle_1',[336,300],{inner:336,outer:300})]
  const browDrawing=regionDrawing(result(brows),{id:'brows',metrics:brows})
  assert.equal(browDrawing.points.length,52)
  for (const id of [70,107,336,300,33,263]) assert.ok(browDrawing.points.some(point=>point.id===id))
})

test('symmetry retains the complete mesh and reflects measured pairs across its actual axis',()=>{
  const symmetry=metric('asymmetry',landmarks.map(p=>p.id),{pairs:[[33,263]]})
  const drawing=regionDrawing(result([symmetry]),{id:'symmetry',metrics:[symmetry]})
  assert.equal(drawing.points.length,478)
  assert.equal(drawing.lines[0].a.x,200-landmarks[33].x)
  assert.equal(drawing.lines[0].a.y,landmarks[33].y)
  assert.deepEqual(drawing.lines[0].b,landmarks[263])
})

test('brow gap uses the measured centers rather than an arbitrary pair of contour points',()=>{
  const gap=metric('brow_gap_0',[70,63,105,33,133],{brow_center:[50,40],eye_center:[55,60]})
  const drawing=regionDrawing(result([gap]),{id:'brows',metrics:[gap]})
  assert.deepEqual(drawing.lines[0].a,{x:50,y:40})
  assert.deepEqual(drawing.lines[0].b,{x:55,y:60})
  assert.equal(regionDrawing({landmarks:[]},{id:'eyes',metrics:[]}).lines.length,0)
})

test('mascot emotions follow the displayed tier including rounding at boundaries',()=>{
  for (const [score,mood] of [[0,'laughing'],[3.9,'laughing'],[3.96,'calm'],[4,'calm'],[7.4,'calm'],[7.46,'respect'],[8.5,'respect'],[9.44,'respect'],[9.46,'shocked'],[10,'shocked']]) assert.equal(mascotMood(score),mood,String(score))
})

test('regional zoom preserves both eyes and stays inside the original image',()=>{
  const eyes=[{id:33,x:160,y:260},{id:133,x:230,y:265},{id:362,x:310,y:265},{id:263,x:380,y:260}]
  const crop=regionViewport(eyes,'eyes',640,800)
  assert.ok(crop.width<640 && crop.height<800)
  for (const p of eyes) assert.ok(p.x>=crop.x && p.x<=crop.x+crop.width && p.y>=crop.y && p.y<=crop.y+crop.height)
  const edge=regionViewport(eyes.map(p=>({...p,y:p.y-250})),'eyes',640,300)
  assert.ok(edge.x>=0 && edge.y>=0 && edge.x+edge.width<=640 && edge.y+edge.height<=300)
  assert.deepEqual(regionViewport([],'eyes',640,800),{x:0,y:0,width:640,height:800})
})
