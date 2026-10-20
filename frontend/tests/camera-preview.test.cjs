const { test } = require('node:test')
const assert = require('node:assert/strict')
const { readFileSync } = require('node:fs')
const { join } = require('node:path')
const { runInNewContext } = require('node:vm')
const ts = require('typescript')

const source = ts.transpileModule(readFileSync(join(__dirname, '../src/cameraPreview.ts'), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function camera({ width = 1280, height = 720, readyState = 2, videoFrames = true } = {}) {
  let callback
  let firstFrames = 0
  let cancelled = 0
  const draws = []
  const clears = []
  const schedule = next => { callback = next; return 17 }
  const cancel = id => { assert.equal(id, 17); cancelled++ }
  const video = {
    videoWidth: width, videoHeight: height, readyState,
    ...(videoFrames ? { requestVideoFrameCallback: schedule, cancelVideoFrameCallback: cancel } : {}),
  }
  const context = {
    drawImage: (...args) => draws.push(args),
    clearRect: (...args) => clears.push(args),
  }
  const canvas = { width: 300, height: 150, getContext: () => context }
  const exports = {}
  runInNewContext(source, { exports, requestAnimationFrame: schedule, cancelAnimationFrame: cancel })
  const stop = exports.startCameraPreview(video, canvas, () => firstFrames++)
  return {
    video, canvas, draws, clears, stop,
    nextFrame: () => callback(),
    firstFrames: () => firstFrames,
    cancelled: () => cancelled,
  }
}

test('landscape and portrait frames keep intrinsic dimensions without a mirror or rotation transform', () => {
  const front = camera()
  assert.equal(front.canvas.width, 1280)
  assert.equal(front.canvas.height, 720)
  assert.deepEqual(front.draws[0], [front.video, 0, 0, 1280, 720])
  front.video.videoWidth = 720
  front.video.videoHeight = 1280
  front.nextFrame()
  assert.equal(front.canvas.width, 720)
  assert.equal(front.canvas.height, 1280)
  assert.deepEqual(front.draws[1], [front.video, 0, 0, 720, 1280])
  assert.equal(front.firstFrames(), 1)
  front.stop()
})

test('capture only becomes ready once usable pixels arrive', () => {
  const pending = camera({ width: 0, height: 0, readyState: 0 })
  assert.equal(pending.draws.length, 0)
  assert.equal(pending.firstFrames(), 0)
  Object.assign(pending.video, { videoWidth: 640, videoHeight: 480, readyState: 2 })
  pending.nextFrame()
  assert.equal(pending.firstFrames(), 1)
  assert.deepEqual(pending.draws[0], [pending.video, 0, 0, 640, 480])
  pending.stop()
})

test('closing or switching cancels rendering, clears the old frame and ignores late callbacks', () => {
  const old = camera()
  old.stop()
  assert.equal(old.cancelled(), 1)
  assert.deepEqual(old.clears, [[0, 0, 1280, 720]])
  old.nextFrame()
  assert.equal(old.draws.length, 1)
  const rear = camera({ width: 1080, height: 1920 })
  assert.deepEqual(rear.draws[0], [rear.video, 0, 0, 1080, 1920])
  rear.stop()
})

test('browsers without video frame callbacks use and cancel animation frames', () => {
  const fallback = camera({ videoFrames: false })
  fallback.nextFrame()
  assert.equal(fallback.draws.length, 2)
  fallback.stop()
  assert.equal(fallback.cancelled(), 1)
  fallback.nextFrame()
  assert.equal(fallback.draws.length, 2)
})
