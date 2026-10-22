import test from 'node:test'
import assert from 'node:assert/strict'
import { ratingTier } from '../src/ratingTier.mjs'

test('shared boundaries start the next user-defined category', () => {
  const cases = [[0,'sub 3'],[1.9,'sub 3'],[2,'sub 5'],[3.9,'sub 5'],[4,'mtn'],[4.9,'mtn'],[5,'lntn'],[5.5,'ntn'],[6,'hltl'],[6.5,'htn'],[7.5,'chadlite'],[8.5,'chad'],[9.5,'true adam'],[10,'true adam']]
  for (const [score, expected] of cases) assert.equal(ratingTier(score)?.label,expected,'score ' + score)
})
test('category agrees with the visible score after rounding', () => {
  assert.equal(ratingTier(8.49)?.label,'chad')
  assert.equal(ratingTier(9.46)?.label,'true adam')
  assert.equal(ratingTier(7.44)?.label,'htn')
})
test('invalid values do not receive a category', () => {
  for (const value of [-1, 10.01, NaN, Infinity]) assert.equal(ratingTier(value),null)
})
