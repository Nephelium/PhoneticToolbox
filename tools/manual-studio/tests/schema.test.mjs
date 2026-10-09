import test from 'node:test';
import assert from 'node:assert/strict';
import { unknownTypes } from '../src/extensions.ts';
import { extensions } from '../src/extensions.ts';
import { getSchema } from '@tiptap/core';
import { toEditorDocument, fromEditorDocument } from '../src/document-codec.ts';

const referenceDocument = () => ({ type: 'doc', content: [
  { type: 'heading', attrs: { id: 'intro', level: 2 }, content: [{ type: 'text', text: '可编辑正文' }] },
  { type: 'paragraph', content: [
    { type: 'text', text: '中文与 IPA ãː，参见 ' },
    { type: 'crossReference', attrs: { id: 'inline-target', chapterId: 'm08', targetId: 'curve', label: '手绘曲线' } },
    { type: 'citation', attrs: { referenceId: 'source', label: '作者（2026）' } },
  ] },
  { type: 'crossReference', attrs: { id: 'block-target', chapterId: 'settings', targetId: 'retention', label: '查看清理设置' } },
  { type: 'citation', attrs: { id: 'block-source', referenceId: 'source', label: '完整方法来源' } },
] });

function retainsOriginal(expected, actual) {
  assert.equal(actual.type, expected.type);
  if (expected.text !== undefined) assert.equal(actual.text, expected.text);
  for (const [key, value] of Object.entries(expected.attrs ?? {})) assert.deepEqual(actual.attrs?.[key], value);
  if (expected.marks) assert.deepEqual(actual.marks, expected.marks);
  assert.equal(actual.content?.length ?? 0, expected.content?.length ?? 0);
  for (const [index, child] of (expected.content ?? []).entries()) retainsOriginal(child, actual.content[index]);
}

function roundTrip(document) {
  const schema = getSchema(extensions());
  const parsed = schema.nodeFromJSON(toEditorDocument(document));
  parsed.check();
  return fromEditorDocument(parsed.toJSON());
}

test('standalone and inline references both permit visual editing', () => {
  assert.deepEqual(unknownTypes(referenceDocument()), []);
});

test('list anchors permit visual editing', () => {
  const document = { type: 'doc', content: [{ type: 'bulletList', attrs: { id: 'workflow-list' }, content: [
    { type: 'listItem', content: [{ type: 'paragraph', content: [{ type: 'text', text: '定位并试听' }] }] },
  ] }] };
  assert.deepEqual(unknownTypes(document), []);
  retainsOriginal(document, roundTrip(document));
});

test('editor round trip retains reference placement, targets, labels, text and stable anchors', () => {
  const document = referenceDocument(), original = structuredClone(document);
  const saved = roundTrip(document);
  retainsOriginal(document, saved);
  assert.deepEqual(unknownTypes(saved), []);
  assert.deepEqual(document, original);
  assert.ok(!JSON.stringify(saved).includes('blockCrossReference'));
  assert.ok(!JSON.stringify(saved).includes('blockCitation'));
});

test('references in supported block containers round trip without wrapping or flattening', () => {
  const citation = { type: 'citation', attrs: { referenceId: 'source', label: '原始引用' } };
  const document = { type: 'doc', content: [
    { type: 'blockquote', content: [structuredClone(citation)] },
    { type: 'footnote', attrs: { label: '1' }, content: [structuredClone(citation)] },
    { type: 'admonition', content: [structuredClone(citation)] },
    { type: 'columns', content: [{ type: 'column', content: [structuredClone(citation)] }, { type: 'column', content: [{ type: 'paragraph' }] }] },
    { type: 'table', content: [{ type: 'tableRow', content: [{ type: 'tableCell', content: [structuredClone(citation)] }, { type: 'tableHeader', content: [structuredClone(citation)] }] }] },
    { type: 'bulletList', content: [{ type: 'listItem', content: [{ type: 'paragraph' }, structuredClone(citation)] }] },
  ] };
  assert.deepEqual(unknownTypes(document), []);
  retainsOriginal(document, roundTrip(document));
});

test('unknown nodes, attributes and editor-only node names remain protected', () => {
  for (const [node, expected] of [
    [{ type: 'futureNode' }, 'futureNode'],
    [{ type: 'paragraph', attrs: { futureFormatting: 'retain me' } }, 'paragraph.futureFormatting'],
    [{ type: 'blockCitation', attrs: { referenceId: 'source' } }, 'blockCitation'],
  ]) {
    const document = { type: 'doc', content: [node] }, original = structuredClone(document);
    assert.ok(unknownTypes(document).includes(expected));
    assert.deepEqual(document, original);
  }
});

test('invalid text placement still fails structural validation', () => {
  assert.ok(unknownTypes({ type: 'doc', content: [{ type: 'text', text: 'invalid root text' }] }).some(error => error.startsWith('结构校验：')));
});
