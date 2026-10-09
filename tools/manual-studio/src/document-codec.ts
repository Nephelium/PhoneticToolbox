import type { JSONContent } from '@tiptap/core';

const blockContainers = new Set(['doc', 'blockquote', 'column', 'admonition', 'footnote', 'tableCell', 'tableHeader', 'listItem']);
const blockReferences: Record<string, string> = { crossReference: 'blockCrossReference', citation: 'blockCitation' };
const sourceReferences: Record<string, string> = { blockCrossReference: 'crossReference', blockCitation: 'citation' };

export const sourceNodeType = (type: string): string => sourceReferences[type] ?? type;

// The reading contract permits references both inside paragraphs and as blocks.
// ProseMirror requires distinct node types for inline and block content. These
// editor-only names never enter the saved chapter or the reading preview.
function convert(node: JSONContent, direction: 'editor' | 'source', parent?: string): JSONContent {
  const type = node.type ?? '';
  const translated = direction === 'source' ? sourceNodeType(type)
    : parent && blockContainers.has(parent) ? blockReferences[type] ?? type : type;
  return {
    ...node,
    ...(translated !== type ? { type: translated } : {}),
    ...(node.content ? { content: node.content.map(child => convert(child, direction, type)) } : {}),
  };
}

export const toEditorDocument = (document: JSONContent): JSONContent => convert(document, 'editor');
export const fromEditorDocument = (document: JSONContent): JSONContent => convert(document, 'source');
