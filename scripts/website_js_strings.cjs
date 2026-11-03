// Read JavaScript literals with the repository's existing TypeScript parser.
const fs = require('node:fs');
const path = require('node:path');
const ts = require(path.join(__dirname, '../frontend/node_modules/typescript/lib/typescript.js'));
const file = process.argv[2], text = fs.readFileSync(file === '-' ? 0 : file, 'utf8').replace(/\r\n?/g, '\n');
const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
if (source.parseDiagnostics.length) throw Error('Invalid JavaScript source: ' + file);
const result = [];
function visit(node) {
  if (ts.isStringLiteralLike(node) || [ts.SyntaxKind.TemplateHead, ts.SyntaxKind.TemplateMiddle, ts.SyntaxKind.TemplateTail].includes(node.kind)) {
    result.push({start: [...text.slice(0, node.getStart(source))].length, end: [...text.slice(0, node.end)].length, text: node.text, kind: ts.isStringLiteralLike(node) ? 'literal' : ts.SyntaxKind[node.kind]});
  }
  ts.forEachChild(node, visit);
}
visit(source);
process.stdout.write(JSON.stringify(result));
