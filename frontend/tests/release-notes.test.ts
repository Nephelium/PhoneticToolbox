import test from 'node:test';
import assert from 'node:assert/strict';
import {renderReleaseNotes} from '../src/platform/release-notes.ts';

test('old plain text remains a list with bold short topic labels',()=>{
 const html=renderReleaseNotes('更新窗口：显示版本说明。\r\n下载完成后校验更新包。');
 assert.match(html,/<ul>/);assert.equal((html.match(/<li>/g)||[]).length,2);assert.match(html,/<strong>更新窗口：<\/strong>显示版本说明。/);
 assert.doesNotMatch(renderReleaseNotes('https://example.com/update'),/<strong>/);assert.equal(renderReleaseNotes(' \n\r\n'),'');
});
test('Markdown preserves ordered, unordered, nested lists and continuations',()=>{
 const html=renderReleaseNotes('- **更新窗口：** 分条展示。\n  较长说明继续换行。\n  - 次级条目。\n\n1. 退出前保存。\n2. 再次确认。');
 assert.match(html,/<strong>更新窗口：<\/strong>/);assert.equal((html.match(/<ul>/g)||[]).length,2);assert.match(html,/<ol>/);assert.match(html,/较长说明继续换行。/);
});
test('inline emphasis, strike-through and code render without changing their content',()=>{
 const html=renderReleaseNotes('- **重点**、__强调__、*斜体*、~~旧说明~~，以及 `版本号`。');
 assert.match(html,/<strong>重点<\/strong>/);assert.match(html,/<strong>强调<\/strong>/);assert.match(html,/<em>斜体<\/em>/);assert.match(html,/<s>旧说明<\/s>/);assert.match(html,/<code>版本号<\/code>/);
 assert.match(renderReleaseNotes('- **未闭合标记'),/\*\*未闭合标记/);
});
test('headings, block quotes, fenced code, tables and rules preserve block structure',()=>{
 const html=renderReleaseNotes('## 本次更新\n\n> 操作提示。\n\n```text\n保存后更新\n```\n\n| 功能 | 变化 |\n| --- | --- |\n| 更新窗口 | 分条展示 |\n\n---');
 assert.match(html,/<h2>本次更新<\/h2>/);assert.match(html,/<blockquote>/);assert.match(html,/<pre><code class="language-text">保存后更新/);assert.match(html,/<table>/);assert.match(html,/<th>功能<\/th>/);assert.match(html,/<hr>/);
 assert.match(renderReleaseNotes('    保留缩进的代码'),/<pre><code>保留缩进的代码/);
});
test('web links open separately and unsafe or local destinations stay text',()=>{
 const html=renderReleaseNotes('- [官网](https://www.phonetictoolbox.com/)\n- [网页](http://example.com/)\n- [脚本](javascript:alert(1))\n- [本机](file:///C:/private)\n- [相对](../private)\n- [数据](data:text/html,evil)');
 assert.equal((html.match(/<a /g)||[]).length,2);assert.equal((html.match(/target="_blank" rel="noopener noreferrer"/g)||[]).length,2);
 assert.doesNotMatch(html,/href="(?:javascript|file|data):|href="\.\./);
});
test('raw HTML and images cannot execute or load external content; code is escaped',()=>{
 const html=renderReleaseNotes('- **提示：** <img src=x onerror=alert(1)>\n\n<script>alert(1)</script>\n\n![图片](https://example.com/secret.png)\n\n```html\n<img src=x onerror=alert(1)>\n```');
 assert.doesNotMatch(html,/<img|<script/);assert.match(html,/&lt;img src=x onerror=alert\(1\)&gt;/);assert.match(html,/&lt;script&gt;/);
});
