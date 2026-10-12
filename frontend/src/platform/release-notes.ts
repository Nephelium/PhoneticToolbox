import MarkdownIt from 'markdown-it';

const markdown = new MarkdownIt({html:false,breaks:true,linkify:false,typographer:false}).disable('image');
const validateLink = markdown.validateLink.bind(markdown);
markdown.validateLink = url => validateLink(url) && /^https?:\/\//i.test(url);
markdown.renderer.rules.link_open = (tokens,index,options,_env,renderer) => {
  tokens[index].attrSet('target','_blank');
  tokens[index].attrSet('rel','noopener noreferrer');
  return renderer.renderToken(tokens,index,options);
};

function legacyLines(notes:string):string[]|undefined {
  // Existing manifests contain one plain-text change per line. Keep those as bullets.
  // Markdown documents keep their authored paragraphs, lists and other block structure.
  if (/^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>|`{3,}|~{3,}|\|)|^ {4}\S|\n[^\n]*\|[^\n]*\n\s*[:| -]+\|/m.test(notes) || /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/m.test(notes) || /^.+\n[=-]{2,}\s*$/m.test(notes)) return undefined;
  return notes.split('\n').map(line => line.trim()).filter(Boolean);
}

/** Render only Markdown-generated markup. Raw HTML and automatic remote images stay disabled. */
export function renderReleaseNotes(notes:string):string {
  const normalized = notes.replace(/\r\n?/g,'\n');
  if (!normalized.trim()) return '';
  const lines = legacyLines(normalized);
  if (!lines) return markdown.render(normalized);
  return '<ul>\n' + lines.map(line => {
    const topic = !/\*\*|__|^[a-z][a-z\d+.-]*:\/\//i.test(line) && line.match(/^([^：:]{1,36}[：:])(.*)$/);
    const content = topic ? `<strong>${markdown.utils.escapeHtml(topic[1])}</strong>${markdown.renderInline(topic[2])}` : markdown.renderInline(line);
    return `<li>${content}</li>`;
  }).join('\n') + '\n</ul>\n';
}
