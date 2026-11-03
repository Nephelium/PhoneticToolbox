// Ordinary links remain usable when JavaScript or browser storage is unavailable.
(() => {
  const picker = document.querySelector('.language-picker');
  if (!picker) return;
  const summary = picker.querySelector('summary');
  const options = [...picker.querySelectorAll('.language-option')];
  const header = picker.closest('.site-header');
  const navigation = header.querySelector('nav');
  let headerFrame = 0;
  function fitHeader() {
    headerFrame = 0;
    if (innerWidth <= 800) { header.dataset.layout = 'stacked'; return; }
    // Measure the existing labels with their real fonts and spacing. An inert,
    // off-screen copy cannot receive input or take a row in the header.
    const measure = navigation.cloneNode(true);
    measure.setAttribute('aria-hidden', 'true');
    measure.inert = true;
    measure.removeAttribute('id');
    measure.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
    Object.assign(measure.style, {position:'fixed',left:'-100000px',top:'0',visibility:'hidden',display:'flex',flexWrap:'nowrap',width:'max-content',maxWidth:'none'});
    measure.querySelectorAll(':scope > a').forEach(node => Object.assign(node.style,{width:'max-content',maxWidth:'none',flex:'none',whiteSpace:'nowrap'}));
    header.append(measure);
    const width = measure.getBoundingClientRect().width;
    measure.remove();
    const style = getComputedStyle(header);
    const available = header.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
    const required = header.querySelector('.brand').getBoundingClientRect().width + header.querySelector('.header-actions').getBoundingClientRect().width + width + 2*parseFloat(style.columnGap);
    header.dataset.layout = required <= available-8 ? 'single' : 'stacked';
  }
  function queueHeader() { if (!headerFrame) headerFrame = requestAnimationFrame(fitHeader); }
  addEventListener('resize', queueHeader, {passive:true});
  document.fonts.ready.then(queueHeader);
  fitHeader();
  summary.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      picker.open = true;
      options[0]?.focus();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && picker.open) {
      picker.open = false;
      summary.focus({preventScroll: true});
    }
  });
  document.addEventListener('pointerdown', event => {
    if (!picker.contains(event.target)) picker.open = false;
  });
  const images = document.getElementById('feedback-images');
  const choose = document.getElementById('feedback-images-choose');
  const name = document.getElementById('feedback-file-name');
  if (images && choose && name) {
    images.hidden = true;
    choose.hidden = name.hidden = false;
    const update = () => { name.textContent = [...images.files].map(file => file.name).join(', ') || name.dataset.emptyLabel; };
    choose.addEventListener('click', () => images.click());
    images.addEventListener('change', update);
    document.getElementById('feedback-images-clear')?.addEventListener('click', () => setTimeout(update, 0));
  }
  options.forEach(link => link.addEventListener('click', () => {
    const target = new URL(link.href);
    target.hash = '';
    link.href = target.href;
    try { localStorage.setItem('ptb-website-language', link.hreflang); } catch {}
  }));
})();
