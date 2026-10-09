// Shared by the Vue shell and the standalone vocal-tract document.
(() => {
  const controls = 'button,summary,[role="button"],[role="tab"],[role="option"],[role="separator"],[role="img"],[data-no-text-selection],svg,canvas,img,input[type="button"],input[type="submit"],input[type="reset"],input[type="checkbox"],input[type="radio"],input[type="range"],input[type="color"]';
  const textInputs = 'textarea,input:not([type="button"]):not([type="submit"]):not([type="reset"]):not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="color"]),[contenteditable=""],[contenteditable="true"]';
  const root = document.documentElement;
  let pointer;
  function target(event) {
    return event.target instanceof Element ? event.target : event.target?.parentElement;
  }
  function protectedTarget(event) {
    const node = target(event);
    return node && !node.closest(textInputs) && node.closest(controls);
  }
  function clearSelection() { document.getSelection()?.removeAllRanges(); }
  function release() { pointer = undefined; root.removeAttribute('data-ptb-selection-lock'); }
  document.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !protectedTarget(event)) return;
    pointer = event.pointerId;
    root.setAttribute('data-ptb-selection-lock', '');
    clearSelection();
  }, true);
  document.addEventListener('mousedown', event => {
    if (event.button !== 0 || (pointer === undefined && !protectedTarget(event))) return;
    // Native sliders/checkboxes retain their browser pointer behavior.
    if (target(event)?.closest('input:not([type="button"]):not([type="submit"]):not([type="reset"]),select')) return;
    // Cancel native range extension, including Shift-click from an old text range.
    // Pointer handlers and click/double-click handlers continue to receive events.
    event.preventDefault();
    const node = target(event)?.closest('button,input,select,summary,[tabindex],a[href]');
    if (node && !node.matches(':disabled') && !node.closest('[inert]')) node.focus?.({preventScroll:true});
    clearSelection();
  }, true);
  document.addEventListener('selectstart', event => {
    if (pointer !== undefined || protectedTarget(event)) event.preventDefault();
  }, true);
  for (const name of ['pointerup', 'pointercancel']) {
    document.addEventListener(name, event => { if (event.pointerId === pointer) release(); }, true);
  }
  document.addEventListener('pointermove', event => {
    if (event.pointerId === pointer && event.buttons === 0) release();
  }, true);
  window.addEventListener('blur', release);
  document.addEventListener('visibilitychange', () => { if (document.hidden) release(); });
})();
