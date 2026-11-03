(() => {
  const box = document.querySelector('.ime-download');
  if (!box) return;
  const trigger = box.querySelector('summary');
  const panel = box.querySelector('.ime-download-body');
  const canHover = matchMedia('(hover: hover) and (pointer: fine)');
  let pinned = false;
  let suppressHover = false;
  let closeTimer;

  function place() {
    if (!box.open) return;
    const anchor = trigger.getBoundingClientRect();
    const below = innerHeight - anchor.bottom;
    const above = anchor.top;
    const required = panel.scrollHeight + panel.offsetHeight - panel.clientHeight;
    const upward = below < required + 14 && above > below;
    box.dataset.placement = upward ? 'above' : 'below';
    box.style.setProperty('--ime-panel-height', `${Math.max(150, (upward ? above : below) - 14)}px`);
  }

  function open() {
    clearTimeout(closeTimer);
    box.open = true;
    trigger.setAttribute('aria-expanded', 'true');
    place();
  }

  function close() {
    clearTimeout(closeTimer);
    box.open = false;
    trigger.setAttribute('aria-expanded', 'false');
  }

  box.open = false;
  trigger.setAttribute('aria-expanded', 'false');
  box.addEventListener('pointerenter', event => {
    if (event.pointerType === 'mouse' && canHover.matches && !suppressHover) open();
  });
  box.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'mouse') return;
    suppressHover = false;
    if (!pinned) closeTimer = setTimeout(() => {
      if (!box.contains(document.activeElement)) close();
    }, 180);
  });
  panel.addEventListener('pointerenter', () => clearTimeout(closeTimer));
  trigger.addEventListener('click', event => {
    event.preventDefault();
    if (pinned) {
      pinned = false;
      suppressHover = true;
      close();
    } else {
      pinned = true;
      open();
    }
  });
  box.addEventListener('focusout', () => {
    setTimeout(() => {
      if (!pinned && !box.contains(document.activeElement) && !box.matches(':hover')) close();
    }, 0);
  });
  document.addEventListener('pointerdown', event => {
    if (!box.contains(event.target)) {
      pinned = false;
      close();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && box.open) {
      pinned = false;
      suppressHover = box.matches(':hover');
      close();
      trigger.focus({preventScroll: true});
      event.preventDefault();
    }
  });
  box.addEventListener('toggle', () => {
    trigger.setAttribute('aria-expanded', String(box.open));
    if (box.open) place();
  });
  addEventListener('resize', place, {passive: true});
  addEventListener('scroll', place, {passive: true});
  function openFromHash() {
    if (location.hash !== '#ipa-ime-download') return;
    pinned = true;
    suppressHover = false;
    open();
    box.scrollIntoView({block: 'start'});
  }
  addEventListener('hashchange', openFromHash);
  openFromHash();
})();
