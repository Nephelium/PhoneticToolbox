// Each browser document owns a streaming connection. Refresh gets a short grace
// period, and closing one of several editor tabs does not close their server.
export function pageLifetime(shutdown, graceMs = 3000) {
  const pages = new Map();
  let timer, closed = false, reservedUntil = 0;
  const cancel = () => { clearTimeout(timer); timer = undefined; };
  const schedule = (delay = graceMs) => {
    cancel();
    if (!closed && !pages.size) timer = setTimeout(() => void shutdown(), Math.max(delay, reservedUntil - Date.now()));
  };
  const detach = (id, response) => {
    if (pages.get(id) !== response) return;
    pages.delete(id);
    response.end();
    schedule();
  };
  return {
    get count() { return pages.size; },
    attach(id, response) {
      cancel();
      reservedUntil = 0;
      const previous = pages.get(id);
      pages.set(id, response);
      previous?.end();
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
      response.write(': editor connected\n\n');
      const heartbeat = setInterval(() => response.write(': alive\n\n'), 15000);
      response.once('close', () => { clearInterval(heartbeat); detach(id, response); });
    },
    leave(id) { const response = pages.get(id); if (response) detach(id, response); },
    reserve() { if (!pages.size) { reservedUntil = Date.now() + 30000; schedule(30000); } },
    dispose() { closed = true; cancel(); for (const response of pages.values()) response.end(); pages.clear(); },
  };
}
