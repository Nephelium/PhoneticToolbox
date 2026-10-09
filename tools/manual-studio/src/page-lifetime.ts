import { session } from './api';

export function bindPageLifetime(onError: (message: string) => void) {
  const id = crypto.randomUUID();
  let controller: AbortController | undefined, leaving = false, disposed = false;
  async function connect() {
    controller?.abort();
    const connection = new AbortController();
    controller = connection;
    try {
      const response = await fetch('/api/page-live?id=' + id, { headers: { Authorization: 'Bearer ' + session }, signal: connection.signal });
      if (!response.ok || !response.body) throw new Error('连接失败');
      const reader = response.body.getReader();
      while (!(await reader.read()).done) { /* Keep this document's connection open. */ }
      if (!leaving && !disposed) onError('编辑器服务已退出，请双击启动器重新打开。');
    } catch {
      if (!leaving && !disposed && !connection.signal.aborted) onError('无法连接编辑器服务，请双击启动器重新打开。');
    }
  }
  function leave() {
    if (leaving) return;
    leaving = true;
    const body = new Blob([JSON.stringify({ capability: session, id })], { type: 'application/json' });
    navigator.sendBeacon('/api/page-close', body);
    controller?.abort();
  }
  function resume(event: PageTransitionEvent) { if (event.persisted && !disposed) { leaving = false; void connect(); } }
  window.addEventListener('pagehide', leave);
  window.addEventListener('pageshow', resume);
  void connect();
  return () => { disposed = true; leave(); window.removeEventListener('pagehide', leave); window.removeEventListener('pageshow', resume); };
}
