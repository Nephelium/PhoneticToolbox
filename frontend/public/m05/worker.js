/* M05 dedicated classic worker: fixed, same-origin MediaPipe. No camera/network upload. */
let detector = null;
let generation = 0;
self.onmessage = async ({ data }) => {
  const id = data.id;
  try {
    if (data.op === 'init') {
      detector?.close(); detector = null;
      const base = new URL('./mediapipe-0.10.14/', self.location.href).href;
      // Official CJS bundle is unmodified; only its CommonJS exports object is supplied.
      self.exports = {};
      importScripts(base + 'vision_bundle.classic.js');
      const { FaceLandmarker, FilesetResolver } = self.exports;
      const started = performance.now();
      detector = await FaceLandmarker.createFromOptions(await FilesetResolver.forVisionTasks(base + 'wasm'), {
        baseOptions: { modelAssetPath: base + 'face_landmarker.task', delegate: data.delegate || 'CPU' },
        runningMode: data.mode || 'VIDEO', numFaces: 1,
        minFaceDetectionConfidence: .5, minFacePresenceConfidence: .5, minTrackingConfidence: .5,
        outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
      });
      generation++;
      let graphics = null;
      try {
        const context = new OffscreenCanvas(1, 1).getContext('webgl2');
        if (context) { const debug = context.getExtension('WEBGL_debug_renderer_info');
          graphics = { probe: 'independent WebGL2 support context; not model profiler',
            renderer: context.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : context.RENDERER),
            vendor: context.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : context.VENDOR) };
          context.getExtension('WEBGL_lose_context')?.loseContext();
        }
      } catch (_) { /* A CPU delegate does not require this diagnostic context. */ }
      self.postMessage({ id, ok: true, value: { generation, delegate: data.delegate || 'CPU', mode: data.mode || 'VIDEO',
        init_ms: performance.now() - started, worker: true, offscreen: typeof OffscreenCanvas !== 'undefined',
        wasm: typeof WebAssembly !== 'undefined', modelSmoothing: (data.mode || 'VIDEO') === 'VIDEO',
        graphics, resource_timing: performance.getEntriesByType('resource').map(r => ({name:r.name,transfer_bytes:r.transferSize,encoded_bytes:r.encodedBodySize,duration_ms:r.duration})),
        backend: 'mediapipe-web/0.10.14/float16-1/candidate' } });
    } else if (data.op === 'frame') {
      if (!detector) throw Error('detector_not_initialized');
      const started = performance.now();
      try {
        const result = data.mode === 'IMAGE' ? detector.detect(data.frame) : detector.detectForVideo(data.frame, data.time_ms);
        self.postMessage({ id, ok: true, value: { points: result.faceLandmarks[0]?.map(p => [p.x, p.y, p.z]) || null,
          inference_ms: performance.now() - started, time_ms: data.time_ms, generation } });
      } finally { data.frame.close(); }
    } else if (data.op === 'close') {
      detector?.close(); detector = null; self.postMessage({ id, ok: true, value: null });
    } else throw Error('unsupported_worker_operation');
  } catch (error) {
    data.frame?.close();
    self.postMessage({ id, ok: false, error: String(error?.message || error) });
  }
};
