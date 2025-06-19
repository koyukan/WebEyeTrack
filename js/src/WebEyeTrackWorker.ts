// import WebEyeTrack from './WebEyeTrack';

// // @ts-ignore
// const ctx: Worker = self as any;

// onmessage = async (event) => {
//     ctx.postMessage(`[WORKER_TS] ping`)
// }
import WebEyeTrack from './WebEyeTrack';

let tracker: WebEyeTrack;

const ctx: Worker = self as any;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'init') {
    tracker = new WebEyeTrack(null, null);
    await tracker.initialize();
    self.postMessage({ type: 'ready' });
  }

  if (type === 'step') {
    const result = await tracker.step(payload.frame as ImageData);
    self.postMessage({ type: 'stepResult', result });
  }
};

export {}; // for TS module mode
