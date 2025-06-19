import WebEyeTrack from './WebEyeTrack';

let tracker: WebEyeTrack;

const ctx: Worker = self as any;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      tracker = new WebEyeTrack();
      await tracker.initialize();
      self.postMessage({ type: 'ready' });
      break;

    case 'step':
      const result = await tracker.step(payload.frame as ImageData);
      self.postMessage({ type: 'stepResult', result });
      break;
    
    case 'click':
      // Handle click event for re-calibration
      tracker.handleClick(payload.x, payload.y);
      break;

    default:
      console.warn(`[WebEyeTrackWorker] Unknown message type: ${type}`);
      break;
  }
};

export {}; // for TS module mode
