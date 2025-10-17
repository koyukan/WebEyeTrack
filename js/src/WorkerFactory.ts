export interface WorkerConfig {
  workerUrl?: string;
}

export function createWebEyeTrackWorker(config?: WorkerConfig): Worker {
  if (typeof Worker === 'undefined') {
    throw new Error(
      'Web Workers are not supported in this environment. ' +
      'WebEyeTrackProxy requires a browser environment with Worker support.'
    );
  }

  if (config?.workerUrl) {
    try {
      return new Worker(config.workerUrl);
    } catch (error) {
      throw new Error(
        `Failed to load worker from custom URL: ${config.workerUrl}. ` +
        `Error: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  try {
    const WebpackWorker = require('worker-loader?inline=no-fallback!./WebEyeTrackWorker.ts');
    return new WebpackWorker.default();
  } catch (webpackError) {
    try {
      if (typeof document !== 'undefined' && document.currentScript) {
        const scriptUrl = (document.currentScript as HTMLScriptElement).src;
        const baseUrl = scriptUrl.substring(0, scriptUrl.lastIndexOf('/'));
        const workerUrl = `${baseUrl}/webeyetrack.worker.js`;
        return new Worker(workerUrl);
      }

      if (typeof self !== 'undefined' && self.location) {
        const baseUrl = self.location.origin + self.location.pathname.substring(0, self.location.pathname.lastIndexOf('/'));
        const workerUrl = `${baseUrl}/webeyetrack.worker.js`;
        return new Worker(workerUrl);
      }

      const workerUrl = './webeyetrack.worker.js';
      return new Worker(workerUrl);
    } catch (fallbackError) {
      throw new Error(
        'Failed to create WebEyeTrack worker. Please provide a custom workerUrl in the config:\n' +
        'new WebEyeTrackProxy(webcamClient, { workerUrl: "/path/to/webeyetrack.worker.js" })\n\n' +
        'Make sure webeyetrack.worker.js is accessible from your application.\n' +
        `Webpack error: ${webpackError instanceof Error ? webpackError.message : String(webpackError)}\n` +
        `Fallback error: ${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)}`
      );
    }
  }
}
