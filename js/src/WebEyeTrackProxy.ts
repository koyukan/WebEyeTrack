export default class WebEyeTrackProxy {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(new URL('./WebEyeTrackWorker.ts', import.meta.url), {
      type: 'module',
    });
    console.log('WebEyeTrackProxy worker initialized');

    this.worker.onmessage = function (e) {
      console.log(e.data)
    }
  }
}
