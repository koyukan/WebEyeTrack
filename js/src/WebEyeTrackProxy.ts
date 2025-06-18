// export default class WebEyeTrackProxy {
//   private worker: Worker;

//   constructor() {
//     this.worker = new Worker(new URL('./WebEyeTrackWorker.js', import.meta.url));
//     console.log('WebEyeTrackProxy worker initialized');

//     this.worker.onmessage = function (e) {
//       console.log(e.data)
//     }
//   }
// }
