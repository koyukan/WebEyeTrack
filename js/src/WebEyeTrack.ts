export default class WebEyeTrack {
    constructor() {
      console.log('WebEyeTrack constructor');
    }
  
    /**
     * Step function that processes incoming results from the facial landmark model.
     *
     * @param {Object} result - The result object from the facial landmark model.
     * @param {Array} result.landmarks - The facial landmarks points.
     * @param {Array} result.transformationMatrix - The facial transformation matrix.
     * @param {Array} result.faceBlendshapes - The face blendshapes.
     * @param {HTMLVideoElement} frame - The original frame being processed.
     * @param {...any} inputArgs - Additional input arguments for further customization.
     */
    step(result: any, frame: HTMLVideoElement) {
    }
}