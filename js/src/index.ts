import WebEyeTrack from './WebEyeTrack'
// import WebEyeTrackProxy from './WebEyeTrackProxy'
import WebcamClient from './WebcamClient'
import FaceLandmarkerClient from './FaceLandmarkerClient'
import BlazeGaze from "./BlazeGaze"

// @ts-ignore
import libTs from "worker-loader?inline=no-fallback!./worker.ts";
class WebWorkerLibTs{
    worker:Worker
    constructor(){
        this.worker = new libTs()
        this.worker.onmessage = (mess) =>{
            console.log(`[WebWorkerLibTs] ${mess.data}`)
        }
    }

    sendMessage = () =>{
        this.worker.postMessage(3 * 1000);
    }
}

export {
    WebWorkerLibTs,
    WebEyeTrack,
    WebcamClient,
    FaceLandmarkerClient,
    BlazeGaze
}