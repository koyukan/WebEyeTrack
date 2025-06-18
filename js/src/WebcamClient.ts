export default class WebcamClient {
    private videoElement: HTMLVideoElement;
    private stream?: MediaStream;
    private frameCallback?: (frame: HTMLVideoElement) => void;

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
        if (!videoElement) {
            throw new Error(`Video element with id '${videoElementId}' not found`);
        }
        this.videoElement = videoElement;
    }

    async startWebcam(frameCallback?: (frame: HTMLVideoElement) => Promise<void>): Promise<void> {
        try {
            const constraints: MediaStreamConstraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            };

            // Request webcam access
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;

            // Set the callback if provided
            if (frameCallback) {
                this.frameCallback = frameCallback;
            }

            // Start video playback
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play();
            };

            this.videoElement.addEventListener('loadeddata', () => {
                this._processFrames();
            });

        } catch (error) {
            console.error("Error accessing the webcam:", error);
        }
    }

    stopWebcam(): void {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = undefined;
        }
    }

    private _processFrames(): void {
        const process = async () => {
            if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) return;
            // console.log("Processing frame...", performance.now());

            // Call the frame callback if provided
            if (this.frameCallback) {
                await this.frameCallback(this.videoElement);
            }

            // Request the next frame
            requestAnimationFrame(process);
        };

        requestAnimationFrame(process);
    }
}