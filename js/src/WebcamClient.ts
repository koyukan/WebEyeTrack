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

    async startWebcam(frameCallback?: (frame: HTMLVideoElement) => void): Promise<void> {
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
                this._processFrames();
            };
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

    // This method can be overridden in a subclass or passed as a callback
    onFrame(frame: HTMLVideoElement): void {
        // Default implementation: simply log that a frame is being processed
        console.log("Processing frame...");
    }

    private async _processFrames(): Promise<void> {
        if (!this.videoElement) return;

        // Process frames at a set interval (e.g., every 100ms)
        const processInterval = 100;
        setInterval(() => {
            if (this.frameCallback) {
                // If a frame callback is provided, call it with the video element
                this.frameCallback(this.videoElement);
            } else {
                // Otherwise, use the default onFrame method
                this.onFrame(this.videoElement);
            }
        }, processInterval);
    }
}
