type FaceLandmark = {
    x: number;
    y: number;
    z: number;
    visibility: number;
    confidence: number;
}

type FaceLandmarkerResults = {
    faceLandmarks: FaceLandmark[];
}