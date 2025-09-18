# WebEyeTrack in JS/TS

Created by <a href="https://edavalosanaya.github.io" target="_blank">Eduardo Davalos</a>, <a href="https://scholar.google.com/citations?user=_E0SGAkAAAAJ&hl=en" target="_blank">Yike Zhang</a>, <a href="https://scholar.google.com/citations?user=GWvdYIoAAAAJ&hl=en&oi=ao" target="_blank">Namrata Srivastava</a>, <a href="https://www.linkedin.com/in/yashvitha/" target="_blank">Yashvitha Thatigolta</a>, <a href="" target="_blank">Jorge A. Salas</a>, <a href="https://www.linkedin.com/in/sara-mcfadden-93162a4/" target="_blank">Sara McFadden</a>, <a href="https://scholar.google.com/citations?user=0SHxelgAAAAJ&hl=en" target="_blank">Cho Sun-Joo</a>, <a href="https://scholar.google.com/citations?user=dZ8X7mMAAAAJ&hl=en" target="_blank">Amanda Goodwin</a>, <a href="https://sites.google.com/view/ashwintudur/home" target="_blank">Ashwin TS</a>, and <a href="https://scholar.google.com/citations?user=-m5wrTkAAAAJ&hl=en" target="_blank">Guatam Biswas</a> from <a href="https://wp0.vanderbilt.edu/oele/" target="_blank">Vanderbilt University</a>, <a href="https://redforestai.github.io" target="_blank">Trinity University</a>, <a href="https://knotlab.github.io/KnotLab/" target="_blank">St. Mary's University</a>

### [Project](https://redforestai.github.io/WebEyeTrack) | [Paper](https://arxiv.org/abs/2508.19544) | [Demo](https://azure-olympie-5.tiiny.site)

<p></p>

[![NPM Version](https://img.shields.io/npm/v/webeyetrack)](https://www.npmjs.com/package/webeyetrack) [![PyPI - Version](https://img.shields.io/pypi/v/webeyetrack)](https://pypi.org/project/webeyetrack/) [![GitHub License](https://img.shields.io/github/license/RedForestAI/webeyetrack)](#license)

The JS/TS implementation of WebEyeTrack uses a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) to offload the AI inference to an isolated worker thread, preventing the main UI thread to become unresponsive. Lastly, we made the ``webeyetrack`` independent of a UI framework such as React, Vue, or Angular.

Additionally, you can combine ``webeyetrack``, a gaze estimation JS library, with [``webfixrt``](https://github.com/redforestai/webfixrt), an online fixation detection JS library, to extract fixation and saccade information as a real-time stream.

# Getting Started

Install the npm package running the following command:

```bash
npm install webeyetrack
```

# Usage

To use WebEyeTrack, we provide a webcam client solution to support the streaming of frames into the tracker.

```ts
const webcamClient = new WebcamClient('video'); // id of HTMLVideoElement
const webEyeTrackProxy = new WebEyeTrackProxy(webcamClient);
```

Then you define the callback function once gaze estimation results are available:

```ts
webEyeTrackProxy.onGazeResults = (gazeResult: GazeResult) => {
  console.log(gazeResult)
}
```

### GazeResult Interface (TypeScript)

| Field Name       | Type                          | Shape / Values      | Description                                         |
|------------------|-------------------------------|----------------------|-----------------------------------------------------|
| facialLandmarks  | `NormalizedLandmark[]`        | [N, 5]              | Facial landmarks, normalized to screen or image     |
| faceRt           | `Matrix`                      | [4, 4]                  | Face rotation-translation matrix                   |
| faceBlendshapes  | `Classifications[]`           | [N, 1]              | Blendshape classification outputs                  |
| eyePatch         | `ImageData`                   | [H=512, W=128, 3]           | RGB image of the eye region                        |
| headVector       | `number[]`                    | [3]                 | Head direction vector in camera coordinates        |
| faceOrigin3D     | `number[]`                    | [3] (X, Y, Z)       | 3D origin of the face in space                     |
| metric_transform | `Matrix`                      | [4, 4]              | Transformation matrix applied to reconstructed face|
| gazeState        | `'open'` \| `'closed'`        | Enum-like Literal   | Current eye state (e.g., blink detection)          |
| normPog          | `number[]`                    | [2] (X, Y)          | Normalized point-of-gaze on the screen             |
| durations        | `Record<string, number>`      | —                   | Timing metadata for each processing stage (seconds)|
| timestamp        | `number`                      | —                   | Timestamp in milliseconds of video start (relative) |

The normalized PoG is from range ``[[-0.5, 0.5], [-0.5, 0.5]]`` where the origin ``(0,0)`` is located at the center of the screen. The positive Y axis is pointing downwards and the positive X axis is pointing toward the right.

# Demo

Try out the demo located within the ``example-app`` directory by running the following commands:

```bash
cd js
# Build the ``webeyetrack`` bundle locally
npm install
npm run build 

# Run the example React app
cd example-app
npm install
npm run start
```

Then you should be able to visit the React application at [``https://localhost:3000``](https://localhost:3000)

# Acknowledgements

The research reported here was supported by the Institute of Education Sciences, U.S. Department of Education, through Grant R305A150199 and R305A210347 to Vanderbilt University. The opinions expressed are those of the authors and do not represent views of the Institute or the U.S. Department of Education.

# Reference

If you use this work in your research, please cite us using the following:

```bibtex
@misc{davalos2025webeyetrack,
	title={WEBEYETRACK: Scalable Eye-Tracking for the Browser via On-Device Few-Shot Personalization},
	author={Eduardo Davalos and Yike Zhang and Namrata Srivastava and Yashvitha Thatigotla and Jorge A. Salas and Sara McFadden and Sun-Joo Cho and Amanda Goodwin and Ashwin TS and Gautam Biswas},
	year={2025},
	eprint={2508.19544},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	url={https://arxiv.org/abs/2508.19544}
}
```

# License

WebEyeTrack is open-sourced under the [MIT License](LICENSE), which permits personal, academic, and commercial use with proper attribution. Feel free to use, modify, and distribute the project.