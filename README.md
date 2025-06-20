# WebEyeTrack
Created by <a href="https://edavalosanaya.github.io" target="_blank">Eduardo Davalos</a>, <a href="https://scholar.google.com/citations?user=_E0SGAkAAAAJ&hl=en" target="_blank">Yike Zhang</a>, <a href="https://www.linkedin.com/in/yashvitha/" target="_blank">Yashvitha Thatigolta</a>, <a href="https://scholar.google.com/citations?user=GWvdYIoAAAAJ&hl=en&oi=ao" target="_blank">Namrata Srivastava</a>, <a href="" target="_blank">Jorge A. Salas</a>, <a href="https://www.linkedin.com/in/sara-mcfadden-93162a4/" target="_blank">Sara McFadden</a>, <a href="https://scholar.google.com/citations?user=0SHxelgAAAAJ&hl=en" target="_blank">Cho Sun-Joo</a>, <a href="https://scholar.google.com/citations?user=dZ8X7mMAAAAJ&hl=en" target="_blank">Amanda Goodwin</a>, and <a href="https://scholar.google.com/citations?user=-m5wrTkAAAAJ&hl=en" target="_blank">Guatam Biswas</a> from <a href="https://wp0.vanderbilt.edu/oele/" target="_blank">Vanderbilt University</a>

### [Project](https://) | [Paper](https://) | [Demo](https://)

WebEyeTrack is a framework that uses a lightweight CNN-based neural network to predict the ``(x,y)`` gaze point on the screen. The framework provides both a Python and JavaScript/TypeScript (client-side) versions to support research/testing and deployment via TS/JS. It performs few-shot gaze estimation by collecting samples on-device to adapt the model to account for unseen persons.

# Getting Started

Deciding which version of WebEyeTrack depends on your purpose and target platform. Here is a table to help you determine which version to use: 

| Feature              | Python Version                       | JavaScript Version                     |
|----------------------|--------------------------------------|----------------------------------------|
| **Purpose**          | Training, Research, and Testing      | Deployment and Production              |
| **Primary Use Case** | Model development and experimentation | Real-time inference in the browser     |
| **Supported Devices**| CPU & GPU (desktop/server)           | CPU (Web browser, mobile)    |
| **Model Access**     | Full access to model internals       | Optimized for on-device inference and training |
| **Extensibility**    | Highly customizable (e.g., few-shot learning, adaptation) | Minimal, focused on performance        |
| **Frameworks**       | TensorFlow / Keras                   | TensorFlow.js                          |
| **Data Handling**    | Direct access to datasets and logs   | Webcam stream, UI input                |

Go to the README (links below) to the corresponding Python/JS version to get stared using these packages.

* [Python ``webeyetrack`` PYPI package](./python/README.md)
* [JavaScript ``webeyetrack`` NPM package](./js/README.md)

# Acknowledgements

The research reported here was supported by the Institute of Education Sciences, U.S. Department of Education, through Grant R305A150199 and R305A210347 to Vanderbilt University. The opinions expressed are those of the authors and do not represent views of the Institute or the U.S. Department of Education.

# License

WebEyeTrack is open-sourced under the [MIT License](LICENSE), which permits personal, academic, and commercial use with proper attribution. Feel free to use, modify, and distribute the project.