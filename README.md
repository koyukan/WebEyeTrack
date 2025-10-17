# WebEyeTrack

Created by <a href="https://edavalosanaya.github.io" target="_blank">Eduardo Davalos</a>, <a href="https://scholar.google.com/citations?user=_E0SGAkAAAAJ&hl=en" target="_blank">Yike Zhang</a>, <a href="https://scholar.google.com/citations?user=GWvdYIoAAAAJ&hl=en&oi=ao" target="_blank">Namrata Srivastava</a>, <a href="https://www.linkedin.com/in/yashvitha/" target="_blank">Yashvitha Thatigolta</a>, <a href="" target="_blank">Jorge A. Salas</a>, <a href="https://www.linkedin.com/in/sara-mcfadden-93162a4/" target="_blank">Sara McFadden</a>, <a href="https://scholar.google.com/citations?user=0SHxelgAAAAJ&hl=en" target="_blank">Cho Sun-Joo</a>, <a href="https://scholar.google.com/citations?user=dZ8X7mMAAAAJ&hl=en" target="_blank">Amanda Goodwin</a>, <a href="https://sites.google.com/view/ashwintudur/home" target="_blank">Ashwin TS</a>, and <a href="https://scholar.google.com/citations?user=-m5wrTkAAAAJ&hl=en" target="_blank">Guatam Biswas</a> from <a href="https://wp0.vanderbilt.edu/oele/" target="_blank">Vanderbilt University</a>, <a href="https://redforestai.github.io" target="_blank">Trinity University</a>, and <a href="https://knotlab.github.io/KnotLab/" target="_blank">St. Mary's University</a>

### [Project](https://redforestai.github.io/WebEyeTrack) | [Paper](https://arxiv.org/abs/2508.19544) | [Demo](https://azure-olympie-5.tiiny.site)

<p></p>

[![NPM Version](https://img.shields.io/npm/v/webeyetrack)](https://www.npmjs.com/package/webeyetrack) [![PyPI - Version](https://img.shields.io/pypi/v/webeyetrack)](https://pypi.org/project/webeyetrack/) [![GitHub License](https://img.shields.io/github/license/RedForestAI/webeyetrack)](#license)

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

* [Python ``webeyetrack`` PYPI package](./python)
* [JavaScript ``webeyetrack`` NPM package](./js)

# Memory Management

WebEyeTrack uses TensorFlow.js for real-time gaze estimation, which requires careful memory management to prevent memory leaks during long-running sessions. All core classes implement the `IDisposable` interface for proper resource cleanup.

## Key Principles

### 1. Always Call `dispose()` When Done

All WebEyeTrack components must be explicitly disposed to release GPU/CPU resources:

```typescript
import { WebcamClient, WebEyeTrackProxy } from 'webeyetrack';

// Initialize
const webcamClient = new WebcamClient('webcam');
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient);

// ... use the tracker ...

// Clean up when done
eyeTrackProxy.dispose();  // Terminates worker, removes event listeners
webcamClient.dispose();   // Stops webcam, cancels animation frames
```

### 2. React Integration Pattern

For React applications, use the cleanup function in `useEffect`:

```typescript
useEffect(() => {
  let webcamClient: WebcamClient | null = null;
  let eyeTrackProxy: WebEyeTrackProxy | null = null;

  const initialize = async () => {
    webcamClient = new WebcamClient('webcam');
    eyeTrackProxy = new WebEyeTrackProxy(webcamClient);
    // ... setup callbacks ...
  };

  initialize();

  // Cleanup on unmount
  return () => {
    eyeTrackProxy?.dispose();
    webcamClient?.dispose();
  };
}, []);
```

### 3. Error Handling with Error Boundaries

Wrap your application with `MemoryCleanupErrorBoundary` to ensure cleanup on errors:

```typescript
import { MemoryCleanupErrorBoundary } from 'webeyetrack';

function App() {
  return (
    <MemoryCleanupErrorBoundary onCleanup={() => console.log('Cleaned up')}>
      <YourApp />
    </MemoryCleanupErrorBoundary>
  );
}
```

### 4. Memory Monitoring

Use the `MemoryMonitor` utility to track TensorFlow.js memory usage:

```typescript
import { MemoryMonitor } from 'webeyetrack';

const monitor = new MemoryMonitor();
monitor.captureBaseline();

// ... perform calibration ...

const report = monitor.checkForLeaks();
console.log(`Tensor leak: ${report.tensorLeak} tensors`);
console.log(`Byte leak: ${report.byteLeak} bytes`);
```

## Best Practices

- **Dispose in reverse order**: Dispose child components before parent components
- **Check `isDisposed`**: Verify disposal state before operations
- **Long sessions**: Monitor memory periodically during extended sessions
- **Component unmounting**: Always dispose resources in React cleanup functions
- **Worker threads**: Ensure workers receive disposal messages before termination

## Common Issues

### Memory Growing Over Time
- Ensure `dispose()` is called on all instances
- Check for circular references preventing garbage collection
- Monitor calibration data accumulation

### Tensors Not Released
- Verify all TensorFlow.js operations use `tf.tidy()` where appropriate
- Ensure custom tensor operations call `tf.dispose()`
- Check that model predictions are properly disposed after use

## Resources Managed

The following resources are automatically managed through `dispose()`:

- **TensorFlow.js tensors**: Model weights, calibration data, intermediate computations
- **Event listeners**: Window events, mouse handlers, message handlers
- **Animation frames**: RequestAnimationFrame loops for video processing
- **Media streams**: Webcam tracks and video elements
- **Worker threads**: Background processing threads

For more details, see the [JavaScript package documentation](./js).

# Worker Configuration

WebEyeTrack runs the eye-tracking model in a Web Worker for better performance. The worker is automatically managed, but you can customize its loading behavior for advanced use cases.

## Default Behavior

By default, WebEyeTrackProxy automatically loads the worker:

```typescript
const webcamClient = new WebcamClient('webcam');
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient);
// Worker is loaded automatically
```

## Custom Worker URL

For production deployments or CDN hosting, specify a custom worker URL:

```typescript
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient, {
  workerUrl: '/static/webeyetrack.worker.js'
});
```

## Vite Configuration

When using Vite, you need to:
1. Exclude webeyetrack from dependency optimization (to avoid webpack-specific code)
2. Copy the worker to your public directory

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import { copyFileSync } from 'fs'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    {
      name: 'copy-worker',
      buildStart() {
        const src = resolve(__dirname, 'node_modules/webeyetrack/dist/webeyetrack.worker.js')
        const dest = resolve(__dirname, 'public/webeyetrack.worker.js')
        copyFileSync(src, dest)
      }
    }
  ],
  optimizeDeps: {
    exclude: ['webeyetrack']  // Important: prevents Vite from pre-bundling
  }
})
```

Then specify the worker URL in your application:

```typescript
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient, {
  workerUrl: '/webeyetrack.worker.js'
});
```

## Webpack / Create React App

Webpack-based projects (including Create React App) automatically bundle the worker. No configuration needed:

```typescript
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient);
// Worker is bundled inline automatically
```

## CDN Deployment

When serving from a CDN, ensure `webeyetrack.worker.js` is accessible:

```typescript
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient, {
  workerUrl: 'https://cdn.example.com/webeyetrack.worker.js'
});
```

## Worker Bundle Details

- **Size**: ~961KB (minified)
- **Dependencies**: TensorFlow.js (bundled), MediaPipe (loaded from CDN)
- **Format**: UMD (works in all environments)
- **Source Maps**: Available for debugging

## Troubleshooting

### Worker fails to load

If you see worker loading errors:

1. Check that `webeyetrack.worker.js` is accessible from your application
2. Verify the worker URL path is correct (absolute or relative to your HTML)
3. Check browser console for CORS errors
4. Ensure the worker file is served with correct MIME type (`application/javascript`)

### Example error and fix:

```
Error: Failed to create WebEyeTrack worker
```

**Solution**: Provide explicit worker URL:

```typescript
const eyeTrackProxy = new WebEyeTrackProxy(webcamClient, {
  workerUrl: window.location.origin + '/webeyetrack.worker.js'
});
```

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