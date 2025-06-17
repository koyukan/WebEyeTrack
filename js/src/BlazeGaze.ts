import * as tf from '@tensorflow/tfjs';

// References
// https://js.tensorflow.org/api/latest/#class:LayersModel

export default class BlazeGaze {
    // private model: tf.GraphModel | null = null;
    private model: tf.LayersModel | null = null;  // Use LayersModel for tf.loadLayersModel

    constructor() {
        // Optionally trigger model load in constructor
    }

    async loadModel(): Promise<void> {
        try {
            // Load model from local directory (adjust path if needed)
            this.model = await tf.loadLayersModel('./web/model.json');
            console.log('✅ BlazeGaze model loaded successfully');
        } catch (error) {
            console.error('❌ Error loading BlazeGaze model at web:', error);
            console.error(error);
            throw error;
        }
        
        // Freeze the ``cnn_model`` layers but keep the gaze_MLP trainable
        this.model.getLayer('cnn_encoder').trainable = false;
    }

    predict(image: tf.Tensor, head_vector: tf.Tensor, face_origin_3d: tf.Tensor): tf.Tensor {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        const inputList: tf.Tensor[] = [image, head_vector, face_origin_3d];

        // Run inference
        const output = this.model.predict(inputList) as tf.Tensor | tf.Tensor[];  // GraphModel always returns Tensor or Tensor[]

        if (Array.isArray(output)) {
            return output[0];  // Return the first tensor if multiple
        }

        return output;
    }
}
