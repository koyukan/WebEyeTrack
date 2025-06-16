import * as tf from '@tensorflow/tfjs';

export default class BlazeGaze {
    private model: tf.GraphModel | null = null;

    constructor() {
        // Optionally trigger model load in constructor
    }

    async loadModel(): Promise<void> {
        try {
            // Load model from local directory (adjust path if needed)
            this.model = await tf.loadGraphModel('./web/model.json');
            console.log('✅ BlazeGaze model loaded successfully');
        } catch (error) {
            console.error('❌ Error loading BlazeGaze model:', error);
            throw error;
        }
    }

    async predict(image: tf.Tensor, head_vector: tf.Tensor, face_origin_3d: tf.Tensor): Promise<tf.Tensor | null> {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Use a tf.NamedTensorMap if your model expects named inputs
        const inputs: tf.NamedTensorMap = {
            image: image,
            head_vector: head_vector,
            face_origin_3d: face_origin_3d
        };

        // Run inference
        const output = this.model.predict(inputs) as tf.Tensor | tf.Tensor[];  // GraphModel always returns Tensor or Tensor[]

        if (Array.isArray(output)) {
            return output[0];  // Return the first tensor if multiple
        }

        return output;
    }
}
