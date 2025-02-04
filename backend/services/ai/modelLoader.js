const ort = require('onnxruntime-node');
const path = require('path');
const sharp = require('sharp');

class ModelLoader {
  constructor() {
    this.model = null;
    this.modelPath = path.join(__dirname, '../../model', 'last.onnx');
    this.targetSize = 640; // Changed from 224 to 640
  }

  async loadModel() {
    try {
      if (!this.model) {
        console.log('Loading model from:', this.modelPath);
        this.model = await ort.InferenceSession.create(this.modelPath);
        console.log('Model input names:', this.model.inputNames);
        console.log('Model loaded successfully');
      }
      return this.model;
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error(`Failed to load AI model: ${error.message}`);
    }
  }

  async preprocessImage(imageBuffer) {
    try {
      const processedImage = await sharp(imageBuffer)
        .resize(this.targetSize, this.targetSize, { fit: 'cover' }) // Now 640x640
        .removeAlpha()
        .raw()
        .toBuffer();

      // Convert HWC to CHW
      const float32Data = new Float32Array(3 * this.targetSize * this.targetSize);
      for (let c = 0; c < 3; c++) {
        for (let h = 0; h < this.targetSize; h++) {
          for (let w = 0; w < this.targetSize; w++) {
            const srcIdx = h * this.targetSize * 3 + w * 3 + c;
            const dstIdx = c * this.targetSize * this.targetSize + h * this.targetSize + w;
            float32Data[dstIdx] = processedImage[srcIdx] / 255.0;
          }
        }
      }

      return float32Data;
    } catch (error) {
      throw new Error(`Image preprocessing failed: ${error.message}`);
    }
  }

  async predict(imageBuffer) {
    try {
      const model = await this.loadModel();
      const preprocessedData = await this.preprocessImage(imageBuffer);

      const inputTensor = new ort.Tensor(
        'float32',
        preprocessedData,
        [1, 3, this.targetSize, this.targetSize] // Now [1, 3, 640, 640]
      );

      const feeds = { [model.inputNames[0]]: inputTensor };
      const results = await model.run(feeds);

      const outputData = Object.values(results)[0].data;
      return this.formatResults(Array.from(outputData));
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Failed to process image: ${error.message}`);
    }
  }

  formatResults(predictions) {
    const conditions = [
      "Acne",
      "Basal-Cell-Carcinoma",
      "Darier's-Disease",
      "Eczema",
      "Epidermolysis-Bullosa-Pruriginosa",
      "Hailey-Hailey-Disease",
      "Capture-d-ecran",
      "Impetigo",
      "LEISHMANIOSE",
      "Lichen",
      "Lupus-Erythematosus-Chronicus-Discoides",
      "Melanoma",
      "Molluscum-Contagiosum",
      "Nevus",
      "Normal",
      "Porokeratosis-Actinic",
      "Psoriasis",
      "Tinea-Corporis",
      "Tungiasis"
    ];

    return predictions.map((probability, index) => ({
      condition: conditions[index] || `Unknown Condition ${index}`,
      probability: parseFloat(probability.toFixed(4)),
    })).sort((a, b) => b.probability - a.probability);
  }
}

module.exports = new ModelLoader();
