const ort = require('onnxruntime-node');
const path = require('path');
const sharp = require('sharp');
const Canvas = require('canvas');
const fs = require('fs');

class ModelLoader {
  constructor() {
    this.model = null;
    this.modelPath = path.join(__dirname, '../../model/last.onnx');
    this.targetSize = 640;
    this.isLoading = false;
    this.conditions = [
      "Acne", "Basal-Cell-Carcinoma", "Darier's-Disease", "Eczema", 
      "Epidermolysis-Bullosa-Pruriginosa", "Hailey-Hailey-Disease", 
      "Capture-d-ecran", "Impetigo", "LEISHMANIOSE", "Lichen", 
      "Lupus-Erythematosus-Chronicus-Discoides", "Melanoma", 
      "Molluscum-Contagiosum", "Nevus", "Normal", 
      "Porokeratosis-Actinic", "Psoriasis", "Tinea-Corporis", "Tungiasis"
    ];
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  async loadModel() {
    try {
      if (!this.model && !this.isLoading) {
        this.isLoading = true;
        const options = {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all'
        };
        this.model = await ort.InferenceSession.create(this.modelPath, options);
        this.isLoading = false;
      }
      return this.model;
    } catch (error) {
      this.isLoading = false;
      throw new Error(`Failed to load AI model: ${error.message}`);
    }
  }

  processBase64Image(base64Image) {
    try {
      // Handle Buffer or Blob directly
      if (base64Image instanceof Buffer) {
        return base64Image;
      }

      // Handle different string formats
      if (typeof base64Image !== 'string') {
        throw new Error('Invalid image input');
      }

      // Remove data URI prefix if present
      const base64Data = base64Image.includes(',') 
        ? base64Image.split(',')[1] 
        : base64Image;

      // Decode Base64 to buffer
      return Buffer.from(base64Data, 'base64');
    } catch (error) {
      throw new Error(`Base64 image processing failed: ${error.message}`);
    }
  }

  async preprocessImage(imageBuffer) {
    try {
      const metadata = await sharp(imageBuffer).metadata();
      const originalWidth = metadata.width;
      const originalHeight = metadata.height;

      const processedImage = await sharp(imageBuffer)
        .resize(this.targetSize, this.targetSize, {
          fit: 'contain',
          background: { r: 114, g: 114, b: 114 }
        })
        .removeAlpha()
        .raw()
        .toBuffer();

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

      return {
        tensor: float32Data,
        originalDimensions: { width: originalWidth, height: originalHeight }
      };
    } catch (error) {
      throw new Error(`Image preprocessing failed: ${error.message}`);
    }
  }

  processDetections(output, originalDimensions, confidenceThreshold = 0.5) {
    const data = Array.from(output.data);
    const numClasses = this.conditions.length;
    const predictions = [];
    
    // Softmax normalization
    const softmaxScores = data.map((score, index) => {
      const classIndex = index % numClasses;
      const exp = Math.exp(score);
      return {
        condition: this.conditions[classIndex],
        score: exp
      };
    });
  
    // Group and sum scores by condition
    const groupedScores = softmaxScores.reduce((acc, item) => {
      if (!acc[item.condition]) {
        acc[item.condition] = 0;
      }
      acc[item.condition] += item.score;
      return acc;
    }, {});
  
    // Normalize probabilities
    const totalScore = Object.values(groupedScores).reduce((a, b) => a + b, 0);
    
    const normalizedPredictions = Object.entries(groupedScores)
      .map(([condition, score]) => ({
        condition,
        probability: score / totalScore
      }))
      .filter(pred => pred.probability > confidenceThreshold)
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3); // Top 3 predictions
  
    return normalizedPredictions;
  }

  async drawDetections(imageBuffer, predictions) {
    try {
      const metadata = await sharp(imageBuffer).metadata();
      const scaledImage = await sharp(imageBuffer)
        .resize(800, 600, { fit: 'inside' })
        .toBuffer();
      
      const canvas = Canvas.createCanvas(800, 600);
      const ctx = canvas.getContext('2d');
      
      const img = new Canvas.Image();
      img.src = scaledImage;
      ctx.drawImage(img, 0, 0);
      
      predictions.forEach(pred => {
        const { condition, probability } = pred;
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 10, 300, 30);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText(
          `${condition}: ${(probability * 100).toFixed(1)}%`,
          20,
          30
        );
      });
      
      const base64Image = canvas.toDataURL('image/jpeg');
      const base64Data = base64Image.split(',')[1];
      const annotatedImageBuffer = Buffer.from(base64Data, 'base64');
      
      return annotatedImageBuffer;
    } catch (error) {
      throw new Error(`Failed to draw detections: ${error.message}`);
    }
  }

  async predict(base64Image) {
    try {
      const imageBuffer = this.processBase64Image(base64Image);
      const model = await this.loadModel();
      const { tensor: preprocessedData, originalDimensions } = await this.preprocessImage(imageBuffer);

      const inputTensor = new ort.Tensor(
        'float32',
        preprocessedData,
        [1, 3, this.targetSize, this.targetSize]
      );

      const feeds = { [model.inputNames[0]]: inputTensor };
      const results = await model.run(feeds);
      
      const output = Object.values(results)[0];
      const predictions = this.processDetections(output, originalDimensions);
      
      const annotatedImage = await this.drawDetections(imageBuffer, predictions);
      
      const outputDir = path.join(__dirname, '../../outputs');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      const filename = `annotated_${Date.now()}.jpg`;
      const fullPath = path.join(outputDir, filename);
      fs.writeFileSync(fullPath, annotatedImage);

      return {
        predictions,
        annotatedImage: annotatedImage.toString('base64')
      };
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Failed to process image: ${error.message}`);
    }
  }
}

module.exports = new ModelLoader();