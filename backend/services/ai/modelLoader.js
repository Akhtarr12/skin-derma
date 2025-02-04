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
    // ImageNet mean and std for normalization
    this.mean = [0.485, 0.456, 0.406];
    this.std = [0.229, 0.224, 0.225];
  }

  async loadModel() {
    try {
      if (!this.model && !this.isLoading) {
        this.isLoading = true;
        console.log('Loading model from:', this.modelPath);
        const options = {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all'
        };
        this.model = await ort.InferenceSession.create(this.modelPath, options);
        console.log('Model loaded successfully');
        console.log('Model input names:', this.model.inputNames);
        console.log('Model output names:', this.model.outputNames);
        this.isLoading = false;
      }
      return this.model;
    } catch (error) {
      this.isLoading = false;
      console.error('Model loading error:', error);
      throw new Error(`Failed to load AI model: ${error.message}`);
    }
  }

  processBase64Image(base64Image) {
    try {
      if (base64Image instanceof Buffer) {
        return base64Image;
      }

      if (typeof base64Image !== 'string') {
        throw new Error('Invalid image input');
      }

      const base64Data = base64Image.includes(',') 
        ? base64Image.split(',')[1] 
        : base64Image;

      return Buffer.from(base64Data, 'base64');
    } catch (error) {
      console.error('Base64 processing error:', error);
      throw new Error(`Base64 image processing failed: ${error.message}`);
    }
  }

  async preprocessImage(imageBuffer) {
    try {
      console.log('Starting image preprocessing');
      const metadata = await sharp(imageBuffer).metadata();
      console.log('Original image dimensions:', { width: metadata.width, height: metadata.height });

      const processedImage = await sharp(imageBuffer)
        .resize(this.targetSize, this.targetSize, {
          fit: 'cover',  // Changed to cover to maintain aspect ratio and fill
          position: 'center'
        })
        .removeAlpha()
        .raw()
        .toBuffer();

      const float32Data = new Float32Array(3 * this.targetSize * this.targetSize);
      
      // Apply normalization with ImageNet mean and std
      for (let c = 0; c < 3; c++) {
        for (let h = 0; h < this.targetSize; h++) {
          for (let w = 0; w < this.targetSize; w++) {
            const srcIdx = h * this.targetSize * 3 + w * 3 + c;
            const dstIdx = c * this.targetSize * this.targetSize + h * this.targetSize + w;
            float32Data[dstIdx] = (processedImage[srcIdx] / 255.0 - this.mean[c]) / this.std[c];
          }
        }
      }

      console.log('Preprocessing completed');
      return {
        tensor: float32Data,
        originalDimensions: { width: metadata.width, height: metadata.height }
      };
    } catch (error) {
      console.error('Preprocessing error:', error);
      throw new Error(`Image preprocessing failed: ${error.message}`);
    }
  }

  processDetections(output, confidenceThreshold = 0.25) {
    try {
      console.log('Processing detections with confidence threshold:', confidenceThreshold);
      const data = Array.from(output.data);
      const numClasses = this.conditions.length;
      
      // softmax with numerical stability
      const scores = [];
      for (let i = 0; i < data.length; i += numClasses) {
        const classScores = data.slice(i, i + numClasses);
        const maxScore = Math.max(...classScores);
        const expScores = classScores.map(score => Math.exp(score - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const softmaxScores = expScores.map(score => score / sumExp);
        scores.push(...softmaxScores);
      }

      //  predictions array with probabilities
      const predictions = this.conditions.map((condition, idx) => ({
        condition,
        probability: scores[idx]
      }))
      .filter(pred => pred.probability > confidenceThreshold)
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);  // top 3 predictions

      console.log('Processed predictions:', predictions);
      return predictions;
    } catch (error) {
      console.error('Detection processing error:', error);
      throw new Error(`Failed to process detections: ${error.message}`);
    }
  }

  async drawDetections(imageBuffer, predictions) {
    try {
      const scaledImage = await sharp(imageBuffer)
        .resize(800, 600, { fit: 'inside' })
        .toBuffer();
      
      const canvas = Canvas.createCanvas(800, 600);
      const ctx = canvas.getContext('2d');
      
      const img = new Canvas.Image();
      img.src = scaledImage;
      ctx.drawImage(img, 0, 0);
      
      //  prediction boxes with improved styling
      predictions.forEach((pred, index) => {
        const yPosition = 30 + (index * 40);  // Stack predictions vertically
        
        // background box
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, yPosition - 20, 300, 30);
        
        //  text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(
          `${pred.condition}: ${(pred.probability * 100).toFixed(1)}%`,
          20,
          yPosition
        );
      });
      
      const base64Image = canvas.toDataURL('image/jpeg', 0.9);  // Increased quality
      const base64Data = base64Image.split(',')[1];
      return Buffer.from(base64Data, 'base64');
    } catch (error) {
      console.error('Drawing error:', error);
      throw new Error(`Failed to draw detections: ${error.message}`);
    }
  }

  async predict(base64Image) {
    try {
      console.log('Starting prediction process');
      const imageBuffer = this.processBase64Image(base64Image);
      const model = await this.loadModel();
      const { tensor: preprocessedData, originalDimensions } = await this.preprocessImage(imageBuffer);

      const inputTensor = new ort.Tensor(
        'float32',
        preprocessedData,
        [1, 3, this.targetSize, this.targetSize]
      );

      console.log('Running model inference');
      const feeds = { [model.inputNames[0]]: inputTensor };
      const results = await model.run(feeds);
      
      const output = Object.values(results)[0];
      console.log('Model output shape:', output.dims);
      console.log('Raw output sample:', Array.from(output.data.slice(0, 5)));

      const predictions = this.processDetections(output);
      
      if (predictions.length === 0) {
        console.warn('No predictions above confidence threshold');
      }

      const annotatedImage = await this.drawDetections(imageBuffer, predictions);
      
      // Save annotated image
      const outputDir = path.join(__dirname, '../../outputs');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      const filename = `annotated_${Date.now()}.jpg`;
      const fullPath = path.join(outputDir, filename);
      fs.writeFileSync(fullPath, annotatedImage);
      console.log('Saved annotated image to:', fullPath);

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