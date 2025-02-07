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
    this.stride = 32;
    this.isLoading = false;

    // List of class names in the same order as the model output
    this.names = [
      'acne', 'Basal-Cell-Carcinoma', 'Darier_s-Disease', 'eczema',
      'Epidermolysis-Bullosa-Pruriginosa', 'Hailey-Hailey-Disease',
      'Capture-d-ecran', 'Impetigo', 'LEISHMANIOSE', 'lichen',
      'Lupus-Erythematosus-Chronicus-Discoides', 'Melanoma',
      'Molluscum-Contagiosum', 'nevus', 'normal', 'Porokeratosis-Actinic',
      'Psoriasis', 'tinea-corporis', 'Tungiasis'
    ];

    if (this.names.length !== 19) {
      throw new Error(`Class count mismatch: Model expects 19 classes, got ${this.names.length}`);
    }
  }

  async loadModel() {
    if (!this.model && !this.isLoading) {
      this.isLoading = true;
      try {
        console.log('Loading model from:', this.modelPath);
        this.model = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all'
        });
        console.log('Model loaded successfully');
      } catch (error) {
        console.error('Model loading failed:', error);
        throw error;
      } finally {
        this.isLoading = false;
      }
    }
    return this.model;
  }

  async preprocessImage(imageBuffer) {
    try {
      console.log('Starting image preprocessing...');
      console.log('Image buffer length:', imageBuffer.length);

      if (!imageBuffer || imageBuffer.length === 0) {
        throw new Error('Empty or invalid image buffer');
      }

      // Obtain original image dimensions
      const metadata = await sharp(imageBuffer).metadata();
      const originalWidth = metadata.width;
      const originalHeight = metadata.height;
      console.log(`Original image dimensions: ${originalWidth} x ${originalHeight}`);

      // Compute scaling factor and new dimensions
      const scale = Math.min(this.targetSize / originalWidth, this.targetSize / originalHeight);
      const newWidth = Math.round(originalWidth * scale);
      const newHeight = Math.round(originalHeight * scale);
      // Compute padding (letterbox) offsets so that the resized image is centered
      const padX = Math.floor((this.targetSize - newWidth) / 2);
      const padY = Math.floor((this.targetSize - newHeight) / 2);
      console.log(`Scale: ${scale}, New dimensions: ${newWidth}x${newHeight}, Padding: (${padX}, ${padY})`);

      // Resize the image to new dimensions and then pad to create a 640x640 image
      const { data, info } = await sharp(imageBuffer)
        .resize(newWidth, newHeight)
        .extend({
          top: padY,
          bottom: this.targetSize - newHeight - padY,
          left: padX,
          right: this.targetSize - newWidth - padX,
          background: { r: 114, g: 114, b: 114 }
        })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      console.log('Image resized and padded to:', info.width, 'x', info.height);

      // Create a Float32Array for the model input.
      // Optionally, add normalization here if your model requires it.
      // For example, if your model was trained on ImageNet normalized inputs:
      // const mean = [0.485, 0.456, 0.406];
      // const std  = [0.229, 0.224, 0.225];
      const float32Data = new Float32Array(3 * this.targetSize * this.targetSize);
      let minVal = Infinity;
      let maxVal = -Infinity;

      for (let i = 0; i < data.length; i += 3) {
        const pixelIndex = i / 3;
        let r = data[i] / 255.0;
        let g = data[i + 1] / 255.0;
        let b = data[i + 2] / 255.0;

        // Uncomment and adjust the following if normalization is needed:
        // r = (r - mean[0]) / std[0];
        // g = (g - mean[1]) / std[1];
        // b = (b - mean[2]) / std[2];

        float32Data[pixelIndex] = r;
        float32Data[pixelIndex + this.targetSize * this.targetSize] = g;
        float32Data[pixelIndex + 2 * this.targetSize * this.targetSize] = b;

        minVal = Math.min(minVal, r, g, b);
        maxVal = Math.max(maxVal, r, g, b);
      }

      console.log('Preprocessed tensor statistics:');
      console.log('Min value:', minVal);
      console.log('Max value:', maxVal);

      const tensor = new ort.Tensor('float32', float32Data, [1, 3, this.targetSize, this.targetSize]);

      return {
        tensor,
        ratio: scale,         // The scale factor from original -> resized image
        padX,                 // Horizontal padding added to the resized image
        padY,                 // Vertical padding added to the resized image
        originalDimensions: { width: originalWidth, height: originalHeight }
      };
    } catch (error) {
      console.error('Image preprocessing failed:', error);
      throw error;
    }
  }
  processOutput(
    outputTensor,
    confidenceThreshold = 0.80,
    iouThreshold = 0.15,
    minBoxArea = 100,
    mergeDistanceThreshold = 30  // <-- Added parameter here
  ) {
    try {
      console.log('Processing model output...');
      const predictions = [];
      const outputData = outputTensor.data;
  
      // Constants based on model output format:
      const BOX_ELEMENTS = 4;
      const MASK_ELEMENTS = 32;
      const OBJ_SCORE_INDEX = BOX_ELEMENTS;
      // With 55 numbers per detection:
      // numClasses = 55 - (4 + 1 + 32) = 18
      const numClasses = outputTensor.dims[1] - (BOX_ELEMENTS + 1 + MASK_ELEMENTS);
      const CLASS_START_INDEX = BOX_ELEMENTS + 1 + MASK_ELEMENTS;
  
      // Sigmoid for objectness score
      const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  
      const numDetections = outputTensor.dims[2];
      for (let i = 0; i < numDetections; i++) {
        const offset = i * outputTensor.dims[1];
  
        // Extract bounding box [center_x, center_y, width, height]
        const bbox = Array.from(outputData.slice(offset, offset + BOX_ELEMENTS));
  
        // Compute objectness score via sigmoid
        const rawObjScore = outputData[offset + OBJ_SCORE_INDEX];
        const objScore = sigmoid(rawObjScore);
  
        // (Optional) Extract mask coefficients if needed
        const maskCoefficients = Array.from(
          outputData.slice(offset + BOX_ELEMENTS + 1, offset + BOX_ELEMENTS + 1 + MASK_ELEMENTS)
        );
  
        // Extract raw class scores and apply softmax:
        const rawClassScores = Array.from(
          outputData.slice(offset + CLASS_START_INDEX, offset + CLASS_START_INDEX + numClasses)
        );
        const maxRaw = Math.max(...rawClassScores);
        const expScores = rawClassScores.map(x => Math.exp(x - maxRaw));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const classScores = expScores.map(x => x / sumExp);
        const maxClassScore = Math.max(...classScores);
        const classId = classScores.indexOf(maxClassScore);
  
        // Combined confidence score
        const confidence = objScore * maxClassScore;
  
        // Keep only high-confidence detections.
        if (confidence > confidenceThreshold) {
          predictions.push({
            bbox,
            confidence,
            classId,
            maskCoefficients,
            rawScores: rawClassScores,
            processedScores: classScores
          });
        }
      }
  
      console.log('\nTotal predictions before filtering:', predictions.length);
  
      // --- Filter out detections with very small areas.
      const filteredByArea = predictions.filter(pred => {
        const [cx, cy, w, h] = pred.bbox;
        return (w * h) >= minBoxArea;
      });
      console.log('Predictions after area filtering:', filteredByArea.length);
  
      // --- Apply Non-Maximum Suppression (NMS)
      const nmsPredictions = this.nms(filteredByArea, iouThreshold);
      console.log('Predictions after NMS:', nmsPredictions.length);
  
      // --- Merge nearby detections of the same class.
      const finalPredictions = this.mergeNearbyDetections(nmsPredictions, mergeDistanceThreshold);
      console.log('Final predictions after merging nearby detections:', finalPredictions.length);
  
      finalPredictions.forEach((pred, idx) => {
        const className = this.names[pred.classId] || 'Unknown';
        console.log(`\nPrediction ${idx + 1}:`);
        console.log('Class:', className);
        console.log('Confidence:', pred.confidence);
        console.log('Bounding Box:', pred.bbox);
      });
  
      return finalPredictions;
    } catch (error) {
      console.error('Output processing failed:', error);
      throw error;
    }
  }
  
  nms(predictions, iouThreshold) {
    // Sort predictions in descending order by confidence.
    predictions.sort((a, b) => b.confidence - a.confidence);
    const selected = [];
  
    while (predictions.length > 0) {
      const current = predictions.shift();
      selected.push(current);
  
      predictions = predictions.filter(pred => {
        const iou = this.calculateIOU(current.bbox, pred.bbox);
        // Remove detections that have an IoU higher than the threshold.
        return iou < iouThreshold;
      });
    }
    return selected;
  }
  
  calculateIOU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
  
    // Convert center coordinates to corner coordinates.
    const x1Min = x1 - w1 / 2, y1Min = y1 - h1 / 2;
    const x1Max = x1 + w1 / 2, y1Max = y1 + h1 / 2;
    const x2Min = x2 - w2 / 2, y2Min = y2 - h2 / 2;
    const x2Max = x2 + w2 / 2, y2Max = y2 + h2 / 2;
  
    const interXMin = Math.max(x1Min, x2Min);
    const interYMin = Math.max(y1Min, y2Min);
    const interXMax = Math.min(x1Max, x2Max);
    const interYMax = Math.min(y1Max, y2Max);
  
    const interWidth = Math.max(0, interXMax - interXMin);
    const interHeight = Math.max(0, interYMax - interYMin);
    const interArea = interWidth * interHeight;
  
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
  
    return interArea / (box1Area + box2Area - interArea);
  }
  
  mergeNearbyDetections(detections, distanceThreshold = 30) {
    // This function clusters detections of the same class that are near each other.
    const mergedDetections = [];
    const used = new Array(detections.length).fill(false);
  
    for (let i = 0; i < detections.length; i++) {
      if (used[i]) continue;
      const current = detections[i];
      const cluster = [current];
      used[i] = true;
  
      for (let j = i + 1; j < detections.length; j++) {
        if (used[j]) continue;
        const candidate = detections[j];
        if (current.classId !== candidate.classId) continue;
  
        // Calculate Euclidean distance between centers.
        const dx = current.bbox[0] - candidate.bbox[0];
        const dy = current.bbox[1] - candidate.bbox[1];
        const dist = Math.sqrt(dx * dx + dy * dy);
  
        if (dist < distanceThreshold) {
          cluster.push(candidate);
          used[j] = true;
        }
      }
  
      // Merge the cluster: average the box coordinates and take the maximum confidence.
      const mergedBox = cluster.reduce(
        (acc, det) => [
          acc[0] + det.bbox[0],
          acc[1] + det.bbox[1],
          acc[2] + det.bbox[2],
          acc[3] + det.bbox[3]
        ],
        [0, 0, 0, 0]
      ).map(val => val / cluster.length);
      const mergedConfidence = Math.max(...cluster.map(det => det.confidence));
  
      mergedDetections.push({
        bbox: mergedBox,
        classId: current.classId,
        confidence: mergedConfidence
      });
    }
  
    return mergedDetections;
  }
  
  async drawDetections(imageBuffer, predictions, ratio, padX, padY, originalDimensions) {
    try {
      // Create a canvas using the original image dimensions
      const canvas = Canvas.createCanvas(originalDimensions.width, originalDimensions.height);
      const ctx = canvas.getContext('2d');

      // (Optionally) convert imageBuffer to JPEG if needed
      const jpegBuffer = await sharp(imageBuffer)
        .jpeg()
        .toBuffer();

      const img = new Canvas.Image();
      await new Promise((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = (err) => reject(new Error('Failed to load image: ' + err));
        const base64Image = jpegBuffer.toString('base64');
        img.src = `data:image/jpeg;base64,${base64Image}`;
      });

      // Draw the original image (without letterbox) on the canvas
      ctx.drawImage(img, 0, 0, originalDimensions.width, originalDimensions.height);

      // IMPORTANT: The network’s bbox predictions are given in the padded (640×640) space.
      // To convert them to the coordinates on the original image, you need to “undo” the letterbox.
      predictions.forEach(pred => {
        // Remove the padding offsets and then scale by the inverse of the ratio.
        // Here we assume the model outputs [center_x, center_y, width, height] in the 640×640 space.
        const x = (pred.bbox[0] - padX) / ratio;
        const y = (pred.bbox[1] - padY) / ratio;
        const w = pred.bbox[2] / ratio;
        const h = pred.bbox[3] / ratio;

        ctx.beginPath();
        ctx.rect(x - w / 2, y - h / 2, w, h);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#FF0000';
        ctx.stroke();

        ctx.fillStyle = '#FF0000';
        ctx.font = '14px Arial';
        const label = `${this.names[pred.classId]} ${(pred.confidence * 100).toFixed(1)}%`;
        const textMetrics = ctx.measureText(label);

        ctx.fillRect(x - w / 2 - 2, y - h / 2 - 14, textMetrics.width + 4, 16);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x - w / 2, y - h / 2 - 2);
      });

      return canvas.toBuffer('image/jpeg', { quality: 90 });
    } catch (error) {
      console.error('Annotation failed:', error);
      throw error;
    }
  }

  async predict(imageBuffer) {
    try {
      console.log('\n=== Starting new prediction ===\n');

      const model = await this.loadModel();
      console.log('Model loaded successfully');

      // Note: preprocessImage now returns additional info (ratio and padX, padY)
      const { tensor, ratio, padX, padY, originalDimensions } = await this.preprocessImage(imageBuffer);
      console.log('Image preprocessed successfully');

      console.log('Running inference...');
      const outputs = await model.run({ images: tensor });
      console.log('Inference completed');

      const predictions = this.processOutput(outputs.output0);
      console.log('Output processed successfully');

      const annotatedImage = await this.drawDetections(
        imageBuffer,
        predictions,
        ratio,
        padX,
        padY,
        originalDimensions
      );
      console.log('Image annotated successfully');

      return {
        status: 'success',
        predictions: predictions.map(p => ({
          class: this.names[p.classId],
          confidence: p.confidence,
          // Convert bbox coordinates from the padded image back to original image space:
          bbox: p.bbox.map(v => v), // Further processing can be done if needed
          classId: p.classId,
          debug: {
            rawScores: p.rawScores,
            processedScores: p.processedScores
          }
        })),
        annotatedImage: annotatedImage.toString('base64'),
        originalDimensions,
        preprocessingRatio: ratio
      };
    } catch (error) {
      console.error('Prediction failed:', error);
      return {
        status: 'error',
        message: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      };
    }
  }
}

module.exports = new ModelLoader();