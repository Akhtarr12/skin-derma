const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const modelLoader = require('../services/ai/modelLoader');

// Enhanced logging middleware
const logRequestDetails = (req, res, next) => {
  console.log('Request Headers:', req.headers);
  console.log('Content-Type:', req.headers['content-type']);
  console.log('Request Body:', req.body);
  next();
};

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

router.post('/analyze', 
  logRequestDetails,
  (req, res, next) => {
    upload.single('image')(req, res, (err) => {
      if (err) {
        console.error('Multer Error:', err);
        return res.status(400).json({ 
          success: false, 
          error: err.message || 'Upload failed' 
        });
      }
      next();
    });
  }, 
  async (req, res) => {
    try {
      console.log('Received File:', req.file);
      
      if (!req.file) {
        return res.status(400).json({ 
          success: false, 
          error: 'No image uploaded' 
        });
      }

      const base64Image = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;

      const result = await modelLoader.predict(base64Image);

      res.json({
        success: true,
        analysis: {
          predictions: result.predictions,
          annotatedImage: result.annotatedImage
        }
      });

    } catch (error) {
      console.error('Full Error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Processing failed'
      });
    }
  }
);

module.exports = router;