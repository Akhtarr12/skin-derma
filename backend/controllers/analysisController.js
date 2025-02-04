const ModelLoader = require('../services/ai/modelLoader');

// controllers/analysisController.js
exports.analyzeSkin = async (req, res) => {
    try {
        if (!req.file?.buffer) {
            return res.status(400).json({ error: 'No image provided' });
        }

        console.log('Received image buffer:', req.file.buffer.length, 'bytes');
        const analysis = await ModelLoader.predict(req.file.buffer);
        
        res.json({ success: true, analysis });
    } catch (error) {
        console.error('Analysis error:', error.message, error.stack); // Detailed logging
        res.status(500).json({ 
            error: 'Failed to analyze image',
            details: error.message // Send error details in development
        });
    }
};