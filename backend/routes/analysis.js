const express = require('express');
const router = express.Router();
const multer = require('multer');

// Configure multer to store file in memory
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only JPEG and PNG allowed.'));
        }
    }
});

const { analyzeSkin } = require('../controllers/analysisController');

// routes/analysis.js (temporary test route)
router.get('/test', (req, res) => {
    res.json({ message: "API connected!" });
  });

router.post('/analyze', upload.single('image'), (req, res, next) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image provided' });
    }
    next();
}, analyzeSkin);

module.exports = router;
