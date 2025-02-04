// check-model.js
const fs = require('fs').promises;
const path = require('path');

async function checkModel() {
    try {
        const modelPath = path.join(__dirname, 'model', 'model.json');
        const modelContent = await fs.readFile(modelPath, 'utf8');
        const modelJson = JSON.parse(modelContent);
        
        console.log('Expected weight files:', modelJson.weightsManifest[0].paths);
        
        // List actual files
        const modelDir = path.join(__dirname, 'model');
        const files = await fs.readdir(modelDir);
        console.log('\nActual files:', files);
        
        // Check for missing files
        const expectedFiles = new Set(modelJson.weightsManifest[0].paths);
        const actualFiles = new Set(files);
        
        console.log('\nMissing expected files:');
        for (const file of expectedFiles) {
            if (!actualFiles.has(file)) {
                console.log(`- ${file}`);
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

checkModel();