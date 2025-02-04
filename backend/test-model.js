// test-model.js
const ModelLoader = require('./services/ai/modelLoader');

async function testModel() {
    try {
        console.log('=== Starting Model Test ===');
        console.log('Current working directory:', process.cwd());
        
        console.log('\nAttempting to load model...');
        const model = await ModelLoader.loadModel();
        
        console.log('\n=== Test Completed Successfully ===');
    } catch (error) {
        console.error('\n=== Test Failed ===');
        console.error('Error:', error.message);
        console.error('Stack:', error.stack);
    }
}

testModel();