// scripts/load_medical_data.js
const fs = require('fs').promises;
const csv = require('csv-parse');
const path = require('path');
const { OpenAI } = require('openai');
const { PineconeClient } = require('@pinecone-database/pinecone');

class MedicalDataProcessor {
    constructor() {
        this.openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        this.pinecone = new PineconeClient();
        this.initialized = false;
    }

    async initialize() {
        if (!this.initialized) {
            await this.pinecone.init({
                environment: process.env.PINECONE_ENVIRONMENT,
                apiKey: process.env.PINECONE_API_KEY
            });
            this.initialized = true;
        }
    }

    async processRawData() {
        const rawDataPath = path.join(__dirname, '../data/medical/raw/dermatology_qa.csv');
        const processedDataPath = path.join(__dirname, '../data/medical/processed/qa_pairs.json');

        try {
            // Read CSV file
            const fileContent = await fs.readFile(rawDataPath, 'utf-8');
            const records = await this.parseCSV(fileContent);

            // Process and structure the data
            const processedData = await this.structureData(records);

            // Generate embeddings
            const dataWithEmbeddings = await this.generateEmbeddings(processedData);

            // Save processed data
            await fs.writeFile(processedDataPath, JSON.stringify(dataWithEmbeddings, null, 2));

            // Store embeddings in Pinecone
            await this.storeEmbeddings(dataWithEmbeddings);

            console.log('Medical data processing completed successfully');
        } catch (error) {
            console.error('Error processing medical data:', error);
            throw error;
        }
    }

    async parseCSV(content) {
        return new Promise((resolve, reject) => {
            csv.parse(content, {
                columns: true,
                skip_empty_lines: true
            }, (err, records) => {
                if (err) reject(err);
                else resolve(records);
            });
        });
    }

    async structureData(records) {
        return records.map(record => ({
            id: record.id,
            question: record.question,
            answer: record.answer,
            category: record.category,
            severity: record.severity,
            tags: record.tags ? record.tags.split(',').map(tag => tag.trim()) : []
        }));
    }

    async generateEmbeddings(data) {
        return Promise.all(data.map(async (item) => {
            const embedding = await this.openai.embeddings.create({
                model: "text-embedding-ada-002",
                input: item.question + " " + item.answer
            });
            
            return {
                ...item,
                embedding: embedding.data[0].embedding
            };
        }));
    }

    async storeEmbeddings(dataWithEmbeddings) {
        await this.initialize();
        const index = this.pinecone.Index("dermatology-qa");

        const vectors = dataWithEmbeddings.map(item => ({
            id: item.id,
            values: item.embedding,
            metadata: {
                question: item.question,
                category: item.category,
                severity: item.severity,
                tags: item.tags
            }
        }));

        // Store in batches of 100
        for (let i = 0; i < vectors.length; i += 100) {
            const batch = vectors.slice(i, i + 100);
            await index.upsert(batch);
        }
    }
}

module.exports = new MedicalDataProcessor();