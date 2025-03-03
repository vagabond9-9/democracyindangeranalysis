import * as tf from '@tensorflow/tfjs';
import { analyzeText } from './nlp';
import { mlStorage, isSupabaseConfigured } from './supabase';

// Simple text classifier using TensorFlow.js
export class AuthoritarianClassifier {
  private model: tf.LayersModel | null = null;
  private wordIndex: Record<string, number> = {};
  private maxSequenceLength = 50; // Reduced from 100 to improve performance
  private vocabSize = 3000; // Reduced from 5000 to improve performance
  private trainingData: {text: string, label: number}[] = [];
  private isModelReady = false;
  private useSupabase = false;
  private trainingStatus = '';
  
  constructor() {
    this.useSupabase = isSupabaseConfigured();
    this.buildModel();
    
    // Try to load existing training data from Supabase
    if (this.useSupabase) {
      this.loadTrainingDataFromSupabase();
    }
  }
  
  // Load training data from Supabase
  private async loadTrainingDataFromSupabase() {
    if (!this.useSupabase) return;
    
    try {
      this.trainingStatus = 'Loading training data from Supabase...';
      const data = await mlStorage.getTrainingData();
      
      if (data.length > 0) {
        this.addTrainingData(data);
        this.trainingStatus = `Loaded ${data.length} training examples from Supabase`;
        console.log(`Loaded ${data.length} training examples from Supabase`);
      } else {
        this.trainingStatus = 'No training data found in Supabase';
      }
    } catch (error) {
      console.error('Error loading training data from Supabase:', error);
      this.trainingStatus = 'Failed to load training data from Supabase';
    }
  }
  
  // Get current training status
  public getTrainingStatus(): string {
    return this.trainingStatus;
  }
  
  private async buildModel() {
    try {
      // Create a simple sequential model
      this.model = tf.sequential();
      
      // Use a simpler model architecture
      // Input layer - dense instead of embedding for better compatibility
      this.model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        inputShape: [this.maxSequenceLength],
        kernelInitializer: 'varianceScaling'
      }));
      
      // Add dropout for regularization
      this.model.add(tf.layers.dropout({ rate: 0.3 }));
      
      // Hidden layer
      this.model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
      
      // Output layer with 5 units (0 = not authoritarian, 1-4 = indicator types)
      this.model.add(tf.layers.dense({
        units: 5,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
      }));
      
      // Compile the model with categorical crossentropy
      this.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      this.isModelReady = true;
      console.log('Model built successfully');
      
      // Try to load model weights from Supabase
      if (this.useSupabase) {
        this.loadModelFromSupabase();
      }
    } catch (error) {
      console.error('Error building model:', error);
      // Rebuild a simpler model as fallback
      this.buildSimpleModel();
    }
  }
  
  // Load model weights from Supabase
  private async loadModelFromSupabase() {
    if (!this.useSupabase || !this.model) return;
    
    try {
      this.trainingStatus = 'Loading model from Supabase...';
      const modelWeights = await mlStorage.getLatestModelWeights();
      
      if (modelWeights) {
        const weights = await tf.io.decodeWeights(modelWeights);
        await this.model.setWeights(Object.values(weights));
        this.trainingStatus = 'Model loaded from Supabase successfully';
        console.log('Model loaded from Supabase successfully');
      } else {
        this.trainingStatus = 'No model found in Supabase';
      }
    } catch (error) {
      console.error('Error loading model from Supabase:', error);
      this.trainingStatus = 'Failed to load model from Supabase';
    }
  }
  
  // Save model weights to Supabase
  private async saveModelToSupabase() {
    if (!this.useSupabase || !this.model) return;
    
    try {
      this.trainingStatus = 'Saving model to Supabase...';
      
      // Get model weights
      const weights = this.model.getWeights();
      const weightData = await tf.io.encodeWeights(weights);
      
      // Save to Supabase
      await mlStorage.storeModelWeights(weightData.data);
      
      this.trainingStatus = 'Model saved to Supabase successfully';
      console.log('Model saved to Supabase successfully');
    } catch (error) {
      console.error('Error saving model to Supabase:', error);
      this.trainingStatus = 'Failed to save model to Supabase';
    }
  }
  
  // Fallback to a simpler model if the main one fails
  private buildSimpleModel() {
    try {
      this.model = tf.sequential();
      
      // Single dense layer model
      this.model.add(tf.layers.dense({ 
        units: 10, 
        activation: 'relu',
        inputShape: [this.maxSequenceLength],
        kernelInitializer: 'varianceScaling'
      }));
      
      this.model.add(tf.layers.dense({ 
        units: 5, 
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
      }));
      
      this.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      this.isModelReady = true;
      console.log('Fallback model built successfully');
    } catch (error) {
      console.error('Error building fallback model:', error);
      this.model = null;
      this.isModelReady = false;
    }
  }
  
  // Process text into a bag-of-words vector
  private textToVector(text: string): number[] {
    try {
      // Create a zero-filled vector
      const vector = new Array(this.maxSequenceLength).fill(0);
      
      // Tokenize the text
      const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
      
      // Count word occurrences (bag of words approach)
      const wordCounts: Record<string, number> = {};
      words.forEach(word => {
        if (word.length > 2) { // Only consider words with more than 2 characters
          wordCounts[word] = (wordCounts[word] || 0) + 1;
        }
      });
      
      // Fill the vector with word counts
      let index = 0;
      for (const [word, count] of Object.entries(wordCounts)) {
        if (index < this.maxSequenceLength) {
          vector[index] = count;
          index++;
        } else {
          break;
        }
      }
      
      return vector;
    } catch (error) {
      console.error('Error in textToVector:', error);
      // Return a zero-filled array as fallback
      return new Array(this.maxSequenceLength).fill(0);
    }
  }
  
  // Add training data
  public addTrainingData(data: {text: string, label: number}[]) {
    try {
      // Filter out invalid data
      const validData = data.filter(item => 
        item && item.text && 
        typeof item.label === 'number' && 
        item.label >= 0 && 
        item.label <= 4
      );
      
      this.trainingData = [...this.trainingData, ...validData];
      
      // Store in Supabase if available and not already there
      if (this.useSupabase && validData.length > 0) {
        this.storeTrainingDataInSupabase(validData);
      }
      
      return this.trainingData.length;
    } catch (error) {
      console.error('Error adding training data:', error);
      return this.trainingData.length;
    }
  }
  
  // Store training data in Supabase
  private async storeTrainingDataInSupabase(data: {text: string, label: number}[]) {
    if (!this.useSupabase) return;
    
    try {
      this.trainingStatus = 'Storing training data in Supabase...';
      await mlStorage.storeTrainingData(data);
      this.trainingStatus = 'Training data stored in Supabase successfully';
    } catch (error) {
      console.error('Error storing training data in Supabase:', error);
      this.trainingStatus = 'Failed to store training data in Supabase';
    }
  }
  
  // Train the model
  public async train(epochs = 3, batchSize = 8) {
    if (!this.model || !this.isModelReady) {
      await this.buildModel();
      if (!this.model) {
        throw new Error('Model not initialized. Please rebuild the model.');
      }
    }
    
    if (this.trainingData.length < 10) {
      throw new Error('Not enough training data. Need at least 10 examples.');
    }
    
    try {
      this.trainingStatus = 'Preparing training data...';
      
      // Clear TensorFlow memory before starting
      tf.engine().startScope();
      
      // Prepare training data using bag-of-words approach
      const vectors = this.trainingData.map(item => this.textToVector(item.text));
      
      // Convert labels to one-hot encoding
      const labels = this.trainingData.map(item => {
        const label = Array(5).fill(0);
        label[item.label] = 1;
        return label;
      });
      
      // Log data for debugging
      console.log('Training data prepared:', {
        vectorsLength: vectors.length,
        firstVector: vectors[0].slice(0, 5),
        firstLabel: labels[0]
      });
      
      // Create tensors with explicit types
      const xs = tf.tensor2d(vectors, [vectors.length, this.maxSequenceLength], 'float32');
      const ys = tf.tensor2d(labels, [labels.length, 5], 'float32');
      
      // Log tensor info for debugging
      console.log('Tensor shapes:', {
        xs: xs.shape,
        ys: ys.shape
      });
      
      // Use a very small batch size and few epochs for initial training
      const actualEpochs = Math.min(epochs, 5);
      const actualBatchSize = Math.min(batchSize, 4);
      
      this.trainingStatus = `Training model with ${this.trainingData.length} examples...`;
      console.log(`Starting training with ${actualEpochs} epochs and batch size ${actualBatchSize}`);
      
      // Train the model
      const history = await this.model.fit(xs, ys, {
        epochs: actualEpochs,
        batchSize: actualBatchSize,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
          onEpochBegin: (epoch) => {
            this.trainingStatus = `Training epoch ${epoch + 1}/${actualEpochs}...`;
          },
          onEpochEnd: (epoch, logs) => {
            const accuracy = logs?.acc ? (logs.acc * 100).toFixed(1) : 'N/A';
            this.trainingStatus = `Epoch ${epoch + 1}/${actualEpochs} complete - accuracy: ${accuracy}%`;
            console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss?.toFixed(4) || 'N/A'}, accuracy = ${logs?.acc?.toFixed(4) || 'N/A'}`);
          }
        }
      });
      
      // Clean up tensors
      xs.dispose();
      ys.dispose();
      
      // End TensorFlow memory scope
      tf.engine().endScope();
      
      // Force garbage collection
      tf.engine().disposeVariables();
      
      // Save model to Supabase if available
      if (this.useSupabase) {
        await this.saveModelToSupabase();
      }
      
      this.trainingStatus = 'Model trained successfully';
      return history;
    } catch (error) {
      console.error('Training error details:', error);
      
      // Try to rebuild the model if training failed
      this.buildSimpleModel();
      
      this.trainingStatus = `Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`;
      throw new Error(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  // Predict the indicator for a given text
  public predict(text: string) {
    if (!this.model || !this.isModelReady) {
      return {
        notAuthoritarian: 1,
        indicator1: 0,
        indicator2: 0,
        indicator3: 0,
        indicator4: 0,
        predictedClass: 0
      };
    }
    
    try {
      return tf.tidy(() => {
        // Convert text to vector
        const vector = this.textToVector(text);
        
        // Make prediction
        const input = tf.tensor2d([vector], [1, this.maxSequenceLength], 'float32');
        const prediction = this.model!.predict(input) as tf.Tensor;
        
        // Get the predicted class
        const predictionData = prediction.dataSync();
        
        // Return probabilities for each class
        return {
          notAuthoritarian: predictionData[0] || 0,
          indicator1: predictionData[1] || 0,
          indicator2: predictionData[2] || 0,
          indicator3: predictionData[3] || 0,
          indicator4: predictionData[4] || 0,
          predictedClass: predictionData.indexOf(Math.max(...predictionData))
        };
      });
    } catch (error) {
      console.error('Prediction error:', error);
      // Return a default prediction if there's an error
      return {
        notAuthoritarian: 1,
        indicator1: 0,
        indicator2: 0,
        indicator3: 0,
        indicator4: 0,
        predictedClass: 0
      };
    }
  }
  
  // Get the current training data size
  public getTrainingDataSize() {
    return this.trainingData.length;
  }
  
  // Generate synthetic training data from examples
  public generateSyntheticData(examples: string[], count = 100) {
    try {
      const syntheticData: {text: string, label: number}[] = [];
      
      for (let i = 0; i < count; i++) {
        // Pick a random example
        const example = examples[Math.floor(Math.random() * examples.length)];
        
        // Analyze it
        const analysis = analyzeText(example);
        const scores = analysis.map(a => a.score);
        const maxScore = Math.max(...scores);
        const label = maxScore > 3 ? analysis.findIndex(a => a.score === maxScore) + 1 : 0;
        
        if (label > 0) {
          syntheticData.push({
            text: example,
            label
          });
        }
      }
      
      return syntheticData;
    } catch (error) {
      console.error('Error generating synthetic data:', error);
      return [];
    }
  }
  
  // Check if Supabase is being used
  public isUsingSupabase(): boolean {
    return this.useSupabase;
  }
}