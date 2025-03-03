import { createClient } from '@supabase/supabase-js';
import { v4 as uuidv4 } from 'uuid';

// Initialize Supabase client with public anon key (safe to expose)
// These are placeholder values - users will need to connect their own Supabase instance
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://your-project.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'your-anon-key';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Check if Supabase is properly configured
export const isSupabaseConfigured = () => {
  return supabaseUrl !== 'https://your-project.supabase.co' && 
         supabaseAnonKey !== 'your-anon-key';
};

// PDF Storage and Processing
export const pdfStorage = {
  // Upload PDF chunks to Supabase storage
  async uploadPdfChunk(pdfData: Uint8Array, fileName: string, chunkIndex: number): Promise<string> {
    try {
      const chunkId = uuidv4();
      const path = `pdf-chunks/${fileName.replace(/\s+/g, '_')}_chunk_${chunkIndex}_${chunkId}`;
      
      const { error } = await supabase.storage
        .from('democracy-analyzer')
        .upload(path, pdfData);
      
      if (error) throw error;
      
      return path;
    } catch (error) {
      console.error('Error uploading PDF chunk:', error);
      throw error;
    }
  },
  
  // Store extracted text in Supabase
  async storeExtractedText(fileName: string, text: string, pageRange: {start: number, end: number}): Promise<string> {
    try {
      const id = uuidv4();
      
      const { error } = await supabase
        .from('extracted_texts')
        .insert({
          id,
          file_name: fileName,
          text,
          page_start: pageRange.start,
          page_end: pageRange.end,
          created_at: new Date().toISOString()
        });
      
      if (error) throw error;
      
      return id;
    } catch (error) {
      console.error('Error storing extracted text:', error);
      throw error;
    }
  },
  
  // Get extracted text from Supabase
  async getExtractedText(id: string): Promise<string> {
    try {
      const { data, error } = await supabase
        .from('extracted_texts')
        .select('text')
        .eq('id', id)
        .single();
      
      if (error) throw error;
      
      return data.text;
    } catch (error) {
      console.error('Error getting extracted text:', error);
      throw error;
    }
  },
  
  // List all extracted texts
  async listExtractedTexts(): Promise<{id: string, fileName: string, pageRange: {start: number, end: number}, createdAt: string}[]> {
    try {
      const { data, error } = await supabase
        .from('extracted_texts')
        .select('id, file_name, page_start, page_end, created_at')
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      
      return data.map(item => ({
        id: item.id,
        fileName: item.file_name,
        pageRange: {
          start: item.page_start,
          end: item.page_end
        },
        createdAt: item.created_at
      }));
    } catch (error) {
      console.error('Error listing extracted texts:', error);
      return [];
    }
  }
};

// ML Model Storage and Training
export const mlStorage = {
  // Store training data
  async storeTrainingData(data: {text: string, label: number}[]): Promise<string> {
    try {
      const batchId = uuidv4();
      
      // Insert in batches to avoid payload size limits
      const BATCH_SIZE = 100;
      for (let i = 0; i < data.length; i += BATCH_SIZE) {
        const batch = data.slice(i, i + BATCH_SIZE).map(item => ({
          id: uuidv4(),
          batch_id: batchId,
          text: item.text,
          label: item.label,
          created_at: new Date().toISOString()
        }));
        
        const { error } = await supabase
          .from('training_data')
          .insert(batch);
        
        if (error) throw error;
      }
      
      return batchId;
    } catch (error) {
      console.error('Error storing training data:', error);
      throw error;
    }
  },
  
  // Get training data
  async getTrainingData(): Promise<{text: string, label: number}[]> {
    try {
      const { data, error } = await supabase
        .from('training_data')
        .select('text, label')
        .limit(5000); // Limit to prevent memory issues
      
      if (error) throw error;
      
      return data.map(item => ({
        text: item.text,
        label: item.label
      }));
    } catch (error) {
      console.error('Error getting training data:', error);
      return [];
    }
  },
  
  // Store trained model weights
  async storeModelWeights(modelWeights: ArrayBuffer): Promise<string> {
    try {
      const modelId = uuidv4();
      const path = `models/${modelId}`;
      
      const { error } = await supabase.storage
        .from('democracy-analyzer')
        .upload(path, modelWeights);
      
      if (error) throw error;
      
      return modelId;
    } catch (error) {
      console.error('Error storing model weights:', error);
      throw error;
    }
  },
  
  // Get trained model weights
  async getLatestModelWeights(): Promise<ArrayBuffer | null> {
    try {
      const { data, error } = await supabase.storage
        .from('democracy-analyzer')
        .list('models', {
          limit: 1,
          sortBy: { column: 'created_at', order: 'desc' }
        });
      
      if (error) throw error;
      
      if (data.length === 0) return null;
      
      const { data: modelData, error: downloadError } = await supabase.storage
        .from('democracy-analyzer')
        .download(`models/${data[0].name}`);
      
      if (downloadError) throw downloadError;
      
      return await modelData.arrayBuffer();
    } catch (error) {
      console.error('Error getting model weights:', error);
      return null;
    }
  }
};