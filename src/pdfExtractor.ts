import * as pdfjsLib from 'pdfjs-dist';
import { INDICATORS } from './indicators';
import { pdfStorage, isSupabaseConfigured } from './supabase';

// Configure PDF.js worker - use a bundled worker approach
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.mjs',
  import.meta.url
).toString();

export class PDFExtractor {
  private pdfData: Uint8Array | null = null;
  private numPages = 0;
  private extractedText = '';
  private processingStatus = '';
  private fileName = '';
  private extractedTextId: string | null = null;
  private useSupabase = false;
  
  constructor() {
    // Check if Supabase is configured
    this.useSupabase = isSupabaseConfigured();
  }
  
  // Load a PDF file
  public async loadPDF(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = async (event) => {
        if (event.target?.result) {
          try {
            const arrayBuffer = event.target.result as ArrayBuffer;
            // Create a copy of the array buffer to prevent detachment issues
            this.pdfData = new Uint8Array(arrayBuffer);
            this.fileName = file.name;
            
            // Create a new copy for the initial loading to avoid detachment
            const dataCopy = new Uint8Array(this.pdfData);
            const loadingTask = pdfjsLib.getDocument({ data: dataCopy });
            const pdf = await loadingTask.promise;
            this.numPages = pdf.numPages;
            console.log(`PDF loaded successfully with ${this.numPages} pages`);
            resolve();
          } catch (error) {
            console.error('PDF loading error:', error);
            reject(error);
          }
        } else {
          reject(new Error('Failed to read file'));
        }
      };
      
      reader.onerror = (error) => {
        reject(error);
      };
      
      reader.readAsArrayBuffer(file);
    });
  }
  
  // Extract text from a single page
  private async extractPageText(pdf: pdfjsLib.PDFDocumentProxy, pageNum: number): Promise<string> {
    try {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();
      
      // Join all the text items
      let lastY = -1;
      let text = '';
      
      for (const item of textContent.items) {
        const textItem = item as pdfjsLib.TextItem;
        
        // Add newlines between different vertical positions (paragraphs)
        if (lastY !== -1 && Math.abs(textItem.transform[5] - lastY) > 5) {
          text += '\n';
        }
        
        text += textItem.str + ' ';
        lastY = textItem.transform[5];
      }
      
      return text;
    } catch (error) {
      console.warn(`Error extracting text from page ${pageNum}:`, error);
      return `[Error extracting page ${pageNum}]`;
    }
  }
  
  // Extract text from the PDF with pagination support
  public async extractText(startPage = 1, endPage?: number): Promise<string> {
    if (!this.pdfData) {
      throw new Error('No PDF loaded');
    }
    
    try {
      // Create a new copy of the data for each extraction to avoid detachment issues
      const dataCopy = new Uint8Array(this.pdfData);
      const loadingTask = pdfjsLib.getDocument({ data: dataCopy });
      const pdf = await loadingTask.promise;
      const maxPage = Math.min(endPage || this.numPages, this.numPages);
      const startPageNum = Math.max(1, startPage);
      
      if (startPageNum > maxPage) {
        throw new Error('Start page cannot be greater than end page');
      }
      
      let extractedText = '';
      
      // Process in smaller batches to avoid memory issues
      const BATCH_SIZE = 3;
      const totalBatches = Math.ceil((maxPage - startPageNum + 1) / BATCH_SIZE);
      
      for (let batch = 0; batch < totalBatches; batch++) {
        const batchStart = startPageNum + (batch * BATCH_SIZE);
        const batchEnd = Math.min(batchStart + BATCH_SIZE - 1, maxPage);
        
        this.processingStatus = `Processing pages ${batchStart}-${batchEnd} of ${maxPage}...`;
        console.log(this.processingStatus);
        
        // Process pages in this batch
        const pagePromises = [];
        for (let i = batchStart; i <= batchEnd; i++) {
          this.processingStatus = `Processing page ${i} of ${maxPage}...`;
          pagePromises.push(this.extractPageText(pdf, i));
        }
        
        // Wait for all pages in the batch to be processed
        const pageTexts = await Promise.all(pagePromises);
        extractedText += pageTexts.join('\n\n');
        
        // Allow UI to update between batches and free up memory
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      this.extractedText = extractedText;
      
      // Store in Supabase if available
      if (this.useSupabase) {
        try {
          this.processingStatus = 'Storing extracted text in Supabase...';
          this.extractedTextId = await pdfStorage.storeExtractedText(
            this.fileName,
            extractedText,
            { start: startPageNum, end: maxPage }
          );
          this.processingStatus = 'Text stored in Supabase successfully';
        } catch (error) {
          console.error('Failed to store text in Supabase:', error);
          // Continue with local processing if Supabase fails
        }
      }
      
      this.processingStatus = '';
      console.log(`Extracted ${extractedText.length} characters of text`);
      return extractedText;
    } catch (error) {
      console.error('Text extraction error:', error);
      this.processingStatus = '';
      throw error;
    }
  }
  
  // Get current processing status
  public getProcessingStatus(): string {
    return this.processingStatus;
  }
  
  // Extract training data from the PDF
  public async extractTrainingData(): Promise<{text: string, label: number}[]> {
    if (!this.extractedText || this.extractedText.trim().length === 0) {
      throw new Error('No text extracted from PDF');
    }
    
    try {
      this.processingStatus = 'Extracting training data...';
      
      // Split the text into paragraphs
      const paragraphs = this.extractedText.split(/\n\s*\n/);
      console.log(`Found ${paragraphs.length} paragraphs for training data extraction`);
      
      // Process each paragraph to extract training data
      const trainingData: {text: string, label: number}[] = [];
      
      // Generate synthetic training data
      for (const paragraph of paragraphs) {
        if (paragraph.trim().length > 50) { // Only process substantial paragraphs
          // For each paragraph, check if it contains keywords from our indicators
          const sentences = paragraph.match(/[^.!?]+[.!?]+/g) || [];
          
          for (const sentence of sentences) {
            if (sentence.trim().length < 20) continue; // Skip very short sentences
            
            // Check if the sentence contains keywords from any indicator
            let foundIndicator = false;
            let indicatorId = 0;
            
            for (let i = 0; i < 4; i++) {
              const indicator = INDICATORS[i];
              for (const keyword of indicator.keywords) {
                if (sentence.toLowerCase().includes(keyword.toLowerCase())) {
                  foundIndicator = true;
                  indicatorId = i + 1;
                  break;
                }
              }
              if (foundIndicator) break;
            }
            
            if (foundIndicator) {
              trainingData.push({
                text: sentence.trim(),
                label: indicatorId
              });
            } else if (Math.random() < 0.1) { // Add some non-authoritarian examples (10% chance)
              trainingData.push({
                text: sentence.trim(),
                label: 0 // Not authoritarian
              });
            }
          }
        }
      }
      
      // Store training data in Supabase if available
      if (this.useSupabase && trainingData.length > 0) {
        try {
          this.processingStatus = 'Storing training data in Supabase...';
          await pdfStorage.storeTrainingData(trainingData);
          this.processingStatus = 'Training data stored in Supabase successfully';
        } catch (error) {
          console.error('Failed to store training data in Supabase:', error);
          // Continue with local processing if Supabase fails
        }
      }
      
      this.processingStatus = '';
      console.log(`Generated ${trainingData.length} training examples`);
      return trainingData;
    } catch (error) {
      console.error('Training data extraction error:', error);
      this.processingStatus = '';
      return [];
    }
  }
  
  // Get the number of pages in the PDF
  public getPageCount(): number {
    return this.numPages;
  }
  
  // Get a summary of the extracted text
  public getTextSummary(): string {
    if (!this.extractedText || this.extractedText.trim().length === 0) {
      return 'No text extracted';
    }
    
    const words = this.extractedText.split(/\s+/).length;
    const paragraphs = this.extractedText.split(/\n\s*\n/).length;
    const characters = this.extractedText.length;
    
    let summary = `Extracted ${words.toLocaleString()} words (${characters.toLocaleString()} characters) in ${paragraphs} paragraphs from ${this.numPages} pages.`;
    
    if (this.useSupabase && this.extractedTextId) {
      summary += ' Text stored in Supabase for better performance.';
    }
    
    return summary;
  }
  
  // Check if text has been extracted
  public hasExtractedText(): boolean {
    return this.extractedText.trim().length > 0;
  }
  
  // Get the extracted text ID from Supabase
  public getExtractedTextId(): string | null {
    return this.extractedTextId;
  }
  
  // Check if Supabase is being used
  public isUsingSupabase(): boolean {
    return this.useSupabase;
  }
}