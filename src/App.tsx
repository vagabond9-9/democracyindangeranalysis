import React, { useState, useRef, useEffect } from 'react';
import { AlertTriangle, Info, Check, Cloud, Server, Brain, Database, Upload, FileText, Loader } from 'lucide-react';
import { INDICATORS } from './indicators';
import { PDFExtractor } from './pdfExtractor';
import { analyzeText, extractEntities, analyzeSentiment, extractKeyPhrases } from './nlp';
import { AuthoritarianClassifier } from './ml';
import { isSupabaseConfigured } from './supabase';

// Initialize PDF extractor and ML classifier
const pdfExtractor = new PDFExtractor();
const classifier = new AuthoritarianClassifier();

function App() {
  // State for UI tabs
  const [activeTab, setActiveTab] = useState('analysis');
  
  // State for text analysis
  const [inputText, setInputText] = useState('');
  const [analysis, setAnalysis] = useState([]);
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [sentiment, setSentiment] = useState(null);
  const [entities, setEntities] = useState(null);
  const [keyPhrases, setKeyPhrases] = useState([]);
  const [mlPrediction, setMlPrediction] = useState(null);
  
  // State for PDF processing
  const fileInputRef = useRef(null);
  const [pdfFileName, setPdfFileName] = useState('');
  const [pdfError, setPdfError] = useState('');
  const [pdfStatus, setPdfStatus] = useState('');
  const [totalPages, setTotalPages] = useState(0);
  const [pageRange, setPageRange] = useState({ start: 1, end: 5 });
  const [processingPages, setProcessingPages] = useState(false);
  const [pdfSummary, setPdfSummary] = useState('');
  
  // State for ML training
  const [trainingDataSize, setTrainingDataSize] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  
  // State for Supabase connection
  const [usingSupabase, setUsingSupabase] = useState(false);
  const [needsSupabaseSetup, setNeedsSupabaseSetup] = useState(false);
  
  // Check if Supabase is configured
  useEffect(() => {
    const supabaseConfigured = isSupabaseConfigured();
    setUsingSupabase(supabaseConfigured);
    setNeedsSupabaseSetup(!supabaseConfigured);
    
    // Check if classifier has training data
    setTrainingDataSize(classifier.getTrainingDataSize());
  }, []);
  
  // Update training status periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (isTraining) {
        setTrainingStatus(classifier.getTrainingStatus());
      }
      
      if (processingPages) {
        setPdfStatus(pdfExtractor.getProcessingStatus());
      }
    }, 500);
    
    return () => clearInterval(interval);
  }, [isTraining, processingPages]);
  
  // Handle PDF file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    if (file.type !== 'application/pdf') {
      setPdfError('Please upload a PDF file');
      return;
    }
    
    try {
      setPdfFileName(file.name);
      setPdfError('');
      setPdfStatus('Loading PDF...');
      
      await pdfExtractor.loadPDF(file);
      
      setTotalPages(pdfExtractor.getPageCount());
      setPageRange({ start: 1, end: Math.min(5, pdfExtractor.getPageCount()) });
      setPdfStatus('');
    } catch (error) {
      console.error('PDF loading error:', error);
      setPdfError('Failed to load PDF. The file may be corrupted or too large.');
      setPdfStatus('');
    }
  };
  
  // Extract text from PDF
  const handleExtractText = async () => {
    try {
      setProcessingPages(true);
      setPdfStatus('Extracting text...');
      
      await pdfExtractor.extractText(pageRange.start, pageRange.end);
      
      setPdfSummary(pdfExtractor.getTextSummary());
      setProcessingPages(false);
      setPdfStatus('');
    } catch (error) {
      console.error('PDF processing error:', error);
      setPdfError('Failed to extract text. Try processing fewer pages at once.');
      setProcessingPages(false);
      setPdfStatus('');
    }
  };
  
  // Extract training data from PDF
  const extractTrainingData = async () => {
    try {
      setPdfStatus('Extracting training data...');
      
      const trainingData = await pdfExtractor.extractTrainingData();
      
      if (trainingData.length > 0) {
        const newSize = classifier.addTrainingData(trainingData);
        setTrainingDataSize(newSize);
        setPdfStatus(`Extracted ${trainingData.length} training examples`);
      } else {
        setPdfStatus('No training examples found. Try different pages.');
      }
      
      setTimeout(() => setPdfStatus(''), 3000);
    } catch (error) {
      console.error('Training data extraction error:', error);
      setPdfError('Failed to extract training data');
      setTimeout(() => setPdfError(''), 3000);
    }
  };
  
  // Train the ML model
  const trainModel = async () => {
    try {
      setIsTraining(true);
      setTrainingStatus('Preparing to train model...');
      
      await classifier.train();
      
      setTrainingStatus('Model trained successfully!');
      setTimeout(() => setTrainingStatus(''), 3000);
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStatus(`Training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };
  
  // Analyze text with advanced features
  const analyzeTextAdvanced = () => {
    if (!inputText.trim()) return;
    
    // Basic analysis
    const results = analyzeText(inputText);
    setAnalysis(results);
    
    // Sentiment analysis
    const sentimentScore = analyzeSentiment(inputText);
    setSentiment(sentimentScore);
    
    // Entity extraction
    const extractedEntities = extractEntities(inputText);
    setEntities(extractedEntities);
    
    // Key phrase extraction
    const phrases = extractKeyPhrases(inputText);
    setKeyPhrases(phrases);
    
    // ML prediction if we have training data
    if (trainingDataSize > 10) {
      const prediction = classifier.predict(inputText);
      setMlPrediction(prediction);
    } else {
      setMlPrediction(null);
    }
    
    setIsAnalyzed(true);
  };
  
  // Reset analysis
  const resetAnalysis = () => {
    setInputText('');
    setAnalysis([]);
    setIsAnalyzed(false);
    setSentiment(null);
    setEntities(null);
    setKeyPhrases([]);
    setMlPrediction(null);
  };
  
  // Get overall risk level
  const getOverallRisk = () => {
    if (!analysis || analysis.length === 0) return 0;
    
    // Calculate weighted average of all indicator scores
    const sum = analysis.reduce((acc, result) => acc + result.score, 0);
    return sum / analysis.length;
  };
  
  // Get risk level text
  const getRiskLevel = (score) => {
    if (score < 2) return 'Low';
    if (score < 5) return 'Moderate';
    if (score < 8) return 'High';
    return 'Extreme';
  };
  
  // Get color based on risk level
  const getRiskColor = (score) => {
    if (score < 2) return 'text-green-600';
    if (score < 5) return 'text-yellow-600';
    if (score < 8) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="flex items-center mb-4 md:mb-0">
              <AlertTriangle className="text-red-600 mr-2" size={24} />
              <h1 className="text-xl font-bold">Authoritarian Language Analyzer</h1>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('analysis')}
                className={`px-4 py-2 rounded-md font-medium ${
                  activeTab === 'analysis'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Analysis
              </button>
              <button
                onClick={() => setActiveTab('training')}
                className={`px-4 py-2 rounded-md font-medium ${
                  activeTab === 'training'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Training
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4 md:p-6">
        {needsSupabaseSetup && (
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-6">
            <div className="flex items-start">
              <Cloud className="text-indigo-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-medium text-indigo-800">Connect to Supabase for Better Performance</h3>
                <p className="text-indigo-700 text-sm mt-1">
                  This application can use Supabase to offload memory-intensive operations from your browser.
                  This will significantly improve performance when processing PDFs and training ML models.
                </p>
                <div className="mt-3">
                  <button 
                    className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center"
                    onClick={() => window.open('https://supabase.com/dashboard/sign-in', '_blank')}
                  >
                    <Server className="mr-2" size={16} />
                    Connect to Supabase
                  </button>
                </div>
                <p className="text-xs text-indigo-600 mt-2">
                  After connecting, add your Supabase URL and anon key to the .env file.
                </p>
              </div>
            </div>
          </div>
        )}

        {usingSupabase && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-6">
            <div className="flex items-center">
              <Cloud className="text-green-600 mr-2" size={18} />
              <p className="text-green-800 text-sm">
                <span className="font-medium">Supabase Connected:</span> Using cloud storage for better performance with large PDFs and ML models.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'analysis' ? (
          <>
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center">
                <Info className="mr-2" size={20} />
                About This Tool
              </h2>
              <p className="text-gray-700 mb-4">
                This analyzer examines text for language that aligns with the four key indicators of authoritarianism 
                identified in "How Democracies Die" by Steven Levitsky and Daniel Ziblatt. 
                Enter any text (speeches, articles, social media posts) to analyze its authoritarian tendencies.
              </p>
              <div className="bg-blue-50 p-4 rounded-md">
                <h3 className="font-medium text-blue-800 mb-2">The Four Indicators:</h3>
                <ul className="list-disc pl-5 text-blue-800">
                  {INDICATORS.map(indicator => (
                    <li key={indicator.id} className="mb-1">{indicator.name}</li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-4 bg-purple-50 p-4 rounded-md">
                <h3 className="font-medium text-purple-800 mb-2 flex items-center">
                  <Brain size={16} className="mr-1" />
                  Advanced Analysis Features:
                </h3>
                <ul className="list-disc pl-5 text-purple-800">
                  <li>Natural Language Processing for deeper text analysis</li>
                  <li>Entity extraction (people, places, organizations)</li>
                  <li>Sentiment analysis</li>
                  <li>Key phrase extraction</li>
                  {trainingDataSize > 10 && <li>Machine learning classification</li>}
                </ul>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-lg font-semibold mb-4">Text Analysis</h2>
              <textarea
                className="w-full p-3 border border-gray-300 rounded-md h-40 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Paste text to analyze (speech, article, social media post, etc.)"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              ></textarea>
              <div className="mt-4 flex gap-2">
                <button
                  onClick={analyzeTextAdvanced}
                  disabled={!inputText.trim()}
                  className={`px-4 py-2 rounded-md font-medium ${
                    !inputText.trim() 
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  Analyze Text
                </button>
                {isAnalyzed && (
                  <button
                    onClick={resetAnalysis}
                    className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md font-medium hover:bg-gray-300"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>

            {isAnalyzed && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="mb-6">
                  <h2 className="text-lg font-semibold mb-2">Analysis Results</h2>
                  <div className="flex items-center">
                    <div className="mr-4">
                      <div className="text-3xl font-bold mb-1 flex items-center">
                        <span className={getRiskColor(getOverallRisk())}>
                          {getRiskLevel(getOverallRisk())} Risk
                        </span>
                        <AlertTriangle 
                          className={`ml-2 ${getRiskColor(getOverallRisk())}`} 
                          size={24} 
                        />
                      </div>
                      <div className="text-sm text-gray-500">
                        Overall authoritarian tendency score: {getOverallRisk().toFixed(1)}/10
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {sentiment !== null && (
                    <div className="bg-gray-50 p-4 rounded-md">
                      <h3 className="font-medium mb-2">Sentiment Analysis</h3>
                      <div className="flex items-center">
                        <div className={`font-bold ${
                          sentiment < -0.2 ? 'text-red-600' : 
                          sentiment > 0.2 ? 'text-green-600' : 'text-yellow-600'
                        }`}>
                          {sentiment < -0.2 ? 'Negative' : 
                           sentiment > 0.2 ? 'Positive' : 'Neutral'}
                        </div>
                        <span className="ml-2 text-sm text-gray-500">
                          ({sentiment.toFixed(2)})
                        </span>
                      </div>
                    </div>
                  )}
                  
                  {entities && (
                    <div className="bg-gray-50 p-4 rounded-md">
                      <h3 className="font-medium mb-2">Entities Detected</h3>
                      <div className="text-sm">
                        {entities.people.length > 0 && (
                          <div className="mb-1">
                            <span className="font-medium">People:</span> {entities.people.join(', ')}
                          </div>
                        )}
                        {entities.organizations.length > 0 && (
                          <div className="mb-1">
                            <span className="font-medium">Organizations:</span> {entities.organizations.join(', ')}
                          </div>
                        )}
                        {entities.places.length > 0 && (
                          <div>
                            <span className="font-medium">Places:</span> {entities.places.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {keyPhrases.length > 0 && (
                    <div className="bg-gray-50 p-4 rounded-md md:col-span-2">
                      <h3 className="font-medium mb-2">Key Phrases</h3>
                      <div className="flex flex-wrap gap-2">
                        {keyPhrases.map((phrase, idx) => (
                          <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                            {phrase.term}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {mlPrediction && (
                    <div className="bg-purple-50 p-4 rounded-md md:col-span-2">
                      <h3 className="font-medium mb-2 flex items-center">
                        <Brain size={16} className="mr-1" />
                        ML Classification
                      </h3>
                      <div>
                        <div className="mb-2">
                          <span className="font-medium">Predicted indicator: </span>
                          {mlPrediction.predictedClass === 0 ? (
                            <span className="text-green-600">Not authoritarian</span>
                          ) : (
                            <span className="text-red-600">
                              {INDICATORS[mlPrediction.predictedClass - 1]?.name || 'Unknown'}
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>Not authoritarian: {(mlPrediction.notAuthoritarian * 100).toFixed(1)}%</div>
                          <div>Indicator 1: {(mlPrediction.indicator1 * 100).toFixed(1)}%</div>
                          <div>Indicator 2: {(mlPrediction.indicator2 * 100).toFixed(1)}%</div>
                          <div>Indicator 3: {(mlPrediction.indicator3 * 100).toFixed(1)}%</div>
                          <div>Indicator 4: {(mlPrediction.indicator4 * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-6">
                  {analysis.map((result) => (
                    <div key={result.indicator.id} className="border-t pt-4">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-semibold text-lg">{result.indicator.name}</h3>
                        <div className={`px-2 py-1 rounded-full text-sm font-medium ${
                          getRiskColor(result.score)
                        }`}>
                          Score: {result.score}/10
                        </div>
                      </div>
                      <p className="text-gray-700 mb-3">{result.indicator.description}</p>
                      
                      <div className="mt-2">
                        {result.matches.length > 0 ? (
                          <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-1">Concerning language found:</h4>
                            <div className="flex flex-wrap gap-1">
                              {result.matches.map((match, idx) => (
                                <span key={idx} className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">
                                  {match}
                                </span>
                              ))}
                            </div>
                            
                            {result.matchDetails && result.matchDetails.length > 0 && (
                              <div className="mt-3">
                                <h4 className="text-sm font-medium text-gray-700 mb-1">Context:</h4>
                                <div className="space-y-2 text-sm">
                                  {result.matchDetails.slice(0, 3).map((detail, idx) => (
                                    <div key={idx} className="bg-gray-50 p-2 rounded">
                                      "...{detail.context}..."
                                    </div>
                                  ))}
                                  {result.matchDetails.length > 3 && (
                                    <div className="text-gray-500 text-xs">
                                      +{result.matchDetails.length - 3} more instances
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="flex items-center text-green-600">
                            <Check size={16} className="mr-1" />
                            <span className="text-sm">No concerning language detected</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Database className="mr-2" size={20} />
              Training Data & Model
            </h2>
            
            <div className="mb-6">
              <p className="text-gray-700 mb-4">
                Upload a PDF of "How Democracies Die" to extract training data and train a machine learning model.
                This will enhance the analyzer's ability to detect authoritarian language patterns.
                {usingSupabase && " Using Supabase for storage allows processing larger documents with less browser memory usage."}
              </p>
              
              <div className="bg-gray-50 p-4 rounded-md mb-4">
                <h3 className="font-medium mb-3">Upload PDF</h3>
                <div className="flex items-center">
                  <input
                    type="file"
                    accept=".pdf"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  >
                    <Upload size={16} className="mr-2" />
                    Select PDF
                  </button>
                  <span className={`ml-3 text-sm ${pdfError ? 'text-red-600 font-medium' : 'text-gray-600'}`}>
                    {pdfError || pdfStatus || (pdfFileName && `Selected: ${pdfFileName}`)}
                  </span>
                </div>
                
                {processingPages && (
                  <div className="mt-3 flex items-center text-blue-600">
                    <Loader size={16} className="mr-2 animate-spin" />
                    <span className="text-sm">Processing PDF... This may take a few minutes.</span>
                  </div>
                )}
                
                {pdfSummary && (
                  <div className="mt-3 text-sm text-gray-700">
                    <FileText size={14} className="inline mr-1" />
                    {pdfSummary}
                  </div>
                )}
              </div>
              
              {totalPages > 0 && (
                <div className="bg-blue-50 p-4 rounded-md mb-4">
                  <h4 className="font-medium text-blue-800 mb-2">Select Pages to Process</h4>
                  <div className="flex flex-wrap items-center gap-4">
                    <div>
                      <label className="block text-xs text-blue-700 mb-1">Start Page</label>
                      <input 
                        type="number" 
                        min="1" 
                        max={totalPages}
                        value={pageRange.start}
                        onChange={(e) => setPageRange({...pageRange, start: Math.max(1, parseInt(e.target.value) || 1)})}
                        className="w-20 p-1 border rounded text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-blue-700 mb-1">End Page</label>
                      <input 
                        type="number" 
                        min={pageRange.start}
                        max={totalPages}
                        value={pageRange.end}
                        onChange={(e) => setPageRange({...pageRange, end: Math.min(totalPages, parseInt(e.target.value) || totalPages)})}
                        className="w-20 p-1 border rounded text-sm"
                      />
                    </div>
                    <button
                      onClick={handleExtractText}
                      disabled={!totalPages || processingPages}
                      className={`px-4 py-2 rounded-md font-medium mt-3 ${
                        !totalPages || processingPages
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-blue-600 text-white hover:bg-blue-700'
                      }`}
                    >
                      {processingPages ? (
                        <>
                          <Loader size={16} className="inline mr-2 animate-spin" />
                          Processing...
                        </>
                      ) : 'Extract Text'}
                    </button>
                  </div>
                  <p className="text-xs text-blue-700 mt-2">
                    Processing large PDFs may cause memory issues. For best results, process 3-5 pages at a time.
                  </p>
                </div>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <h3 className="font-medium mb-3">Extract Training Data</h3>
                  <p className="text-sm text-gray-600 mb-3">
                    Extract examples from the PDF to train the model.
                    {usingSupabase && " Data will be stored in Supabase."}
                  </p>
                  <button
                    onClick={extractTrainingData}
                    disabled={!pdfExtractor.hasExtractedText()}
                    className={`px-4 py-2 rounded-md font-medium ${
                      !pdfExtractor.hasExtractedText()
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                     Extract Training Data
                  </button>
                  
                  {trainingDataSize > 0 && (
                    <div className="mt-3 text-sm">
                      <span className="font-medium">Training examples:</span> {trainingDataSize}
                    </div>
                  )}
                </div>
                
                <div className="bg-gray-50 p-4 rounded-md">
                  <h3 className="font-medium mb-3">Train Model</h3>
                  <p className="text-sm text-gray-600 mb-3">
                    Train a machine learning model on the extracted data.
                    {usingSupabase && " Model weights will be saved to Supabase."}
                  </p>
                  <button
                    onClick={trainModel}
                    disabled={trainingDataSize < 10 || isTraining}
                    className={`px-4 py-2 rounded-md font-medium ${
                      trainingDataSize < 10 || isTraining
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-purple-600 text-white hover:bg-purple-700'
                    }`}
                  >
                    {isTraining ? (
                      <>
                        <Loader size={16} className="inline mr-2 animate-spin" />
                        Training...
                      </>
                    ) : 'Train Model'}
                  </button>
                  
                  {trainingStatus && (
                    <div className="mt-3 text-sm">
                      <span className="font-medium">Status:</span> {trainingStatus}
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="bg-blue-50 p-4 rounded-md">
              <h3 className="font-medium text-blue-800 mb-2">How Training Works:</h3>
              <ol className="list-decimal pl-5 text-blue-800 space-y-1">
                <li>Upload the PDF of "How Democracies Die"</li>
                <li>Extract text from the PDF (process in smaller chunks for large PDFs)</li>
                <li>Process the text to identify examples of authoritarian language</li>
                <li>Train a machine learning model on these examples</li>
                <li>Use the model to analyze new text with greater accuracy</li>
              </ol>
              <p className="mt-3 text-sm text-blue-700">
                Once trained, switch to "Analysis" mode to see machine learning predictions for your text.
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gray-100 border-t mt-8 py-4">
        <div className="container mx-auto px-4 text-center text-gray-600 text-sm">
          <p>This tool is for educational purposes only. It provides analysis based on NLP techniques and machine learning.</p>
          <p className="mt-1">For a comprehensive understanding, please read "How Democracies Die" by Steven Levitsky & Daniel Ziblatt.</p>
          {usingSupabase && (
            <p className="mt-1 text-blue-600">
              <Cloud size={14} className="inline mr-1" />
              Using Supabase for cloud storage and processing
            </p>
          )}
        </div>
      </footer>
    </div>
  );
}

export default App;