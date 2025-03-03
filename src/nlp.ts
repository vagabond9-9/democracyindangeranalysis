import nlp from 'compromise';
import { INDICATORS } from './indicators';

// Prepare keywords for each indicator
const prepareKeywords = () => {
  return INDICATORS.map(indicator => ({
    ...indicator,
    processedKeywords: indicator.keywords.map(keyword => keyword.toLowerCase())
  }));
};

const processedIndicators = prepareKeywords();

// Extract entities and concepts from text
export const extractEntities = (text: string) => {
  const doc = nlp(text);
  
  return {
    people: doc.people().out('array'),
    places: doc.places().out('array'),
    organizations: doc.organizations().out('array'),
    topics: doc.topics().out('array'),
    nouns: doc.nouns().out('array')
  };
};

// Simple sentiment analysis
export const analyzeSentiment = (text: string) => {
  // A very basic sentiment analysis using positive/negative word lists
  const positiveWords = ['good', 'great', 'excellent', 'positive', 'wonderful', 'best', 'love', 'happy', 'right', 'free'];
  const negativeWords = ['bad', 'terrible', 'awful', 'negative', 'worst', 'hate', 'sad', 'wrong', 'evil', 'corrupt'];
  
  const words = text.toLowerCase().split(/\W+/);
  let score = 0;
  
  words.forEach(word => {
    if (positiveWords.includes(word)) score += 0.1;
    if (negativeWords.includes(word)) score -= 0.1;
  });
  
  return Math.max(-1, Math.min(1, score)); // Clamp between -1 and 1
};

// Extract key phrases using term frequency
export const extractKeyPhrases = (text: string, numPhrases = 5) => {
  const doc = nlp(text);
  const nouns = doc.nouns().out('array');
  const verbs = doc.verbs().out('array');
  const adjectives = doc.adjectives().out('array');
  
  // Count term frequencies
  const termFreq: Record<string, number> = {};
  [...nouns, ...verbs, ...adjectives].forEach(term => {
    if (term.length > 3) { // Only consider terms with more than 3 characters
      termFreq[term] = (termFreq[term] || 0) + 1;
    }
  });
  
  // Sort by frequency
  const sortedTerms = Object.entries(termFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, numPhrases)
    .map(([term, freq]) => ({ term, tfidf: freq }));
  
  return sortedTerms;
};

// Advanced text analysis
export const analyzeText = (text: string) => {
  if (!text || text.trim().length === 0) return [];
  
  // Analyze each indicator
  return processedIndicators.map(indicator => {
    const matches: string[] = [];
    const matchDetails: {original: string, context: string}[] = [];
    
    // Find keyword matches
    indicator.processedKeywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      const found = text.match(regex);
      
      if (found) {
        matches.push(...found);
        
        // Get context for each match
        let match;
        const tempRegex = new RegExp(`\\b${keyword}\\b`, 'gi');
        while ((match = tempRegex.exec(text)) !== null) {
          const start = Math.max(0, match.index - 30);
          const end = Math.min(text.length, match.index + keyword.length + 30);
          const context = text.substring(start, end);
          
          matchDetails.push({
            original: match[0],
            context
          });
        }
      }
    });
    
    // Calculate a score based on matches and their significance
    const uniqueMatches = [...new Set(matches)];
    
    // Base score on number of unique matches
    let score = Math.min(10, uniqueMatches.length * 2);
    
    // Adjust score based on sentiment if there are matches
    if (uniqueMatches.length > 0) {
      const sentiment = analyzeSentiment(text);
      // Negative sentiment increases the score for authoritarian indicators
      if (sentiment < -0.2) {
        score = Math.min(10, score + 1);
      }
    }
    
    return {
      indicator: indicator,
      score,
      matches: uniqueMatches,
      matchDetails: matchDetails.filter((v, i, a) => 
        a.findIndex(t => t.original === v.original && t.context === v.context) === i
      )
    };
  });
};

// Function to extract training data from text (like a book chapter)
export const extractTrainingData = (text: string) => {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
  
  return sentences.map(sentence => {
    const analysis = analyzeText(sentence);
    const highestScore = Math.max(...analysis.map(a => a.score));
    const primaryIndicator = analysis.find(a => a.score === highestScore)?.indicator.id || 0;
    
    return {
      text: sentence.trim(),
      label: primaryIndicator > 0 ? primaryIndicator : 0, // 0 means no indicator detected
      scores: analysis.map(a => a.score)
    };
  }).filter(item => item.label > 0); // Only keep sentences with detected indicators
};