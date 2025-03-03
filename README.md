# Democracy Analyzer

An interactive tool for analyzing text for authoritarian language patterns based on "How Democracies Die" by Steven Levitsky and Daniel Ziblatt.

## Features

- Analyze text for the four key indicators of authoritarianism
- Extract entities, sentiment, and key phrases using NLP
- Train a machine learning model on examples from "How Democracies Die"
- Process PDFs to extract training data
- Cloud storage integration with Supabase for better performance

## Setup

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

## Supabase Integration (Optional)

For better performance with large PDFs and ML models, you can connect to Supabase:

1. Create a Supabase account at [supabase.com](https://supabase.com)
2. Create a new project
3. Copy the `.env.example` file to `.env`
4. Update the `.env` file with your Supabase URL and anon key
5. Run the SQL migration in `supabase/migrations/create_democracy_analyzer_tables.sql`

## How It Works

The Democracy Analyzer examines text for language that aligns with the four key indicators of authoritarianism identified in "How Democracies Die":

1. Rejection of democratic rules
2. Denial of legitimacy of opponents
3. Toleration of violence
4. Readiness to curtail civil liberties

The tool uses natural language processing and machine learning to identify these patterns in text.

## Training the Model

1. Upload a PDF of "How Democracies Die"
2. Extract text from the PDF (process in smaller chunks for large PDFs)
3. Extract training data from the text
4. Train the machine learning model
5. Use the model to analyze new text

## Technologies Used

- React
- TypeScript
- TensorFlow.js
- PDF.js
- Compromise NLP
- Supabase (optional)
- Tailwind CSS