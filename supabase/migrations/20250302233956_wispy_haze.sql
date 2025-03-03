/*
  # Democracy Analyzer Schema

  1. New Tables
    - `extracted_texts`
      - `id` (uuid, primary key)
      - `file_name` (text)
      - `text` (text)
      - `page_start` (integer)
      - `page_end` (integer)
      - `created_at` (timestamp)
    - `training_data`
      - `id` (uuid, primary key)
      - `batch_id` (uuid)
      - `text` (text)
      - `label` (integer)
      - `created_at` (timestamp)
  
  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users
*/

-- Create storage bucket for PDF chunks and models
INSERT INTO storage.buckets (id, name, public) 
VALUES ('democracy-analyzer', 'democracy-analyzer', false)
ON CONFLICT (id) DO NOTHING;

-- Create extracted_texts table
CREATE TABLE IF NOT EXISTS extracted_texts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  file_name TEXT NOT NULL,
  text TEXT NOT NULL,
  page_start INTEGER NOT NULL,
  page_end INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Create training_data table
CREATE TABLE IF NOT EXISTS training_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_id UUID NOT NULL,
  text TEXT NOT NULL,
  label INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE extracted_texts ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_data ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users
CREATE POLICY "Allow authenticated users to read extracted_texts"
  ON extracted_texts
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to insert extracted_texts"
  ON extracted_texts
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Allow authenticated users to read training_data"
  ON training_data
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to insert training_data"
  ON training_data
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

-- Create policies for storage
CREATE POLICY "Allow authenticated users to read democracy-analyzer bucket"
  ON storage.objects
  FOR SELECT
  TO authenticated
  USING (bucket_id = 'democracy-analyzer');

CREATE POLICY "Allow authenticated users to insert into democracy-analyzer bucket"
  ON storage.objects
  FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'democracy-analyzer');