-- scripts/db/schema.sql

-- ============================================
-- RAG System Database Schema
-- PostgreSQL 14+
-- ============================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- PARENTS TABLE
-- Stores parent chunks (full context)
-- ============================================
CREATE TABLE IF NOT EXISTS parents (
    -- Identity
    parent_id VARCHAR(100) PRIMARY KEY,
    
    -- Content
    parent_text TEXT NOT NULL,
    parent_type VARCHAR(20) NOT NULL CHECK (parent_type IN ('text', 'table', 'heading', 'list')),
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    
    -- Table-specific metadata
    table_header TEXT,  -- NULL for non-table parents
    is_table_split BOOLEAN DEFAULT FALSE,
    split_part INTEGER,
    total_parts INTEGER DEFAULT 1,
    
    -- Source metadata
    source_file VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    page_number INTEGER,
    section_header VARCHAR(500),
    element_id INTEGER,  -- From Upstage parser
    
    -- Row tracking (for table parents)
    start_row_idx INTEGER,
    end_row_idx INTEGER,
    
    -- Access Control
    acl_users TEXT[],  -- Array of user IDs who can access
    acl_groups TEXT[], -- Array of group IDs who can access
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for parents
CREATE INDEX IF NOT EXISTS idx_parents_type ON parents(parent_type);
CREATE INDEX IF NOT EXISTS idx_parents_source ON parents(source_file);
CREATE INDEX IF NOT EXISTS idx_parents_file_name ON parents(file_name);
CREATE INDEX IF NOT EXISTS idx_parents_acl_users ON parents USING GIN(acl_users);
CREATE INDEX IF NOT EXISTS idx_parents_created ON parents(created_at DESC);


-- ============================================
-- FULL-TEXT SEARCH (FALLBACK/METADATA ONLY)
-- ============================================
-- NOTE: This TSV column is NOT used for core retrieval.
-- 
-- Usage:
--   - Exact phrase matching ("Project Alpha")
--   - Metadata filtering by keywords
--   - Fallback when BM25 index is rebuilding
--
-- Core retrieval uses bm25_index table (higher quality BM25 ranking)
-- ============================================

ALTER TABLE parents ADD COLUMN IF NOT EXISTS tsv TSVECTOR;
CREATE INDEX IF NOT EXISTS idx_parents_fts ON parents USING GIN(tsv);

-- Trigger to update tsv on insert/update
CREATE OR REPLACE FUNCTION parents_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.parent_text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER parents_tsv_update 
    BEFORE INSERT OR UPDATE ON parents
    FOR EACH ROW EXECUTE FUNCTION parents_tsv_trigger();


-- ============================================
-- CHILDREN TABLE
-- Stores child chunks (for retrieval)
-- ============================================
CREATE TABLE IF NOT EXISTS children (
    -- Identity
    child_id VARCHAR(100) PRIMARY KEY,
    parent_id VARCHAR(100) NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
    
    -- Content
    child_text TEXT NOT NULL,
    child_type VARCHAR(20) NOT NULL CHECK (child_type IN ('text_chunk', 'table_row_group')),
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    
    -- Positioning
    chunk_index INTEGER NOT NULL,  -- Position within parent
    row_indices INTEGER[],  -- For table children: which rows from original table
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for children
CREATE INDEX IF NOT EXISTS idx_children_parent ON children(parent_id);
CREATE INDEX IF NOT EXISTS idx_children_type ON children(child_type);


-- ============================================
-- BM25 STATISTICS TABLE
-- Global statistics for BM25 scoring
-- ============================================
CREATE TABLE IF NOT EXISTS bm25_stats (
    -- Statistics
    stat_key VARCHAR(50) PRIMARY KEY,
    stat_value NUMERIC NOT NULL,
    
    -- Timestamps
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Initialize with default values
INSERT INTO bm25_stats (stat_key, stat_value) VALUES
    ('total_documents', 0),
    ('avg_doc_length', 0),
    ('total_terms', 0)
ON CONFLICT (stat_key) DO NOTHING;


-- ============================================
-- BM25 DOCUMENT FREQUENCIES TABLE
-- Stores document frequency for each term
-- ============================================
CREATE TABLE IF NOT EXISTS bm25_df (
    -- Term
    term VARCHAR(255) PRIMARY KEY,
    
    -- Document frequency (how many documents contain this term)
    df INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_bm25_df_term ON bm25_df(term);


-- ============================================
-- BM25 INDEX TABLE
-- Stores term frequencies for BM25 retrieval
-- ============================================
CREATE TABLE IF NOT EXISTS bm25_index (
    -- Identity
    child_id VARCHAR(100) PRIMARY KEY REFERENCES children(child_id) ON DELETE CASCADE,
    
    -- BM25 data
    term_frequencies JSONB NOT NULL,  -- {"term1": 5, "term2": 3, ...}
    doc_length INTEGER NOT NULL,  -- Total number of terms in document
    
    -- Metadata
    parent_id VARCHAR(100) NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for BM25
CREATE INDEX IF NOT EXISTS idx_bm25_parent ON bm25_index(parent_id);
CREATE INDEX IF NOT EXISTS idx_bm25_tf ON bm25_index USING GIN(term_frequencies);
CREATE INDEX IF NOT EXISTS idx_bm25_doc_length ON bm25_index(doc_length);


-- ============================================
-- BM25 DF UPDATE TRIGGERS
-- ============================================

-- Function to update document frequencies when BM25 index changes
CREATE OR REPLACE FUNCTION update_bm25_df()
RETURNS TRIGGER AS $$
DECLARE
    term_key TEXT;
    term_freq INTEGER;
BEGIN
    IF (TG_OP = 'INSERT') THEN
        -- Increment DF for each term in the new document
        FOR term_key, term_freq IN 
            SELECT key, value::INTEGER 
            FROM jsonb_each_text(NEW.term_frequencies)
        LOOP
            INSERT INTO bm25_df (term, df)
            VALUES (term_key, 1)
            ON CONFLICT (term) 
            DO UPDATE SET 
                df = bm25_df.df + 1,
                updated_at = NOW();
        END LOOP;
        
        RETURN NEW;
        
    ELSIF (TG_OP = 'DELETE') THEN
        -- Decrement DF for each term in the deleted document
        FOR term_key, term_freq IN 
            SELECT key, value::INTEGER 
            FROM jsonb_each_text(OLD.term_frequencies)
        LOOP
            UPDATE bm25_df 
            SET 
                df = GREATEST(df - 1, 0),  -- Don't go below 0
                updated_at = NOW()
            WHERE term = term_key;
            
            -- Optionally: Delete terms with df=0 (cleanup)
            DELETE FROM bm25_df WHERE term = term_key AND df = 0;
        END LOOP;
        
        RETURN OLD;
        
    ELSIF (TG_OP = 'UPDATE') THEN
        -- Handle update as DELETE old + INSERT new
        -- First, decrement DF for old terms
        FOR term_key, term_freq IN 
            SELECT key, value::INTEGER 
            FROM jsonb_each_text(OLD.term_frequencies)
        LOOP
            UPDATE bm25_df 
            SET 
                df = GREATEST(df - 1, 0),
                updated_at = NOW()
            WHERE term = term_key;
        END LOOP;
        
        -- Then, increment DF for new terms
        FOR term_key, term_freq IN 
            SELECT key, value::INTEGER 
            FROM jsonb_each_text(NEW.term_frequencies)
        LOOP
            INSERT INTO bm25_df (term, df)
            VALUES (term_key, 1)
            ON CONFLICT (term) 
            DO UPDATE SET 
                df = bm25_df.df + 1,
                updated_at = NOW();
        END LOOP;
        
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on bm25_index table
CREATE TRIGGER bm25_df_update
    AFTER INSERT OR UPDATE OR DELETE ON bm25_index
    FOR EACH ROW
    EXECUTE FUNCTION update_bm25_df();


-- ============================================
-- INGESTION LOG TABLE
-- Track ingestion jobs
-- ============================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    -- Identity
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Job info
    source_file VARCHAR(500) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    
    -- Statistics
    total_parents INTEGER DEFAULT 0,
    total_children INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Timestamps
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Index for tracking
CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_log(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_file ON ingestion_log(source_file);


-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get parent context by child_id
CREATE OR REPLACE FUNCTION get_parent_context(p_child_id VARCHAR)
RETURNS TABLE (
    parent_id VARCHAR,
    parent_text TEXT,
    source_file VARCHAR,
    section_header VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT p.parent_id, p.parent_text, p.source_file, p.section_header
    FROM parents p
    INNER JOIN children c ON c.parent_id = p.parent_id
    WHERE c.child_id = p_child_id;
END;
$$ LANGUAGE plpgsql;


-- Function to update BM25 statistics
CREATE OR REPLACE FUNCTION update_bm25_stats()
RETURNS void AS $$
DECLARE
    doc_count INTEGER;
    avg_length NUMERIC;
BEGIN
    -- Count total documents
    SELECT COUNT(*) INTO doc_count FROM bm25_index;
    
    -- Calculate average document length
    SELECT AVG(doc_length) INTO avg_length FROM bm25_index;
    
    -- Update stats
    UPDATE bm25_stats SET stat_value = doc_count, updated_at = NOW() WHERE stat_key = 'total_documents';
    UPDATE bm25_stats SET stat_value = COALESCE(avg_length, 0), updated_at = NOW() WHERE stat_key = 'avg_doc_length';
END;
$$ LANGUAGE plpgsql;


-- Function to get IDF for a term
CREATE OR REPLACE FUNCTION get_idf(p_term TEXT)
RETURNS NUMERIC AS $$
DECLARE
    total_docs NUMERIC;
    doc_freq NUMERIC;
    idf_score NUMERIC;
BEGIN
    -- Get total documents
    SELECT stat_value INTO total_docs 
    FROM bm25_stats 
    WHERE stat_key = 'total_documents';
    
    -- Get document frequency for term
    SELECT df INTO doc_freq 
    FROM bm25_df 
    WHERE term = p_term;
    
    -- If term not found, return 0
    IF doc_freq IS NULL OR doc_freq = 0 THEN
        RETURN 0;
    END IF;
    
    -- Calculate IDF: log((N - df + 0.5) / (df + 0.5))
    idf_score := LN((total_docs - doc_freq + 0.5) / (doc_freq + 0.5));
    
    RETURN GREATEST(idf_score, 0);  -- IDF should not be negative
END;
$$ LANGUAGE plpgsql;


-- Function to calculate BM25 score for a document
CREATE OR REPLACE FUNCTION calculate_bm25_score(
    p_child_id VARCHAR,
    p_query_terms TEXT[],
    p_k1 NUMERIC DEFAULT 1.5,
    p_b NUMERIC DEFAULT 0.75
)
RETURNS NUMERIC AS $$
DECLARE
    score NUMERIC := 0;
    term TEXT;
    tf INTEGER;
    idf NUMERIC;
    doc_len INTEGER;
    avg_len NUMERIC;
    tf_component NUMERIC;
    term_freqs JSONB;
BEGIN
    -- Get document data
    SELECT term_frequencies, doc_length 
    INTO term_freqs, doc_len
    FROM bm25_index
    WHERE child_id = p_child_id;
    
    IF term_freqs IS NULL THEN
        RETURN 0;
    END IF;
    
    -- Get average document length
    SELECT stat_value INTO avg_len
    FROM bm25_stats
    WHERE stat_key = 'avg_doc_length';
    
    IF avg_len IS NULL OR avg_len = 0 THEN
        avg_len := 1;  -- Prevent division by zero
    END IF;
    
    -- Calculate score for each query term
    FOREACH term IN ARRAY p_query_terms
    LOOP
        -- Get term frequency in document
        tf := COALESCE((term_freqs ->> term)::INTEGER, 0);
        
        IF tf > 0 THEN
            -- Get IDF for term
            idf := get_idf(term);
            
            -- BM25 formula:
            -- score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
            tf_component := (tf * (p_k1 + 1)) / 
                           (tf + p_k1 * (1 - p_b + p_b * (doc_len / avg_len)));
            
            score := score + (idf * tf_component);
        END IF;
    END LOOP;
    
    RETURN score;
END;
$$ LANGUAGE plpgsql;


-- ============================================
-- TRIGGERS
-- ============================================

-- Trigger to update BM25 stats when index changes
CREATE OR REPLACE FUNCTION bm25_stats_trigger()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_bm25_stats();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER bm25_stats_update
    AFTER INSERT OR DELETE ON bm25_index
    FOR EACH STATEMENT
    EXECUTE FUNCTION bm25_stats_trigger();


-- ============================================
-- VIEWS
-- ============================================

-- View: Document overview
CREATE OR REPLACE VIEW document_overview AS
SELECT 
    p.source_file,
    COUNT(DISTINCT p.parent_id) as total_parents,  -- p. important
    COUNT(DISTINCT c.child_id) as total_children,
    MIN(p.created_at) as ingested_at
FROM parents p
LEFT JOIN children c ON c.parent_id = p.parent_id
GROUP BY p.source_file;  --  p. important


-- View: BM25 index health
CREATE OR REPLACE VIEW bm25_health AS
SELECT 
    (SELECT stat_value FROM bm25_stats WHERE stat_key = 'total_documents') as total_docs,
    (SELECT stat_value FROM bm25_stats WHERE stat_key = 'avg_doc_length') as avg_doc_length,
    (SELECT COUNT(*) FROM bm25_df) as unique_terms,
    (SELECT MAX(updated_at) FROM bm25_stats) as last_updated;