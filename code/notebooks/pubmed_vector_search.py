import os
import logging
import json
import dotenv
import psycopg2
import xmltodict
import numpy as np
import openai
from typing import List, Dict, Any
import pdb

# Load environment variables
dotenv.load_dotenv()
openai.api_key = ""
client = openai.OpenAI()
# OpenAI and Database Configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'localhost')
}

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_dir (str): Directory to store log files
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('pubmed_vector_search')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'pubmed_vector_search.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Function to save list of dictionaries to a file
def save_dict_list(filename, data):
    """
    Save a list of dictionaries to a JSON file.
    
    :param filename: Name of the file to save
    :param data: List of dictionaries to save
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {filename}")

# Function to read list of dictionaries from a file
def read_dict_list(filename):
    """
    Read a list of dictionaries from a JSON file.
    
    :param filename: Name of the file to read
    :return: List of dictionaries
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def parse_pubmed_xml(file_path: str, logger: logging.Logger = None, parsed_filename = '../../data/parsed_articles.json') -> List[Dict[str, Any]]:
    """
    Parse PubMed XML and extract comprehensive article metadata
    
    Args:
        file_path (str): Path to PubMed XML file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: List of dictionaries containing article metadata
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logging()
    
    # Log start of parsing
    
    print("Parsed filename:", parsed_filename)
    print(os.path.exists(parsed_filename))
    if os.path.exists(parsed_filename):
        logger.info(f"Reading from parsed file: {parsed_filename}")
        processed_articles = read_dict_list(parsed_filename)
        logger.info(f"Total articles found: {len(processed_articles)}")
        processed_articles = processed_articles[0:10]
        logger.info(f"Total articles subset: {len(processed_articles)}")
        return processed_articles
    return
    logger.info(f"Starting to parse XML file: {file_path}")
    
    try:
        # Read XML file
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        
        # Log file size
        logger.info(f"XML file size: {len(xml_content)} bytes")
        
        # Parse XML
        parsed_data = xmltodict.parse(xml_content)
        articles = parsed_data.get('PubmedArticleSet', {}).get('PubmedArticle', [])
        
        # Ensure articles is a list
        if not isinstance(articles, list):
            articles = [articles]
        
        logger.info(f"Total articles found: {len(articles)}")
        
        processed_articles = []
        parsing_errors = 0
        
        for idx, article in enumerate(articles, 1):
            try:
                # Extract Medline Citation
                medline_citation = article.get('MedlineCitation', {})
                article_data = medline_citation.get('Article', {})
                
                # PMID
                pmid = medline_citation.get('PMID', {}).get('#text', 'Unknown')
                
                # Title
                title = article_data.get('ArticleTitle', 'No Title')
                if isinstance(title, dict):
                    title = title.get('#text', 'No Title')
                
                # Abstract
                abstract = article_data.get('Abstract', {})
                abstract_text = ''
                if abstract:
                    # Handle multiple AbstractText elements
                    if isinstance(abstract.get('AbstractText'), list):
                        abstract_text = ' '.join([
                            text.get('#text', '') if isinstance(text, dict) else text
                            for text in abstract.get('AbstractText', [])
                        ])
                    elif isinstance(abstract.get('AbstractText'), dict):
                        abstract_text = abstract.get('AbstractText', {}).get('#text', '')
                    elif isinstance(abstract.get('AbstractText'), str):
                        abstract_text = abstract.get('AbstractText', '')
                
                # Authors
                authors = article_data.get('AuthorList', {}).get('Author', [])
                if not isinstance(authors, list):
                    authors = [authors]
                
                author_names = []
                for author in authors:
                    if isinstance(author, dict):
                        # Construct full name
                        last_name = author.get('LastName', '')
                        fore_name = author.get('ForeName', '')
                        initials = author.get('Initials', '')
                        
                        # Prefer full name, fall back to alternatives
                        full_name = f"{last_name} {fore_name}".strip() or \
                                    f"{last_name} {initials}".strip() or \
                                    last_name
                        
                        author_names.append(full_name)
                
                # Journal Information
                journal = article_data.get('Journal', {})
                journal_title = journal.get('Title', 'Unknown Journal')
                
                # Publication Date
                pub_date = journal.get('JournalIssue', {}).get('PubDate', {})
                publication_year = pub_date.get('Year', 'Unknown')
                publication_month = pub_date.get('Month', '')
                
                # MeSH Headings
                mesh_headings = medline_citation.get('MeshHeadingList', {}).get('MeshHeading', [])
                if not isinstance(mesh_headings, list):
                    mesh_headings = [mesh_headings]
                
                mesh_terms = [
                    heading.get('DescriptorName', {}).get('#text', '')
                    for heading in mesh_headings
                    if isinstance(heading, dict)
                ]
                
                article_entry = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract_text,
                    'authors': ', '.join(author_names[:3]),  # Limit to first 3 authors
                    'journal': journal_title,
                    'publication_year': publication_year,
                    'publication_month': publication_month,
                    'mesh_terms': ', '.join(mesh_terms)
                }
                
                processed_articles.append(article_entry)
                
                # Log successful parsing of each article
                logger.info(f"Successfully parsed article {idx}: PMID {pmid}")
            
            except Exception as e:
                parsing_errors += 1
                logger.error(f"Error processing article {idx}: {e}")
                logger.debug(f"Problematic article data: {article}")
        
        # Log overall parsing summary
        logger.info(f"Parsing complete. Total articles processed: {len(processed_articles)}")
        if parsing_errors > 0:
            logger.warning(f"Number of articles with parsing errors: {parsing_errors}")

        save_dict_list(parsed_filename, processed_articles)
        return processed_articles
    
    except Exception as e:
        logger.error(f"Critical error parsing XML file: {e}")
        return []

def generate_embedding(text: str, model: str = 'text-embedding-ada-002', logger: logging.Logger = None) -> List[float]:
    """
    Generate vector embedding for given text using OpenAI API
    
    Args:
        text (str): Input text to embed
        model (str): OpenAI embedding model to use
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: Vector embedding
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logging()
    
    try:
        # Remove newlines and extra whitespace
        text = text.replace('\n', ' ').strip()
        
        # Truncate text to max token limit (8191 tokens)
        text = text[:8000]
        
        # Log embedding generation
        logger.info(f"Generating embedding for text (length: {len(text)})")
        
        response = openai.embeddings.create(
            input=[text],
            model=model
        )
        
        embedding = response.data[0].embedding
        
        # Log successful embedding generation
        logger.info(f"Successfully generated embedding")
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def create_vector_table(conn, logger: logging.Logger = None):
    """
    Create table for storing PubMed article vectors
    
    Args:
        conn (psycopg2.connection): Database connection
        logger (logging.Logger, optional): Logger instance
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logging()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pubmed_vectors (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    journal TEXT,
                    publication_year TEXT,
                    publication_month TEXT,
                    mesh_terms TEXT,
                    embedding vector(1536)
                )
            """)
        conn.commit()
        logger.info("Vector table created successfully")
    except Exception as e:
        logger.error(f"Error creating vector table: {e}")
        conn.rollback()

def ingest_pubmed_vectors(conn, articles: List[Dict[str, Any]], logger: logging.Logger = None):
    """
    Ingest articles with their vector embeddings into database
    
    Args:
        conn (psycopg2.connection): Database connection
        articles (list): List of articles to ingest
        logger (logging.Logger, optional): Logger instance
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logging()

    logger.info("Loading data into the table..")
    
    try:
        with conn.cursor() as cur:
            embeddings_generated = 0
            embeddings_failed = 0
            
            for i, article in enumerate(articles):
                logger.info("loading article # " + str(i))
                try:
                    # Combine text for embedding
                    text_for_embedding = (
                        f"{article['title']} {article['abstract']} "
                        f"Keywords: {article['mesh_terms']}"
                    )
                    
                    # Generate embedding
                    embedding = generate_embedding(text_for_embedding, logger=logger)
                    if embedding:
                        cur.execute("""
                            INSERT INTO pubmed_vectors 
                            (pmid, title, abstract, authors, journal, 
                             publication_year, publication_month, mesh_terms, embedding) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (pmid) DO UPDATE 
                            SET title = EXCLUDED.title,
                                abstract = EXCLUDED.abstract,
                                authors = EXCLUDED.authors,
                                journal = EXCLUDED.journal,
                                publication_year = EXCLUDED.publication_year,
                                publication_month = EXCLUDED.publication_month,
                                mesh_terms = EXCLUDED.mesh_terms,
                                embedding = EXCLUDED.embedding
                        """, (
                            article['pmid'], 
                            article['title'], 
                            article['abstract'], 
                            article['authors'],
                            article['journal'],
                            article['publication_year'],
                            article['publication_month'],
                            article['mesh_terms'],
                            embedding
                        ))
                        embeddings_generated += 1
                    else:
                        embeddings_failed += 1
                
                except Exception as e:
                    logger.error(f"Error ingesting article {article.get('pmid')}: {e}")
                    logger.exception({e})
                    embeddings_failed += 1
            
            conn.commit()
            
            # Log ingestion summary
            logger.info(f"Ingestion complete. "
                        f"Embeddings generated: {embeddings_generated}, "
                        f"Embeddings failed: {embeddings_failed}")
    
    except Exception as e:
        logger.error(f"Critical error during vector ingestion: {e}")
        conn.rollback()

def semantic_search(conn, query: str, top_n: int = 5, logger: logging.Logger = None):
    """
    Perform semantic search on PubMed vectors
    
    Args:
        conn (psycopg2.connection): Database connection
        query (str): Search query
        top_n (int): Number of top results to return
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: Top similar articles
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logging()
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query, logger=logger)
        
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    pmid, 
                    title, 
                    abstract, 
                    authors,
                    journal,
                    publication_year,
                    publication_month,
                    mesh_terms,
                    1 - (embedding <=> %s::vector) as similarity
                FROM pubmed_vectors
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding, top_n))
            
            results = cur.fetchall()
        
        # Log search results
        logger.info(f"Semantic search completed. Found {len(results)} results")
        
        # Format results
        formatted_results = [
            {
                'pmid': result[0],
                'title': result[1],
                'abstract': result[2],
                'authors': result[3],
                'journal': result[4],
                'publication_year': result[5],
                'publication_month': result[6],
                'mesh_terms': result[7],
                'similarity_score': result[8]
            }
            for result in results
        ]
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        return []

def main(pubmed_xml_path):
    # Set up logging
    logger = setup_logging()
    
    try:
        # Establish database connection
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Path to PubMed daily update XML
        #pubmed_xml_path = os.getenv('PUBMED_XML_PATH')
        
        # Create vector table
        create_vector_table(conn, logger)
        # Parse XML and get articles
        articles = parse_pubmed_xml(pubmed_xml_path, logger)
        # Ingest articles with embeddings
        ingest_pubmed_vectors(conn, articles, logger)
        #return conn
        ## Interactive search
        #while True:
        query = input("\nEnter your search query (or 'exit' to quit): ").strip()
        #    
        #if query.lower() == 'exit':
        #    break
        #    
        # Perform semantic search
        results = semantic_search(conn, query, logger=logger)
        #    
        # Display results
        print("\n--- Top Matching Articles ---")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity Score: {result['similarity_score']:.2f}")
            print(f"   Title: {result['title']}")
            print(f"   Authors: {result['authors']}")
            print(f"   Journal: {result['journal']} ({result['publication_year']})")
            print(f"   PMID: {result['pmid']}")
            print(f"   Abstract: {result['abstract'][:300]}...")
            print(f"   Keywords: {result['mesh_terms']}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main()
