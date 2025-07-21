from together import Together
import json
import re
import os
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import threading
from ...config.config import Config

# Configuration
DOCUMENT_PATHS = {
    "constitution": r"app\models\law\data\Constitution Articles.txt",
    "ppc": r"app\models\law\data\PPC_sections.txt"
}

API_KEY = Config.MODEL_API_KEY

LEGAL_GLOSSARY = {
    # Constitutional Terms
    "آرٹيڪل": "Article",
    "آئين": "Constitution",
    "آزاديءَ جي اظهار": "Freedom of Expression",
    "منصفاڻو ٽرائل": "Fair Trial",
    "حريم جو تحفظ": "Inviolability of Home",
    "ملڪيت جو تحفظ": "Protection of Property",
    "اقليت": "Minority",
    "حقوق": "Rights",
    "عدليه": "Judiciary",
    "مقننه": "Legislature",
    "شھري": "Citizen",

    # Pakistan Penal Code (PPC) Terms
    "سيڪشن": "Section",
    "جرم": "Crime",
    "سزا": "Punishment",
    "قتل": "Murder",
    "چوري": "Theft",
    "جبري گھسن": "Criminal Trespass",
    "توهين رسالت": "Blasphemy",
    "نقصان جي نيت سان گھسن": "House-Trespass with Intent to Harm",
    "بدنامي": "Defamation",
    "بغاوت": "Sedition",
    "دھشتگردي": "Terrorism",

    # Code of Criminal Procedure (CrPC) Terms
    "گرفتاري": "Arrest",
    "وارنٽ": "Warrant",
    "ضمانت": "Bail",
    "تحقيق": "Investigation",
    "عدالتي حڪم": "Court Order",
    "قابل دستياب جرم": "Cognizable Offense",
    "غير قابل دستياب جرم": "Non-Cognizable Offense",
    "ايف آءِ آر": "FIR (First Information Report)",
    "چالان": "Challan",
    "شڪايت": "Complaint",

    # Police and Procedural Terms
    "پوليس": "Police",
    "عدالت": "Court",
    "جج": "Judge",
    "وڪيل": "Lawyer",
    "مقدمو": "Case",
    "سماع": "Hearing",
    "فيصلو": "Judgment",
    "اپيل": "Appeal",
    "رٽ": "Writ",

    # General Legal Terms
    "قانون": "Law",
    "انصاف": "Justice",
    "وضاحت": "Explanation",
    "استثنا": "Exception",
    "نيڪ نيتي": "Good Faith",
    "عوامي مفاد": "Public Interest",
    "تفسير": "Interpretation",
    "اختيار": "Authority",
    "جواز": "Justification",
    "حدون": "Limits"
}

INTRODUCTION_TEXT = """آءُ ھڪ آن لائین مشیني وڪیل آھیان، جدید مصنوعي ذهانت جي بڻیاد تي ٺھیل چیٹ بوٹ، جیڪو بنیادي قانوني معلومات سنڌي ٻولیءَ ۾ مھیا ڪرڻ لاءِ تیار ڪیو ویو آهي.

هي چیٹ بوٹ عبدالماجد ڀرڳڙي انسٹیٹیوٹ آف لینگویج انجنیئرنگ حیدرآباد، جیڪو ثقافت، سیاحت، آثار قدیمه ۽ آرکائیوز کاتي، سنڌ حڪومت پاران قائم ڪیل هڪ خودمختیار ادارو آهي ۽ ایڪس فلو ریسرچ انڪارپوریٹیڈ، جیڪو سافٹ ویئر ڈیفائنڈ نیٹ ورڪنگ ۽ اوپن اسٹیڪ جي واڌاري ۾ عالمي سطح تي تسلیم ٿیل ادارو آهي انهن ٻنهي جي گڏیل رٿا جو نتیجو آهي."""

# Cache configuration
CACHE_DIR = Path("app/models/law/cache")
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl"
ARTICLES_CACHE = CACHE_DIR / "articles_cache.pkl" 
SECTIONS_CACHE = CACHE_DIR / "sections_cache.pkl"
MODEL_CACHE = CACHE_DIR / "model_cache.pkl"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalSection:
    """Structured representation of legal section"""
    section_id: str
    document_type: str  # 'constitution' or 'ppc'
    section_number: str
    title: str
    content: str
    chapter: str = ""
    part: str = ""
    keywords: List[str] = None
    embedding: np.ndarray = None
    content_hash: str = ""
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.content_hash == "":
            self.content_hash = hashlib.md5(f"{self.title}{self.content}".encode()).hexdigest()

class DocumentProcessor:
    """Production-grade document processor with robust extraction"""
    
    @staticmethod
    def extract_constitution_articles(file_path: str) -> List[LegalSection]:
        """Extract constitution articles with enhanced pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read constitution file: {e}")
            return []
        
        articles = []
        lines = content.split('\n')
        
        # Enhanced patterns for constitution articles
        article_patterns = [
            r'^\s*آرٹیکل\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*آرٽیڪل\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*Article\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*ARTICLE\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*(\d+[A-Za-z]?)[:：.\-]\s*(.+)$'  # Simple number pattern
        ]
        
        current_part = ""
        current_chapter = ""
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Track structural elements
            if 'حصو' in line or 'PART' in line.upper():
                current_part = line
                i += 1
                continue
            elif 'باب' in line or 'CHAPTER' in line.upper():
                current_chapter = line
                i += 1
                continue
            
            # Try to match article patterns
            for pattern in article_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    article_num = match.group(1).strip()
                    title = match.group(2).strip()
                    
                    # Extract article content
                    content_lines = []
                    j = i + 1
                    
                    # Look ahead for content until next article or end
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        
                        # Stop if we hit another article
                        if any(re.match(p, next_line, re.IGNORECASE) for p in article_patterns):
                            break
                        
                        # Stop at structural elements
                        if 'حصو' in next_line or 'PART' in next_line.upper() or \
                           'باب' in next_line or 'CHAPTER' in next_line.upper():
                            break
                        
                        content_lines.append(next_line)
                        j += 1
                        
                        # Limit content extraction
                        if len(content_lines) > 20:
                            break
                    
                    article_content = ' '.join(content_lines).strip()
                    
                    # Only include if we have substantial content
                    if len(article_content) > 30:
                        article = LegalSection(
                            section_id=f"const_{article_num}",
                            document_type="constitution",
                            section_number=article_num,
                            title=title,
                            content=article_content,
                            chapter=current_chapter,
                            part=current_part,
                            keywords=DocumentProcessor._extract_keywords(f"{title} {article_content}")
                        )
                        articles.append(article)
                    
                    i = j - 1  # Continue from where we left off
                    break
            
            i += 1
        
        logger.info(f"Extracted {len(articles)} constitution articles")
        return articles
    
    @staticmethod
    def extract_ppc_sections(file_path: str) -> List[LegalSection]:
        """Extract PPC sections with enhanced pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read PPC file: {e}")
            return []
        
        sections = []
        lines = content.split('\n')
        
        # Enhanced patterns for PPC sections
        section_patterns = [
            r'^\s*سیڪشن\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*سيڪشن\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*Section\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*SECTION\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*دفعہ\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\s*دفع\s*(\d+[A-Za-z]?)[:：.\-\s]*(.+)$',
            r'^\*\*Section\s*(\d+[A-Za-z]?)\*\*[:：.\-\s]*(.+)$'
        ]
        
        current_chapter = ""
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Track chapters
            if ('باب' in line and any(char.isdigit() for char in line)) or \
               ('CHAPTER' in line.upper() and any(char.isdigit() for char in line)):
                current_chapter = line
                i += 1
                continue
            
            # Try to match section patterns
            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    section_num = match.group(1).strip()
                    title = match.group(2).strip()
                    
                    # Extract section content
                    content_lines = []
                    j = i + 1
                    
                    # Look ahead for content
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        
                        # Stop if we hit another section
                        if any(re.match(p, next_line, re.IGNORECASE) for p in section_patterns):
                            break
                        
                        # Stop at chapter boundaries
                        if ('باب' in next_line and any(char.isdigit() for char in next_line)) or \
                           ('CHAPTER' in next_line.upper() and any(char.isdigit() for char in next_line)):
                            break
                        
                        content_lines.append(next_line)
                        j += 1
                        
                        # Limit content extraction
                        if len(content_lines) > 15:
                            break
                    
                    section_content = ' '.join(content_lines).strip()
                    
                    # Include even shorter content for PPC sections
                    if len(section_content) > 15 or len(title) > 10:
                        section = LegalSection(
                            section_id=f"ppc_{section_num}",
                            document_type="ppc",
                            section_number=section_num,
                            title=title,
                            content=section_content if section_content else title,
                            chapter=current_chapter,
                            keywords=DocumentProcessor._extract_keywords(f"{title} {section_content}")
                        )
                        sections.append(section)
                    
                    i = j - 1
                    break
            
            i += 1
        
        logger.info(f"Extracted {len(sections)} PPC sections")
        return sections
    
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'کي', 'جي', 'جو', 'کان', '۾', 'تي', 'سان', 'the', 'and', 'or', 'is', 'of', 'to', 'in', 'for'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))

class EmbeddingEngine:
    """Production-grade embedding engine with persistent caching"""
    
    def __init__(self):
        self.model = None
        self.model_name = 'distiluse-base-multilingual-cased-v2'
        self.embedding_dim = None
        self.embeddings_cache = {}
        self.cache_version = "v2.0"
        self.initialize_model()
        self.load_cache()
    
    def initialize_model(self):
        """Initialize embedding model with proper error handling"""
        try:
            if EMBEDDINGS_CACHE.exists():
                try:
                    with open(EMBEDDINGS_CACHE, 'rb') as f:
                        cache_data = pickle.load(f)
                    if cache_data.get('version') != self.cache_version:
                        EMBEDDINGS_CACHE.unlink()
                        logger.info("Cleared outdated embeddings cache")
                except:
                    EMBEDDINGS_CACHE.unlink()
                    logger.info("Cleared corrupted embeddings cache")
            
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model ready: {self.embedding_dim}D embeddings")
            
            with open(MODEL_CACHE, 'wb') as f:
                pickle.dump({
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim,
                    'version': self.cache_version,
                    'timestamp': datetime.now()
                }, f)
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model = None
            self.embedding_dim = 384
    
    def load_cache(self):
        """Load cached embeddings with validation"""
        if not EMBEDDINGS_CACHE.exists() or not self.embedding_dim:
            return
        
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
            
            if (cache_data.get('version') == self.cache_version and 
                cache_data.get('embedding_dim') == self.embedding_dim):
                self.embeddings_cache = cache_data.get('embeddings', {})
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            else:
                logger.info("Cache version mismatch, starting fresh")
                
        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            self.embeddings_cache = {}
    
    def save_cache(self):
        """Save embeddings cache with metadata"""
        if not self.embeddings_cache:
            return
        
        try:
            cache_data = {
                'version': self.cache_version,
                'embedding_dim': self.embedding_dim,
                'model_name': self.model_name,
                'timestamp': datetime.now(),
                'embeddings': self.embeddings_cache
            }
            
            with open(EMBEDDINGS_CACHE, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Cache saving failed: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching and validation"""
        if not self.model or not text.strip():
            return np.zeros(self.embedding_dim or 384)
        
        cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        
        if cache_key in self.embeddings_cache:
            cached = self.embeddings_cache[cache_key]
            if cached.shape[0] == self.embedding_dim:
                return cached
            else:
                del self.embeddings_cache[cache_key]
        
        try:
            embedding = self.model.encode(text)
            
            if embedding.shape[0] == self.embedding_dim:
                self.embeddings_cache[cache_key] = embedding
                
                if len(self.embeddings_cache) % 100 == 0:
                    self.save_cache()
                
                return embedding
            else:
                logger.warning(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.embedding_dim}")
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_sections(self, sections: List[LegalSection]) -> List[LegalSection]:
        """Generate embeddings for all sections"""
        embedded_sections = []
        total = len(sections)
        
        for i, section in enumerate(sections):
            searchable_text = f"{section.title} {section.content}"
            embedding = self.get_embedding(searchable_text)
            section.embedding = embedding
            embedded_sections.append(section)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Embedded {i + 1}/{total} sections")
        
        self.save_cache()
        return embedded_sections

class EnhancedSemanticSearchEngine:
    """Production-grade semantic search with accurate relevance scoring"""
    
    def __init__(self, sections: List[LegalSection], embedding_engine: EmbeddingEngine):
        self.sections = sections
        self.embedding_engine = embedding_engine
        self.section_embeddings = None
        self.keyword_index = self._build_keyword_index()
        self._prepare_search_infrastructure()
    
    def _build_keyword_index(self) -> Dict[str, List[int]]:
        """Build inverted index for keyword-based filtering"""
        keyword_index = {}
        
        for idx, section in enumerate(self.sections):
            text = f"{section.title} {section.content}".lower()
            words = re.findall(r'\b\w+\b', text)
            
            for word in words:
                if len(word) > 2:
                    if word not in keyword_index:
                        keyword_index[word] = []
                    keyword_index[word].append(idx)
        
        return keyword_index
    
    def _prepare_search_infrastructure(self):
        """Prepare embedding matrix with validation"""
        embeddings = []
        valid_sections = []
        
        for section in self.sections:
            if section.embedding is not None and section.embedding.size > 0:
                embeddings.append(section.embedding)
                valid_sections.append(section)
            else:
                text = f"{section.title} {section.content}"
                embedding = self.embedding_engine.get_embedding(text)
                if embedding.size > 0:
                    section.embedding = embedding
                    embeddings.append(embedding)
                    valid_sections.append(section)
        
        self.sections = valid_sections
        self.section_embeddings = np.vstack(embeddings) if embeddings else np.array([])
        
        logger.info(f"Search engine ready: {len(self.sections)} sections indexed")
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.15) -> List[Tuple[LegalSection, float]]:
        """Multi-stage search with accurate relevance scoring"""
        if self.section_embeddings.size == 0:
            return []
        
        candidate_indices = self._get_keyword_candidates(query)
        query_embedding = self.embedding_engine.get_embedding(query)
        if query_embedding.size == 0:
            return []
        
        if candidate_indices:
            candidate_embeddings = self.section_embeddings[candidate_indices]
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
            similarity_results = [(candidate_indices[i], similarities[i]) for i in range(len(similarities))]
        else:
            similarities = cosine_similarity([query_embedding], self.section_embeddings)[0]
            similarity_results = [(i, similarities[i]) for i in range(len(similarities))]
        
        enhanced_results = []
        for idx, sim_score in similarity_results:
            if sim_score >= min_similarity:
                section = self.sections[idx]
                enhanced_score = self._calculate_enhanced_relevance(query, section, sim_score)
                enhanced_results.append((section, enhanced_score))
        
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:top_k]
    
    def _get_keyword_candidates(self, query: str) -> List[int]:
        """Get candidate sections using keyword matching"""
        query_words = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]
        candidate_sets = []
        
        for word in query_words:
            if word in self.keyword_index:
                candidate_sets.append(set(self.keyword_index[word]))
        
        if not candidate_sets:
            return []
        
        candidates = set()
        for candidate_set in candidate_sets:
            candidates.update(candidate_set)
        
        return list(candidates)
    
    def _calculate_enhanced_relevance(self, query: str, section: LegalSection, base_similarity: float) -> float:
        """Calculate enhanced relevance score with multiple factors"""
        query_lower = query.lower()
        section_text = f"{section.title} {section.content}".lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        section_words = set(re.findall(r'\b\w+\b', section_text))
        
        score = base_similarity
        
        if query_words and section_words:
            overlap_ratio = len(query_words & section_words) / len(query_words)
            score += overlap_ratio * 0.3
        
        title_words = set(re.findall(r'\b\w+\b', section.title.lower()))
        if query_words and title_words:
            title_overlap = len(query_words & title_words) / len(query_words)
            score += title_overlap * 0.2
        
        for word in query_words:
            if word in section_text:
                score += 0.05
        
        query_numbers = re.findall(r'\d+', query)
        if query_numbers and any(num in section.section_number for num in query_numbers):
            score += 0.4
        
        if section.document_type == "constitution" and any(word in query_lower for word in ['آرٹیکل', 'article', 'حق', 'آئین']):
            score += 0.1
        
        if len(section.content) > 100:
            score += 0.05
        
        return min(score, 1.0)

class CacheManager:
    """Manages persistent caching for articles and sections"""
    
    @staticmethod
    def save_sections(sections: List[LegalSection], cache_file: Path):
        """Save sections to cache"""
        try:
            cache_data = {
                'version': '2.0',
                'timestamp': datetime.now(),
                'sections': [asdict(section) for section in sections]
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Cache saving failed for {cache_file}: {e}")
    
    @staticmethod
    def load_sections(cache_file: Path, file_path: str) -> Optional[List[LegalSection]]:
        """Load sections from cache if valid"""
        if not cache_file.exists():
            return None
        
        try:
            source_mtime = os.path.getmtime(file_path)
            cache_mtime = os.path.getmtime(cache_file)
            
            if source_mtime > cache_mtime:
                logger.info(f"Source file updated, invalidating cache for {cache_file.name}")
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('version') != '2.0':
                logger.info(f"Cache version mismatch for {cache_file.name}")
                return None
            
            sections = []
            for section_data in cache_data.get('sections', []):
                embedding = section_data.pop('embedding', None)
                section = LegalSection(**section_data)
                if embedding is not None:
                    section.embedding = np.array(embedding)
                sections.append(section)
            
            logger.info(f"Loaded {len(sections)} sections from cache: {cache_file.name}")
            return sections
            
        except Exception as e:
            logger.warning(f"Cache loading failed for {cache_file}: {e}")
            return None

class LegalAssistant:
    """Clean legal assistant with function-based interface"""
    
    def __init__(self, api_key: str):
        self.client = Together(api_key=api_key)
        self.embedding_engine = EmbeddingEngine()
        self.search_engine = None
        self.all_sections = []
        self.last_search_results = []
        
        self._initialize_documents()
    
    def _initialize_documents(self):
        """Initialize documents with enhanced extraction"""
        logger.info("Initializing legal document database...")
        
        constitution_sections = self._load_or_extract_constitution()
        ppc_sections = self._load_or_extract_ppc()
        
        self.all_sections = constitution_sections + ppc_sections
        
        if not self.all_sections:
            logger.error("No legal sections loaded")
            return
        
        self.search_engine = EnhancedSemanticSearchEngine(self.all_sections, self.embedding_engine)
        
        total_const = len(constitution_sections)
        total_ppc = len(ppc_sections)
        logger.info(f"Database ready: {total_const} constitution articles, {total_ppc} PPC sections")
    
    def _load_or_extract_constitution(self) -> List[LegalSection]:
        """Load constitution with enhanced extraction patterns"""
        cached = CacheManager.load_sections(ARTICLES_CACHE, DOCUMENT_PATHS["constitution"])
        if cached:
            return cached
        
        logger.info("Processing constitution...")
        sections = DocumentProcessor.extract_constitution_articles(DOCUMENT_PATHS["constitution"])
        if sections:
            sections = self.embedding_engine.embed_sections(sections)
            CacheManager.save_sections(sections, ARTICLES_CACHE)
        return sections
    
    def _load_or_extract_ppc(self) -> List[LegalSection]:
        """Load PPC with enhanced extraction patterns"""
        cached = CacheManager.load_sections(SECTIONS_CACHE, DOCUMENT_PATHS["ppc"])
        if cached:
            return cached
        
        logger.info("Processing PPC...")
        sections = DocumentProcessor.extract_ppc_sections(DOCUMENT_PATHS["ppc"])
        if sections:
            sections = self.embedding_engine.embed_sections(sections)
            CacheManager.save_sections(sections, SECTIONS_CACHE)
        return sections
    
    def search_legal_content(self, query: str, top_k: int = 5) -> List[Tuple[LegalSection, float]]:
        """Search with result tracking"""
        if not self.search_engine:
            return []
        results = self.search_engine.search(query, top_k)
        self.last_search_results = results
        return results
    
    def _classify_query(self, query: str) -> str:
        """Enhanced query classification"""
        query_lower = query.lower()
        
        intro_patterns = ['تواھان ڪير', 'توهان ڪير', 'who are you', 'تعارف', 'ڪن ٺاهيو', 'ڪير بڻايو']
        greeting_patterns = ['سلام', 'hello', 'آداب', 'السلام', 'hi', 'ھیلو']
        legal_patterns = [
            'قانون', 'سزا', 'آرٹیکل', 'سیڪشن', 'دفعہ', 'آئین', 'حق', 'عدالت',
            'law', 'section', 'article', 'punishment', 'constitution', 'court',
            'جرم', 'ڏوھ', 'سپریم ڪورٹ', 'ہائی ڪورٹ', 'قاضي'
        ]
        
        non_legal_patterns = [
            'شاه لطيف', 'شاھ لطيف', 'ڀٽائي', 'شاعر', 'شاعري', 'ادب', 'کلام',
            'صوفي', 'روحانيت', 'فلسفو', 'تاريخ', 'ثقافت', 'سنڌي ادب',
            'poet', 'poetry', 'literature', 'culture', 'history', 'philosophy'
        ]
        
        if any(p in query_lower for p in intro_patterns):
            return "introduction"
        elif any(p in query_lower for p in greeting_patterns):
            return "greeting"
        elif any(p in query_lower for p in non_legal_patterns):
            return "non_legal"
        elif any(p in query_lower for p in legal_patterns):
            return "legal"
        else:
            return "non_legal"
    
    def _build_context(self, search_results: List[Tuple[LegalSection, float]]) -> str:
        """Build context from search results"""
        if not search_results:
            return ""
        
        context_parts = []
        for i, (section, score) in enumerate(search_results[:3], 1):
            doc_name = "آئین پاکستان" if section.document_type == "constitution" else "پاکستان پینل کوڈ"
            
            context_parts.append(f"""
{i}. {doc_name} - سیکشن {section.section_number}
عنوان: {section.title}
تفصیل: {section.content}
مطابقت: {score*100:.1f}%
""")
        
        return "\n".join(context_parts)
    
    def _generate_llm_response(self, query: str, context: str) -> str:
        """Generate complete LLM response"""
        if not context.strip():
            return "معاف ڪريو، توهان جي سوال سان لاڳاپيل ڪا معلومات نه ملي۔"
        
        prompt = f"""سوال: {query}

دستیاب قانوني معلومات:
{context}

ھدايتون:
1. تون پاڪستان پينل ڪوڊ (PPC)، ڪوڊ آف ڪرمنل پروسيجر (CrPC)، ۽ پاڪستان جي آئين (1973) جو ماهر آهين. قانوني سوالن جا جواب سادي ۽ درست سنڌي ۾ ڏي، قانوني اصطلاحن کي واضح ۽ سمجهڻ ۾ آسان انداز ۾ بيان ڪر.
2. هر جواب ۾ درست قانوني حوالا شامل ڪر (مثال طور، آئين جو آرٽيڪل، PPC يا CrPC جو سيڪشن نمبر)، ۽ حوالن جي تصديق {context} مان ڪر.
3. جواب صرف {context} ۾ موجود معلومات يا PPC، CrPC، ۽ آئين جي بنياد تي ڏي. جيڪڏهن معلومات موجود نه هجي، واضح ڪر ته "معلومات موجود ناهي" ۽ متبادل قانوني رھنمائي نه ڏي.
4. جواب ۾ صرف قانوني معلومات شامل ڪر. غير قانوني، غير متعلق، يا حساس غير قانوني موضوعن (جهڙوڪ سياست، ذاتي راءِ) کان پاسو ڪر.
5. سوال جو درست ۽ جامع جواب ڏي، پر ذاتي راءِ، قياس، يا غير ضروري تفصيل شامل نه ڪر.
6. جواب صرف سنڌي ۾ ڏي. اردو، انگريزي، يا ٻي ٻولي جا لفظ استعمال نه ڪر، پر انگريزي قانوني اصطلاحن (جهڙوڪ Section، Article) کي سنڌي ۾ ترجمو ڪر (مثال طور، سيڪشن، آرٽيڪل).
7. جيڪڏهن انگريزي دستاويزن (PPC، CrPC، آئين) مان معلومات استعمال ڪجي، ته درست سنڌي ترجمو ڪر ۽ قانوني اصطلاحن جي درستگي کي يقيني بڻائڻ لاءِ {LEGAL_GLOSSARY} استعمال ڪر.
8. صرف قانوني يا آئيني سوالن جا جواب ڏي. غير قانوني سوالن لاءِ معذرت ڪر ۽ چؤ ته "معاف ڪريو، هي سوال قانوني ناهي. مهرباني ڪري قانوني سوال پڇو."
9. جيڪڏهن سوال توهان يا اداري جي تعارف بابت هجي، ته {INTRODUCTION_TEXT} استعمال ڪر ۽ ٻيو ڪو اضافي جواب نه ڏي.
10. جواب ۾ غير جانبدار رهو، صرف قانوني حقائق ۽ تشريح پيش ڪر، ۽ حساس موضوعن (جهڙوڪ فوج، مذهب) تي غير جانبدار ۽ احترام وارو لهجو اختيار ڪر.

جواب:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{
                    "role": "system",
                    "content": "تون پاڪستاني قانون جو ماهر آہین۔ ڏنل معلومات جي بنیاد تي درست جواب ڏي۔"
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.1,
                max_tokens=600,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return "معاف ڪريو، جواب تيار ڪرڻ ۾ خرابي آئي."

# --- API functions for Flask integration ---
# Thread pool for fast embedding/search
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_embedding_lock = threading.Lock()

# Patch EmbeddingEngine to use thread pool for embedding sections
_original_embed_sections = EmbeddingEngine.embed_sections

def threaded_embed_sections(self, sections: List[LegalSection]) -> List[LegalSection]:
    # Use thread pool to parallelize embedding
    def embed_one(section):
        searchable_text = f"{section.title} {section.content}"
        with _embedding_lock:
            embedding = self.get_embedding(searchable_text)
        section.embedding = embedding
        return section
    results = list(_executor.map(embed_one, sections))
    self.save_cache()
    return results

EmbeddingEngine.embed_sections = threaded_embed_sections

# Patch EnhancedSemanticSearchEngine to use thread pool for search
_original_search = EnhancedSemanticSearchEngine.search

def threaded_search(self, query: str, top_k: int = 5, min_similarity: float = 0.15):
    # Embed query in a thread
    future = _executor.submit(self.embedding_engine.get_embedding, query)
    query_embedding = future.result()
    if self.section_embeddings.size == 0 or query_embedding.size == 0:
        return []
    candidate_indices = self._get_keyword_candidates(query)
    if candidate_indices:
        candidate_embeddings = self.section_embeddings[candidate_indices]
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        similarity_results = [(candidate_indices[i], similarities[i]) for i in range(len(similarities))]
    else:
        similarities = cosine_similarity([query_embedding], self.section_embeddings)[0]
        similarity_results = [(i, similarities[i]) for i in range(len(similarities))]
    # Parallelize enhanced relevance calculation
    def enhance(idx_sim):
        idx, sim_score = idx_sim
        if sim_score >= min_similarity:
            section = self.sections[idx]
            enhanced_score = self._calculate_enhanced_relevance(query, section, sim_score)
            return (section, enhanced_score)
        return None
    enhanced_results = list(filter(None, _executor.map(enhance, similarity_results)))
    enhanced_results.sort(key=lambda x: x[1], reverse=True)
    return enhanced_results[:top_k]

EnhancedSemanticSearchEngine.search = threaded_search

# Add timeout to LLM call
import functools
import signal

def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
            return result
        return wrapper
    return decorator

LegalAssistant._generate_llm_response = timeout(15)(LegalAssistant._generate_llm_response)

def query_legal_assistant(input_query: str) -> str:
    """
    Main function to query the legal assistant
    
    Args:
        input_query (str): The legal question in Sindhi or English
        
    Returns:
        str: The assistant's response in Sindhi
    """
    # Initialize assistant if not already done
    if not hasattr(query_legal_assistant, 'assistant'):
        logger.info("Initializing Legal Assistant...")
        query_legal_assistant.assistant = LegalAssistant(API_KEY)
        logger.info("Legal Assistant ready")
    
    assistant = query_legal_assistant.assistant
    
    # Classify and handle query
    query_type = assistant._classify_query(input_query)
    
    if query_type == "introduction":
        return INTRODUCTION_TEXT
    elif query_type == "greeting":
        return "سلام وعلیکم! قانوني مدد لاءِ پڇو۔"
    elif query_type == "non_legal":
        return "معاف ڪريو، آئون صرف قانوني سوالن جا جواب ڏیان ٿو۔ مهرباني ڪري پاڪستاني قانون يا آئین بابت پڇو۔"
    
    # Search for relevant content
    search_results = assistant.search_legal_content(input_query)
    
    if not search_results:
        return "معاف ڪريو، توهان جي سوال بابت ڪا معلومات نه ملي۔"
    
    # Build context and generate response
    context = assistant._build_context(search_results)
    
    try:
        response = assistant._generate_llm_response(input_query, context)
        return response
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        
        # Fallback response using first search result
        if search_results:
            section, score = search_results[0]
            doc_name = "آئین پاکستان" if section.document_type == "constitution" else "پاکستان پینل کوڈ"
            
            return f"""**{section.title}**

{section.content}

**حوالہ:** {doc_name} - سیڪشن {section.section_number}
"""
        else:
            return "معاف ڪريو، ڪا معلومات نه ملي۔"

# --- API functions for Flask integration ---
def initialize_assistant():
    """
    Initializes and returns the singleton LegalAssistant instance.
    """
    if not hasattr(initialize_assistant, '_assistant'):
        initialize_assistant._assistant = LegalAssistant(API_KEY)
    return initialize_assistant._assistant

def get_legal_response(query: str) -> str:
    """
    Returns the legal response for a given query using the singleton assistant.
    """
    assistant = initialize_assistant()
    return query_legal_assistant(query)

# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "آرٽيڪل 19 ڇا آھي؟",
        "بنیادي حق",
        "قتل جي سزا",
        "چوري جو قانون"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Response: {query_legal_assistant(query)}")
        print("-" * 50)