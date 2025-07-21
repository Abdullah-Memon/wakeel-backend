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
from dataclasses import dataclass
from pathlib import Path
from ...config.config import Config

# Configuration
DOCUMENT_PATHS = {
    "constitution": r"app\models\law\data\Constitution Articles.txt",
    "ppc": r"app\models\law\data\PPC_sections.txt"
}
intro= """
آءُ سنڌي قانوني مشير آهيان جيڪو آن لائين جديد مصنوعي ذهانت جي بڻياد تي ٺھيل ھڪ چيٽ بوٽ آهي، جيڪو بنيادي قانوني معلومات سنڌي ٻوليءَ ۾ مھيا ڪرڻ لاءِ تيار ڪيو ويو آھي
.ي چيٽ بوٽ عبدالماجد ڀرڳڙي انسٽيٽيوٽ آف لينگويج انجنيئرنگ حيدرآباد، جيڪو ثقافت، سياحت، آثار قديمه ۽ آرڪائيوز کاتي، سنڌ حڪومت پاران قائم ڪيل هڪ خودمختيار ادارو آهي ۽ ايڪس فلو ريسرچ انڪارپوريٽيڊ، جيڪو سافٽ ويئر ڊيفائنڊ نيٽ ورڪنگ ۽ اوپن اسٽيڪ جي واڌاري ۾ عالمي سطح تي تسليم ٿيل ادارو آهي انهن ٻنهي جي گڏيل رٿا جو نتيجو آهي. انسٽيٽيوٽ سنڌي ٻوليءَ کي سڀني ڪمپيوٽيشنل پروگرامن ۽ قدرتي ٻوليءَ جي انجڻڪارين ۾ آڻي ان کي بين الاقوامي ٻولين جي قد تي آڻڻ لاءِ قائم ڪيو ويو آهي
"""
API_KEY = Config.MODEL_API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Clean legal document structure"""
    id: str
    type: str  # 'constitution' or 'ppc'
    number: str
    title: str
    content: str
    full_text: str
    embedding: np.ndarray = None

class DocumentLoader:
    """Efficient document loading and processing"""
    
    @staticmethod
    def load_constitution(file_path: str) -> List[LegalDocument]:
        """Load constitution articles with robust pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"Constitution file not found: {file_path}")
            return []
        
        documents = []
        
        # Multiple patterns for flexibility
        patterns = [
            r'آرٹیڪل\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=آرٹیڪل|\Z)',
            r'آرٽیڪل\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=آرٽیڪل|\Z)',
            r'Article\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=Article|\Z)',
            r'ARTICLE\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=ARTICLE|\Z)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                article_num = match.group(1).strip()
                article_content = match.group(2).strip()
                
                # Extract title (first line) and content
                lines = [line.strip() for line in article_content.split('\n') if line.strip()]
                if lines:
                    title = lines[0]
                    content_text = ' '.join(lines[1:]) if len(lines) > 1 else title
                    
                    if len(content_text) > 20:  # Filter meaningful content
                        doc = LegalDocument(
                            id=f"const_{article_num}",
                            type="constitution",
                            number=article_num,
                            title=title,
                            content=content_text,
                            full_text=f"آرٹیڪل {article_num}: {title} - {content_text}"
                        )
                        documents.append(doc)
        
        # Remove duplicates
        seen_numbers = set()
        unique_docs = []
        for doc in documents:
            if doc.number not in seen_numbers:
                seen_numbers.add(doc.number)
                unique_docs.append(doc)
        
        logger.info(f"Loaded {len(unique_docs)} constitution articles")
        return unique_docs
    
    @staticmethod
    def load_ppc(file_path: str) -> List[LegalDocument]:
        """Load PPC sections with robust pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"PPC file not found: {file_path}")
            return []
        
        documents = []
        
        patterns = [
            r'سیڪشن\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=سیڪشن|\Z)',
            r'سيڪشن\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=سيڪشن|\Z)',
            r'Section\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=Section|\Z)',
            r'SECTION\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=SECTION|\Z)',
            r'دفعہ\s*(\d+[A-Za-z]?)[:.\-\s]*(.+?)(?=دفعہ|\Z)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_num = match.group(1).strip()
                section_content = match.group(2).strip()
                
                lines = [line.strip() for line in section_content.split('\n') if line.strip()]
                if lines:
                    title = lines[0]
                    content_text = ' '.join(lines[1:]) if len(lines) > 1 else title
                    
                    if len(content_text) > 15:
                        doc = LegalDocument(
                            id=f"ppc_{section_num}",
                            type="ppc",
                            number=section_num,
                            title=title,
                            content=content_text,
                            full_text=f"سیڪشن {section_num}: {title} - {content_text}"
                        )
                        documents.append(doc)
        
        # Remove duplicates
        seen_numbers = set()
        unique_docs = []
        for doc in documents:
            if doc.number not in seen_numbers:
                seen_numbers.add(doc.number)
                unique_docs.append(doc)
        
        logger.info(f"Loaded {len(unique_docs)} PPC sections")
        return unique_docs

def get_legal_response(query: str) -> str:
    """
    Returns the legal response for a given query using the SindhiLegalAssistant singleton.
    """
    return query_sindhi_legal_assistant(query)

    
class SmartEmbedding:
    """Efficient embedding system with caching"""
    
    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_file = Path("app/models/law/cache/legal_embeddings.pkl")
        self._load_model()
        self._load_cache()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def _load_cache(self):
        """Load embedding cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if not self.model:
            return np.zeros(384)
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            embedding = self.model.encode(text)
            self.cache[cache_key] = embedding
            
            # Save cache periodically
            if len(self.cache) % 50 == 0:
                self._save_cache()
            
            return embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return np.zeros(384)
    
    def embed_documents(self, documents: List[LegalDocument]):
        """Generate embeddings for all documents"""
        for i, doc in enumerate(documents):
            doc.embedding = self.get_embedding(doc.full_text)
            if (i + 1) % 25 == 0:
                logger.info(f"Embedded {i + 1}/{len(documents)} documents")
        
        self._save_cache()

class LegalSearchEngine:
    """High-performance legal document search"""
    
    def __init__(self, documents: List[LegalDocument], embedding_system: SmartEmbedding):
        self.documents = documents
        self.embedding_system = embedding_system
        self.embeddings_matrix = None
        self._prepare_search_index()
    
    def _prepare_search_index(self):
        """Prepare search index"""
        valid_docs = [doc for doc in self.documents if doc.embedding is not None]
        if valid_docs:
            self.documents = valid_docs
            self.embeddings_matrix = np.vstack([doc.embedding for doc in valid_docs])
            logger.info(f"Search index ready: {len(valid_docs)} documents")
        else:
            logger.error("No valid documents for search index")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[LegalDocument, float]]:
        """Search for relevant legal documents"""
        if self.embeddings_matrix is None:
            return []
        
        query_embedding = self.embedding_system.get_embedding(query)
        if query_embedding.size == 0:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get top results with minimum threshold
        results = []
        for i, sim in enumerate(similarities):
            if sim > 0.1:  # Minimum similarity threshold
                results.append((self.documents[i], float(sim)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class SindhiLegalAssistant:
    """Main legal assistant focused on practical solutions"""
    
    def __init__(self, api_key: str):
        self.client = Together(api_key=api_key)
        self.embedding_system = SmartEmbedding()
        self.search_engine = None
        self.documents = []
        self._initialize()
    
    def _initialize(self):
        """Initialize the legal assistant"""
        logger.info("Initializing Sindhi Legal Assistant...")
        
        # Load documents
        constitution_docs = DocumentLoader.load_constitution(DOCUMENT_PATHS["constitution"])
        ppc_docs = DocumentLoader.load_ppc(DOCUMENT_PATHS["ppc"])
        
        self.documents = constitution_docs + ppc_docs
        
        if not self.documents:
            logger.error("No legal documents loaded!")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embedding_system.embed_documents(self.documents)
        
        # Initialize search engine
        self.search_engine = LegalSearchEngine(self.documents, self.embedding_system)
        
        logger.info(f"Legal Assistant ready with {len(self.documents)} documents")
    
    def process_query(self, query: str) -> str:
        """Process legal query and provide solution"""
        
        # Handle basic interactions
        # if self._is_greeting(query):
        #     return self._get_greeting_response()
        
        if self._is_introduction_query(query):
            return self._get_introduction()
        
        # Search for relevant legal content
        search_results = self.search_engine.search(query, top_k=3)
        
        if not search_results:
            return self._handle_no_results(query)
        
        # Generate comprehensive response
        return self._generate_legal_response(query, search_results)
    
    # def _is_greeting(self, query: str) -> bool:
    #     """Check if query is a greeting"""
    #     greetings = ['سلام', 'hello', 'hi', 'السلام', 'ھيلو']
    #     return any(greeting in query.lower() for greeting in greetings)
    
    def _is_introduction_query(self, query: str) -> bool:
        """Check if query asks for introduction"""
        intro_phrases = ['تون ڪير', 'توهان ڪير', 'who are', 'تعارف', 'introduce']
        return any(phrase in query.lower() for phrase in intro_phrases)
    
    def _get_greeting_response(self) -> str:
        """Get greeting response"""
        return """
🏛️ **سلام علیکم!**

آءُ توهان جو قانوني مشير آهيان۔ آءُ پاڪستان جي قانون بابت صحيح ۽ تفصيلي معلومات سنڌي ۾ فراهم ڪندو آهيان۔

**آءُ ڪهڙي مدد ڪري سگهان ٿو:**
• آئيني آرٽيڪل جي تشريح
• پينل ڪوڊ سيڪشن جي تفصيل  
• قانوني حق ۽ فرضن جي معلومات
• عملي قانوني مشورا

توهان جو قانوني سوال پوڇو!
"""
    
    def _get_introduction(self) -> str:
        """Get detailed introduction"""
        return """
🏛️ **آءُ سنڌي قانوني مشير آهيان جيڪو آن لائين جديد مصنوعي ذهانت جي بڻياد تي ٺھيل ھڪ چيٽ بوٽ آهي، آءَ بنيادي قانوني معلومات سنڌي ٻوليءَ ۾ مھيا ڪرڻ لاءِ تيار ڪيو ويو آھيان
. هي چيٽ بوٽ عبدالماجد ڀرڳڙي انسٽيٽيوٽ آف لينگويج انجنيئرنگ حيدرآباد، جيڪو ثقافت، سياحت، آثار قديمه ۽ آرڪائيوز کاتي، سنڌ حڪومت پاران قائم ڪيل هڪ خودمختيار ادارو آهي ۽ ايڪس فلو ريسرچ انڪارپوريٽيڊ، جيڪو سافٽ ويئر ڊيفائنڊ نيٽ ورڪنگ ۽ اوپن اسٽيڪ جي واڌاري ۾ عالمي سطح تي تسليم ٿيل ادارو آهي انهن ٻنهي جي گڏيل رٿا جو نتيجو آهي. انسٽيٽيوٽ سنڌي ٻوليءَ کي سڀني ڪمپيوٽيشنل پروگرامن ۽ قدرتي ٻوليءَ جي انجڻڪارين ۾ آڻي ان کي بين الاقوامي ٻولين جي قد تي آڻڻ لاءِ قائم ڪيو ويو آهي.**منهنجون خصوصيات**
✅ پاڪستان آئين 1973 جو مڪمل ڊيٽابيس
✅ پاڪستان پينل ڪوڊ جي تمام سيڪشنز
✅ صحيح قانوني حوالا ۽ سيڪشن نمبر
✅ آسان سنڌي ۾ پيچيده قانون جي وضاحت
✅ عملي قانوني مشورا ۽ حل

---
*عبدالماجد ڀرڳڙي انسٽيٽيوٽ ۽ ايڪس فلو ريسرچ جو گڏيل پروجيڪٽ*
"""
    
    def _handle_no_results(self, query: str) -> str:
        """Handle queries with no results"""
        return f"""
🔍 **توهان جي سوال لاءِ معذرت**

**سوال:** "{query}"

**ممڪن وجوهات:**
• سوال ۾ واضح قانوني اصطلاح نه آهي
• ڊيٽابيس ۾ لاڳاپيل معلومات موجود ناهي
• مختلف لفظن ۾ سوال پوڇڻ جي ضرورت

**بهتر نتائج لاءِ:**
✅ آرٽيڪل يا سيڪشن نمبر شامل ڪريو
✅ واضح قانوني لفظ استعمال ڪريو
✅ مثال: "آرٽيڪل 25 آزادي" بجاءِ "آزادي"

**فوري مدد:**
📞 قانوني هيلپ لائن: 1090
🏛️ نزديڪي عدالت يا وڪيل سان رابطو

مون کي ٻيو سوال پوڇو!
"""
    
    def _generate_legal_response(self, query: str, search_results: List[Tuple[LegalDocument, float]]) -> str:
        """Generate comprehensive legal response"""
        
        # Build context from search results
        context = self._build_context(search_results)
        
        # Create advanced prompt for legal response
        prompt = f"""
تون پاڪستان جو بهترين قانوني ماهر آهين جيڪو سنڌي ۾ واضح ۽ عملي قانوني مشورو ڏيندو آهي۔

صارف جو سوال: "{query}"

دستياب قانوني معلومات:
{context}

هدايتون:
1. صحيح سنڌي  گرامر ۾ جواب ڏيو
2. آرٽيڪل/سيڪشن نمبر بالڪل صحيح لکو
3. قانوني اصطلاحن جي آسان وضاحت ڪريو
4. عملي مشورو ڏيو
5. جواب کي مختصر ۽ جامع رکو
6. اگر ڪو تواھان جي باري ۾ پُڇي تہ ھي چئو{intro} ھن ۾ حوالا نہ ڏي
7. پھري سوال کي سمجھ پوءِ جواب ڏي
8. اردو نہ صرف سنڌي
9. اگر ڪو انگريزي ۾ سوال پڇي تہ ان کي سنڌي ۾ جواب ڏيو
10. صحيح آرٽيڪل آئين سيڪشن نمبر استعمال ڪريو جواب ۾
11. تعرف ۾ حوالا نہ ڏيو

جواب جي ڍانچو:
**مختصر جواب** - سوال جو سڌو جواب
** تفصيل** - لاڳاپيل قانوني دفعات
**عملي مشورو** - ڇا ڪرڻ گهرجي
**اهم نوٽ** - ضروري احتياط
** حوالا** - آرٽيڪل/سيڪشن نمبر

هاڻي مڪمل جواب ڏيو:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{
                    "role": "system",
                    "content": """تون پاڪستان جو سڀ کان ماهر قانوني صلاحڪار آهين۔ توهان جو ڪم آهي:

• صحيح قانوني معلومات فراهم ڪرڻ
• آسان سنڌي ۾ پيچيده قانون جي وضاحت
• عملي ۽ مفيد مشورا ڏيڻ
• صحيح آرٽيڪل/سيڪشن نمبر استعمال ڪرڻ

هميشه ياد رکو: صرف سنڌي ۾ جواب، صحيح حوالا، عملي مدد۔"""
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=1000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            return self._generate_fallback_response(search_results)
    
    def _build_context(self, search_results: List[Tuple[LegalDocument, float]]) -> str:
        """Build context from search results"""
        context_parts = []
        
        for i, (doc, score) in enumerate(search_results, 1):
            doc_type = "آئین پاڪستان" if doc.type == "constitution" else "پاڪستان پینل کوڊ"
            
            context_parts.append(f"""
{i}. {doc_type}
نمبر: {doc.number}
عنوان: {doc.title}
تفصيل: {doc.content}
مطابقت: {score*100:.1f}%
---""")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, search_results: List[Tuple[LegalDocument, float]]) -> str:
        """Generate fallback response when LLM fails"""
        if not search_results:
            return "معاف ڪريو، تڪنيڪي خرابي جي ڪري جواب تيار نه ٿي سگھيو۔"
        
        # Use first result
        doc, score = search_results[0]
        doc_type = "آئین پاڪستان" if doc.type == "constitution" else "پاڪستان پینل کوڊ"
        
        return f"""
**📖 قانوني معلومات**

**ذريعو:** {doc_type}
**نمبر:** {doc.number}  
**عنوان:** {doc.title}

**تفصيل:**
{doc.content}

**مطابقت:** {score*100:.1f}%

**نوٽ:** تفصيلي قانوني مشوري لاءِ قانوني ماهر سان رابطو ڪريو۔

**فوري مدد:** 📞 1090
"""

# Main function for easy integration
def query_sindhi_legal_assistant(question: str) -> str:
    """
    Main function to query the Sindhi Legal Assistant
    
    Args:
        question (str): Legal question in Sindhi or English
        
    Returns:
        str: Comprehensive legal guidance in Sindhi
    """
    
    # Initialize assistant (singleton pattern)
    if not hasattr(query_sindhi_legal_assistant, '_assistant'):
        logger.info("Initializing Sindhi Legal Assistant...")
        query_sindhi_legal_assistant._assistant = SindhiLegalAssistant(API_KEY)
        
        if not query_sindhi_legal_assistant._assistant.documents:
            return """
❌ **قانوني ڊيٽابيس لوڊ نه ٿي سگھيو**

براہ کرم:
1. دستاويز فائلن جي path چيڪ ڪريو
2. Constitution Articles.txt موجود آهي؟
3. PPC_sections.txt موجود آهي؟

فوري مدد لاءِ: 📞 1090
"""
        
        logger.info("Sindhi Legal Assistant ready!")
    
    try:
        return query_sindhi_legal_assistant._assistant.process_query(question)
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return f"""
⚠️ **تڪنيڪي خرابي**

معاف ڪريو، توهان جو سوال پروسيس ڪرڻ ۾ مسئلو آهي۔

**توهان جو سوال:** "{question}"

**متبادل:**
📞 قانوني هيلپ لائن: 1090
🏛️ نزديڪي عدالت ۾ رابطو
💼 قانوني ماهر سان صلاح

ٿورو دير بعد ٻيهر ڪوشش ڪريو۔
"""

# Example usage and testing
if __name__ == "__main__":
    
    # Test cases covering different legal scenarios
    test_cases = [
        "سلام",  # Greeting
        "تون ڪير آهين؟",  # Introduction
        "آرٽيڪل 19 ڇا آهي؟",  # Constitutional law
        # "چوري جي سزا ڇا آهي؟",  # Criminal law
        # "سيڪشن 302 قتل",  # Specific PPC section
        # "بنيادي حق ڪهڙا آهن؟",  # Fundamental rights
        # "گرفتاري ڪيئن ٿيندي آهي؟",  # Procedural law
        # "ملڪيت جو حق",  # Property rights
        # "منهنجي زمين تي ڪو قبضو ڪيو آهي، ڇا ڪندس؟",  # Practical problem
        "مُکي منھجو مُڙس مکي ماري ٿو، ڇا ڪجي؟",  # Domestic violence
        'ڇا آءَ پيھنجي پاڻ کي ماري سگھان ٿو؟',
    ]
    
    print("Testing Sindhi Legal Assistant")
    print("=" * 60)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n Test {i}: {question}")
        print("-" * 40)
        
        try:
            answer = query_sindhi_legal_assistant(question)
            print(f"Answer:\n{answer}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 60)