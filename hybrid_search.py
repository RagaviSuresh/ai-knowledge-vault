import re
import json
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from .vector_store import VectorStore
from .embedding_engine import EmbeddingEngine, get_sentence_transformer
from .hnsw_index import cosine_similarity
from .document_representation import DocumentRepresentation, ChunkRepresentation

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

TAXONOMY = {
    "categories": {
        "Career": {
            "description": "Professional development, job applications, curriculum vitae, hiring, employment history, work experience, cover letters.",
            "keywords": ["resume", "cv", "experience", "career", "employment", "internship", "projects", "skills", "contact", "education"]
        },
        "Academic Records": {
            "description": "Official records of student performance, transcript, grade card, mark sheet, academic credits, results, examinations.",
            "keywords": ["marksheet", "grade", "gpa", "cgpa", "marks", "semester", "transcript", "results", "credits", "register number", "roll number"]
        },
        "AI/ML": {
            "description": "Artificial intelligence, machine learning, neural networks, deep learning, NLP, computer vision, reinforcement learning, transformers, LLMs, pytorch, tensorflow, model training.",
            "keywords": ["machine learning", "deep learning", "neural network", "artificial intelligence", "nlp", "llm", "transformers", "pytorch", "tensorflow", "model", "classification", "regression", "weights", "epochs"]
        },
        "Data Science": {
            "description": "Data analysis, statistical modeling, data visualization, datasets, analysis scripts, pandas, matplotlib, numpy, data processing.",
            "keywords": ["data science", "statistics", "data analysis", "pandas", "numpy", "matplotlib", "plot", "scatter", "distribution", "mean", "median", "variance"]
        },
        "Programming": {
            "description": "Software development, code syntax, algorithms, data structures, programming paradigms, databases, SQL queries, system networks.",
            "keywords": ["programming", "coding", "software", "database", "sql", "java", "python", "html", "css", "javascript", "code", "function", "variable", "class"]
        },
        "Civil Engineering": {
            "description": "Structural engineering, soil mechanics, concrete materials, geotechnical surveys, retaining walls, earth pressure, hydrology, foundation designs.",
            "keywords": ["civil engineering", "soil", "concrete", "geotechnical", "foundation", "retaining", "wall", "walls", "pressure", "cohesive", "stress", "sand", "clay", "structural"]
        },
        "Computer Science": {
            "description": "Computer hardware architectures, GPU programming, CUDA cores, parallel computing systems, memory hierarchies, processor threads, CPU, streaming multiprocessors.",
            "keywords": ["gpu", "cuda", "parallel", "architecture", "shared memory", "multiprocessor", "threads", "cores", "opencl", "host", "device", "registers", "cache"]
        },
        "Electronics": {
            "description": "Analog and digital electronics, electrical circuits, microcontrollers, signal processing, hardware designs, electrical engineering, microchips.",
            "keywords": ["circuit", "voltage", "current", "signal", "electronics", "analog", "digital", "microcontroller", "hardware", "frequency", "capacitor", "resistor"]
        },
        "Mathematics": {
            "description": "Math concepts, calculus, probability, linear algebra, formulas, equations, proofs, matrices, vector mathematics, percentages.",
            "keywords": ["math", "mathematics", "probability", "calculus", "matrix", "vector", "percentage", "equation", "formula", "proof", "solve", "algebra"]
        },
        "Others": {
            "description": "General topics, miscellaneous contents, personal files, other documents.",
            "keywords": []
        }
    },
    "types": {
        "Resume": {
            "description": "Curriculum vitae, resume, listing contact info, skills, education, work experience, projects, internships, for job application.",
            "keywords": ["resume", "cv", "curriculum vitae", "work experience", "skills", "projects", "education", "experience", "contact"]
        },
        "Statement of Purpose": {
            "description": "Statement of Purpose, SOP, personal statement, or admission essay detailing a student's research interests, university application, career goals, and academic projects in narrative form.",
            "keywords": ["statement of purpose", "sop", "personal statement", "admission essay", "letter of intent", "career goals", "motivation", "academic background", "aspire", "interest"]
        },
        "Research Paper": {
            "description": "Academic research paper, journal publication, conference proceedings, scientific article with abstract, methodology, results, discussion, and references.",
            "keywords": ["abstract", "introduction", "methodology", "results", "conclusion", "references", "citations", "proceedings", "journal", "arxiv", "doi"]
        },
        "Project Report": {
            "description": "Technical project report, development report, documenting system architecture, software implementation, design, and project outcomes.",
            "keywords": ["project report", "system architecture", "implementation", "design", "objectives", "requirements", "flowchart", "scope", "development"]
        },
        "Book": {
            "description": "Published book, textbook chapter, with ISBN, publisher credits, table of contents, syllabus, index.",
            "keywords": ["isbn", "publisher", "table of contents", "index", "chapter", "textbook", "volume", "edition"]
        },
        "Lab Manual": {
            "description": "Laboratory instructions manual, practical record guide, documenting experiments, aims, apparatus, procedures, observations, and program outputs.",
            "keywords": ["aim", "algorithm", "procedure", "experiment", "lab manual", "lab record", "observation", "apparatus", "manual", "record"]
        },
        "Notes": {
            "description": "Class notes, lecture summaries, revision guides, study material, cheat sheets, explaining concepts.",
            "keywords": ["lecture notes", "study guide", "revision", "cheat sheet", "notes", "concepts", "summary", "unit", "syllabus"]
        },
        "Tutorial": {
            "description": "Step-by-step guide, programming walkthrough, instructional tutorial, how-to guide.",
            "keywords": ["tutorial", "step-by-step", "walkthrough", "guide", "how-to", "instructions", "setup"]
        },
        "Assignment": {
            "description": "Student homework submission, coursework assignment, solving problem sets, submitted by student.",
            "keywords": ["assignment", "homework", "coursework", "problem set", "submitted by", "submitted to", "roll no"]
        },
        "Question Bank": {
            "description": "Collection of exam questions, repeated university questions, test bank, question bank divided by units or marks.",
            "keywords": ["question bank", "questions", "marks", "exam", "test", "repeated", "unit", "syllabus", "2 marks", "16 marks"]
        },
        "Answer Key": {
            "description": "Question answers, model answers, test solutions, exam key, answer sheet, test key, examination answers, assessment keys, question-and-answer keys.",
            "keywords": ["key", "answers", "solutions", "exam key", "correct option", "solving", "solution key", "correct", "option", "solution", "assessment", "test"]
        },
        "Certificate": {
            "description": "Certificate of completion, credential award, appreciation certificate, presenting completion of course or internship.",
            "keywords": ["certificate", "certify", "completion", "completed", "presented to", "awarded to", "appreciation"]
        },
        "Marksheet": {
            "description": "Academic transcript, semester mark sheet, grade sheet, showing subjects, marks, grades, cgpa, gpa, register number.",
            "keywords": ["marksheet", "marks", "grades", "cgpa", "gpa", "semester results", "transcript", "grade card", "grade sheet", "register number"]
        },
        "Reference Material": {
            "description": "Standard reference sheet, formula compilation, scientific table, charts, data references.",
            "keywords": ["reference sheet", "formula sheet", "formulae", "charts", "constants", "conversion", "reference table"]
        },
        "Presentation Slides": {
            "description": "Lecture slides, presentation deck, pptx slides, structured with slide titles, bulleted points, diagrams, and summaries.",
            "keywords": ["slide", "presentation", "bullet points", "slides", "overview", "introduction", "summary"]
        },
        "Workshop Brochure": {
            "description": "Flyer or brochure for workshop, seminar, course details, registration fees, speakers, contact address.",
            "keywords": ["workshop", "registration", "brochure", "flyer", "seminar", "speakers", "venue", "fees", "contact", "organized by"]
        }
    }
}

_taxonomy_embeddings = None

def get_taxonomy_embeddings():
    """Initializes and returns lazy loaded taxonomy descriptions embeddings."""
    global _taxonomy_embeddings
    if _taxonomy_embeddings is None:
        model = get_sentence_transformer()
        _taxonomy_embeddings = {
            "categories": {cat: model.encode(data["description"]).tolist() for cat, data in TAXONOMY["categories"].items()},
            "types": {dtype: model.encode(data["description"]).tolist() for dtype, data in TAXONOMY["types"].items()}
        }
    return _taxonomy_embeddings

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

@lru_cache(maxsize=65536)
def is_fuzzy_match(token: str, word: str) -> bool:
    """Check if token matches word exactly or fuzzy matches with Levenshtein distance <= 2 or shared prefix >= 4."""
    if token in word or word in token:
        return True
    if len(token) >= 5 and len(word) >= 5 and token[:4] == word[:4]:
        return True
    if abs(len(token) - len(word)) <= 2:
        dist = calculate_levenshtein_distance(token, word)
        if dist <= 2:
            return True
    return False

def parse_explicit_filters(query: str) -> Tuple[str, Dict[str, List[str]]]:
    """Parses explicit metadata filters such as type:resume, category:ai/ml, tag:etc."""
    filters = {
        "type": [],
        "category": [],
        "subject": [],
        "tag": []
    }
    
    clean_query = query
    for field in ["type", "category", "subject", "tag"]:
        # Match quoted values type:"Research Paper"
        pattern = r'\b' + re.escape(field) + r'\s*:\s*(?:"([^"]+)"|\'([^\']+)\')'
        matches = re.findall(pattern, clean_query, re.IGNORECASE)
        for val in matches:
            v = val[0] or val[1]
            filters[field].append(v.strip().lower())
            full_match = r'\b' + re.escape(field) + r'\s*:\s*(?:"' + re.escape(v) + r'"|\'' + re.escape(v) + r'\')'
            clean_query = re.sub(full_match, '', clean_query, flags=re.IGNORECASE)
            
        all_terms = []
        if field in ["category", "subject"]:
            all_terms = list(TAXONOMY["categories"].keys())
        elif field == "type":
            all_terms = list(TAXONOMY["types"].keys())
            
        for term in all_terms:
            term_pattern = r'\b' + re.escape(field) + r'\s*:\s*' + re.escape(term) + r'\b'
            if re.search(term_pattern, clean_query, re.IGNORECASE):
                filters[field].append(term.lower())
                clean_query = re.sub(term_pattern, '', clean_query, flags=re.IGNORECASE)
                
        single_word_pattern = r'\b' + re.escape(field) + r'\s*:\s*([a-zA-Z0-9_-]+)'
        matches_single = re.findall(single_word_pattern, clean_query, re.IGNORECASE)
        for val in matches_single:
            filters[field].append(val.strip().lower())
            clean_query = re.sub(r'\b' + re.escape(field) + r'\s*:\s*' + re.escape(val) + r'\b', '', clean_query, flags=re.IGNORECASE)
            
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    return clean_query, filters

def filter_documents_by_metadata(doc_list: List[DocumentRepresentation], filters: Dict[str, List[str]]) -> List[DocumentRepresentation]:
    """Filters document list against parsed explicit metadata filters."""
    filtered_docs = []
    for doc in doc_list:
        match_all = True
        
        if filters["type"]:
            doc_type = (doc.document_type or "").lower().strip()
            if not any(t in doc_type or doc_type in t for t in filters["type"]):
                match_all = False
                
        cat_filters = filters["category"] + filters["subject"]
        if cat_filters and match_all:
            doc_cat = (doc.category or "").lower().strip()
            try:
                struct = json.loads(doc.document_structure or "{}")
            except Exception:
                struct = {}
            doc_subj = struct.get("subject_area", "").lower().strip()
            
            if not any(c in doc_cat or doc_cat in c or c in doc_subj or doc_subj in c for c in cat_filters):
                match_all = False
                
        if filters["tag"] and match_all:
            doc_tags = [t.lower().strip() for t in (doc.ai_keywords or "").split(",")] if doc.ai_keywords else []
            extra_tags = []
            if doc.skills:
                try: extra_tags.extend([s.lower().strip() for s in json.loads(doc.skills)])
                except Exception: pass
            if doc.technologies:
                try: extra_tags.extend([t.lower().strip() for t in json.loads(doc.technologies)])
                except Exception: pass
            if doc.topics:
                try: extra_tags.extend([t.lower().strip() for t in json.loads(doc.topics)])
                except Exception: pass
                
            all_doc_tags = set(doc_tags + extra_tags)
            if not any(tf in all_doc_tags or any(tf in tag or tag in tf for tag in all_doc_tags) for tf in filters["tag"]):
                match_all = False
                
        if match_all:
            filtered_docs.append(doc)
            
    return filtered_docs

def detect_query_category_and_type(query_text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Detects taxonomic category and type implicitly using keyword overlap and description embeddings."""
    query_lower = query_text.lower().strip()
    
    target_cat = None
    target_type = None
    target_topics = []
    
    # 1. Parse explicit metadata syntax inside detect (if any remains)
    type_matches = re.findall(r'\btype:\s*["\']?([^"\']+)["\']?', query_lower)
    if type_matches:
        raw_type = type_matches[0].strip()
        for dtype in TAXONOMY["types"]:
            if raw_type == dtype.lower() or raw_type in [kw.lower() for kw in TAXONOMY["types"][dtype]["keywords"]]:
                target_type = dtype
                break
        if not target_type:
            target_type = raw_type.title()
            
    cat_matches = re.findall(r'\b(?:category|subject):\s*["\']?([^"\']+)["\']?', query_lower)
    if cat_matches:
        raw_cat = cat_matches[0].strip()
        for cat in TAXONOMY["categories"]:
            if raw_cat == cat.lower() or raw_cat in [kw.lower() for kw in TAXONOMY["categories"][cat]["keywords"]]:
                target_cat = cat
                break
        if not target_cat:
            target_cat = raw_cat.title()
            
    tag_matches = re.findall(r'\btag:\s*["\']?([^"\']+)["\']?', query_lower)
    if tag_matches:
        target_topics.extend([t.strip() for t in tag_matches])
        
    clean_query = re.sub(r'\b(?:type|category|subject|tag):\s*["\']?[^"\']+["\']?', '', query_lower).strip()
    
    # 2. Implicit Keyword matching
    if not target_type:
        type_scores = {}
        for dtype, data in TAXONOMY["types"].items():
            kws = data["keywords"]
            matches = sum(1 for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', clean_query))
            if matches > 0:
                type_scores[dtype] = matches
        if type_scores:
            target_type = max(type_scores, key=type_scores.get)
            
    if not target_cat:
        cat_scores = {}
        for cat, data in TAXONOMY["categories"].items():
            kws = data["keywords"]
            matches = sum(1 for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', clean_query))
            if matches > 0:
                cat_scores[cat] = matches
        if cat_scores:
            target_cat = max(cat_scores, key=cat_scores.get)
            
    if any(w in clean_query for w in ["ai", "ml", "machine learning", "deep learning", "nlp", "artificial intelligence", "agent", "agents", "llm", "llms", "transformer", "transformers", "data science"]):
        target_topics.append("ai")
    if any(w in clean_query for w in ["soil", "mechanics", "geotechnical", "civil", "foundation", "earth pressure", "retaining wall"]):
        target_topics.append("civil")
        
    # 3. Fallback to embedding cosine similarities
    if not target_cat and not target_type:
        try:
            model = get_sentence_transformer()
            q_emb = model.encode(clean_query).tolist()
            tax_embeddings = get_taxonomy_embeddings()
            
            cat_sims = {cat: cosine_similarity(q_emb, tax_embeddings["categories"][cat]) for cat in TAXONOMY["categories"]}
            best_cat = max(cat_sims, key=cat_sims.get)
            if cat_sims[best_cat] >= 0.55:
                target_cat = best_cat
                
            type_sims = {dtype: cosine_similarity(q_emb, tax_embeddings["types"][dtype]) for dtype in TAXONOMY["types"]}
            best_type = max(type_sims, key=type_sims.get)
            if type_sims[best_type] >= 0.55:
                target_type = best_type
        except Exception as e:
            print(f"HybridSearch: Error in embedding similarity map: {e}")
            
    return target_cat, target_type, target_topics

def detect_query_intent(query: str) -> Optional[str]:
    """Detects intended document type from query words."""
    query_lower = query.lower().strip()
    mapping = {
        "question bank": "Question Bank",
        "question-bank": "Question Bank",
        "exam papers": "Question Bank",
        "question papers": "Question Bank",
        "test paper": "Question Bank",
        "research paper": "Research Paper",
        "project report": "Project Report",
        "class notes": "Notes",
        "lecture notes": "Notes",
        "study guide": "Notes",
        "revision notes": "Notes",
        "lab manual": "Lab Manual",
        "lab record": "Lab Manual",
        "statement of purpose": "Statement of Purpose",
        "presentation slides": "Presentation Slides",
        "workshop brochure": "Workshop Brochure",
        "reference material": "Reference Material",
        "answer key": "Answer Key",
        "resume": "Resume",
        "cv": "Resume",
        "curriculum vitae": "Resume",
        "certificate": "Certificate",
        "certifications": "Certificate",
        "notes": "Notes",
        "project": "Project Report",
        "book": "Book",
        "textbook": "Book",
        "paper": "Research Paper",
        "journal": "Research Paper",
        "manuscript": "Research Paper",
        "lab": "Lab Manual",
        "assignment": "Assignment",
        "homework": "Assignment",
        "coursework": "Assignment",
        "marksheet": "Marksheet",
    }
    for kw, doc_type in mapping.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
            return doc_type
    return None

def extract_query_subject(query: str, detected_type: str | None) -> List[str]:
    """Extracts subject keywords by stripping out stop words and document type verbs."""
    query_lower = query.lower().strip()
    type_kws = []
    mapping = {
        "question bank": "Question Bank",
        "question-bank": "Question Bank",
        "exam papers": "Question Bank",
        "question papers": "Question Bank",
        "test paper": "Question Bank",
        "research paper": "Research Paper",
        "project report": "Project Report",
        "class notes": "Notes",
        "lecture notes": "Notes",
        "study guide": "Notes",
        "revision notes": "Notes",
        "lab manual": "Lab Manual",
        "lab record": "Lab Manual",
        "statement of purpose": "Statement of Purpose",
        "presentation slides": "Presentation Slides",
        "workshop brochure": "Workshop Brochure",
        "reference material": "Reference Material",
        "answer key": "Answer Key",
        "resume": "Resume",
        "cv": "Resume",
        "curriculum vitae": "Resume",
        "certificate": "Certificate",
        "certifications": "Certificate",
        "notes": "Notes",
        "project": "Project Report",
        "book": "Book",
        "textbook": "Book",
        "paper": "Research Paper",
        "journal": "Research Paper",
        "manuscript": "Research Paper",
        "lab": "Lab Manual",
        "assignment": "Assignment",
        "homework": "Assignment",
        "coursework": "Assignment",
        "marksheet": "Marksheet",
    }
    if detected_type:
        for kw, dtype in mapping.items():
            if dtype == detected_type:
                type_kws.extend(kw.split())
                
    ignored_words = {
        "show", "open", "display", "view", "tell", "summarize", "find", "search", "get", "retrieve", "list",
        "the", "a", "an", "my", "file", "document", "pdf", "please", "any", "explain", "describe",
        "question", "answer", "documents", "selected", "what", "how", "why", "who", "which", "compare",
        "difference", "differences", "similarity", "similarities", "common", "both", "between", "across",
        "first", "second", "third", "fourth", "fifth", "one", "two", "three", "four", "five", "former", "latter",
        "previous", "next", "above", "below", "last", "files", "here", "there", "on", "of", "in", "for", "with", 
        "and", "or", "to", "at", "by", "about"
    }
    
    words = re.sub(r'[^a-z0-9\s]', ' ', query_lower).split()
    subject_words = []
    for w in words:
        if w not in ignored_words and w not in type_kws and len(w) > 1:
            subject_words.append(w)
    return subject_words

def check_document_type_match(doc: DocumentRepresentation, target_type: str) -> bool:
    """Checks if the document matches target document type by fields, filename, or keywords."""
    if not target_type:
        return False
    if doc.document_type and doc.document_type.lower() == target_type.lower():
        return True
    filename_lower = (doc.filename or "").lower().replace('_', ' ').replace('-', ' ')
    type_kws = {
        "Question Bank": ["question bank", "qb", "question paper", "exam paper"],
        "Resume": ["resume", "cv", "curriculum vitae"],
        "Certificate": ["certificate", "completion cert", "credential"],
        "Notes": ["notes", "lecture notes", "class notes", "revision"],
        "Project Report": ["project report", "project", "report"],
        "Book": ["book", "textbook"],
        "Research Paper": ["research paper", "paper", "journal", "arxiv"],
        "Lab Manual": ["lab manual", "lab record", "practical"],
        "Assignment": ["assignment", "homework", "coursework"],
        "Marksheet": ["marksheet", "grade card", "grade sheet", "transcript"],
    }
    kws = type_kws.get(target_type, [target_type.lower()])
    for kw in kws:
        if re.search(r'\b' + re.escape(kw) + r'\b', filename_lower):
            return True
    ai_kws_lower = (doc.ai_keywords or "").lower()
    if target_type.lower() in ai_kws_lower:
        return True
    return False

def check_topic_match(doc: DocumentRepresentation, subject_words: List[str]) -> bool:
    """Checks if subject words are present in doc metadata, json lists, or content prefix."""
    if not subject_words:
        return False
    metadata_text = []
    if doc.filename:
        metadata_text.append(doc.filename.lower())
    if doc.category:
        metadata_text.append(doc.category.lower())
    if doc.document_type:
        metadata_text.append(doc.document_type.lower())
    if doc.ai_keywords:
        metadata_text.append(doc.ai_keywords.lower())
    
    for field in [doc.topics, doc.keywords, doc.skills, doc.technologies]:
        if field:
            try:
                data = json.loads(field)
                if isinstance(data, list):
                    metadata_text.extend([str(item).lower() for item in data])
                elif isinstance(data, dict):
                    metadata_text.extend([str(val).lower() for val in data.values()])
            except:
                metadata_text.append(str(field).lower())
                
    combined_metadata = " ".join(metadata_text)
    extracted_text_lower = (doc.extracted_text or "").lower()
    
    for word in subject_words:
        words_to_try = [word]
        if word.endswith('s') and len(word) > 3:
            words_to_try.append(word[:-1])
        elif len(word) > 2:
            words_to_try.append(word + 's')
            
        for w in words_to_try:
            pattern = r'\b' + re.escape(w) + r'\b'
            if re.search(pattern, combined_metadata):
                return True
            # Scan top 5000 chars of extracted text
            if re.search(pattern, extracted_text_lower[:5000]):
                return True
    return False

def extract_headings_from_text(text: str) -> List[str]:
    """Helper to extract top heading candidate lines from text."""
    if not text:
        return []
    headings = []
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:100]:
        if 3 < len(line) < 60 and not line.endswith('.') and any(c.isupper() for c in line):
            if line.isupper() or line.istitle() or re.match(r'^\d+(\.\d+)*\s+[A-Z]', line) or line.startswith(('CHAPTER', 'Section', 'Unit', 'Module')):
                headings.append(line.lower())
    return headings

def calculate_exact_keyword_bonus(doc: DocumentRepresentation, query_clean_tokens: List[str]) -> float:
    """Computes exact keyword bonus for search matching doc structure or headers."""
    if not query_clean_tokens:
        return 0.0
    fields_to_check = []
    struct = {}
    if doc.document_structure:
        try:
            struct = json.loads(doc.document_structure)
        except:
            pass
    title = struct.get("title", "")
    if title:
        fields_to_check.append(title.lower())
    if doc.filename:
        fields_to_check.append(doc.filename.lower())
    if doc.ai_keywords:
        fields_to_check.append(doc.ai_keywords.lower())
        
    for field in [doc.keywords, doc.topics, doc.skills, doc.technologies]:
        if field:
            try:
                data = json.loads(field)
                if isinstance(data, list):
                    fields_to_check.extend([str(item).lower() for item in data])
                elif isinstance(data, dict):
                    fields_to_check.extend([str(val).lower() for val in data.values()])
            except:
                fields_to_check.append(str(field).lower())
                
    dynamic_headings = extract_headings_from_text(doc.extracted_text)
    fields_to_check.extend(dynamic_headings)
    combined_fields = " | ".join(fields_to_check)
    
    bonus = 0.0
    for token in query_clean_tokens:
        if re.search(r'\b' + re.escape(token) + r'\b', combined_fields):
            bonus += 5.0
    clean_query_phrase = " ".join(query_clean_tokens)
    if len(query_clean_tokens) > 1 and clean_query_phrase in combined_fields:
        bonus += 15.0
    return bonus

class HybridSearch:
    """
    HybridSearch coordinates HNSW semantic search results with
    metadata queries, taxonomy scoring, and custom relevance bonuses.
    """
    def __init__(self, vector_store: VectorStore, embedding_engine: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    def search(
        self,
        query: str,
        doc_list: List[DocumentRepresentation],
        top_n_chunks: int = 6
    ) -> List[Dict[str, Any]]:
        # 1. Parse explicit metadata filters
        clean_query, filters = parse_explicit_filters(query)
        
        # 2. Filter documents list
        filtered_docs = filter_documents_by_metadata(doc_list, filters)
        if not filtered_docs:
            return []
            
        # 3. Document pre-ranking if too many documents
        if len(filtered_docs) > 6:
            query_lower = query.lower().strip()
            query_tokens = [w for w in re.sub(r'[^a-z0-9\s]', ' ', query_lower).split() if len(w) > 2]
            scored_docs = []
            for doc in filtered_docs:
                score = 0.0
                fname = (doc.filename or "").lower()
                for t in query_tokens:
                    if t in fname:
                        score += 100.0
                cat = (doc.category or "").lower()
                dtype = (doc.document_type or "").lower()
                for t in query_tokens:
                    if t in cat:
                        score += 50.0
                    if t in dtype:
                        score += 50.0
                text = (doc.extracted_text or "").lower()
                for t in query_tokens:
                    if t in text:
                        score += 10.0 + min(5, text.count(t)) * 2.0
                scored_docs.append((score, doc))
            scored_docs.sort(key=lambda x: -x[0])
            filtered_docs = [d for s, d in scored_docs[:6]]

        # Keep set of filtered doc IDs for faster lookup
        allowed_doc_ids = {d.doc_id for d in filtered_docs}
        
        # 4. Clean tokens for keyword matches
        query_lower = clean_query.lower().strip()
        ignored_words = {
            "show", "open", "display", "view", "tell", "summarize", "find", "search", "get", "retrieve", "list",
            "the", "a", "an", "my", "file", "document", "pdf", "please", "any", "explain", "describe",
            "question", "answer", "documents", "selected", "what", "how", "why", "who", "which", "compare",
            "difference", "differences", "similarity", "similarities", "common", "both", "between", "across",
            "first", "second", "third", "fourth", "fifth", "one", "two", "three", "four", "five", "former", "latter",
            "previous", "next", "above", "below", "last", "files", "here", "there"
        }
        query_clean_tokens = [
            w for w in re.sub(r'[^a-z0-9\s]', ' ', query_lower).split() 
            if w not in ignored_words and w not in STOP_WORDS and len(w) > 2
        ]
        
        # 5. Implicit metadata extraction from clean query
        target_cat, target_type, target_topics = detect_query_category_and_type(clean_query)
        intent_doc_type = detect_query_intent(clean_query)
        subject_words = extract_query_subject(clean_query, intent_doc_type)
        if intent_doc_type:
            target_type = intent_doc_type

        # 6. Generate Query Embedding
        query_vector = None
        try:
            query_vector = self.embedding_engine.get_query_embedding(clean_query)
        except Exception as e:
            print(f"HybridSearch: Failed to generate query embedding: {e}")

        # 7. Index verification and batch extraction
        # Ensure all chunks of these filtered documents are inside the VectorStore
        # Note: If VectorStore doesn't have the document representation or chunks, we index them!
        for doc in filtered_docs:
            if doc.doc_id not in self.vector_store.documents:
                # Need embeddings
                chunk_texts = [c.text for c in doc.chunks]
                embeddings = self.embedding_engine.get_batch_embeddings(chunk_texts)
                self.vector_store.add_document(doc, embeddings)

        # 8. Query HNSW Index (retrieve candidate hits)
        # Search for a larger pool of candidates than top_n_chunks to allow reranker to work properly
        search_pool_size = max(50, top_n_chunks * 5)
        hits = []
        if query_vector:
            hits = self.vector_store.search(query_vector, k=search_pool_size)

        # Filter hits to include ONLY chunks belonging to allowed filtered documents
        filtered_hits = [h for h in hits if h["doc"].doc_id in allowed_doc_ids]
        
        # If hits are empty, fallback to searching all chunks in memory flatly
        if not filtered_hits:
            # Re-generate search from active filtered documents in flat mode
            flat_hits = []
            for doc in filtered_docs:
                for chunk in doc.chunks:
                    # Calculate similarity manually
                    chunk_emb = self.embedding_engine.cache.get(chunk.text)
                    sim = 0.0
                    if chunk_emb and query_vector:
                        sim = max(0.0, cosine_similarity(query_vector, chunk_emb))
                    flat_hits.append({
                        "chunk": chunk,
                        "score": sim,
                        "doc": doc
                    })
            filtered_hits = flat_hits

        # 9. Precompute Document Metadata and Boosts
        doc_metadata_cache = {}
        for doc in filtered_docs:
            doc_metadata_words = set()
            if doc.category:
                doc_metadata_words.add(doc.category.lower())
            if doc.document_type:
                doc_metadata_words.add(doc.document_type.lower())
            if doc.ai_keywords:
                for t in doc.ai_keywords.split(','):
                    doc_metadata_words.add(t.strip().lower())
            try:
                struct = json.loads(doc.document_structure or "{}")
            except Exception:
                struct = {}
            doc_concepts = struct.get("important_concepts", [])
            for c in doc_concepts:
                doc_metadata_words.add(c.lower())
            doc_subject = struct.get("subject_area", "").lower()
            if doc_subject:
                doc_metadata_words.add(doc_subject)
                
            for field in [doc.keywords, doc.topics, doc.skills, doc.technologies]:
                if field:
                    try:
                        for kw in json.loads(field):
                            doc_metadata_words.add(kw.lower())
                    except: pass
                    
            # A. Taxonomic category match score
            cat_match = 0.0
            if target_cat:
                if (doc.category and doc.category.lower() == target_cat.lower()) or doc_subject == target_cat.lower():
                    cat_match = 1.0
                    
            # B. Taxonomic document type match score
            type_match = 0.0
            if target_type:
                if check_document_type_match(doc, target_type):
                    type_match = 1.0
                    
            # C. Keyword/Concept overlap with metadata
            metadata_keyword_overlap = 0.0
            if query_clean_tokens:
                matches = 0
                for token in query_clean_tokens:
                    if any(token in m or m in token for m in doc_metadata_words):
                        matches += 1
                metadata_keyword_overlap = matches / len(query_clean_tokens)
                
            weights = []
            scores = []
            if target_cat:
                weights.append(0.3)
                scores.append(cat_match)
            if target_type:
                weights.append(0.3)
                scores.append(type_match)
            weights.append(0.4 if (target_cat or target_type) else 1.0)
            scores.append(metadata_keyword_overlap)
            
            total_w = sum(weights)
            metadata_score = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
            
            # Intent Boosting
            intent_bonus = 0.0
            if intent_doc_type:
                type_matches_intent = check_document_type_match(doc, intent_doc_type)
                if type_matches_intent:
                    intent_bonus += 35.0
                    if subject_words and check_topic_match(doc, subject_words):
                        intent_bonus += 25.0
                if subject_words and not check_topic_match(doc, subject_words):
                    intent_bonus -= 50.0
                    
            # Exact Keyword Bonus
            exact_keyword_bonus = calculate_exact_keyword_bonus(doc, query_clean_tokens)
            
            doc_metadata_cache[doc.doc_id] = (metadata_score, intent_bonus, exact_keyword_bonus)

        # 10. Combine scores for retrieved chunks
        scored_chunks = []
        for hit in filtered_hits:
            chunk = hit["chunk"]
            doc = hit["doc"]
            
            # A. Chunk-level semantic similarity (70%)
            chunk_sim = hit["score"]
            
            # B. Chunk-level keyword overlap (10%)
            chunk_overlap = 0.0
            if query_clean_tokens:
                chunk_text_lower = chunk.text.lower()
                chunk_words = set(w for w in re.sub(r'[^a-z0-9\s]', ' ', chunk_text_lower).split() if len(w) > 2)
                matches = 0
                for t in query_clean_tokens:
                    if any(is_fuzzy_match(t, w) for w in chunk_words):
                        matches += 1
                chunk_overlap = matches / len(query_clean_tokens)
                
            metadata_score, intent_bonus, exact_keyword_bonus = doc_metadata_cache[doc.doc_id]
            
            base_hybrid = (0.7 * chunk_sim + 0.2 * metadata_score + 0.1 * chunk_overlap) * 100.0
            hybrid_score = base_hybrid + intent_bonus + exact_keyword_bonus
            
            scored_chunks.append({
                "chunk": chunk,
                "score": hybrid_score,
                "semantic_sim": chunk_sim,
                "metadata_score": metadata_score,
                "keyword_overlap": chunk_overlap,
                "semantic_score_exposed": chunk_sim * 100.0,
                "metadata_score_exposed": metadata_score * 100.0,
                "keyword_score_exposed": (chunk_overlap * 100.0) + exact_keyword_bonus,
                "intent_bonus": intent_bonus,
                "exact_keyword_bonus": exact_keyword_bonus,
                "doc": doc
            })
            
        return scored_chunks
