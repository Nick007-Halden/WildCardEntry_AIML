import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import os
from typing import Optional, List, Dict
import tempfile

class ResumeKeywordExtractor:
    def __init__(self):
        print("Initializing job-focused keyword extractor...")
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Installing spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        self.tesseract_config = {
            'lang': 'eng',
            'config': '--psm 1'  # Automatic page segmentation with OSD
        }

        # Enhanced job-specific keywords
        self.job_related_skills = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'C', 'C++' 'c++', 'c#', 'ruby', 'php', 'golang',
                'rust', 'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab', 'sql', 'shell',
                'objective-c', 'dart', 'lua', 'haskell', 'erlang', 'cobol', 'fortran',
                'visual basic', 'assembly', 'groovy', 'julia', 'lisp', 'prolog'
            ],
            'web_technologies': [
                'html5', 'css3', 'sass', 'less', 'webpack', 'babel', 'rest', 'graphql',
                'websocket', 'ajax', 'jquery', 'xml', 'json', 'oauth', 'api', 'webrtc',
                'pwa', 'service workers', 'web components', 'web assembly', 'web gl',
                'web sockets', 'web workers', 'sse', 'cors', 'jwt', 'oauth2', 'saml',
                'soap', 'microservices', 'grpc', 'web3', 'progressive enhancement'
            ],
            'frameworks_libraries': {
                'frontend': [
                    'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt', 'gatsby',
                    'bootstrap', 'tailwind', 'material-ui', 'chakra ui', 'jquery',
                    'ember', 'backbone', 'alpine.js', 'lit', 'preact', 'stimulus'
                ],
                'backend': [
                    'django', 'flask', 'fastapi', 'spring', 'node.js', 'express',
                    'asp.net', 'laravel', 'symfony', 'rails', 'nest.js', 'strapi',
                    'phoenix', 'fiber', 'gin', 'echo', 'koa', 'hapi', 'meteor'
                ],
                'mobile': [
                    'react native', 'flutter', 'ionic', 'xamarin', 'android sdk',
                    'ios sdk', 'swift ui', 'jetpack compose', 'kotlin multiplatform'
                ]
            },
            'databases': {
                'relational': [
                    'mysql', 'postgresql', 'oracle', 'sql server', 'sqlite', 'mariadb',
                    'db2', 'cockroachdb', 'timescaledb'
                ],
                'nosql': [
                    'mongodb', 'cassandra', 'couchdb', 'redis', 'neo4j', 'elasticsearch',
                    'dynamodb', 'firebase', 'cosmosdb', 'hbase', 'influxdb'
                ],
                'data_warehousing': [
                    'snowflake', 'redshift', 'bigquery', 'synapse', 'vertica',
                    'clickhouse', 'greenplum'
                ]
            },
            'cloud_devops': {
                'platforms': [
                    'aws', 'azure', 'gcp', 'alibaba cloud', 'oracle cloud',
                    'digitalocean', 'heroku', 'openstack', 'vmware'
                ],
                'containerization': [
                    'docker', 'kubernetes', 'openshift', 'rancher', 'podman',
                    'containerd', 'docker-compose', 'helm'
                ],
                'ci_cd': [
                    'jenkins', 'gitlab ci', 'github actions', 'travis ci', 'circle ci',
                    'teamcity', 'bamboo', 'azure pipelines', 'argocd', 'tekton'
                ],
                'infrastructure': [
                    'terraform', 'ansible', 'puppet', 'chef', 'cloudformation',
                    'pulumi', 'salt', 'vagrant', 'packer'
                ],
                'monitoring': [
                    'prometheus', 'grafana', 'datadog', 'nagios', 'zabbix',
                    'new relic', 'splunk', 'elastic stack', 'dynatrace'
                ]
            },
            'data_science_ai': {
                'machine_learning': [
                    'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'xgboost',
                    'lightgbm', 'catboost', 'rapids', 'mxnet', 'theano'
                ],
                'deep_learning': [
                    'neural networks', 'cnn', 'rnn', 'lstm', 'transformers',
                    'bert', 'gpt', 'reinforcement learning', 'gan', 'attention mechanism'
                ],
                'data_processing': [
                    'pandas', 'numpy', 'scipy', 'dask', 'spark', 'hadoop',
                    'databricks', 'airflow', 'kubeflow', 'mlflow'
                ],
                'visualization': [
                    'matplotlib', 'seaborn', 'plotly', 'bokeh', 'tableau',
                    'power bi', 'looker', 'd3.js', 'highcharts'
                ]
            },
            'security': {
                'web_security': [
                    'owasp', 'penetration testing', 'csrf', 'xss', 'sql injection',
                    'authentication', 'authorization', 'encryption', 'ssl/tls'
                ],
                'tools': [
                    'burp suite', 'wireshark', 'metasploit', 'nmap', 'kali linux',
                    'snort', 'ossec', 'vault', 'keycloak'
                ],
                'compliance': [
                    'gdpr', 'hipaa', 'sox', 'pci dss', 'iso 27001', 'ccpa',
                    'security auditing', 'risk assessment'
                ]
            }
        }
        
        # Enhanced soft skills with industry context
        self.professional_skills = {
            'leadership': [
                'team leadership', 'project management', 'mentoring', 'strategic planning',
                'decision making', 'conflict resolution', 'team building', 'coaching',
                'change management', 'organizational development', 'cross-functional leadership',
                'stakeholder management', 'program management', 'resource allocation',
                'performance management', 'talent development'
            ],
            'communication': [
                'technical writing', 'presentation skills', 'stakeholder communication',
                'client relations', 'documentation', 'requirements gathering',
                'public speaking', 'interpersonal skills', 'cross-cultural communication',
                'negotiation', 'facilitation', 'knowledge transfer', 'technical documentation',
                'api documentation', 'user stories', 'business requirements'
            ],
            'analytical': [
                'problem solving', 'critical thinking', 'system design', 'data analysis',
                'debugging', 'performance optimization', 'root cause analysis',
                'quantitative analysis', 'qualitative analysis', 'systems thinking',
                'requirement analysis', 'competitive analysis', 'market analysis',
                'business intelligence', 'metrics analysis', 'code review'
            ],
            'management': [
                'risk management', 'resource planning', 'budget management', 'vendor management',
                'product management', 'quality assurance', 'process improvement',
                'portfolio management', 'release management', 'capacity planning',
                'service management', 'incident management', 'change control',
                'business continuity', 'operational excellence'
            ]
        }
        
        # Job level indicators
        self.job_levels = {
            'entry_level': [
                'junior', 'entry level', 'associate', 'trainee', 'intern', 'graduate'
            ],
            'mid_level': [
                'mid level', 'intermediate', 'experienced', 'senior', 'lead'
            ],
            'senior_level': [
                'principal', 'staff', 'architect', 'expert', 'specialist', 'director',
                'manager', 'head', 'chief', 'vp', 'executive'
            ]
        }
        
        # Compile all keywords
        self.all_keywords = set()
        for category in self.job_related_skills.values():
            self.all_keywords.update(category)
        for category in self.professional_skills.values():
            self.all_keywords.update(category)
        for category in self.job_levels.values():
            self.all_keywords.update(category)

    def validate_pdf_path(self, pdf_path: str) -> bool:
        """Validate if the PDF file exists and is accessible."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"The file {pdf_path} is not a PDF file.")
        if not os.access(pdf_path, os.R_OK):
            raise PermissionError(f"Unable to read {pdf_path}. Check file permissions.")
        return True
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using Tesseract OCR."""
        try:
            # Convert PDF to images
            print("Converting PDF to images for OCR processing...")
            images = convert_from_path(pdf_path)
            
            # Process each page with OCR
            text = ""
            total_pages = len(images)
            
            for i, image in enumerate(images, 1):
                print(f"Processing page {i}/{total_pages} with OCR", end='\r')
                # Use Tesseract to extract text from the image
                page_text = pytesseract.image_to_string(
                    image, 
                    lang=self.tesseract_config['lang'],
                    config=self.tesseract_config['config']
                )
                text += page_text + "\n"
            
            print("\nOCR processing completed.")
            return text
        
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF, trying both direct extraction and OCR."""
        try:
            # First try direct text extraction
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                total_pages = len(pdf_reader.pages)
                
                print("Attempting direct text extraction...")
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    print(f"Processing page {page_num}/{total_pages}", end='\r')
                    page_text = page.extract_text()
                    text += page_text
                
                # If direct extraction yields very little text, try OCR
                if len(text.strip()) < 100:  # Arbitrary threshold
                    print("\nDirect extraction yielded limited text. Switching to OCR...")
                    text = self.extract_text_with_ocr(pdf_path)
                else:
                    print("\nDirect text extraction completed successfully.")
                
                return text
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            print("Attempting OCR as fallback...")
            return self.extract_text_with_ocr(pdf_path)

    def preprocess_text(self, text):
        """Preprocess the extracted text."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_job_requirements(self, text):
        """Extract and refine job requirement patterns from text."""
        requirement_patterns = [
            # Match bullet points with requirement-related words
            r'(?:[\•\-\★]\s*)([^.\n]+(?:required|experience|expertise|proficiency|knowledge|understanding|skills)[^.\n]+)',
            # Match "years of experience in X" patterns
            r'(?:^|\n)(?:\d+\+?\s*)?years?\s+(?:of\s+)?experience\s+(?:in|with)\s+([^.\n]+)',
            # Match phrases like "Proven experience in X"
            r'(?:proven|demonstrated|strong)\s+(?:experience|expertise|background)\s+(?:in|with)\s+([^.\n]+)',
            # Match "Proficient in X"
            r'(?:proficient|fluent|skilled)\s+(?:in|with)\s+([^.\n]+)',
            # Match "Working knowledge of X"
            r'working knowledge of ([^.\n]+)'
        ]
    
        stopwords = [
            "job description", "responsibilities", "role overview", "about the company",
            "work environment", "perks and benefits", "qualifications", "eligibility",
            "ideal candidate", "application process"
        ]
    
        # Initialize list to collect valid requirements
        requirements = []
    
        # Apply patterns
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1).strip()
                # Exclude irrelevant phrases
                if any(stopword in requirement.lower() for stopword in stopwords):
                    continue
                # Filter out overly short matches
                if len(requirement.split()) < 4:  # Less than 4 words is likely irrelevant
                    continue
                # Add to list if it passes all checks
                requirements.append(requirement)
    
        # Deduplicate and return refined requirements
        return list(set(requirements))


    def extract_experience_details(self, text):
        """Extract detailed experience information."""
        experience_patterns = {
            'years': r'\b(\d+)[\+]?\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b',
            'position_duration': r'(?:^|\n)(?:[\•\-\★]\s*)?(\w+\s+\w+(?:\s+\w+)?)\s*(?:\(|-)?\s*(\d{1,2}(?:\.\d)?)\s*(?:years?|yrs?)',
            'achievements': r'(?:^|\n)(?:[\•\-\★]\s*)(?:developed|implemented|led|managed|created|designed|improved|reduced|increased)[^.\n]+'
        }
        
        experience = {
            'total_years': None,
            'positions': [],
            'key_achievements': []
        }
        
        # Extract total years of experience
        years_matches = re.findall(experience_patterns['years'], text.lower())
        if years_matches:
            experience['total_years'] = max(map(float, years_matches))
        
        # Extract position durations
        position_matches = re.finditer(experience_patterns['position_duration'], text)
        for match in position_matches:
            position = {
                'title': match.group(1).strip(),
                'duration': float(match.group(2))
            }
            experience['positions'].append(position)
        
        # Extract key achievements
        achievement_matches = re.finditer(experience_patterns['achievements'], text)
        for match in achievement_matches:
            achievement = match.group().strip()
            if len(achievement.split()) >= 5:  # Filter out too short achievements
                experience['key_achievements'].append(achievement)
        
        return experience

    def extract_keywords(self, pdf_path):
        """Enhanced keyword extraction method with improved pattern matching."""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        print("Analyzing resume for job-relevant content...")
        processed_text = self.preprocess_text(text)
        
        # Initialize results structure
        keywords = {
            'technical_expertise': {},
            'professional_skills': {},
            'seniority_indicators': [],
            'experience_details': self.extract_experience_details(text),
            'job_requirements': self.extract_job_requirements(text)
        }
        
        # Enhanced pattern matching for technical skills
        for category, skills in self.job_related_skills.items():
            if isinstance(skills, dict):
                # Handle nested categories
                subcategories = {}
                for subcategory, subskills in skills.items():
                    found_skills = self._find_skills_with_context(processed_text, subskills)
                    if found_skills:
                        subcategories[subcategory] = found_skills
                if subcategories:
                    keywords['technical_expertise'][category] = subcategories
            else:
                # Handle flat categories
                found_skills = self._find_skills_with_context(processed_text, skills)
                if found_skills:
                    keywords['technical_expertise'][category] = found_skills
        
        # Enhanced pattern matching for professional skills
        for category, skills in self.professional_skills.items():
            found_skills = self._find_skills_with_context(processed_text, skills)
            if found_skills:
                keywords['professional_skills'][category] = found_skills
        
        return keywords
    
    def _find_skills_with_context(self, text, skills):
        """Enhanced skill detection with context awareness."""
        found_skills = []
        programming_context = r'\b(?:programming|language|coding|develop|framework|technology)\b'

        for skill in skills:
            # Define context patterns
            skill_patterns = [
                # Direct mention
                rf'\b{re.escape(skill)}\b',
                # Experience with
                rf'experience (?:in|with) .*?\b{re.escape(skill)}\b',
                # Proficient/skilled in
                rf'(?:proficient|skilled|expertise) (?:in|with) .*?\b{re.escape(skill)}\b',
                # Years of experience
                rf'\d+[\+]? years?.*?\b{re.escape(skill)}\b',
                # Project or work involving
                rf'(?:developed|implemented|built|designed|maintained).*?\b{re.escape(skill)}\b',
                # Certification or training
                rf'(?:certified|trained|courses?).*?\b{re.escape(skill)}\b'
                # Match skill in relevant programming context
                rf'\b{re.escape(skill)}\b(?:\s+{programming_context})?',
                rf'{programming_context}.*?\b{re.escape(skill)}\b',
            ]
            
            # Check for any pattern match
            for pattern in skill_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_skills.append({
                        'skill': skill,
                        'context': self._extract_context(text, skill)
                    })
                    break  
        return found_skills if found_skills else None

    def _extract_context(self, text, skill):
        """Extract surrounding context for a skill mention."""
        # Find the first occurrence of the skill
        skill_index = text.lower().find(skill.lower())
        if skill_index == -1:
            return None
        
        # Extract surrounding context (50 characters before and after)
        start = max(0, skill_index - 50)
        end = min(len(text), skill_index + len(skill) + 50)
        context = text[start:end].strip()
        
        # Clean up the context
        context = re.sub(r'\s+', ' ', context)
        return context

def print_results(keywords):
    """Print the extracted keywords in a detailed, organized format."""
    if not keywords:
        print("No keywords were extracted. Please check the PDF file.")
        return

    print("\n=== Job-Focused Resume Analysis ===\n")
    
    # Print technical expertise by category
    print("Technical Expertise:")
    for category, skills in keywords['technical_expertise'].items():
        print(f"  {category.replace('_', ' ').title()}:")
        if isinstance(skills, dict):
            for subcategory, subskills in skills.items():
                print(f"    {subcategory.title()}:")
                print(f"      {', '.join([skill['skill'] for skill in subskills])}")
        else:
            print(f"    {', '.join([skill['skill'] for skill in skills])}")
    
    # Print professional skills by category
    print("\nProfessional Skills:")
    for category, skills in keywords['professional_skills'].items():
        print(f"  {category.title()}:")
        print(f"    {', '.join([skill['skill'] for skill in skills])}")

    # Print seniority indicators
    print("\nSeniority Level Indicators:")
    for level_info in keywords.get('seniority_indicators', []):
        print(f"  {level_info['level'].replace('_', ' ').title()}:")
        print(f"    {', '.join(level_info['indicators'])}")
    
    # Print experience details
    print("\nExperience Profile:")
    exp_details = keywords['experience_details']
    if exp_details['total_years']:
        print(f"  Total Years of Experience: {exp_details['total_years']}")
    
    if exp_details['positions']:
        print("  Position History:")
        for position in exp_details['positions']:
            print(f"    {position['title']}: {position['duration']} years")
    
    if exp_details['key_achievements']:
        print("\nKey Achievements:")
        for achievement in exp_details['key_achievements']:
            print(f"  • {achievement}")
    
    # Print job requirements
    if keywords['job_requirements']:
        print("\nIdentified Job Requirements:")
        for req in keywords['job_requirements']:
            print(f"  • {req}")


def get_pdf_path():
    """
    Get PDF path from user input with validation.
    Returns validated PDF path or None if user cancels.
    """
    while True:
        pdf_path = input("\nPlease enter the path to your resume PDF file: ").strip()
        
        # Remove quotes if user included them
        pdf_path = pdf_path.strip('"\'')
        
        try:
            if os.path.exists(pdf_path):
                if pdf_path.lower().endswith('.pdf'):
                    if os.access(pdf_path, os.R_OK):
                        return pdf_path
                    else:
                        print("Error: Unable to read the file. Please check file permissions.")
                else:
                    print("Error: The file must be a PDF file.")
            else:
                print("Error: File does not exist. Please check the path and try again.")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        retry = input("Would you like to try again? (y/n): ").lower()
        if retry != 'y':
            return None

def main():
    print("Welcome to Job-Focused Resume Keyword Extractor!")
    print("This tool will analyze your resume and extract relevant job-related keywords and experience.")
    
    pdf_path = get_pdf_path()
    if not pdf_path:
        print("Program terminated.")
        return
    
    try:
        extractor = ResumeKeywordExtractor()
        keywords = extractor.extract_keywords(pdf_path)
        
        if keywords:
            print_results(keywords)
            
            save = input("\nWould you like to save the results to a file? (y/n): ").lower()
            if save == 'y':
                output_path = input("Enter the path for the output file (default: resume_analysis.json): ").strip() or "resume_analysis.json"
                try:
                    import json
                    with open(output_path, 'w') as f:
                        json.dump(keywords, f, indent=2)
                    print(f"Results saved to {output_path}")
                except Exception as e:
                    print(f"Error saving results: {str(e)}")
        else:
                                print("No keywords were extracted. Please check the PDF file.")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()