import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import os
from typing import Optional, List, Dict, Union
import tempfile
import json
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

class ResumeKeywordExtractor:
    def __init__(self):
        """Initialize the keyword extractor with enhanced NLP models and patterns."""
        self.console = Console()
        
        # Create a single progress instance for initialization
        self.progress = Progress()
        with self.progress:
            task1 = self.progress.add_task("[green]Initializing NLP components...", total=3)
            
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.progress.update(task1, advance=1)
            
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
            self.progress.update(task1, advance=1)
            
            self._initialize_skill_patterns()
            self.progress.update(task1, advance=1)

        self.stopwords = set(stopwords.words('english'))
        self.tesseract_config = {
            'lang': 'eng',
            'config': '--psm 1 --oem 3'
        }

    def _initialize_skill_patterns(self):
        """Initialize comprehensive skill patterns with expanded categories."""
        # Enhanced context patterns for better matching
        self.context_patterns = {
            'programming_context': r'(?i)(?:programming|develop(?:ed|ing)|coding|implemented|architected|built|designed|using|language[s]?|framework[s]?|stack|development|developer)\s+(?:in|with)?\s*',
            'experience_context': r'(?i)(?:experience|expertise|proficiency|skilled|fluent|competent|knowledge|background)\s+(?:in|with|of)\s+',
            'tool_context': r'(?i)(?:using|utilized|leveraged|worked with|familiar with|proficient in|tools?|platforms?|technologies)\s+',
            'version_context': r'\d+(?:\.\d+)*',
            'certification_context': r'(?i)(?:certified|certification|certificate|qualified|trained|accredited)\s+(?:in|as|by)?\s*',
            'achievement_context': r'(?i)(?:achieved|accomplished|delivered|improved|optimized|enhanced|reduced|increased)\s+',
            'project_context': r'(?i)(?:project|initiative|implementation|development|deployment|migration|integration)\s+'
        }

        # Expanded skill categories
        self.job_related_skills = {
            'programming_languages': {
                'python': [
                    r'(?i)python\s*(?:\d+(?:\.\d+)*)?(?:\s+programming|\s+development|\s+scripting)?',
                    r'(?i)(?:django|flask|fastapi|pytest|pandas|numpy|scipy|tensorflow|pytorch|scikit-learn)',
                    r'(?i)(?:pip|virtualenv|conda|jupyter|pyspark|celery|asyncio)'
                ],
                'java': [
                    r'(?i)java\s*(?:\d+(?:\.\d+)*)?(?:\s+programming|\s+development)?',
                    r'(?i)(?:spring|hibernate|maven|gradle|junit|tomcat|jdbc|kotlin)',
                    r'(?i)(?:j2ee|java\s+ee|jakarta\s+ee|microservices|quarkus|micronaut)'
                ],
                'javascript': [
                    r'(?i)(?:javascript|js|ecmascript|es6|typescript)(?:\s+programming|\s+development)?',
                    r'(?i)(?:node\.?js|react|angular|vue|next\.js|nuxt\.js|svelte)',
                    r'(?i)(?:webpack|babel|npm|yarn|deno|express|nestjs)'
                ],
                'go': [
                    r'(?i)(?:golang|go\s+programming|go\s+lang)',
                    r'(?i)(?:gin|echo|fiber|gorm|gorilla)',
                    r'(?i)(?:go\s+modules|go\s+routines)'
                ],
                'rust': [
                    r'(?i)rust(?:\s+programming|\s+development)?',
                    r'(?i)(?:cargo|tokio|actix|rocket|wasm)',
                    r'(?i)(?:rust\s+analyzer|clippy)'
                ]
            },
            'cloud_and_devops': {
                'aws': [
                    r'(?i)(?:aws|amazon\s+web\s+services)(?:\s+cloud|\s+services)?',
                    r'(?i)(?:ec2|s3|lambda|cloudformation|eks|rds|dynamodb)',
                    r'(?i)(?:cloudwatch|route53|iam|vpc|fargate|sagemaker)'
                ],
                'azure': [
                    r'(?i)(?:microsoft\s+azure|azure)(?:\s+cloud|\s+services)?',
                    r'(?i)(?:azure\s+devops|azure\s+functions|aks|cosmos\s+db)',
                    r'(?i)(?:azure\s+ad|azure\s+pipeline|app\s+service)'
                ],
                'gcp': [
                    r'(?i)(?:google\s+cloud|gcp|gke)',
                    r'(?i)(?:bigquery|cloud\s+run|cloud\s+functions)',
                    r'(?i)(?:cloud\s+storage|dataflow|vertex\s+ai)'
                ],
                'devops': [
                    r'(?i)(?:devops|ci/cd|continuous\s+(?:integration|deployment|delivery))(?:\s+practices|\s+tools)?',
                    r'(?i)(?:jenkins|gitlab|github\s+actions|travis|circleci|argocd)',
                    r'(?i)(?:docker|kubernetes|k8s|helm|terraform|ansible|pulumi)'
                ]
            },
            'data_engineering': {
                'batch_processing': [
                    r'(?i)(?:spark|hadoop|mapreduce|airflow|luigi)',
                    r'(?i)(?:etl|data\s+pipeline|data\s+workflow)',
                    r'(?i)(?:batch\s+processing|distributed\s+computing)'
                ],
                'streaming': [
                    r'(?i)(?:kafka|rabbitmq|kinesis|pubsub)',
                    r'(?i)(?:stream\s+processing|real-time|event-driven)',
                    r'(?i)(?:flink|storm|beam)'
                ],
                'data_warehouse': [
                    r'(?i)(?:snowflake|redshift|bigquery|synapse)',
                    r'(?i)(?:data\s+modeling|data\s+warehouse|dwh)',
                    r'(?i)(?:dimensional\s+modeling|star\s+schema)'
                ]
            },
            'databases': {
                'sql': [
                    r'(?i)(?:sql|relational\s+database)(?:\s+development|\s+design)?',
                    r'(?i)(?:mysql|postgresql|oracle|sql\s+server|sqlite)',
                    r'(?i)(?:pl/sql|t-sql|stored\s+procedures)'
                ],
                'nosql': [
                    r'(?i)(?:nosql|document\s+database|key-value\s+store)',
                    r'(?i)(?:mongodb|cassandra|redis|elasticsearch)',
                    r'(?i)(?:dynamodb|cosmosdb|couchbase)'
                ],
                'graph': [
                    r'(?i)(?:graph\s+database|graph\s+query)',
                    r'(?i)(?:neo4j|amazon\s+neptune|arangodb)',
                    r'(?i)(?:cypher|gremlin|sparql)'
                ]
            }
        }

        self.professional_skills = {
            'leadership': [
                r'(?i)(?:team\s+lead(?:er)?|project\s+manag(?:er|ement)|leadership)',
                r'(?i)(?:managed|directed|supervised|mentored)\s+(?:team|project|department)',
                r'(?i)(?:strategic|execution|decision-making|delegation|cross-functional)'
            ],
            'agile_methodologies': [
                r'(?i)(?:agile|scrum|kanban|safe|lean)\s+(?:methodology|framework)?',
                r'(?i)(?:sprint|backlog|refinement|retrospective|ceremonies)',
                r'(?i)(?:product\s+owner|scrum\s+master|agile\s+coach|scaled\s+agile)'
            ],
            'business_skills': [
                r'(?i)(?:stakeholder|client)\s+(?:management|communication|interaction)',
                r'(?i)(?:requirements\s+gathering|business\s+analysis|solution\s+design)',
                r'(?i)(?:problem-solving|analytical|strategic\s+thinking|innovation)'
            ],
            'soft_skills': [
                r'(?i)(?:communication|collaboration|teamwork|interpersonal)',
                r'(?i)(?:presentation|public\s+speaking|negotiation)',
                r'(?i)(?:time\s+management|organization|adaptability|flexibility)'
            ]
        }

    def _calculate_relevance_score(self, text: str, context: str) -> float:
        """
        Calculate an enhanced relevance score based on multiple factors.
        
        Args:
            text: The full text of the resume
            context: The specific context around a skill mention
            
        Returns:
            float: Normalized relevance score between 0 and 1
        """
        score = 0.0
        doc = self.nlp(context)
        
        # Context pattern matching (weighted)
        for pattern_type, pattern in self.context_patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                if 'programming' in pattern_type or 'tool' in pattern_type:
                    score += 2.5
                elif 'experience' in pattern_type or 'achievement' in pattern_type:
                    score += 2.0
                else:
                    score += 1.5

        # Entity recognition (weighted)
        entity_weights = {
            'ORG': 2.0,      # Organizations
            'PRODUCT': 1.8,   # Products/Technologies
            'DATE': 1.5,      # Temporal references
            'GPE': 1.2,       # Locations
            'PERSON': 1.0     # People references
        }
        
        for ent in doc.ents:
            if ent.label_ in entity_weights:
                score += entity_weights[ent.label_]

        # Semantic analysis
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        action_verbs = {'implement', 'develop', 'create', 'design', 'manage', 'lead', 'optimize', 'improve'}
        score += len(set(verbs) & action_verbs) * 1.5

        # Context position analysis
        if context.lower().startswith(('responsible for', 'led', 'managed', 'developed')):
            score += 2.0

        # Normalize score to 0-1 range
        max_possible_score = 20.0  # Adjusted based on maximum possible points
        normalized_score = min(score / max_possible_score, 1.0)
        
        return round(normalized_score, 2)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced text extraction with better handling of formatting and layout."""
        text = ""
        
        # Use a single progress instance
        task = self.progress.add_task("[cyan]Extracting text from PDF...", total=2)
        
        # Try PyPDF2 first
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = '\n'.join(
                    page.extract_text(layout=True)
                    for page in reader.pages
                )
            self.progress.update(task, advance=1)
            
            # Validate text quality
            if len(text.strip()) < 100 or self._assess_text_quality(text) < 0.3:
                # Fall back to OCR
                self.console.print("[yellow]Low quality text detected, attempting OCR...")
                images = convert_from_path(pdf_path)
                ocr_texts = []
                
                for img in images:
                    img = img.convert('L')
                    ocr_text = pytesseract.image_to_string(img, **self.tesseract_config)
                    ocr_texts.append(ocr_text)
                
                text = '\n'.join(ocr_texts)
            
            self.progress.update(task, advance=1)
            
        except Exception as e:
            self.console.print(f"[red]Error extracting text: {str(e)}")
            return ''

        return self._clean_extracted_text(text)

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')
        text = re.sub(r'(?<=[a-z])\.(?=[A-Z])', '. ', text)
        
        # Normalize bullet points
        text = re.sub(r'[•·⋅○●]', '• ', text)
        
        # Fix sentence spacing
        text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', ' ', text)
        
        return text.strip()

    def _format_results(self, results: Dict) -> None:
        """Format and display results using rich formatting."""
        # Technical Skills Table
        tech_table = Table(title="Technical Skills", show_header=True, header_style="bold magenta")
        tech_table.add_column("Category", style="cyan")
        tech_table.add_column("Skill", style="green")
        tech_table.add_column("Confidence", justify="right", style="yellow")
        
        for category, skills in results['technical_skills'].items():
            for skill, details in skills.items():
                tech_table.add_row(
                    category.replace('_', ' ').title(),
                    skill,
                    f"{details['confidence']*100:.1f}%"
                )
        
        self.console.print(tech_table)
        
        # Professional Skills
        prof_table = Table(title="Professional Skills", show_header=True, header_style="bold magenta")
        prof_table.add_column("Category", style="cyan")
        prof_table.add_column("Skill", style="green")
        prof_table.add_column("Confidence", justify="right", style="yellow")
        
        for category, skills in results['professional_skills'].items():
            if skills:
                for skill in skills:
                    prof_table.add_row(
                        category.replace('_', ' ').title(),
                        skill['text'],
                        f"{skill['confidence']*100:.1f}%"
                    )
        
        self.console.print("\n", prof_table)
        
        # Experience Details Panel
        if results['experience_details']:
            exp_details = results['experience_details']
            
            # Format experience years
            years_text = ""
            for year_detail in exp_details['total_years']:
                years_text += f"• {year_detail['years']} years\n"
            
            # Format positions
            positions_text = ""
            for position in exp_details['positions']:
                positions_text += f"• {position['title']}\n"
            
            # Format achievements
            achievements_text = ""
            for achievement in exp_details['key_achievements']:
                achievements_text += f"• {achievement}\n"
            
            experience_panel = Panel(
                f"[bold cyan]Years of Experience:[/]\n{years_text}\n"
                f"[bold cyan]Positions Held:[/]\n{positions_text}\n"
                f"[bold cyan]Key Achievements:[/]\n{achievements_text}",
                title="Experience Details",
                border_style="magenta"
            )
            
            self.console.print("\n", experience_panel)

    def _extract_validated_experience(self, text: str) -> Dict:
        """Enhanced experience extraction with better validation and context analysis."""
        experience = {
            'total_years': [],
            'positions': [],
            'key_achievements': []
        }
        
        # Enhanced year patterns with better context
        year_patterns = [
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|expertise)',
            r'(?i)(?:experience|expertise).{0,20}?(\d+)\+?\s*(?:years?|yrs?)',
            r'(?i)(?:career|professional).{0,20}?(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in year_patterns:
            for match in re.finditer(pattern, text):
                years = int(match.group(1))
                context = self._extract_skill_context(text, match.start(), match.end())
                if years > 0 and years < 50:  # Basic validation
                    experience['total_years'].append({
                        'years': years,
                        'context': context
                    })
        
        # Enhanced position patterns with hierarchy and specialization
        position_patterns = [
            # Technical positions
            r'(?i)(senior|lead|principal|staff|chief|head|director)?\s*'
            r'(?:software|systems?|data|cloud|devops|full.?stack|machine\s+learning|ai)'
            r'\s+(?:engineer|developer|architect|specialist)',
            
            # Management positions
            r'(?i)(tech(?:nical)?|engineering|development|product)\s*'
            r'(?:lead(?:er)?|manager|director|head)',
            
            # Specialized roles
            r'(?i)(senior|lead|principal)?\s*'
            r'(?:solution|security|database|infrastructure|platform|site\s+reliability)'
            r'\s+(?:architect|engineer|specialist)'
        ]
        
        for pattern in position_patterns:
            for match in re.finditer(pattern, text):
                position = match.group().strip()
                context = self._extract_skill_context(text, match.start(), match.end())
                
                # Validate position
                if len(position) > 3 and not any(char.isdigit() for char in position):
                    experience['positions'].append({
                        'title': position,
                        'context': context
                    })
        
        # Enhanced achievement extraction
        achievement_patterns = [
            # Quantifiable achievements
            r'(?i)(?:led|managed|developed|implemented|created|designed|architected)'
            r'.{10,150}?(?:resulting in|achieving|improving|reducing)\s+'
            r'(?:\d+(?:\.\d+)?%|\$[\d,]+k?|\d+\s*x)',
            
            # Project impacts
            r'(?i)(?:increased|improved|reduced|optimized|enhanced)'
            r'.{10,150}?(?:by|to)\s+'
            r'(?:\d+(?:\.\d+)?%|\$[\d,]+k?|\d+\s*x)',
            
            # Team achievements
            r'(?i)(?:led|managed|mentored)'
            r'.{10,150}?(?:team|group|department)'
            r'.{10,150}?(?:delivered|completed|achieved|launched)'
        ]
        
        for pattern in achievement_patterns:
            for match in re.finditer(pattern, text):
                achievement = match.group().strip()
                # Validate achievement
                if (len(achievement) > 30 and  # Meaningful length
                    len(achievement) < 200 and  # Not too long
                    any(char.isdigit() for char in achievement)):  # Contains metrics
                    experience['key_achievements'].append(achievement)
        
        # Remove duplicates while preserving order
        experience['key_achievements'] = list(dict.fromkeys(experience['key_achievements']))
        
        return experience

def main():
    """Enhanced main function with better user interaction and error handling."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]Resume Keyword Extractor[/]\n"
        "[cyan]Analyzes resumes to extract key skills, experience, and achievements[/]",
        border_style="green"
    ))
    
    while True:
        pdf_path = console.input("\nEnter the path to your PDF resume (or 'q' to quit): ").strip()
        
        if pdf_path.lower() == 'q':
            console.print("\n[yellow]Thank you for using Resume Keyword Extractor![/]")
            sys.exit(0)
        
        pdf_path = os.path.expanduser(pdf_path)
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"File '{pdf_path}' does not exist.")
                
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError(f"File '{pdf_path}' is not a PDF file.")
                
            if not os.access(pdf_path, os.R_OK):
                raise PermissionError(f"File '{pdf_path}' is not readable. Check file permissions.")
            
            # Create a single Progress instance
            progress = Progress()
            with progress:
                task = progress.add_task("[green]Processing resume...", total=100)
                
                extractor = ResumeKeywordExtractor()
                progress.update(task, advance=30)
                
                results = extractor.extract_keywords(pdf_path)
                progress.update(task, advance=70)
            
            if results is None:
                console.print("\n[red]No text could be extracted from the PDF. Please check the file and try again.[/]")
                continue
            
            # Format and display results
            extractor._format_results(results)
            
            # Save results option
            save_choice = console.input("\nWould you like to save the results to a JSON file? (yes/no): ").strip().lower()
            if save_choice in ('yes', 'y'):
                output_path = os.path.join(os.getcwd(), 'resume_analysis.json')
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"\n[green]Results saved to:[/] {output_path}")
            
            break
            
        except Exception as e:
            console.print(f"\n[red]Error:[/] {str(e)}")
            continue

if __name__ == "__main__":
    main()