import asyncio
import aiohttp
from typing import List, Dict, Set
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
from datetime import datetime
from bs4 import BeautifulSoup
import json
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import logging

@dataclass
class JobPosting:
    id: str
    title: str
    company: str
    description: str
    location: str
    required_years: float
    salary_range: Dict[str, float]
    required_skills: Set[str]
    nice_to_have_skills: Set[str]
    posting_date: datetime
    source: str
    url: str

class BaseJobScraper:
    async def scrape(self, search_params: Dict) -> List[JobPosting]:
        raise NotImplementedError

class LinkedInScraper(BaseJobScraper):
    async def scrape(self, search_params: Dict) -> List[JobPosting]:
        await asyncio.sleep(1)  # Simulate API call
        return [
            JobPosting(
                id="li_1",
                title="Senior Python Developer",
                company="TechCorp",
                description="Looking for an experienced Python developer...",
                location="San Francisco",
                required_years=5.0,
                salary_range={"min": 120000, "max": 180000},
                required_skills={"python", "django", "aws"},
                nice_to_have_skills={"docker", "kubernetes"},
                posting_date=datetime.now(),
                source="linkedin",
                url="https://linkedin.com/jobs/1"
            )
        ]

class IndeedScraper(BaseJobScraper):
    async def scrape(self, search_params: Dict) -> List[JobPosting]:
        await asyncio.sleep(1)  # Simulate API call
        return [
            JobPosting(
                id="indeed_1",
                title="Python Software Engineer",
                company="StartupInc",
                description="Join our fast-growing team...",
                location="San Francisco",
                required_years=3.0,
                salary_range={"min": 110000, "max": 160000},
                required_skills={"python", "flask", "sql"},
                nice_to_have_skills={"react", "mongodb"},
                posting_date=datetime.now(),
                source="indeed",
                url="https://indeed.com/jobs/1"
            )
        ]

class GlassdoorScraper(BaseJobScraper):
    async def scrape(self, search_params: Dict) -> List[JobPosting]:
        await asyncio.sleep(1)  # Simulate API call
        return [
            JobPosting(
                id="glassdoor_1",
                title="Full Stack Python Developer",
                company="BigTech Ltd",
                description="Looking for a full stack developer...",
                location="San Francisco",
                required_years=4.0,
                salary_range={"min": 130000, "max": 190000},
                required_skills={"python", "javascript", "react"},
                nice_to_have_skills={"typescript", "graphql"},
                posting_date=datetime.now(),
                source="glassdoor",
                url="https://glassdoor.com/jobs/1"
            )
        ]

class JobScraperFactory:
    @staticmethod
    def get_scraper(platform: str) -> BaseJobScraper:
        scrapers = {
            'linkedin': LinkedInScraper(),
            'indeed': IndeedScraper(),
            'glassdoor': GlassdoorScraper()
        }
        return scrapers.get(platform)

class SkillTaxonomy:
    def __init__(self):
        self.skill_graph = self._load_skill_graph()
        self.bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
    def _load_skill_graph(self) -> Dict:
        return {
            'programming': {
                'languages': {'python', 'java', 'javascript', 'c++'},
                'frameworks': {'react', 'django', 'spring', 'angular'},
                'related_concepts': {'oop', 'functional programming', 'algorithms'}
            },
            'data_science': {
                'core': {'machine learning', 'statistics', 'data analysis'},
                'tools': {'pandas', 'sklearn', 'tensorflow', 'pytorch'},
                'techniques': {'regression', 'classification', 'clustering'}
            }
        }
    
    def find_related_skills(self, skill: str) -> Set[str]:
        related = set()
        for category, subcategories in self.skill_graph.items():
            for subcategory, skills in subcategories.items():
                if skill in skills:
                    related.update(skills)
        return related
    
    async def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        embeddings = self.bert_model.encode([skill1, skill2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

class EnhancedJobMatchingSystem:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.skill_taxonomy = SkillTaxonomy()
        self.scrapers = JobScraperFactory()
        self.logger = logging.getLogger(__name__)
        
    def calculate_experience_match(self, required_years: float, user_years: float) -> float:
        """Calculate how well the user's experience matches the job requirements."""
        if user_years >= required_years:
            return 1.0
        elif user_years >= required_years * 0.8:
            return 0.8
        elif user_years >= required_years * 0.6:
            return 0.6
        return 0.4
        
    async def fetch_jobs(self, search_params: Dict, platforms: List[str]) -> List[JobPosting]:
        tasks = []
        for platform in platforms:
            scraper = self.scrapers.get_scraper(platform)
            if scraper:
                tasks.append(asyncio.create_task(scraper.scrape(search_params)))
        
        results = await asyncio.gather(*tasks)
        return [job for platform_jobs in results for job in platform_jobs]
    
    async def calculate_skill_match_score(self, job_skills: Set[str], user_skills: Set[str]) -> Dict:
        exact_matches = job_skills.intersection(user_skills)
        related_matches = set()
        similarity_scores = []
        
        for job_skill in job_skills:
            related = self.skill_taxonomy.find_related_skills(job_skill)
            related_matches.update(user_skills.intersection(related))
            
            for user_skill in user_skills:
                if user_skill not in exact_matches:
                    similarity = await self.skill_taxonomy.calculate_skill_similarity(job_skill, user_skill)
                    if similarity > 0.8:
                        similarity_scores.append((job_skill, user_skill, similarity))
        
        return {
            'exact_matches': exact_matches,
            'related_matches': related_matches,
            'semantic_matches': similarity_scores,
            'match_score': (len(exact_matches) + 0.5 * len(related_matches)) / len(job_skills) if job_skills else 0
        }
    
    def calculate_compensation_match(self, job_salary: Dict, user_expectations: Dict) -> float:
        if not (job_salary and user_expectations):
            return 0.5
            
        job_mid = (job_salary['min'] + job_salary['max']) / 2
        user_mid = (user_expectations['min'] + user_expectations['max']) / 2
        
        if job_mid >= user_expectations['min'] and job_mid <= user_expectations['max']:
            return 1.0
        
        diff = abs(job_mid - user_mid) / user_mid
        return max(0, 1 - diff)
    
    def calculate_location_score(self, job_location: str, user_location: str, remote_preference: bool) -> float:
        if 'remote' in job_location.lower() and remote_preference:
            return 1.0
        return 1.0 if job_location == user_location else 0.5
    
    async def analyze_job_posting(self, posting: JobPosting, user_profile: Dict) -> Dict:
        skill_match = await self.calculate_skill_match_score(
            posting.required_skills,
            set(user_profile['skills'])
        )
        
        experience_match = self.calculate_experience_match(
            posting.required_years,
            user_profile['years_experience']
        )
        
        compensation_match = self.calculate_compensation_match(
            posting.salary_range,
            user_profile['salary_expectations']
        )
        
        location_match = self.calculate_location_score(
            posting.location,
            user_profile['location'],
            user_profile.get('remote_preference', False)
        )
        
        weights = {
            'skill_match': 0.35,
            'experience_match': 0.25,
            'compensation_match': 0.20,
            'location_match': 0.20
        }
        
        overall_score = sum(
            score * weights[metric] for metric, score in {
                'skill_match': skill_match['match_score'],
                'experience_match': experience_match,
                'compensation_match': compensation_match,
                'location_match': location_match
            }.items()
        )
        
        return {
            'job_id': posting.id,
            'title': posting.title,
            'company': posting.company,
            'source': posting.source,
            'url': posting.url,
            'overall_match_score': round(overall_score * 100, 2),
            'skill_analysis': {
                'exact_matches': list(skill_match['exact_matches']),
                'related_matches': list(skill_match['related_matches']),
                'semantic_matches': skill_match['semantic_matches'],
                'missing_critical_skills': list(posting.required_skills - set(user_profile['skills'])),
                'score': round(skill_match['match_score'] * 100, 2)
            },
            'experience_match_score': round(experience_match * 100, 2),
            'compensation_match_score': round(compensation_match * 100, 2),
            'location_match_score': round(location_match * 100, 2),
            'posting_recency': (datetime.now() - posting.posting_date).days
        }
    
    async def analyze_multiple_postings(self, search_params: Dict, user_profile: Dict, platforms: List[str]) -> List[Dict]:
        job_postings = await self.fetch_jobs(search_params, platforms)
        
        analysis_tasks = [
            self.analyze_job_posting(posting, user_profile)
            for posting in job_postings
        ]
        results = await asyncio.gather(*analysis_tasks)
        
        filtered_results = [
            result for result in results
            if result['overall_match_score'] >= 60
            and len(result['skill_analysis']['missing_critical_skills']) <= 2
            and result['posting_recency'] <= 30
        ]
        
        filtered_results.sort(key=lambda x: (
            x['overall_match_score'],
            -len(x['skill_analysis']['missing_critical_skills']),
            -x['posting_recency']
        ), reverse=True)
        
        return filtered_results

async def main():
    # Initialize the system
    matcher = EnhancedJobMatchingSystem()
    
    # Search parameters
    search_params = {
        'keywords': ['software engineer', 'python developer'],
        'location': 'San Francisco',
        'experience_level': 'senior'
    }
    
    # User profile
    user_profile = {
        'skills': ['python', 'aws', 'machine learning', 'react'],
        'years_experience': 5,
        'location': 'San Francisco',
        'remote_preference': True,
        'salary_expectations': {'min': 120000, 'max': 180000}
    }
    
    # Analyze jobs from multiple platforms
    results = await matcher.analyze_multiple_postings(
        search_params,
        user_profile,
        platforms=['linkedin', 'indeed', 'glassdoor']
    )
    
    # Display results
    for match in results[:5]:
        print(f"\nJob: {match['title']} at {match['company']}")
        print(f"Overall Match: {match['overall_match_score']}%")
        print(f"Source: {match['source']}")
        print(f"URL: {match['url']}")
        print("Skill Analysis:", json.dumps(match['skill_analysis'], indent=2))

if __name__ == "__main__":
    asyncio.run(main())