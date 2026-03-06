# ChatBot.py - Enhanced with Google Gemini API
import io
import json
import os
import tempfile
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
from dataclasses import dataclass
import csv
import time

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class Symptom:
    name: str
    severity: float
    duration: Optional[str] = None


@dataclass
class Diagnosis:
    condition: str
    confidence: float
    symptoms_matched: List[str]
    recommendations: List[str]
    specialist: Optional[str] = None
    severity: Optional[str] = None
    urgency: Optional[str] = None


class GeminiMedicalAPI:
    """Wrapper for Google Gemini API integration for medical analysis"""

    def __init__(self):
        # REPLACE THIS WITH YOUR ACTUAL GEMINI API KEY FROM https://ai.google.dev/
        self.api_key = "YOUR ACTUAL GEMINI API"
        self.model_name = "gemini-1.5-flash"  # Free tier model
        self.is_configured_flag = False

        # Debug information
        print(f"🔍 Checking API key...")
        print(f"🔍 Key length: {len(self.api_key)}")
        print(f"🔍 Key starts with: {self.api_key[:15]}...")

        # Check if API key is configured (not the placeholder)
        if self.api_key != "YOUR_GEMINI_API_KEY_HERE" and self.api_key and len(self.api_key) > 20:
            try:
                print("🔄 Configuring Gemini API...")
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)

                # Test the connection with a simple request
                print("🧪 Testing API connection...")
                test_response = self.model.generate_content("Hello, respond with 'API working'")
                if test_response.text:
                    self.is_configured_flag = True
                    print("✅ Gemini API configured and tested successfully!")
                else:
                    print("⚠️ API responded but with empty content")
                    self.model = None

            except Exception as e:
                print(f"❌ Error configuring Gemini API: {e}")
                if "API_KEY_INVALID" in str(e):
                    print("🔑 Your API key appears to be invalid. Please check it at https://ai.google.dev/")
                elif "PERMISSION_DENIED" in str(e):
                    print("🚫 Permission denied. Make sure your API key has proper permissions.")
                elif "QUOTA_EXCEEDED" in str(e):
                    print("📊 Quota exceeded. You may have hit your rate limit.")
                self.model = None
        else:
            print("ℹ️ Gemini API not configured - using local analysis only")
            print("💡 To enable AI features, get your free API key from https://ai.google.dev/")
            self.model = None

    def is_configured(self) -> bool:
        """Check if API is properly configured"""
        return self.is_configured_flag

    def analyze_symptoms(self, symptoms_text: str, age: int = 30, gender: str = "not specified") -> Dict:
        """Analyze symptoms using Gemini AI"""
        if not self.is_configured():
            return {}

        try:
            prompt = f"""
            You are a medical AI assistant. Analyze the following symptoms and provide a professional assessment.

            Patient Information:
            - Symptoms: {symptoms_text}
            - Age: {age}
            - Gender: {gender}

            Please analyze and respond in JSON format:
            {{
                "primary_condition": "Most likely condition name",
                "confidence": 0.85,
                "alternative_conditions": ["condition1", "condition2"],
                "matched_symptoms": ["symptom1", "symptom2"],
                "specialist": "Recommended specialist type",
                "urgency": "routine/urgent/emergency",
                "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
                "severity": "mild/moderate/severe",
                "explanation": "Brief explanation"
            }}

            Guidelines:
            - Be conservative and suggest professional medical advice
            - Focus on common conditions over rare diseases
            - Consider age and gender in assessment
            - Mark as "emergency" only for life-threatening symptoms
            - Provide practical, actionable recommendations
            - Always emphasize this is not a substitute for professional care

            Respond with ONLY the JSON, no other text.
            """

            response = self.model.generate_content(prompt)

            # Clean and parse JSON response
            response_text = response.text.strip()

            # Remove markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            try:
                analysis = json.loads(response_text)
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                print("⚠️ JSON parsing failed, creating fallback response")
                return {
                    "primary_condition": "Medical condition requiring evaluation",
                    "confidence": 0.7,
                    "alternative_conditions": [],
                    "matched_symptoms": symptoms_text.split(),
                    "specialist": "General Practitioner",
                    "urgency": "routine",
                    "recommendations": [
                        "Consult a healthcare provider for proper diagnosis",
                        "Monitor your symptoms carefully",
                        "Seek immediate care if symptoms worsen"
                    ],
                    "severity": "moderate",
                    "explanation": "AI analysis completed. Professional medical consultation recommended."
                }

        except Exception as e:
            print(f"❌ Error in  analysis: {e}")
            return {}

    def get_detailed_recommendations(self, condition: str, symptoms: str) -> List[str]:
        """Get detailed recommendations for a specific condition"""
        if not self.is_configured():
            return ["Consult a healthcare provider for proper diagnosis and treatment"]

        try:
            prompt = f"""
            Provide specific, actionable recommendations for someone who may have "{condition}" with symptoms: {symptoms}

            Give 4-6 practical recommendations including:
            - Immediate care steps
            - When to seek medical attention
            - Lifestyle adjustments
            - Monitoring advice

            Format as a simple list, one recommendation per line.
            Be conservative and always recommend professional medical consultation.
            """

            response = self.model.generate_content(prompt)
            recommendations = []

            for line in response.text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up list formatting
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        line = line[1:].strip()
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                        line = line[2:].strip()
                    if line:
                        recommendations.append(line)

            # Ensure we have at least basic recommendations
            if not recommendations:
                recommendations = [
                    "Consult a healthcare provider for proper diagnosis",
                    "Monitor your symptoms carefully",
                    "Get adequate rest and stay hydrated",
                    "Seek immediate medical attention if symptoms worsen"
                ]

            return recommendations[:6]  # Limit to 6 recommendations

        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            return ["Consult a healthcare provider for proper diagnosis and treatment"]


class MedicalChatbot:
    def __init__(self):
        # Initialize Gemini API
        print("🚀 Initializing Enhanced Medical Chatbot...")
        self.gemini = GeminiMedicalAPI()

        # Initialize transformer with explicit model (keeping your original)
        model_name = "facebook/bart-large-mnli"
        try:
            print("📚 Loading local analysis model...")
            self.symptom_classifier = pipeline("zero-shot-classification", model=model_name)
            print("✅ Local model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Local model not loaded: {e}")
            self.symptom_classifier = None

        # Initialize language support
        self.supported_languages = {
            'en': 'english',
            'fr': 'french',
            'es': 'spanish',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ar': 'arabic',
        }
        self.current_language = 'en'

        # Initialize speech components
        self.setup_audio_components()

        # Load medical knowledge base
        self.load_medical_data()

        # Load and preprocess the dataset
        self.load_dataset()

        # Load specialist recommendations
        self.specialist_data = self.load_specialist_data()

        print("🎉 Medical Chatbot initialization complete!")

    def load_specialist_data(self):
        """Load specialist recommendation data from CSV file"""
        try:
            specialist_dict = {}
            with open('specialist-recommendations.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    specialist_dict[row['Disease'].lower()] = {
                        'symptoms': row['Symptom'].lower(),
                        'specialist': row['Specialist Recommended']
                    }
            print("📋 Specialist recommendations loaded from CSV")
            return specialist_dict
        except FileNotFoundError:
            print("⚠️ specialist-recommendations.csv not found. Using built-in data.")
            # Enhanced specialist mapping with more conditions
            return {
                'common cold': {'specialist': 'General Practitioner'},
                'flu': {'specialist': 'General Practitioner'},
                'influenza': {'specialist': 'General Practitioner'},
                'pneumonia': {'specialist': 'Pulmonologist'},
                'bronchitis': {'specialist': 'Pulmonologist'},
                'asthma': {'specialist': 'Pulmonologist'},
                'covid-19': {'specialist': 'Internal Medicine'},
                'coronavirus': {'specialist': 'Internal Medicine'},
                'migraine': {'specialist': 'Neurologist'},
                'tension headache': {'specialist': 'General Practitioner'},
                'cluster headache': {'specialist': 'Neurologist'},
                'sinusitis': {'specialist': 'ENT Specialist'},
                'allergic rhinitis': {'specialist': 'Allergist'},
                'hay fever': {'specialist': 'Allergist'},
                'gastroenteritis': {'specialist': 'Gastroenterologist'},
                'food poisoning': {'specialist': 'General Practitioner'},
                'acid reflux': {'specialist': 'Gastroenterologist'},
                'gerd': {'specialist': 'Gastroenterologist'},
                'urinary tract infection': {'specialist': 'Urologist'},
                'uti': {'specialist': 'Urologist'},
                'kidney stones': {'specialist': 'Urologist'},
                'hypertension': {'specialist': 'Cardiologist'},
                'high blood pressure': {'specialist': 'Cardiologist'},
                'diabetes': {'specialist': 'Endocrinologist'},
                'thyroid': {'specialist': 'Endocrinologist'},
                'hyperthyroidism': {'specialist': 'Endocrinologist'},
                'hypothyroidism': {'specialist': 'Endocrinologist'},
                'depression': {'specialist': 'Psychiatrist'},
                'anxiety': {'specialist': 'Psychiatrist'},
                'panic disorder': {'specialist': 'Psychiatrist'},
                'skin rash': {'specialist': 'Dermatologist'},
                'eczema': {'specialist': 'Dermatologist'},
                'psoriasis': {'specialist': 'Dermatologist'},
                'dermatitis': {'specialist': 'Dermatologist'},
                'back pain': {'specialist': 'Orthopedist'},
                'arthritis': {'specialist': 'Rheumatologist'},
                'fibromyalgia': {'specialist': 'Rheumatologist'},
                'joint pain': {'specialist': 'Rheumatologist'},
                'anemia': {'specialist': 'Hematologist'},
                'allergies': {'specialist': 'Allergist'},
                'appendicitis': {'specialist': 'Emergency Medicine'},
                'concussion': {'specialist': 'Neurologist'},
                'chest pain': {'specialist': 'Cardiologist'},
                'shortness of breath': {'specialist': 'Pulmonologist'},
                'heart attack': {'specialist': 'Emergency Medicine'},
                'stroke': {'specialist': 'Emergency Medicine'},
                'seizure': {'specialist': 'Neurologist'},
                'epilepsy': {'specialist': 'Neurologist'}
            }

    def setup_audio_components(self):
        """Initialize text-to-speech and speech recognition components"""
        try:
            self.speech_engine = pyttsx3.init()
            self.recognizer = sr.Recognizer()
            print("🔊 Audio components initialized")
        except Exception as e:
            print(f"⚠️ Warning: Speech components not initialized: {e}")
            self.speech_engine = None
            self.recognizer = None

    def load_dataset(self):
        """Load and preprocess the medical dataset"""
        try:
            self.df = pd.read_csv('Symptom_dataset.csv')
            print(f"📊 Dataset loaded: {len(self.df)} cases, {len(self.df['Disease'].unique())} diseases")

            # Create symptom-disease relationship dictionary
            self.disease_symptoms = {}
            symptom_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']

            for _, row in self.df.iterrows():
                disease = row['Disease']
                if disease not in self.disease_symptoms:
                    self.disease_symptoms[disease] = {
                        'symptoms': [],
                        'age_range': set(),
                        'gender_dist': {'Male': 0, 'Female': 0},
                        'blood_pressure': set(),
                        'cholesterol': set()
                    }

                # Add symptoms that are present (marked as 'Yes')
                symptoms = [col.lower() for col in symptom_columns if row[col] == 'Yes']
                self.disease_symptoms[disease]['symptoms'].extend(symptoms)

                # Add demographic and health information
                self.disease_symptoms[disease]['age_range'].add(row['Age'])
                self.disease_symptoms[disease]['gender_dist'][row['Gender']] += 1
                self.disease_symptoms[disease]['blood_pressure'].add(row['Blood Pressure'])
                self.disease_symptoms[disease]['cholesterol'].add(row['Cholesterol Level'])

            # Clean up symptoms lists and remove duplicates
            for disease in self.disease_symptoms:
                self.disease_symptoms[disease]['symptoms'] = list(set(self.disease_symptoms[disease]['symptoms']))

            print("✅ Dataset processed successfully!")
        except Exception as e:
            print(f"⚠️ Error loading dataset: {e}")
            self.disease_symptoms = {}

    def load_medical_data(self):
        """Load and prepare medical knowledge base"""
        self.medical_data = {
            "conditions": {
                "common_cold": {
                    "symptoms": ["runny nose", "cough", "sore throat", "fever", "sneezing", "congestion"],
                    "severity": "mild",
                    "recommendations": [
                        "Rest well and get plenty of sleep",
                        "Stay hydrated with fluids",
                        "Over-the-counter cold medication if needed",
                        "Use a humidifier or breathe steam",
                        "Gargle with warm salt water"
                    ]
                },
                "flu": {
                    "symptoms": ["high fever", "body aches", "fatigue", "cough", "headache", "chills"],
                    "severity": "moderate",
                    "recommendations": [
                        "Get plenty of rest",
                        "Take fever reducers as directed",
                        "Stay hydrated with fluids",
                        "Consult doctor if symptoms worsen",
                        "Consider antiviral medication if early onset"
                    ]
                },
                "allergies": {
                    "symptoms": ["sneezing", "itchy eyes", "runny nose", "congestion", "watery eyes"],
                    "severity": "mild",
                    "recommendations": [
                        "Avoid known allergens",
                        "Take antihistamines as needed",
                        "Use air purifier",
                        "Consider allergy testing",
                        "Keep windows closed during high pollen days"
                    ]
                },
                "migraine": {
                    "symptoms": ["severe headache", "nausea", "light sensitivity", "sound sensitivity", "aura"],
                    "severity": "moderate",
                    "recommendations": [
                        "Rest in a dark, quiet room",
                        "Apply cold compress to head",
                        "Stay hydrated",
                        "Take prescribed migraine medication",
                        "Identify and avoid triggers"
                    ]
                },
                "gastroenteritis": {
                    "symptoms": ["nausea", "vomiting", "diarrhea", "stomach pain", "fever"],
                    "severity": "moderate",
                    "recommendations": [
                        "Stay hydrated with small sips of fluids",
                        "Rest and avoid solid foods initially",
                        "Try clear broths or electrolyte solutions",
                        "Gradually reintroduce bland foods",
                        "Seek medical care if severe dehydration"
                    ]
                }
            }
        }
        print("📚 Medical knowledge base loaded")

    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.supported_languages else 'en'
        except Exception as e:
            print(f"⚠️ Language detection error: {e}")
            return 'en'

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language"""
        if target_lang not in self.supported_languages:
            return text

        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return text

    def generate_audio_response(self, text: str, language: str) -> str:
        """Generate audio response in specified language"""
        try:
            # Create temporary file with a more specific path
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_audio')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Clean text for audio (remove markdown formatting)
            clean_text = text.replace('**', '').replace('*', '').replace('#', '')
            # Remove emoji and special characters for better TTS
            import re
            clean_text = re.sub(r'[^\w\s.,!?-]', '', clean_text)

            temp_file_path = os.path.join(temp_dir, f'audio_{os.getpid()}_{hash(clean_text)}.mp3')
            tts = gTTS(text=clean_text, lang=language, slow=False)
            tts.save(temp_file_path)
            return temp_file_path
        except Exception as e:
            print(f"⚠️ Error generating audio: {e}")
            return None

    def extract_symptoms_enhanced(self, user_input: str) -> Tuple[List[Symptom], Dict]:
        """Enhanced symptom extraction using both local and Gemini analysis"""
        # Get basic symptoms using local method first
        local_symptoms = self.extract_symptoms(user_input)

        # If Gemini is available, get enhanced analysis
        gemini_analysis = {}
        if self.gemini.is_configured():
            try:
                print("🧠 Using Gemini AI for symptom analysis...")
                gemini_analysis = self.gemini.analyze_symptoms(user_input)
            except Exception as e:
                print(f"⚠️ Error using Gemini analysis: {e}")

        # Combine results - prioritize Gemini analysis if available
        if gemini_analysis and gemini_analysis.get('matched_symptoms'):
            # Add Gemini-detected symptoms
            for symptom_name in gemini_analysis['matched_symptoms']:
                if not any(s.name.lower() == symptom_name.lower() for s in local_symptoms):
                    local_symptoms.append(
                        Symptom(
                            name=symptom_name.lower(),
                            severity=0.7  # Default severity for AI-detected symptoms
                        )
                    )

        return local_symptoms, gemini_analysis

    def extract_symptoms(self, user_input: str) -> List[Symptom]:
        """Extract symptoms from user input using enhanced pattern matching"""
        user_input = user_input.lower()
        symptoms = []

        # Enhanced symptom variations with more comprehensive matching
        symptom_variations = {
            "fever": ["fever", "temperature", "hot", "burning up", "chills", "feverish", "high temp"],
            "cough": ["cough", "coughing", "hack", "hacking", "persistent cough", "dry cough", "wet cough"],
            "headache": ["headache", "head pain", "migraine", "head ache", "cranial pain", "head hurts"],
            "sore throat": ["sore throat", "throat pain", "scratchy throat", "throat ache", "swollen throat"],
            "runny nose": ["runny nose", "nasal congestion", "stuffy nose", "blocked nose", "nasal discharge"],
            "fatigue": ["fatigue", "tired", "exhausted", "weakness", "weary", "worn out", "low energy"],
            "body aches": ["body aches", "muscle pain", "aches", "soreness", "joint pain", "muscle aches"],
            "sneezing": ["sneeze", "sneezing", "achoo", "sneeze fits"],
            "itchy eyes": ["itchy eyes", "eye irritation", "watery eyes", "burning eyes", "red eyes"],
            "nausea": ["nausea", "sick", "queasy", "vomiting", "throw up", "nauseated", "feel sick"],
            "diarrhea": ["diarrhea", "loose stools", "stomach upset", "loose bowels", "watery stools"],
            "chest pain": ["chest pain", "chest tightness", "chest pressure", "heart pain"],
            "shortness of breath": ["shortness of breath", "difficulty breathing", "breathless", "out of breath",
                                    "hard to breathe"],
            "dizziness": ["dizzy", "dizziness", "lightheaded", "vertigo", "spinning", "unsteady"],
            "rash": ["rash", "skin irritation", "itchy skin", "skin redness", "bumps on skin"],
            "abdominal pain": ["stomach pain", "belly ache", "abdominal pain", "tummy ache", "stomach cramps"],
            "back pain": ["back pain", "lower back pain", "spine pain", "back ache"],
            "joint pain": ["joint pain", "arthritis", "stiff joints", "swollen joints", "joint aches"],
            "insomnia": ["can't sleep", "insomnia", "trouble sleeping", "sleepless", "no sleep"],
            "loss of appetite": ["no appetite", "don't want to eat", "loss of appetite", "not hungry"],
            "constipation": ["constipated", "constipation", "can't poop", "no bowel movement"],
            "ear pain": ["ear pain", "earache", "ear hurts", "ear infection"],
            "eye pain": ["eye pain", "eyes hurt", "eye ache", "painful eyes"],
            "leg pain": ["leg pain", "leg ache", "legs hurt", "leg cramps"],
            "arm pain": ["arm pain", "arm ache", "arms hurt", "shoulder pain"]
        }

        for main_symptom, variations in symptom_variations.items():
            if any(var in user_input for var in variations):
                severity = self._analyze_severity(user_input, main_symptom)
                symptoms.append(Symptom(name=main_symptom, severity=severity))

        return symptoms

    def _analyze_severity(self, text: str, symptom: str = "") -> float:
        """Analyze the severity of symptoms based on text"""
        severity_modifiers = {
            "mild": 0.3, "slight": 0.3, "little": 0.3, "minor": 0.3, "light": 0.3,
            "moderate": 0.6, "medium": 0.6, "average": 0.6,
            "severe": 0.9, "extreme": 1.0, "intense": 0.9, "strong": 0.8,
            "very": 0.8, "really": 0.8, "incredibly": 0.9, "extremely": 1.0,
            "terrible": 0.9, "awful": 0.9, "horrible": 0.9, "excruciating": 1.0,
            "unbearable": 1.0, "crushing": 1.0, "devastating": 1.0,
            "chronic": 0.7, "persistent": 0.7, "constant": 0.8, "ongoing": 0.7,
            "sharp": 0.8, "throbbing": 0.7, "burning": 0.7, "stabbing": 0.9,
            "dull": 0.4, "aching": 0.5, "nagging": 0.5
        }

        max_severity = 0.5  # Default
        for modifier, value in severity_modifiers.items():
            if modifier in text:
                max_severity = max(max_severity, value)

        # Context-specific adjustments
        if symptom == "fever" and ("high" in text or "102" in text or "103" in text):
            max_severity = max(max_severity, 0.8)
        elif symptom == "chest pain" and ("crushing" in text or "pressure" in text):
            max_severity = max(max_severity, 0.9)

        return max_severity

    def diagnose_enhanced(self, symptoms: List[Symptom], gemini_analysis: Dict = None) -> Optional[Diagnosis]:
        """Enhanced diagnosis using Gemini AI and local knowledge"""
        if not symptoms:
            return None

        # If Gemini analysis is available and comprehensive, use it
        if gemini_analysis and gemini_analysis.get('primary_condition'):
            try:
                condition = gemini_analysis.get('primary_condition', 'Unknown condition')
                confidence = float(gemini_analysis.get('confidence', 0.7))

                # Get specialist recommendation
                specialist = gemini_analysis.get('specialist') or self._get_specialist_for_condition(condition)

                # Get recommendations (try Gemini first, then fall back)
                recommendations = gemini_analysis.get('recommendations', [])
                if not recommendations or len(recommendations) < 3:
                    # Get enhanced recommendations from Gemini
                    symptoms_text = ", ".join([s.name for s in symptoms])
                    recommendations = self.gemini.get_detailed_recommendations(condition, symptoms_text)

                return Diagnosis(
                    condition=condition,
                    confidence=confidence,
                    symptoms_matched=[s.name for s in symptoms],
                    recommendations=recommendations,
                    specialist=specialist,
                    severity=gemini_analysis.get('severity', 'moderate'),
                    urgency=gemini_analysis.get('urgency', 'routine')
                )
            except Exception as e:
                print(f"⚠️ Error processing Gemini diagnosis: {e}")

        # Fallback to local diagnosis method
        return self.diagnose_local(symptoms)

    def diagnose_local(self, symptoms: List[Symptom]) -> Optional[Diagnosis]:
        """Local diagnosis using original knowledge base and dataset"""
        if not symptoms:
            return None

        best_match = None
        highest_confidence = 0.0
        matched_symptoms = []

        # Combine knowledge base and dataset for diagnosis
        all_conditions = {**self.medical_data["conditions"]}

        # Add dataset conditions if available
        if hasattr(self, 'disease_symptoms'):
            for disease, data in self.disease_symptoms.items():
                if disease.lower() not in all_conditions:
                    all_conditions[disease.lower()] = {
                        'symptoms': data['symptoms'],
                        'severity': self._determine_severity(data),
                        'recommendations': self._generate_recommendations(data)
                    }

        for condition, data in all_conditions.items():
            confidence, matches = self._calculate_condition_match(
                symptoms, data["symptoms"]
            )

            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = condition
                matched_symptoms = matches

        if best_match:
            specialist = self._get_specialist_for_condition(best_match)

            return Diagnosis(
                condition=best_match,
                confidence=highest_confidence,
                symptoms_matched=matched_symptoms,
                recommendations=all_conditions[best_match]["recommendations"],
                specialist=specialist,
                severity=all_conditions[best_match].get("severity", "moderate")
            )

        return None

    def _get_specialist_for_condition(self, condition_name: str) -> str:
        """Get specialist recommendation for a given condition"""
        condition_lower = condition_name.lower()

        # Check our specialist data first
        if condition_lower in self.specialist_data:
            return self.specialist_data[condition_lower]['specialist']

        # Use pattern matching for common conditions
        specialist_patterns = {
            'respiratory': ['Pulmonologist',
                            ['lung', 'breath', 'cough', 'pneumonia', 'asthma', 'bronchitis', 'respiratory', 'chest',
                             'breathing']],
            'cardiac': ['Cardiologist',
                        ['heart', 'chest pain', 'hypertension', 'cardiac', 'blood pressure', 'cardiovascular']],
            'neurological': ['Neurologist',
                             ['headache', 'migraine', 'seizure', 'stroke', 'neurological', 'brain', 'concussion',
                              'epilepsy']],
            'gastro': ['Gastroenterologist',
                       ['stomach', 'digestive', 'gastro', 'intestinal', 'bowel', 'nausea', 'vomiting', 'diarrhea']],
            'skin': ['Dermatologist', ['skin', 'rash', 'dermatitis', 'eczema', 'acne', 'psoriasis', 'dermatological']],
            'mental': ['Psychiatrist', ['depression', 'anxiety', 'mental', 'psychiatric', 'panic', 'mood']],
            'orthopedic': ['Orthopedist',
                           ['bone', 'joint', 'muscle', 'back pain', 'arthritis', 'fracture', 'orthopedic']],
            'ent': ['ENT Specialist', ['ear', 'nose', 'throat', 'sinus', 'hearing', 'tinnitus', 'sinusitis']],
            'urological': ['Urologist', ['urinary', 'kidney', 'bladder', 'urine', 'urological']],
            'endocrine': ['Endocrinologist', ['diabetes', 'thyroid', 'hormone', 'endocrine', 'metabolism']],
            'emergency': ['Emergency Medicine', ['emergency', 'urgent', 'life-threatening', 'severe', 'acute']]
        }

        for specialty, (specialist, keywords) in specialist_patterns.items():
            if any(keyword in condition_lower for keyword in keywords):
                return specialist

        return 'General Practitioner'

    def _determine_severity(self, disease_data: Dict) -> str:
        """Determine disease severity based on symptoms and patient data"""
        severity_score = 0

        # Count number of symptoms
        severity_score += len(disease_data.get('symptoms', [])) * 0.2

        # Check blood pressure distribution
        if 'High' in disease_data.get('blood_pressure', set()):
            severity_score += 0.3

        # Check cholesterol distribution
        if 'High' in disease_data.get('cholesterol', set()):
            severity_score += 0.2

        if severity_score >= 0.7:
            return "severe"
        elif severity_score >= 0.4:
            return "moderate"
        else:
            return "mild"

    def _generate_recommendations(self, disease_data: Dict) -> List[str]:
        """Generate recommendations based on disease data"""
        recommendations = [
            "Consult a healthcare provider for proper diagnosis and treatment"
        ]

        if 'High' in disease_data.get('blood_pressure', set()):
            recommendations.append("Monitor blood pressure regularly")

        if 'High' in disease_data.get('cholesterol', set()):
            recommendations.append("Consider cholesterol management strategies")

        if len(disease_data.get('symptoms', [])) >= 3:
            recommendations.append("Get adequate rest and monitor symptoms closely")

        return recommendations

    def _calculate_condition_match(self, user_symptoms: List[Symptom], condition_symptoms: List[str]) -> Tuple[
        float, List[str]]:
        """Calculate how well user symptoms match a condition"""
        matched_symptoms = []
        match_score = 0.0

        for user_symptom in user_symptoms:
            for condition_symptom in condition_symptoms:
                if user_symptom.name in condition_symptom or condition_symptom in user_symptom.name:
                    matched_symptoms.append(user_symptom.name)
                    match_score += user_symptom.severity
                    break

        confidence = match_score / len(condition_symptoms) if condition_symptoms else 0.0
        return confidence, matched_symptoms

    def generate_response(self, diagnosis: Optional[Diagnosis]) -> str:
        """Generate a comprehensive user-friendly response"""
        if not diagnosis:
            return ("I need more information about your symptoms to provide a proper analysis. "
                    "Could you please describe them in more detail, including when they started and their severity?")

        # Build enhanced response
        response_parts = []

        # Main diagnosis with confidence
        if isinstance(diagnosis.confidence, float):
            confidence_text = f"{diagnosis.confidence:.0%}"
        else:
            confidence_text = f"{int(float(diagnosis.confidence) * 100)}%"

        response_parts.append(f"**🔍 Medical Analysis Results:**")
        response_parts.append(
            f"Based on your symptoms, you may have **{diagnosis.condition}** (Confidence: {confidence_text})")

        # Matched symptoms
        if diagnosis.symptoms_matched:
            response_parts.append(f"\n**📋 Symptoms Identified:** {', '.join(diagnosis.symptoms_matched)}")

        # Urgency/Severity assessment
        if diagnosis.urgency:
            urgency_icons = {
                'emergency': '🚨 EMERGENCY - Seek immediate medical attention!',
                'urgent': '⚠️ URGENT - See a doctor today',
                'semi_urgent': '📋 SEMI-URGENT - Schedule appointment within 24-48 hours',
                'routine': '📅 ROUTINE - Schedule regular appointment when convenient'
            }
            urgency_text = urgency_icons.get(diagnosis.urgency, f"📊 {diagnosis.urgency.upper()}")
            response_parts.append(f"\n**🚦 Priority Level:** {urgency_text}")
        elif diagnosis.severity:
            severity_icons = {
                'severe': '🔴 HIGH SEVERITY - Monitor closely',
                'moderate': '🟡 MODERATE SEVERITY - Keep watch',
                'mild': '🟢 MILD SEVERITY - Generally manageable'
            }
            severity_text = severity_icons.get(diagnosis.severity, f"📊 {diagnosis.severity.upper()}")
            response_parts.append(f"\n**📊 Severity Level:** {severity_text}")

        # Specialist recommendation
        if diagnosis.specialist:
            response_parts.append(f"\n**👨‍⚕️ Recommended Specialist:** {diagnosis.specialist}")

        # Detailed recommendations
        if diagnosis.recommendations:
            response_parts.append(f"\n**💡 Recommended Actions:**")
            for i, rec in enumerate(diagnosis.recommendations, 1):
                response_parts.append(f"{i}. {rec}")

        # Enhanced disclaimer with urgency-specific advice
        if diagnosis.urgency == 'emergency':
            response_parts.append(f"\n🚨 **URGENT MEDICAL ATTENTION REQUIRED** 🚨")
            response_parts.append(f"Please seek immediate emergency care or call emergency services.")
        else:
            response_parts.append(f"\n⚠️ **Important Medical Disclaimer:**")
            response_parts.append(
                f"This AI analysis is for informational purposes only and should not replace professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment.")

        return '\n'.join(response_parts)

    def process_input(self, user_input: str) -> Tuple[str, str, str]:
        """Process user input with enhanced AI-powered diagnosis"""
        print(f"🔄 Processing input: {user_input[:50]}...")

        # Detect input language
        detected_lang = self.detect_language(user_input)
        print(f"🌍 Detected language: {detected_lang}")

        # Translate to English if needed for analysis
        if detected_lang != 'en':
            english_input = self.translate_text(user_input, 'en')
            print(f"🔤 Translated to English for analysis")
        else:
            english_input = user_input

        # Enhanced symptom extraction and diagnosis with AI
        print("🔍 Extracting symptoms...")
        symptoms, gemini_analysis = self.extract_symptoms_enhanced(english_input)
        print(f"📋 Found {len(symptoms)} symptoms")

        print("🩺 Generating diagnosis...")
        diagnosis = self.diagnose_enhanced(symptoms, gemini_analysis)

        print("📝 Formatting response...")
        response = self.generate_response(diagnosis)

        # Translate response back to original language if needed
        if detected_lang != 'en':
            print(f"🔤 Translating response back to {detected_lang}")
            translated_response = self.translate_text(response, detected_lang)
        else:
            translated_response = response

        # Generate audio response
        print("🔊 Generating audio response...")
        audio_path = self.generate_audio_response(translated_response, detected_lang)

        print("✅ Processing complete!")
        return translated_response, detected_lang, audio_path