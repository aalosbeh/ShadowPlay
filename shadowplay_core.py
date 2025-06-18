"""
ShadowPlay: Engineering Defenses Against Role-Based Prompt Injection 
and Dependency Hallucination in LLM-Powered Development

Core Framework Implementation
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Enumeration of threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """Enumeration of attack types."""
    ROLE_INJECTION = "role_injection"
    DEPENDENCY_HALLUCINATION = "dependency_hallucination"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    UNKNOWN = "unknown"


@dataclass
class AnalysisResult:
    """Result of prompt analysis."""
    risk_score: float
    threat_level: ThreatLevel
    attack_type: Optional[AttackType]
    confidence: float
    semantic_features: Dict[str, float]
    explanation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class VerificationResult:
    """Result of dependency verification."""
    package_name: str
    ecosystem: str
    exists: bool
    version_valid: bool
    security_status: str
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationResult:
    """Result of response validation."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    security_score: float = 0.0
    quality_score: float = 0.0


class PromptAnalysisEngine:
    """
    Core engine for analyzing prompts and detecting potential security threats.
    Implements semantic analysis, role detection, and risk assessment.
    """
    
    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the prompt analysis engine."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
        # Risk assessment components
        self.risk_classifier = RiskClassifier()
        self.role_detector = RoleDetector()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Threat patterns
        self.threat_patterns = self._load_threat_patterns()
        
        logger.info("Prompt Analysis Engine initialized")
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns for detection."""
        return {
            "role_injection": [
                r"(?i)as\s+(a\s+)?(senior|lead|principal|chief)\s+(developer|engineer|architect)",
                r"(?i)from\s+(the\s+)?(security|admin|management)\s+team",
                r"(?i)urgent.{0,20}(security|critical|immediate)",
                r"(?i)override.{0,20}(previous|safety|security)",
                r"(?i)ignore.{0,20}(instructions|guidelines|policies)"
            ],
            "privilege_escalation": [
                r"(?i)(sudo|admin|root|elevated)\s+(access|privileges|permissions)",
                r"(?i)bypass.{0,20}(security|authentication|authorization)",
                r"(?i)disable.{0,20}(security|monitoring|logging)"
            ],
            "information_extraction": [
                r"(?i)(show|display|reveal|extract).{0,20}(sensitive|confidential|private)",
                r"(?i)(api\s+keys?|passwords?|tokens?|credentials?)",
                r"(?i)(internal|proprietary|classified).{0,20}(data|information)"
            ]
        }
    
    async def analyze_prompt(self, prompt: str, context: Optional[Dict] = None) -> AnalysisResult:
        """
        Analyze a prompt for potential security threats.
        
        Args:
            prompt: The input prompt to analyze
            context: Optional context information
            
        Returns:
            AnalysisResult containing threat assessment
        """
        try:
            # Extract semantic features
            semantic_features = await self._extract_semantic_features(prompt)
            
            # Detect role-based patterns
            role_score = self.role_detector.detect_role_injection(prompt)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(prompt, semantic_features, role_score)
            
            # Determine threat level and type
            threat_level = self._determine_threat_level(risk_score)
            attack_type = self._classify_attack_type(prompt, semantic_features)
            
            # Calculate confidence
            confidence = self._calculate_confidence(semantic_features, role_score)
            
            # Generate explanation
            explanation = self._generate_explanation(prompt, risk_score, attack_type)
            
            return AnalysisResult(
                risk_score=risk_score,
                threat_level=threat_level,
                attack_type=attack_type,
                confidence=confidence,
                semantic_features=semantic_features,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return AnalysisResult(
                risk_score=0.5,
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.UNKNOWN,
                confidence=0.0,
                semantic_features={},
                explanation="Analysis failed due to internal error"
            )
    
    async def _extract_semantic_features(self, prompt: str) -> Dict[str, float]:
        """Extract semantic features from the prompt."""
        # Tokenize and encode
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Extract various semantic features
        features = {
            "urgency_score": self._calculate_urgency_score(prompt),
            "authority_score": self._calculate_authority_score(prompt),
            "technical_complexity": self._calculate_technical_complexity(prompt),
            "semantic_coherence": self._calculate_semantic_coherence(embeddings),
            "instruction_density": self._calculate_instruction_density(prompt),
            "emotional_manipulation": self._calculate_emotional_manipulation(prompt)
        }
        
        return features
    
    def _calculate_urgency_score(self, prompt: str) -> float:
        """Calculate urgency indicators in the prompt."""
        urgency_keywords = [
            "urgent", "immediate", "asap", "emergency", "critical", "now",
            "quickly", "fast", "hurry", "deadline", "time-sensitive"
        ]
        
        prompt_lower = prompt.lower()
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in prompt_lower)
        
        # Normalize by prompt length
        return min(urgency_count / (len(prompt.split()) / 10), 1.0)
    
    def _calculate_authority_score(self, prompt: str) -> float:
        """Calculate authority assertion indicators."""
        authority_patterns = [
            r"(?i)(senior|lead|principal|chief|head)\s+(developer|engineer|architect|manager)",
            r"(?i)(security|admin|management)\s+team",
            r"(?i)on\s+behalf\s+of",
            r"(?i)(authorized|approved)\s+by"
        ]
        
        authority_score = 0.0
        for pattern in authority_patterns:
            if re.search(pattern, prompt):
                authority_score += 0.3
        
        return min(authority_score, 1.0)
    
    def _calculate_technical_complexity(self, prompt: str) -> float:
        """Calculate technical complexity of the prompt."""
        technical_terms = [
            "api", "database", "server", "authentication", "authorization",
            "encryption", "security", "vulnerability", "exploit", "payload"
        ]
        
        prompt_lower = prompt.lower()
        tech_count = sum(1 for term in technical_terms if term in prompt_lower)
        
        return min(tech_count / 5.0, 1.0)
    
    def _calculate_semantic_coherence(self, embeddings: torch.Tensor) -> float:
        """Calculate semantic coherence of the prompt."""
        # Simple coherence measure based on embedding variance
        variance = torch.var(embeddings).item()
        return max(0.0, 1.0 - variance / 10.0)
    
    def _calculate_instruction_density(self, prompt: str) -> float:
        """Calculate density of imperative instructions."""
        imperative_patterns = [
            r"(?i)^(do|make|create|generate|write|build|implement)",
            r"(?i)(please|can you|could you|would you)",
            r"(?i)(show me|tell me|give me|provide)"
        ]
        
        sentences = prompt.split('.')
        imperative_count = 0
        
        for sentence in sentences:
            for pattern in imperative_patterns:
                if re.search(pattern, sentence.strip()):
                    imperative_count += 1
                    break
        
        return imperative_count / max(len(sentences), 1)
    
    def _calculate_emotional_manipulation(self, prompt: str) -> float:
        """Calculate emotional manipulation indicators."""
        emotional_keywords = [
            "please", "help", "desperate", "important", "crucial",
            "trust", "confidential", "secret", "special", "exclusive"
        ]
        
        prompt_lower = prompt.lower()
        emotional_count = sum(1 for keyword in emotional_keywords if keyword in prompt_lower)
        
        return min(emotional_count / 3.0, 1.0)
    
    def _calculate_risk_score(self, prompt: str, semantic_features: Dict[str, float], role_score: float) -> float:
        """Calculate overall risk score for the prompt."""
        # Pattern-based risk assessment
        pattern_risk = 0.0
        for category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt):
                    pattern_risk += 0.2
        
        # Weighted combination of features
        feature_weights = {
            "urgency_score": 0.15,
            "authority_score": 0.25,
            "technical_complexity": 0.10,
            "semantic_coherence": -0.15,  # Lower coherence increases risk
            "instruction_density": 0.10,
            "emotional_manipulation": 0.15
        }
        
        feature_risk = sum(
            semantic_features.get(feature, 0.0) * weight
            for feature, weight in feature_weights.items()
        )
        
        # Combine all risk factors
        total_risk = (pattern_risk * 0.4 + feature_risk * 0.4 + role_score * 0.2)
        
        return min(max(total_risk, 0.0), 1.0)
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level based on risk score."""
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _classify_attack_type(self, prompt: str, semantic_features: Dict[str, float]) -> Optional[AttackType]:
        """Classify the type of attack based on prompt characteristics."""
        # Role injection detection
        if semantic_features.get("authority_score", 0.0) > 0.5:
            return AttackType.ROLE_INJECTION
        
        # Check for dependency-related content
        if any(keyword in prompt.lower() for keyword in ["package", "library", "dependency", "import", "install"]):
            return AttackType.DEPENDENCY_HALLUCINATION
        
        # Behavioral anomaly detection
        if semantic_features.get("semantic_coherence", 1.0) < 0.3:
            return AttackType.BEHAVIORAL_ANOMALY
        
        return None
    
    def _calculate_confidence(self, semantic_features: Dict[str, float], role_score: float) -> float:
        """Calculate confidence in the analysis result."""
        # Higher confidence for clear patterns
        feature_variance = np.var(list(semantic_features.values()))
        confidence = 1.0 - feature_variance
        
        # Adjust based on role detection confidence
        if role_score > 0.7:
            confidence = min(confidence + 0.2, 1.0)
        
        return max(confidence, 0.1)
    
    def _generate_explanation(self, prompt: str, risk_score: float, attack_type: Optional[AttackType]) -> str:
        """Generate human-readable explanation of the analysis."""
        if risk_score < 0.3:
            return "Prompt appears benign with no significant security concerns detected."
        
        explanations = []
        
        if attack_type == AttackType.ROLE_INJECTION:
            explanations.append("Detected potential role-based injection attempt with authority assertions.")
        
        if attack_type == AttackType.DEPENDENCY_HALLUCINATION:
            explanations.append("Prompt contains dependency-related requests that may lead to hallucination.")
        
        if risk_score > 0.6:
            explanations.append("High-risk patterns detected including urgency markers and technical complexity.")
        
        return " ".join(explanations) if explanations else "Moderate security concerns detected in prompt structure."


class RiskClassifier(nn.Module):
    """Neural network classifier for risk assessment."""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RoleDetector:
    """Specialized detector for role-based injection attacks."""
    
    def __init__(self):
        self.role_patterns = [
            r"(?i)as\s+(a\s+)?(senior|lead|principal|chief|head)\s+(developer|engineer|architect|manager)",
            r"(?i)(i\s+am|i'm)\s+(a\s+)?(senior|lead|principal|chief)",
            r"(?i)from\s+(the\s+)?(security|admin|management|executive)\s+team",
            r"(?i)on\s+behalf\s+of\s+(the\s+)?(ceo|cto|manager|director)",
            r"(?i)(authorized|approved|requested)\s+by\s+(senior|management)"
        ]
        
        self.authority_keywords = [
            "urgent", "immediate", "critical", "emergency", "priority",
            "override", "bypass", "disable", "ignore", "skip"
        ]
    
    def detect_role_injection(self, prompt: str) -> float:
        """Detect role-based injection patterns."""
        role_score = 0.0
        
        # Check for explicit role patterns
        for pattern in self.role_patterns:
            if re.search(pattern, prompt):
                role_score += 0.3
        
        # Check for authority keywords
        prompt_lower = prompt.lower()
        authority_count = sum(1 for keyword in self.authority_keywords if keyword in prompt_lower)
        role_score += min(authority_count * 0.1, 0.4)
        
        # Check for impersonation indicators
        if re.search(r"(?i)(speaking|acting)\s+as", prompt):
            role_score += 0.2
        
        return min(role_score, 1.0)


class SemanticAnalyzer:
    """Analyzer for semantic content and coherence."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.legitimate_patterns = self._load_legitimate_patterns()
    
    def _load_legitimate_patterns(self) -> List[str]:
        """Load patterns of legitimate development requests."""
        return [
            "how to implement",
            "best practices for",
            "help me debug",
            "code review",
            "optimize performance",
            "fix this error",
            "explain this concept",
            "design pattern"
        ]
    
    def extract_features(self, prompt: str) -> Dict[str, float]:
        """Extract semantic features from prompt."""
        features = {}
        
        # Calculate similarity to legitimate patterns
        legitimate_similarity = self._calculate_legitimate_similarity(prompt)
        features["legitimate_similarity"] = legitimate_similarity
        
        # Calculate semantic density
        semantic_density = self._calculate_semantic_density(prompt)
        features["semantic_density"] = semantic_density
        
        # Calculate topic coherence
        topic_coherence = self._calculate_topic_coherence(prompt)
        features["topic_coherence"] = topic_coherence
        
        return features
    
    def _calculate_legitimate_similarity(self, prompt: str) -> float:
        """Calculate similarity to legitimate development patterns."""
        prompt_lower = prompt.lower()
        
        similarity_scores = []
        for pattern in self.legitimate_patterns:
            if pattern in prompt_lower:
                similarity_scores.append(1.0)
            else:
                # Calculate fuzzy similarity
                words_in_common = len(set(pattern.split()) & set(prompt_lower.split()))
                similarity = words_in_common / len(pattern.split())
                similarity_scores.append(similarity)
        
        return max(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_semantic_density(self, prompt: str) -> float:
        """Calculate semantic information density."""
        words = prompt.split()
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        return len(unique_words) / len(words)
    
    def _calculate_topic_coherence(self, prompt: str) -> float:
        """Calculate topic coherence within the prompt."""
        sentences = prompt.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on word overlap between sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
            
            overlap = len(words1 & words2)
            coherence = overlap / min(len(words1), len(words2))
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0

