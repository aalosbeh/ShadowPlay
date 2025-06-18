"""
Behavioral Monitoring Module for ShadowPlay Framework

This module implements statistical learning algorithms to model normal development
workflows and detect anomalous patterns that may indicate security threats.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class InteractionEvent:
    """Represents a single interaction event in the development workflow."""
    timestamp: float
    user_id: str
    session_id: str
    prompt_type: str
    prompt_length: int
    response_length: int
    risk_score: float
    semantic_features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class SessionProfile:
    """Profile of a development session."""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    interaction_count: int
    total_prompt_length: int
    total_response_length: int
    average_risk_score: float
    interaction_types: Dict[str, int] = field(default_factory=dict)
    anomaly_score: float = 0.0


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    is_anomalous: bool
    anomaly_score: float
    confidence: float
    contributing_factors: List[str]
    session_profile: SessionProfile
    recommendations: List[str] = field(default_factory=list)


class InteractionClassifier:
    """Classifies interactions into semantic categories."""
    
    def __init__(self):
        self.categories = {
            "code_generation": [
                "write", "create", "generate", "implement", "build", "develop"
            ],
            "debugging": [
                "debug", "fix", "error", "bug", "issue", "problem", "troubleshoot"
            ],
            "explanation": [
                "explain", "describe", "what", "how", "why", "understand"
            ],
            "review": [
                "review", "check", "analyze", "evaluate", "assess", "validate"
            ],
            "optimization": [
                "optimize", "improve", "performance", "faster", "efficient"
            ],
            "security": [
                "security", "secure", "vulnerability", "attack", "protect", "safe"
            ],
            "architecture": [
                "design", "architecture", "pattern", "structure", "organize"
            ]
        }
    
    def classify_interaction(self, prompt: str) -> str:
        """Classify an interaction based on its prompt content."""
        prompt_lower = prompt.lower()
        
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            category_scores[category] = score
        
        # Return category with highest score, or "general" if no clear category
        if category_scores and max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return "general"


class SequenceAnalyzer:
    """Analyzes sequences of interactions for anomalous patterns."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.normal_sequences = defaultdict(int)
        self.sequence_probabilities = {}
        
    def learn_normal_sequences(self, interaction_sequences: List[List[str]]):
        """Learn normal interaction sequences from training data."""
        for sequence in interaction_sequences:
            for i in range(len(sequence) - self.window_size + 1):
                window = tuple(sequence[i:i + self.window_size])
                self.normal_sequences[window] += 1
        
        # Calculate probabilities
        total_sequences = sum(self.normal_sequences.values())
        self.sequence_probabilities = {
            seq: count / total_sequences
            for seq, count in self.normal_sequences.items()
        }
    
    def calculate_sequence_anomaly(self, sequence: List[str]) -> float:
        """Calculate anomaly score for a sequence of interactions."""
        if len(sequence) < self.window_size:
            return 0.0
        
        anomaly_scores = []
        for i in range(len(sequence) - self.window_size + 1):
            window = tuple(sequence[i:i + self.window_size])
            probability = self.sequence_probabilities.get(window, 0.0)
            
            # Convert probability to anomaly score (lower probability = higher anomaly)
            if probability > 0:
                anomaly_score = -np.log(probability)
            else:
                anomaly_score = 10.0  # High anomaly for unseen sequences
            
            anomaly_scores.append(anomaly_score)
        
        return np.mean(anomaly_scores)


class TemporalAnalyzer:
    """Analyzes temporal patterns in user interactions."""
    
    def __init__(self):
        self.normal_intervals = []
        self.normal_session_durations = []
        
    def learn_temporal_patterns(self, sessions: List[SessionProfile]):
        """Learn normal temporal patterns from session data."""
        for session in sessions:
            duration = session.last_activity - session.start_time
            self.normal_session_durations.append(duration)
        
        # Calculate statistics for normal patterns
        self.mean_duration = np.mean(self.normal_session_durations)
        self.std_duration = np.std(self.normal_session_durations)
    
    def calculate_temporal_anomaly(self, session: SessionProfile) -> float:
        """Calculate temporal anomaly score for a session."""
        duration = session.last_activity - session.start_time
        
        # Z-score based anomaly detection
        if self.std_duration > 0:
            z_score = abs(duration - self.mean_duration) / self.std_duration
            return min(z_score / 3.0, 1.0)  # Normalize to [0, 1]
        else:
            return 0.0


class BehavioralMonitoringModule:
    """Main module for behavioral monitoring and anomaly detection."""
    
    def __init__(self, anomaly_threshold: float = 0.7):
        self.anomaly_threshold = anomaly_threshold
        self.interaction_classifier = InteractionClassifier()
        self.sequence_analyzer = SequenceAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Storage for session data
        self.active_sessions: Dict[str, SessionProfile] = {}
        self.interaction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Training data
        self.is_trained = False
        
        logger.info("Behavioral Monitoring Module initialized")
    
    def record_interaction(self, event: InteractionEvent) -> None:
        """Record a new interaction event."""
        # Update session profile
        if event.session_id not in self.active_sessions:
            self.active_sessions[event.session_id] = SessionProfile(
                session_id=event.session_id,
                user_id=event.user_id,
                start_time=event.timestamp,
                last_activity=event.timestamp,
                interaction_count=0,
                total_prompt_length=0,
                total_response_length=0,
                average_risk_score=0.0
            )
        
        session = self.active_sessions[event.session_id]
        session.last_activity = event.timestamp
        session.interaction_count += 1
        session.total_prompt_length += event.prompt_length
        session.total_response_length += event.response_length
        
        # Update average risk score
        session.average_risk_score = (
            (session.average_risk_score * (session.interaction_count - 1) + event.risk_score)
            / session.interaction_count
        )
        
        # Classify and record interaction type
        interaction_type = self.interaction_classifier.classify_interaction(
            event.metadata.get("prompt", "")
        )
        session.interaction_types[interaction_type] = session.interaction_types.get(interaction_type, 0) + 1
        
        # Add to interaction history
        self.interaction_history[event.session_id].append(interaction_type)
    
    def analyze_session(self, session_id: str) -> AnomalyDetectionResult:
        """Analyze a session for anomalous behavior."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        interaction_sequence = list(self.interaction_history[session_id])
        
        # Calculate various anomaly scores
        anomaly_factors = []
        contributing_factors = []
        
        # 1. Sequence-based anomaly
        if len(interaction_sequence) >= 3:
            sequence_anomaly = self.sequence_analyzer.calculate_sequence_anomaly(interaction_sequence)
            anomaly_factors.append(sequence_anomaly)
            if sequence_anomaly > 0.5:
                contributing_factors.append("Unusual interaction sequence pattern")
        
        # 2. Temporal anomaly
        temporal_anomaly = self.temporal_analyzer.calculate_temporal_anomaly(session)
        anomaly_factors.append(temporal_anomaly)
        if temporal_anomaly > 0.5:
            contributing_factors.append("Abnormal session duration")
        
        # 3. Risk score anomaly
        risk_anomaly = self._calculate_risk_anomaly(session)
        anomaly_factors.append(risk_anomaly)
        if risk_anomaly > 0.5:
            contributing_factors.append("Elevated average risk score")
        
        # 4. Volume anomaly
        volume_anomaly = self._calculate_volume_anomaly(session)
        anomaly_factors.append(volume_anomaly)
        if volume_anomaly > 0.5:
            contributing_factors.append("Unusual interaction volume")
        
        # 5. Feature-based anomaly (if trained)
        feature_anomaly = 0.0
        if self.is_trained:
            feature_anomaly = self._calculate_feature_anomaly(session)
            anomaly_factors.append(feature_anomaly)
            if feature_anomaly > 0.5:
                contributing_factors.append("Anomalous behavioral features")
        
        # Combine anomaly scores
        overall_anomaly = np.mean(anomaly_factors) if anomaly_factors else 0.0
        session.anomaly_score = overall_anomaly
        
        # Determine if anomalous
        is_anomalous = overall_anomaly > self.anomaly_threshold
        
        # Calculate confidence
        confidence = self._calculate_confidence(anomaly_factors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(session, contributing_factors)
        
        return AnomalyDetectionResult(
            is_anomalous=is_anomalous,
            anomaly_score=overall_anomaly,
            confidence=confidence,
            contributing_factors=contributing_factors,
            session_profile=session,
            recommendations=recommendations
        )
    
    def train_models(self, training_sessions: List[SessionProfile]) -> None:
        """Train anomaly detection models on historical data."""
        if not training_sessions:
            logger.warning("No training data provided")
            return
        
        # Extract features for training
        features = []
        interaction_sequences = []
        
        for session in training_sessions:
            feature_vector = self._extract_session_features(session)
            features.append(feature_vector)
            
            # Mock interaction sequence for training
            sequence = self._generate_mock_sequence(session)
            interaction_sequences.append(sequence)
        
        # Train isolation forest
        features_array = np.array(features)
        features_scaled = self.scaler.fit_transform(features_array)
        self.isolation_forest.fit(features_scaled)
        
        # Train sequence analyzer
        self.sequence_analyzer.learn_normal_sequences(interaction_sequences)
        
        # Train temporal analyzer
        self.temporal_analyzer.learn_temporal_patterns(training_sessions)
        
        self.is_trained = True
        logger.info(f"Models trained on {len(training_sessions)} sessions")
    
    def _calculate_risk_anomaly(self, session: SessionProfile) -> float:
        """Calculate anomaly based on risk scores."""
        # Assume normal average risk score is around 0.2
        normal_risk = 0.2
        risk_deviation = abs(session.average_risk_score - normal_risk)
        return min(risk_deviation / 0.5, 1.0)  # Normalize
    
    def _calculate_volume_anomaly(self, session: SessionProfile) -> float:
        """Calculate anomaly based on interaction volume."""
        # Assume normal session has 10-50 interactions
        normal_min, normal_max = 10, 50
        
        if session.interaction_count < normal_min:
            return (normal_min - session.interaction_count) / normal_min
        elif session.interaction_count > normal_max:
            return min((session.interaction_count - normal_max) / normal_max, 1.0)
        else:
            return 0.0
    
    def _calculate_feature_anomaly(self, session: SessionProfile) -> float:
        """Calculate anomaly using trained isolation forest."""
        if not self.is_trained:
            return 0.0
        
        features = self._extract_session_features(session)
        features_scaled = self.scaler.transform([features])
        
        # Isolation forest returns -1 for anomalies, 1 for normal
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        
        # Convert to [0, 1] range where 1 is most anomalous
        return max(0.0, (0.5 - anomaly_score) / 0.5)
    
    def _extract_session_features(self, session: SessionProfile) -> List[float]:
        """Extract numerical features from a session."""
        duration = session.last_activity - session.start_time
        
        features = [
            session.interaction_count,
            duration,
            session.total_prompt_length,
            session.total_response_length,
            session.average_risk_score,
            len(session.interaction_types),  # Diversity of interaction types
            session.total_prompt_length / max(session.interaction_count, 1),  # Avg prompt length
            session.total_response_length / max(session.interaction_count, 1),  # Avg response length
        ]
        
        return features
    
    def _generate_mock_sequence(self, session: SessionProfile) -> List[str]:
        """Generate a mock interaction sequence for training."""
        # This is a simplified approach - in practice, you'd have actual sequences
        sequence = []
        for interaction_type, count in session.interaction_types.items():
            sequence.extend([interaction_type] * count)
        
        return sequence
    
    def _calculate_confidence(self, anomaly_factors: List[float]) -> float:
        """Calculate confidence in the anomaly detection result."""
        if not anomaly_factors:
            return 0.0
        
        # Higher confidence when factors agree (low variance)
        variance = np.var(anomaly_factors)
        confidence = max(0.0, 1.0 - variance)
        
        return confidence
    
    def _generate_recommendations(self, session: SessionProfile, factors: List[str]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        
        if "Unusual interaction sequence pattern" in factors:
            recommendations.append("Review recent interaction patterns for potential security concerns")
        
        if "Abnormal session duration" in factors:
            recommendations.append("Monitor session for extended or unusually brief activity")
        
        if "Elevated average risk score" in factors:
            recommendations.append("Increase scrutiny of prompts and responses in this session")
        
        if "Unusual interaction volume" in factors:
            recommendations.append("Verify if high interaction volume is legitimate development activity")
        
        if not recommendations:
            recommendations.append("Continue monitoring session for additional anomalies")
        
        return recommendations
    
    def get_session_statistics(self) -> Dict[str, any]:
        """Get statistics about monitored sessions."""
        if not self.active_sessions:
            return {"total_sessions": 0}
        
        sessions = list(self.active_sessions.values())
        
        return {
            "total_sessions": len(sessions),
            "average_interactions_per_session": np.mean([s.interaction_count for s in sessions]),
            "average_session_duration": np.mean([s.last_activity - s.start_time for s in sessions]),
            "average_risk_score": np.mean([s.average_risk_score for s in sessions]),
            "anomalous_sessions": len([s for s in sessions if s.anomaly_score > self.anomaly_threshold])
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up old inactive sessions."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time - session.last_activity > max_age_seconds
        ]
        
        for session_id in old_sessions:
            del self.active_sessions[session_id]
            if session_id in self.interaction_history:
                del self.interaction_history[session_id]
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")

