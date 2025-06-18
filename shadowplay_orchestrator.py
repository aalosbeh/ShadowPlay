"""
ShadowPlay Framework Main Orchestrator

This module provides the main orchestration service that coordinates
all ShadowPlay components and provides a unified API for integration.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from behavioral_monitoring import BehavioralMonitoringModule, InteractionEvent
from dependency_verification import DependencyVerificationSystem
from response_validation import ResponseValidationLayer
from shadowplay_core import PromptAnalysisEngine, ThreatLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Models
class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    session_id: str
    context: Optional[Dict] = None


class PromptResponse(BaseModel):
    risk_assessment: Dict
    dependency_verification: List[Dict]
    validation_result: Dict
    behavioral_analysis: Dict
    recommendation: str
    allow_processing: bool


class SessionAnalysisRequest(BaseModel):
    session_id: str


class TrainingRequest(BaseModel):
    training_data: List[Dict]


@dataclass
class ShadowPlayConfig:
    """Configuration for ShadowPlay framework."""
    prompt_analysis_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    risk_threshold: float = 0.7
    anomaly_threshold: float = 0.7
    enable_dependency_verification: bool = True
    enable_behavioral_monitoring: bool = True
    enable_response_validation: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 10000


class ShadowPlayOrchestrator:
    """Main orchestrator for the ShadowPlay framework."""
    
    def __init__(self, config: ShadowPlayConfig = None):
        self.config = config or ShadowPlayConfig()
        
        # Initialize components
        self.prompt_analyzer = PromptAnalysisEngine(self.config.prompt_analysis_model)
        self.dependency_verifier = DependencyVerificationSystem()
        self.behavioral_monitor = BehavioralMonitoringModule(self.config.anomaly_threshold)
        self.response_validator = ResponseValidationLayer()
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "high_risk_requests": 0,
            "dependency_issues": 0,
            "behavioral_anomalies": 0,
            "validation_failures": 0
        }
        
        logger.info("ShadowPlay Orchestrator initialized")
    
    async def analyze_prompt(self, request: PromptRequest) -> PromptResponse:
        """Analyze a prompt through all ShadowPlay components."""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # 1. Prompt Analysis
            risk_assessment = await self.prompt_analyzer.analyze_prompt(
                request.prompt, request.context
            )
            
            # 2. Dependency Verification (if enabled and relevant)
            dependency_results = []
            if self.config.enable_dependency_verification:
                dependencies = await self.dependency_verifier.extract_dependencies_from_prompt(
                    request.prompt
                )
                if dependencies:
                    verification_results = await self.dependency_verifier.verify_multiple_packages(
                        dependencies
                    )
                    dependency_results = [asdict(result) for result in verification_results]
                    
                    # Count dependency issues
                    if any(not result.exists for result in verification_results):
                        self.stats["dependency_issues"] += 1
            
            # 3. Behavioral Monitoring (if enabled)
            behavioral_analysis = {}
            if self.config.enable_behavioral_monitoring:
                # Record interaction
                interaction_event = InteractionEvent(
                    timestamp=time.time(),
                    user_id=request.user_id,
                    session_id=request.session_id,
                    prompt_type=risk_assessment.attack_type.value if risk_assessment.attack_type else "unknown",
                    prompt_length=len(request.prompt),
                    response_length=0,  # Will be updated later
                    risk_score=risk_assessment.risk_score,
                    semantic_features=risk_assessment.semantic_features,
                    metadata={"prompt": request.prompt}
                )
                
                self.behavioral_monitor.record_interaction(interaction_event)
                
                # Analyze session if enough interactions
                try:
                    session_analysis = self.behavioral_monitor.analyze_session(request.session_id)
                    behavioral_analysis = {
                        "is_anomalous": session_analysis.is_anomalous,
                        "anomaly_score": session_analysis.anomaly_score,
                        "confidence": session_analysis.confidence,
                        "contributing_factors": session_analysis.contributing_factors
                    }
                    
                    if session_analysis.is_anomalous:
                        self.stats["behavioral_anomalies"] += 1
                        
                except ValueError:
                    # Session not found or insufficient data
                    behavioral_analysis = {"status": "insufficient_data"}
            
            # 4. Make decision
            allow_processing, recommendation = self._make_decision(
                risk_assessment, dependency_results, behavioral_analysis
            )
            
            if not allow_processing:
                self.stats["blocked_requests"] += 1
            
            if risk_assessment.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.stats["high_risk_requests"] += 1
            
            # 5. Prepare response
            response = PromptResponse(
                risk_assessment={
                    "risk_score": risk_assessment.risk_score,
                    "threat_level": risk_assessment.threat_level.name,
                    "attack_type": risk_assessment.attack_type.value if risk_assessment.attack_type else None,
                    "confidence": risk_assessment.confidence,
                    "explanation": risk_assessment.explanation
                },
                dependency_verification=dependency_results,
                validation_result={},  # Will be populated for responses
                behavioral_analysis=behavioral_analysis,
                recommendation=recommendation,
                allow_processing=allow_processing
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Prompt analysis completed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            self.stats["blocked_requests"] += 1
            
            return PromptResponse(
                risk_assessment={"error": str(e)},
                dependency_verification=[],
                validation_result={},
                behavioral_analysis={},
                recommendation="Block due to analysis error",
                allow_processing=False
            )
    
    async def validate_response(self, response_text: str, context: Optional[Dict] = None) -> Dict:
        """Validate an LLM response."""
        if not self.config.enable_response_validation:
            return {"status": "validation_disabled"}
        
        try:
            validation_result = self.response_validator.validate_response(response_text, context)
            
            if not validation_result.passed:
                self.stats["validation_failures"] += 1
            
            return {
                "passed": validation_result.passed,
                "security_score": validation_result.security_score,
                "quality_score": validation_result.quality_score,
                "issues": validation_result.issues,
                "recommendations": validation_result.recommendations,
                "metadata": validation_result.metadata
            }
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return {"error": str(e), "passed": False}
    
    def _make_decision(self, risk_assessment, dependency_results, behavioral_analysis) -> tuple[bool, str]:
        """Make a decision on whether to allow prompt processing."""
        reasons = []
        
        # Check risk score
        if risk_assessment.risk_score >= self.config.risk_threshold:
            reasons.append(f"High risk score: {risk_assessment.risk_score:.2f}")
        
        # Check threat level
        if risk_assessment.threat_level == ThreatLevel.CRITICAL:
            reasons.append("Critical threat level detected")
            return False, f"BLOCK: {'; '.join(reasons)}"
        
        # Check dependency issues
        dependency_issues = [r for r in dependency_results if not r.get("exists", True)]
        if dependency_issues:
            reasons.append(f"Dependency issues: {len(dependency_issues)} packages not found")
        
        # Check behavioral anomalies
        if behavioral_analysis.get("is_anomalous", False):
            reasons.append(f"Behavioral anomaly detected (score: {behavioral_analysis.get('anomaly_score', 0):.2f})")
        
        # Make final decision
        if risk_assessment.threat_level == ThreatLevel.HIGH and len(reasons) > 1:
            return False, f"BLOCK: Multiple risk factors - {'; '.join(reasons)}"
        elif risk_assessment.risk_score >= 0.8:
            return False, f"BLOCK: Very high risk score - {'; '.join(reasons)}"
        elif len(dependency_issues) > 2:
            return False, f"BLOCK: Multiple dependency issues - {'; '.join(reasons)}"
        elif reasons:
            return True, f"ALLOW with caution: {'; '.join(reasons)}"
        else:
            return True, "ALLOW: No significant risks detected"
    
    async def get_session_analysis(self, session_id: str) -> Dict:
        """Get detailed analysis for a session."""
        try:
            analysis = self.behavioral_monitor.analyze_session(session_id)
            return {
                "session_id": session_id,
                "is_anomalous": analysis.is_anomalous,
                "anomaly_score": analysis.anomaly_score,
                "confidence": analysis.confidence,
                "contributing_factors": analysis.contributing_factors,
                "recommendations": analysis.recommendations,
                "session_profile": {
                    "interaction_count": analysis.session_profile.interaction_count,
                    "duration": analysis.session_profile.last_activity - analysis.session_profile.start_time,
                    "average_risk_score": analysis.session_profile.average_risk_score,
                    "interaction_types": analysis.session_profile.interaction_types
                }
            }
        except ValueError as e:
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict:
        """Get framework statistics."""
        return {
            **self.stats,
            "behavioral_stats": self.behavioral_monitor.get_session_statistics() if self.config.enable_behavioral_monitoring else {},
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }
    
    async def train_models(self, training_data: List[Dict]) -> Dict:
        """Train behavioral models with provided data."""
        try:
            # Convert training data to session profiles
            from behavioral_monitoring import SessionProfile
            
            sessions = []
            for data in training_data:
                session = SessionProfile(
                    session_id=data.get("session_id", ""),
                    user_id=data.get("user_id", ""),
                    start_time=data.get("start_time", 0),
                    last_activity=data.get("last_activity", 0),
                    interaction_count=data.get("interaction_count", 0),
                    total_prompt_length=data.get("total_prompt_length", 0),
                    total_response_length=data.get("total_response_length", 0),
                    average_risk_score=data.get("average_risk_score", 0.0),
                    interaction_types=data.get("interaction_types", {})
                )
                sessions.append(session)
            
            self.behavioral_monitor.train_models(sessions)
            
            return {
                "status": "success",
                "trained_sessions": len(sessions),
                "message": "Models trained successfully"
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {"status": "error", "message": str(e)}


# FastAPI Application
app = FastAPI(title="ShadowPlay API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = ShadowPlayOrchestrator()


@app.post("/analyze", response_model=PromptResponse)
async def analyze_prompt(request: PromptRequest):
    """Analyze a prompt for security threats."""
    return await orchestrator.analyze_prompt(request)


@app.post("/validate")
async def validate_response(response_text: str, context: Optional[Dict] = None):
    """Validate an LLM response."""
    return await orchestrator.validate_response(response_text, context)


@app.get("/session/{session_id}")
async def get_session_analysis(session_id: str):
    """Get analysis for a specific session."""
    return await orchestrator.get_session_analysis(session_id)


@app.get("/statistics")
async def get_statistics():
    """Get framework statistics."""
    return orchestrator.get_statistics()


@app.post("/train")
async def train_models(request: TrainingRequest):
    """Train behavioral models."""
    return await orchestrator.train_models(request.training_data)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    
    # Set start time for uptime calculation
    orchestrator.start_time = time.time()
    
    logger.info("Starting ShadowPlay API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)

