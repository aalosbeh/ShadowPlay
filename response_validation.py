"""
Response Validation Layer for ShadowPlay Framework

This module implements comprehensive analysis of LLM outputs to ensure
security and quality standards before delivery to developers.
"""

import ast
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import bandit
from bandit.core import config as bandit_config
from bandit.core import manager as bandit_manager

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""
    severity: str
    confidence: str
    issue_type: str
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None


@dataclass
class QualityMetric:
    """Represents a code quality metric."""
    metric_name: str
    value: float
    threshold: float
    passed: bool
    description: str


@dataclass
class ValidationResult:
    """Result of response validation."""
    passed: bool
    security_score: float
    quality_score: float
    issues: List[str] = field(default_factory=list)
    security_issues: List[SecurityIssue] = field(default_factory=list)
    quality_metrics: List[QualityMetric] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)


class StaticCodeAnalyzer:
    """Performs static analysis on generated code."""
    
    def __init__(self):
        self.supported_languages = {"python", "javascript", "java", "go"}
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security vulnerability patterns."""
        return {
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%.*['\"]",
                r"query\s*\(\s*['\"].*\+.*['\"]",
                r"SELECT.*\+.*FROM",
                r"INSERT.*\+.*VALUES"
            ],
            "command_injection": [
                r"os\.system\s*\(\s*.*\+",
                r"subprocess\.(call|run|Popen)\s*\(\s*.*\+",
                r"exec\s*\(\s*.*\+",
                r"eval\s*\(\s*.*\+"
            ],
            "path_traversal": [
                r"open\s*\(\s*.*\.\./",
                r"file\s*\(\s*.*\.\./",
                r"include\s*\(\s*.*\.\./"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]"
            ]
        }
    
    def analyze_code(self, code: str, language: str = "python") -> List[SecurityIssue]:
        """Analyze code for security vulnerabilities."""
        issues = []
        
        if language.lower() not in self.supported_languages:
            logger.warning(f"Unsupported language for analysis: {language}")
            return issues
        
        # Pattern-based analysis
        issues.extend(self._pattern_based_analysis(code))
        
        # Language-specific analysis
        if language.lower() == "python":
            issues.extend(self._analyze_python_code(code))
        elif language.lower() == "javascript":
            issues.extend(self._analyze_javascript_code(code))
        
        return issues
    
    def _pattern_based_analysis(self, code: str) -> List[SecurityIssue]:
        """Perform pattern-based security analysis."""
        issues = []
        lines = code.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity="HIGH",
                            confidence="MEDIUM",
                            issue_type=category,
                            description=f"Potential {category.replace('_', ' ')} vulnerability detected",
                            line_number=line_num,
                            code_snippet=line.strip()
                        ))
        
        return issues
    
    def _analyze_python_code(self, code: str) -> List[SecurityIssue]:
        """Analyze Python code using AST and Bandit."""
        issues = []
        
        try:
            # Parse AST for structural analysis
            tree = ast.parse(code)
            issues.extend(self._analyze_python_ast(tree))
            
            # Use Bandit for comprehensive security analysis
            issues.extend(self._run_bandit_analysis(code))
            
        except SyntaxError as e:
            issues.append(SecurityIssue(
                severity="LOW",
                confidence="HIGH",
                issue_type="syntax_error",
                description=f"Syntax error in Python code: {e}",
                line_number=getattr(e, 'lineno', None)
            ))
        
        return issues
    
    def _analyze_python_ast(self, tree: ast.AST) -> List[SecurityIssue]:
        """Analyze Python AST for security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ["eval", "exec", "compile"]:
                        issues.append(SecurityIssue(
                            severity="HIGH",
                            confidence="HIGH",
                            issue_type="dangerous_function",
                            description=f"Use of dangerous function: {func_name}",
                            line_number=getattr(node, 'lineno', None)
                        ))
                
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["system", "popen"]:
                        issues.append(SecurityIssue(
                            severity="HIGH",
                            confidence="MEDIUM",
                            issue_type="command_execution",
                            description=f"Potential command execution: {node.func.attr}",
                            line_number=getattr(node, 'lineno', None)
                        ))
            
            # Check for hardcoded strings that might be secrets
            if isinstance(node, ast.Str):
                if self._looks_like_secret(node.s):
                    issues.append(SecurityIssue(
                        severity="MEDIUM",
                        confidence="LOW",
                        issue_type="hardcoded_secret",
                        description="Potential hardcoded secret detected",
                        line_number=getattr(node, 'lineno', None)
                    ))
        
        return issues
    
    def _run_bandit_analysis(self, code: str) -> List[SecurityIssue]:
        """Run Bandit security analysis on Python code."""
        issues = []
        
        try:
            # Create temporary file for Bandit analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Configure Bandit
            conf = bandit_config.BanditConfig()
            b_mgr = bandit_manager.BanditManager(conf, 'file')
            
            # Run Bandit analysis
            b_mgr.discover_files([temp_file])
            b_mgr.run_tests()
            
            # Extract results
            for result in b_mgr.get_issue_list():
                issues.append(SecurityIssue(
                    severity=result.severity,
                    confidence=result.confidence,
                    issue_type=result.test,
                    description=result.text,
                    line_number=result.lineno
                ))
            
        except Exception as e:
            logger.error(f"Error running Bandit analysis: {e}")
        
        return issues
    
    def _analyze_javascript_code(self, code: str) -> List[SecurityIssue]:
        """Analyze JavaScript code for security issues."""
        issues = []
        lines = code.split('\n')
        
        # JavaScript-specific patterns
        js_patterns = {
            "xss": [
                r"innerHTML\s*=\s*.*\+",
                r"document\.write\s*\(\s*.*\+",
                r"eval\s*\(\s*.*\+"
            ],
            "prototype_pollution": [
                r"__proto__",
                r"constructor\.prototype"
            ]
        }
        
        for category, patterns in js_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity="MEDIUM",
                            confidence="MEDIUM",
                            issue_type=category,
                            description=f"Potential {category.replace('_', ' ')} vulnerability",
                            line_number=line_num,
                            code_snippet=line.strip()
                        ))
        
        return issues
    
    def _looks_like_secret(self, string: str) -> bool:
        """Check if a string looks like a secret."""
        if len(string) < 8:
            return False
        
        # Check for patterns that look like secrets
        secret_patterns = [
            r'^[A-Za-z0-9+/]{20,}={0,2}$',  # Base64
            r'^[a-f0-9]{32,}$',  # Hex
            r'^[A-Z0-9]{20,}$',  # API key pattern
        ]
        
        for pattern in secret_patterns:
            if re.match(pattern, string):
                return True
        
        return False


class SecurityScanner:
    """Scans responses for security-related content."""
    
    def __init__(self):
        self.vulnerability_keywords = [
            "sql injection", "xss", "csrf", "buffer overflow", "privilege escalation",
            "code injection", "path traversal", "authentication bypass", "session hijacking"
        ]
        
        self.suspicious_patterns = [
            r"(?i)bypass\s+(security|authentication|authorization)",
            r"(?i)(disable|turn\s+off)\s+(security|logging|monitoring)",
            r"(?i)(exploit|vulnerability|backdoor|malware)",
            r"(?i)(password|credential|token)\s+(steal|extract|dump)"
        ]
    
    def scan_response(self, response: str, context: Optional[Dict] = None) -> List[SecurityIssue]:
        """Scan response for security concerns."""
        issues = []
        
        # Check for vulnerability discussions
        response_lower = response.lower()
        for keyword in self.vulnerability_keywords:
            if keyword in response_lower:
                issues.append(SecurityIssue(
                    severity="LOW",
                    confidence="LOW",
                    issue_type="vulnerability_discussion",
                    description=f"Response discusses {keyword}",
                ))
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, response):
                issues.append(SecurityIssue(
                    severity="MEDIUM",
                    confidence="MEDIUM",
                    issue_type="suspicious_content",
                    description="Response contains suspicious security-related content"
                ))
        
        # Check for potential data leakage
        if self._check_data_leakage(response):
            issues.append(SecurityIssue(
                severity="HIGH",
                confidence="MEDIUM",
                issue_type="data_leakage",
                description="Response may contain sensitive information"
            ))
        
        return issues
    
    def _check_data_leakage(self, response: str) -> bool:
        """Check for potential data leakage in response."""
        # Patterns that might indicate data leakage
        leakage_patterns = [
            r"(?i)(api[_\s]?key|secret[_\s]?key|access[_\s]?token):\s*['\"]?[a-zA-Z0-9+/=]{10,}",
            r"(?i)(password|passwd):\s*['\"]?[a-zA-Z0-9!@#$%^&*]{6,}",
            r"(?i)(database|db)[_\s]?(url|connection):\s*['\"]?[a-zA-Z0-9:/._-]+",
            r"(?i)private[_\s]?key:\s*-----BEGIN"
        ]
        
        for pattern in leakage_patterns:
            if re.search(pattern, response):
                return True
        
        return False


class QualityAssessor:
    """Assesses the quality of generated responses."""
    
    def __init__(self):
        self.quality_thresholds = {
            "readability": 0.7,
            "completeness": 0.8,
            "accuracy": 0.9,
            "relevance": 0.8
        }
    
    def assess_response(self, response: str, context: Optional[Dict] = None) -> List[QualityMetric]:
        """Assess the quality of a response."""
        metrics = []
        
        # Readability assessment
        readability = self._assess_readability(response)
        metrics.append(QualityMetric(
            metric_name="readability",
            value=readability,
            threshold=self.quality_thresholds["readability"],
            passed=readability >= self.quality_thresholds["readability"],
            description="Measures how readable and well-structured the response is"
        ))
        
        # Completeness assessment
        completeness = self._assess_completeness(response, context)
        metrics.append(QualityMetric(
            metric_name="completeness",
            value=completeness,
            threshold=self.quality_thresholds["completeness"],
            passed=completeness >= self.quality_thresholds["completeness"],
            description="Measures how complete the response is relative to the request"
        ))
        
        # Technical accuracy assessment
        accuracy = self._assess_technical_accuracy(response)
        metrics.append(QualityMetric(
            metric_name="accuracy",
            value=accuracy,
            threshold=self.quality_thresholds["accuracy"],
            passed=accuracy >= self.quality_thresholds["accuracy"],
            description="Measures technical accuracy of code and explanations"
        ))
        
        # Relevance assessment
        relevance = self._assess_relevance(response, context)
        metrics.append(QualityMetric(
            metric_name="relevance",
            value=relevance,
            threshold=self.quality_thresholds["relevance"],
            passed=relevance >= self.quality_thresholds["relevance"],
            description="Measures how relevant the response is to the original request"
        ))
        
        return metrics
    
    def _assess_readability(self, response: str) -> float:
        """Assess readability of the response."""
        if not response.strip():
            return 0.0
        
        # Simple readability metrics
        sentences = response.split('.')
        words = response.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Penalize very long or very short sentences
        if avg_sentence_length < 5 or avg_sentence_length > 30:
            readability = 0.5
        else:
            readability = 0.8
        
        # Check for code formatting
        if "```" in response or "    " in response:
            readability += 0.2
        
        return min(readability, 1.0)
    
    def _assess_completeness(self, response: str, context: Optional[Dict]) -> float:
        """Assess completeness of the response."""
        if not response.strip():
            return 0.0
        
        # Basic completeness check
        completeness = 0.5  # Base score
        
        # Check for code examples if requested
        if context and "code" in context.get("request_type", "").lower():
            if "```" in response or "def " in response or "function " in response:
                completeness += 0.3
        
        # Check for explanations
        if len(response.split()) > 50:
            completeness += 0.2
        
        return min(completeness, 1.0)
    
    def _assess_technical_accuracy(self, response: str) -> float:
        """Assess technical accuracy of the response."""
        # This is a simplified assessment
        accuracy = 0.8  # Default assumption of accuracy
        
        # Check for obvious technical errors
        error_indicators = [
            "undefined function", "syntax error", "import error",
            "module not found", "invalid syntax"
        ]
        
        response_lower = response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                accuracy -= 0.2
        
        return max(accuracy, 0.0)
    
    def _assess_relevance(self, response: str, context: Optional[Dict]) -> float:
        """Assess relevance of the response to the request."""
        if not context or not response.strip():
            return 0.5  # Neutral score when context is unavailable
        
        request = context.get("original_request", "").lower()
        response_lower = response.lower()
        
        if not request:
            return 0.5
        
        # Simple keyword overlap
        request_words = set(request.split())
        response_words = set(response_lower.split())
        
        overlap = len(request_words & response_words)
        relevance = overlap / max(len(request_words), 1)
        
        return min(relevance * 2, 1.0)  # Scale up the score


class ResponseValidationLayer:
    """Main validation layer for LLM responses."""
    
    def __init__(self):
        self.code_analyzer = StaticCodeAnalyzer()
        self.security_scanner = SecurityScanner()
        self.quality_assessor = QualityAssessor()
        
        logger.info("Response Validation Layer initialized")
    
    def validate_response(self, response: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate an LLM response comprehensively."""
        try:
            issues = []
            security_issues = []
            recommendations = []
            
            # Extract code blocks if present
            code_blocks = self._extract_code_blocks(response)
            
            # Analyze code blocks
            for code, language in code_blocks:
                code_issues = self.code_analyzer.analyze_code(code, language)
                security_issues.extend(code_issues)
            
            # Scan entire response for security concerns
            response_security_issues = self.security_scanner.scan_response(response, context)
            security_issues.extend(response_security_issues)
            
            # Assess quality
            quality_metrics = self.quality_assessor.assess_response(response, context)
            
            # Calculate scores
            security_score = self._calculate_security_score(security_issues)
            quality_score = self._calculate_quality_score(quality_metrics)
            
            # Determine if validation passed
            passed = security_score >= 0.7 and quality_score >= 0.6
            
            # Generate issues and recommendations
            if security_score < 0.7:
                issues.append("Security concerns detected in response")
                recommendations.append("Review response for potential security vulnerabilities")
            
            if quality_score < 0.6:
                issues.append("Quality standards not met")
                recommendations.append("Consider requesting clarification or additional details")
            
            # Add specific recommendations based on issues
            for issue in security_issues:
                if issue.severity == "HIGH":
                    recommendations.append(f"Address high-severity {issue.issue_type} issue")
            
            return ValidationResult(
                passed=passed,
                security_score=security_score,
                quality_score=quality_score,
                issues=issues,
                security_issues=security_issues,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                metadata={
                    "code_blocks_analyzed": len(code_blocks),
                    "total_security_issues": len(security_issues),
                    "high_severity_issues": len([i for i in security_issues if i.severity == "HIGH"])
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return ValidationResult(
                passed=False,
                security_score=0.0,
                quality_score=0.0,
                issues=[f"Validation failed: {str(e)}"],
                recommendations=["Manual review required due to validation error"]
            )
    
    def _extract_code_blocks(self, response: str) -> List[Tuple[str, str]]:
        """Extract code blocks from response."""
        code_blocks = []
        
        # Extract fenced code blocks
        fenced_pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.finditer(fenced_pattern, response, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or "unknown"
            code = match.group(2)
            code_blocks.append((code, language))
        
        # Extract indented code blocks (simple heuristic)
        lines = response.split('\n')
        current_block = []
        in_code_block = False
        
        for line in lines:
            if line.startswith('    ') and line.strip():  # Indented non-empty line
                current_block.append(line[4:])  # Remove indentation
                in_code_block = True
            else:
                if in_code_block and current_block:
                    code_blocks.append(('\n'.join(current_block), "unknown"))
                    current_block = []
                in_code_block = False
        
        # Add final block if exists
        if current_block:
            code_blocks.append(('\n'.join(current_block), "unknown"))
        
        return code_blocks
    
    def _calculate_security_score(self, security_issues: List[SecurityIssue]) -> float:
        """Calculate overall security score."""
        if not security_issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6, "CRITICAL": 1.0}
        
        total_weight = sum(severity_weights.get(issue.severity, 0.3) for issue in security_issues)
        max_possible_weight = len(security_issues) * 1.0
        
        # Convert to score (higher is better)
        security_score = max(0.0, 1.0 - (total_weight / max_possible_weight))
        
        return security_score
    
    def _calculate_quality_score(self, quality_metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score."""
        if not quality_metrics:
            return 0.5
        
        # Average of all quality metrics
        total_score = sum(metric.value for metric in quality_metrics)
        return total_score / len(quality_metrics)

