"""
Dependency Verification System for ShadowPlay Framework

This module implements real-time package verification against authoritative repositories
to prevent dependency hallucination attacks.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class PackageInfo:
    """Information about a software package."""
    name: str
    version: Optional[str]
    ecosystem: str
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    downloads: Optional[int] = None
    last_updated: Optional[str] = None
    security_advisories: List[str] = field(default_factory=list)


@dataclass
class VerificationCache:
    """Cache for package verification results."""
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.cache: Dict[str, Tuple[any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[any]:
        """Get cached result if still valid."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: any) -> None:
        """Cache a result with timestamp."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())


class RepositoryClient(ABC):
    """Abstract base class for repository clients."""
    
    @abstractmethod
    async def verify_package(self, package_name: str, version: Optional[str] = None) -> PackageInfo:
        """Verify package existence and retrieve metadata."""
        pass
    
    @abstractmethod
    async def search_packages(self, query: str, limit: int = 10) -> List[PackageInfo]:
        """Search for packages matching the query."""
        pass


class NPMClient(RepositoryClient):
    """Client for NPM package repository."""
    
    def __init__(self):
        self.base_url = "https://registry.npmjs.org"
        self.search_url = "https://api.npms.io/v2/search"
    
    async def verify_package(self, package_name: str, version: Optional[str] = None) -> Optional[PackageInfo]:
        """Verify NPM package existence."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/{package_name}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_npm_package(data)
                    elif response.status == 404:
                        return None
                    else:
                        logger.warning(f"NPM API returned status {response.status} for {package_name}")
                        return None
        except Exception as e:
            logger.error(f"Error verifying NPM package {package_name}: {e}")
            return None
    
    async def search_packages(self, query: str, limit: int = 10) -> List[PackageInfo]:
        """Search NPM packages."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"q": query, "size": limit}
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._parse_npm_search_result(result) for result in data.get("results", [])]
                    else:
                        logger.warning(f"NPM search API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error searching NPM packages: {e}")
            return []
    
    def _parse_npm_package(self, data: Dict) -> PackageInfo:
        """Parse NPM package data."""
        latest_version = data.get("dist-tags", {}).get("latest", "unknown")
        latest_info = data.get("versions", {}).get(latest_version, {})
        
        return PackageInfo(
            name=data.get("name", ""),
            version=latest_version,
            ecosystem="npm",
            description=latest_info.get("description", ""),
            author=self._extract_author(latest_info.get("author")),
            license=latest_info.get("license", ""),
            homepage=latest_info.get("homepage", ""),
            repository=self._extract_repository(latest_info.get("repository")),
            last_updated=data.get("time", {}).get(latest_version, "")
        )
    
    def _parse_npm_search_result(self, result: Dict) -> PackageInfo:
        """Parse NPM search result."""
        package = result.get("package", {})
        return PackageInfo(
            name=package.get("name", ""),
            version=package.get("version", ""),
            ecosystem="npm",
            description=package.get("description", ""),
            author=self._extract_author(package.get("author")),
            license=package.get("license", ""),
            homepage=package.get("homepage", ""),
            repository=self._extract_repository(package.get("repository"))
        )
    
    def _extract_author(self, author_data) -> Optional[str]:
        """Extract author name from various formats."""
        if isinstance(author_data, str):
            return author_data
        elif isinstance(author_data, dict):
            return author_data.get("name", "")
        return None
    
    def _extract_repository(self, repo_data) -> Optional[str]:
        """Extract repository URL from various formats."""
        if isinstance(repo_data, str):
            return repo_data
        elif isinstance(repo_data, dict):
            return repo_data.get("url", "")
        return None


class PyPIClient(RepositoryClient):
    """Client for PyPI package repository."""
    
    def __init__(self):
        self.base_url = "https://pypi.org/pypi"
        self.search_url = "https://pypi.org/search/"
    
    async def verify_package(self, package_name: str, version: Optional[str] = None) -> Optional[PackageInfo]:
        """Verify PyPI package existence."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/{package_name}/json"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_pypi_package(data)
                    elif response.status == 404:
                        return None
                    else:
                        logger.warning(f"PyPI API returned status {response.status} for {package_name}")
                        return None
        except Exception as e:
            logger.error(f"Error verifying PyPI package {package_name}: {e}")
            return None
    
    async def search_packages(self, query: str, limit: int = 10) -> List[PackageInfo]:
        """Search PyPI packages (simplified implementation)."""
        # Note: PyPI doesn't have a simple JSON search API
        # This is a simplified implementation
        return []
    
    def _parse_pypi_package(self, data: Dict) -> PackageInfo:
        """Parse PyPI package data."""
        info = data.get("info", {})
        
        return PackageInfo(
            name=info.get("name", ""),
            version=info.get("version", ""),
            ecosystem="pypi",
            description=info.get("summary", ""),
            author=info.get("author", ""),
            license=info.get("license", ""),
            homepage=info.get("home_page", ""),
            repository=self._extract_project_url(info.get("project_urls", {}))
        )
    
    def _extract_project_url(self, project_urls: Dict) -> Optional[str]:
        """Extract repository URL from project URLs."""
        for key in ["Repository", "Source", "Homepage"]:
            if key in project_urls:
                return project_urls[key]
        return None


class MavenClient(RepositoryClient):
    """Client for Maven Central repository."""
    
    def __init__(self):
        self.search_url = "https://search.maven.org/solrsearch/select"
    
    async def verify_package(self, package_name: str, version: Optional[str] = None) -> Optional[PackageInfo]:
        """Verify Maven package existence."""
        try:
            # Parse group:artifact format
            if ":" in package_name:
                group_id, artifact_id = package_name.split(":", 1)
            else:
                # Assume artifact_id only
                group_id = ""
                artifact_id = package_name
            
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": f"g:{group_id} AND a:{artifact_id}" if group_id else f"a:{artifact_id}",
                    "rows": 1,
                    "wt": "json"
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        docs = data.get("response", {}).get("docs", [])
                        if docs:
                            return self._parse_maven_package(docs[0])
                        else:
                            return None
                    else:
                        logger.warning(f"Maven search API returned status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error verifying Maven package {package_name}: {e}")
            return None
    
    async def search_packages(self, query: str, limit: int = 10) -> List[PackageInfo]:
        """Search Maven packages."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "rows": limit,
                    "wt": "json"
                }
                
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        docs = data.get("response", {}).get("docs", [])
                        return [self._parse_maven_package(doc) for doc in docs]
                    else:
                        logger.warning(f"Maven search API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error searching Maven packages: {e}")
            return []
    
    def _parse_maven_package(self, doc: Dict) -> PackageInfo:
        """Parse Maven package data."""
        return PackageInfo(
            name=f"{doc.get('g', '')}:{doc.get('a', '')}",
            version=doc.get("latestVersion", ""),
            ecosystem="maven",
            description="",  # Maven search doesn't provide descriptions
            last_updated=doc.get("timestamp", "")
        )


class FuzzyMatcher:
    """Fuzzy matching for package names to suggest alternatives."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
    
    async def find_similar(self, package_name: str, ecosystem: str, candidates: List[str] = None) -> List[str]:
        """Find similar package names."""
        if candidates is None:
            # In a real implementation, this would query a database of known packages
            candidates = self._get_common_packages(ecosystem)
        
        similar_packages = []
        
        for candidate in candidates:
            similarity = self._calculate_similarity(package_name, candidate)
            if similarity >= self.similarity_threshold:
                similar_packages.append((candidate, similarity))
        
        # Sort by similarity and return top matches
        similar_packages.sort(key=lambda x: x[1], reverse=True)
        return [pkg for pkg, _ in similar_packages[:5]]
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two package names."""
        # Use SequenceMatcher for basic similarity
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def _get_common_packages(self, ecosystem: str) -> List[str]:
        """Get list of common packages for the ecosystem."""
        common_packages = {
            "npm": [
                "react", "vue", "angular", "lodash", "express", "axios", "moment",
                "webpack", "babel", "eslint", "typescript", "jest", "mocha"
            ],
            "pypi": [
                "requests", "numpy", "pandas", "flask", "django", "tensorflow",
                "pytorch", "scikit-learn", "matplotlib", "pillow", "beautifulsoup4"
            ],
            "maven": [
                "org.springframework:spring-core", "junit:junit", "org.apache.commons:commons-lang3",
                "com.google.guava:guava", "org.slf4j:slf4j-api", "org.apache.httpcomponents:httpclient"
            ]
        }
        
        return common_packages.get(ecosystem, [])


class DependencyVerificationSystem:
    """Main system for verifying package dependencies."""
    
    def __init__(self):
        self.clients = {
            "npm": NPMClient(),
            "pypi": PyPIClient(),
            "maven": MavenClient()
        }
        self.cache = VerificationCache()
        self.fuzzy_matcher = FuzzyMatcher()
        
        logger.info("Dependency Verification System initialized")
    
    async def verify_package(self, package_name: str, ecosystem: str, version: Optional[str] = None) -> VerificationResult:
        """Verify a package against its repository."""
        cache_key = f"{ecosystem}:{package_name}:{version or 'latest'}"
        
        # Check cache first
        if cached_result := self.cache.get(cache_key):
            return cached_result
        
        try:
            client = self.clients.get(ecosystem)
            if not client:
                return VerificationResult(
                    package_name=package_name,
                    ecosystem=ecosystem,
                    exists=False,
                    version_valid=False,
                    security_status="unknown",
                    suggestions=[],
                    metadata={"error": f"Unsupported ecosystem: {ecosystem}"}
                )
            
            # Verify package existence
            package_info = await client.verify_package(package_name, version)
            
            if package_info:
                result = VerificationResult(
                    package_name=package_name,
                    ecosystem=ecosystem,
                    exists=True,
                    version_valid=self._validate_version(package_info.version, version),
                    security_status=self._assess_security_status(package_info),
                    suggestions=[],
                    metadata={
                        "description": package_info.description,
                        "author": package_info.author,
                        "license": package_info.license,
                        "last_updated": package_info.last_updated
                    }
                )
            else:
                # Package not found, suggest alternatives
                suggestions = await self.fuzzy_matcher.find_similar(package_name, ecosystem)
                
                result = VerificationResult(
                    package_name=package_name,
                    ecosystem=ecosystem,
                    exists=False,
                    version_valid=False,
                    security_status="not_found",
                    suggestions=suggestions,
                    metadata={"error": "Package not found in repository"}
                )
            
            # Cache the result
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error verifying package {package_name}: {e}")
            return VerificationResult(
                package_name=package_name,
                ecosystem=ecosystem,
                exists=False,
                version_valid=False,
                security_status="error",
                suggestions=[],
                metadata={"error": str(e)}
            )
    
    async def verify_multiple_packages(self, packages: List[Tuple[str, str, Optional[str]]]) -> List[VerificationResult]:
        """Verify multiple packages concurrently."""
        tasks = [
            self.verify_package(name, ecosystem, version)
            for name, ecosystem, version in packages
        ]
        
        return await asyncio.gather(*tasks)
    
    def _validate_version(self, available_version: Optional[str], requested_version: Optional[str]) -> bool:
        """Validate if the requested version is available."""
        if not requested_version:
            return True  # Any version is acceptable
        
        if not available_version:
            return False
        
        # Simple version comparison (in practice, would use semantic versioning)
        return available_version == requested_version
    
    def _assess_security_status(self, package_info: PackageInfo) -> str:
        """Assess the security status of a package."""
        if package_info.security_advisories:
            return "vulnerable"
        
        # Check for other security indicators
        if not package_info.author or not package_info.license:
            return "suspicious"
        
        return "safe"
    
    async def extract_dependencies_from_prompt(self, prompt: str) -> List[Tuple[str, str, Optional[str]]]:
        """Extract package dependencies mentioned in a prompt."""
        dependencies = []
        
        # Patterns for different package managers
        patterns = {
            "npm": [
                r"npm\s+install\s+([a-zA-Z0-9\-_@/]+)(?:@([0-9\.]+))?",
                r"yarn\s+add\s+([a-zA-Z0-9\-_@/]+)(?:@([0-9\.]+))?",
                r"import\s+.*\s+from\s+['\"]([a-zA-Z0-9\-_@/]+)['\"]",
                r"require\(['\"]([a-zA-Z0-9\-_@/]+)['\"]\)"
            ],
            "pypi": [
                r"pip\s+install\s+([a-zA-Z0-9\-_]+)(?:==([0-9\.]+))?",
                r"import\s+([a-zA-Z0-9\-_]+)",
                r"from\s+([a-zA-Z0-9\-_]+)\s+import"
            ],
            "maven": [
                r"<artifactId>([a-zA-Z0-9\-_\.]+)</artifactId>",
                r"([a-zA-Z0-9\-_\.]+):([a-zA-Z0-9\-_\.]+):([0-9\.]+)"
            ]
        }
        
        for ecosystem, ecosystem_patterns in patterns.items():
            for pattern in ecosystem_patterns:
                matches = re.finditer(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    if ecosystem == "maven" and len(match.groups()) == 3:
                        # Maven group:artifact:version format
                        group_id, artifact_id, version = match.groups()
                        package_name = f"{group_id}:{artifact_id}"
                        dependencies.append((package_name, ecosystem, version))
                    else:
                        package_name = match.group(1)
                        version = match.group(2) if len(match.groups()) > 1 else None
                        dependencies.append((package_name, ecosystem, version))
        
        return dependencies

