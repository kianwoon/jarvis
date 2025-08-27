"""
General Purpose Completeness Verification Framework

Ensures that comprehensive queries return complete results regardless of data type,
query phrasing, or content structure. Works with any type of entity or attribute.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CompletenessResult:
    is_complete: bool
    confidence: float
    missing_indicators: List[str]
    found_count: int
    expected_count: Optional[int]
    gaps_detected: List[str]
    recommendations: List[str]

class GeneralCompletenessVerifier:
    """
    General purpose completeness verifier that works for any type of comprehensive query.
    
    Detects:
    - Sequential gaps (years, numbers, dates)
    - Pattern inconsistencies (naming, formatting)
    - Category coverage (companies, types, statuses)
    - Quantity mismatches vs source availability
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def verify_completeness(
        self, 
        response_text: str,
        sources: List[Dict],
        query: str,
        expected_entity_count: Optional[int] = None
    ) -> CompletenessResult:
        """
        General purpose completeness verification.
        
        Args:
            response_text: The LLM response text
            sources: Available source documents
            query: Original user query
            expected_entity_count: Known entity count if available
            
        Returns:
            CompletenessResult with detailed analysis
        """
        
        # Extract entities from response using general patterns
        entities = self._extract_entities_from_response(response_text)
        
        # Detect entity type from query and response
        entity_type = self._detect_entity_type(query, response_text)
        
        # Check for sequential gaps (years, numbers, IDs)
        sequential_gaps = self._detect_sequential_gaps(entities, entity_type)
        
        # Check for pattern inconsistencies
        pattern_issues = self._detect_pattern_inconsistencies(entities)
        
        # Compare with source availability
        source_coverage = self._analyze_source_coverage(entities, sources)
        
        # Calculate completeness confidence
        confidence = self._calculate_completeness_confidence(
            entities, sources, sequential_gaps, pattern_issues, expected_entity_count
        )
        
        # Determine if complete
        is_complete = (
            confidence > 0.9 and
            len(sequential_gaps) == 0 and
            len(pattern_issues) == 0 and
            (expected_entity_count is None or len(entities) >= expected_entity_count * 0.95)
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            entities, sequential_gaps, pattern_issues, source_coverage
        )
        
        self.logger.info(
            f"[COMPLETENESS] Verified {len(entities)} entities, "
            f"confidence: {confidence:.2f}, complete: {is_complete}"
        )
        
        return CompletenessResult(
            is_complete=is_complete,
            confidence=confidence,
            missing_indicators=sequential_gaps + pattern_issues,
            found_count=len(entities),
            expected_count=expected_entity_count,
            gaps_detected=sequential_gaps,
            recommendations=recommendations
        )
    
    def _extract_entities_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract entities from response using general patterns."""
        entities = []
        
        # Pattern 1: Table format detection
        if "|" in response_text and "---" in response_text:
            entities.extend(self._extract_table_entities(response_text))
        
        # Pattern 2: Numbered list format
        elif re.search(r'^\d+\.', response_text, re.MULTILINE):
            entities.extend(self._extract_numbered_list_entities(response_text))
        
        # Pattern 3: Bullet points
        elif re.search(r'^[-*]\s', response_text, re.MULTILINE):
            entities.extend(self._extract_bullet_entities(response_text))
        
        # Pattern 4: General entity detection
        else:
            entities.extend(self._extract_general_entities(response_text))
        
        return entities
    
    def _extract_table_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from markdown table format."""
        entities = []
        lines = text.split('\n')
        
        header_found = False
        headers = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|') == False:
                continue
                
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
            
            if not header_found and not line.startswith('|---'):
                headers = cells
                header_found = True
            elif line.startswith('|---'):
                continue  # Skip separator line
            elif header_found and cells and cells[0]:  # Skip empty rows
                entity = {}
                for i, cell in enumerate(cells):
                    if i < len(headers) and cell.strip():
                        entity[headers[i].lower().strip()] = cell.strip()
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _extract_numbered_list_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from numbered list format."""
        entities = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                content = match.group(1)
                # Try to parse structured content
                entity = {'content': content}
                
                # Look for common patterns: Name - Company (Year)
                patterns = [
                    r'^(.+?)\s*-\s*(.+?)\s*\((\d{4}.*?)\)',  # Name - Company (Year)
                    r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)',       # Name | Company | Year
                    r'^(.+?)\s*,\s*(.+?)\s*,\s*(.+)',        # Name, Company, Year
                ]
                
                for pattern in patterns:
                    match = re.match(pattern, content)
                    if match:
                        entity = {
                            'name': match.group(1).strip(),
                            'company': match.group(2).strip(),
                            'year': match.group(3).strip()
                        }
                        break
                
                entities.append(entity)
        
        return entities
    
    def _extract_bullet_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from bullet point format."""
        entities = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^[-*]\s*(.+)', line)
            if match:
                entities.append({'content': match.group(1)})
        
        return entities
    
    def _extract_general_entities(self, text: str) -> List[Dict[str, Any]]:
        """General entity extraction for unstructured text."""
        # Look for common entity indicators
        entities = []
        
        # Company names (often capitalized, may have Ltd, Inc, etc.)
        companies = re.findall(r'\b[A-Z][a-zA-Z&\s]+(?:Ltd|Inc|Corp|Group|Bank|Authority|Ministry)\b', text)
        
        # Years (4-digit numbers that look like years)
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        
        # Project/system names (often in quotes or title case)
        projects = re.findall(r'"([^"]+)"|([A-Z][a-zA-Z\s]+(?:System|Platform|App|Portal|Framework))', text)
        
        if companies or years or projects:
            entities.append({
                'companies': list(set(companies)),
                'years': list(set(years)),
                'projects': list(set([p[0] or p[1] for p in projects]))
            })
        
        return entities
    
    def _detect_entity_type(self, query: str, response: str) -> str:
        """Detect what type of entities we're dealing with."""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for common entity types
        entity_indicators = {
            'projects': ['project', 'work', 'initiative', 'system', 'platform', 'application'],
            'companies': ['company', 'client', 'organization', 'firm', 'business'],
            'documents': ['document', 'file', 'report', 'paper'],
            'people': ['person', 'employee', 'staff', 'team member', 'user'],
            'events': ['event', 'meeting', 'conference', 'session']
        }
        
        for entity_type, indicators in entity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return entity_type
        
        return 'general'
    
    def _detect_sequential_gaps(self, entities: List[Dict], entity_type: str) -> List[str]:
        """Detect gaps in sequential data (years, numbers, etc.)."""
        gaps = []
        
        # Extract years if present
        years = set()
        for entity in entities:
            if isinstance(entity, dict):
                for key, value in entity.items():
                    if 'year' in key.lower():
                        # Extract years from strings like "2020-2021" or "2020"
                        year_matches = re.findall(r'\b(19|20)\d{2}\b', str(value))
                        years.update(year_matches)
        
        if len(years) > 3:  # Only check for gaps if we have several years
            year_ints = sorted([int(y) for y in years])
            for i in range(len(year_ints) - 1):
                if year_ints[i+1] - year_ints[i] > 1:
                    gap_years = list(range(year_ints[i] + 1, year_ints[i+1]))
                    gaps.append(f"Missing years: {', '.join(map(str, gap_years))}")
        
        # Check for numbering gaps in numbered lists
        numbers = set()
        for entity in entities:
            if isinstance(entity, dict) and 'content' in entity:
                num_match = re.match(r'^(\d+)', entity['content'])
                if num_match:
                    numbers.add(int(num_match.group(1)))
        
        if len(numbers) > 2:
            num_list = sorted(numbers)
            for i in range(len(num_list) - 1):
                if num_list[i+1] - num_list[i] > 1:
                    gaps.append(f"Missing numbers: {num_list[i] + 1} to {num_list[i+1] - 1}")
        
        return gaps
    
    def _detect_pattern_inconsistencies(self, entities: List[Dict]) -> List[str]:
        """Detect inconsistencies in data patterns."""
        issues = []
        
        if not entities:
            return issues
        
        # Check for formatting inconsistencies
        if len(entities) > 5:  # Only check if we have enough data
            # Look for missing company names
            companies = []
            for entity in entities:
                if isinstance(entity, dict):
                    company = entity.get('company', '') or entity.get('client', '')
                    companies.append(company.strip())
            
            empty_companies = sum(1 for c in companies if not c or c == '—' or c.lower() == 'n/a')
            if empty_companies > len(companies) * 0.3:  # More than 30% missing
                issues.append(f"Many entities missing company information ({empty_companies}/{len(companies)})")
        
        return issues
    
    def _analyze_source_coverage(self, entities: List[Dict], sources: List[Dict]) -> Dict[str, Any]:
        """Analyze how well entities match available sources."""
        return {
            'entity_count': len(entities),
            'source_count': len(sources),
            'coverage_ratio': len(entities) / max(len(sources), 1),
            'likely_complete': len(entities) >= len(sources) * 0.8
        }
    
    def _calculate_completeness_confidence(
        self,
        entities: List[Dict],
        sources: List[Dict],
        gaps: List[str],
        issues: List[str],
        expected_count: Optional[int]
    ) -> float:
        """Calculate confidence that results are complete."""
        
        base_confidence = 0.8  # Start optimistic
        
        # Penalize for gaps
        gap_penalty = len(gaps) * 0.1
        base_confidence -= gap_penalty
        
        # Penalize for pattern issues
        issue_penalty = len(issues) * 0.05
        base_confidence -= issue_penalty
        
        # Boost if entity count matches or exceeds source count
        if entities and sources:
            coverage_ratio = len(entities) / len(sources)
            if coverage_ratio >= 0.9:
                base_confidence += 0.1
            elif coverage_ratio < 0.5:
                base_confidence -= 0.2
        
        # Consider expected count if provided
        if expected_count and entities:
            expected_ratio = len(entities) / expected_count
            if expected_ratio >= 0.95:
                base_confidence += 0.1
            elif expected_ratio < 0.8:
                base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_recommendations(
        self,
        entities: List[Dict],
        gaps: List[str],
        issues: List[str],
        coverage: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving completeness."""
        recommendations = []
        
        if gaps:
            recommendations.append("Expand search to cover missing sequential elements")
        
        if issues:
            recommendations.append("Review data formatting and fill missing information")
        
        if coverage['coverage_ratio'] < 0.8:
            recommendations.append("Increase retrieval scope to capture more available sources")
        
        if not entities:
            recommendations.append("Query may be too specific or sources may need reindexing")
        
        return recommendations

# Global instance
completeness_verifier = GeneralCompletenessVerifier()