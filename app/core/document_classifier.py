"""
Document Classifier Module

This module provides intelligent document classification to automatically
route documents to appropriate collections based on their content and metadata.
"""

import re
import logging
from typing import Dict, Any, Tuple, List, Optional
from app.core.collection_registry_cache import get_all_collections
# from app.llm.inference import LLMInference  # TODO: Fix import when LLM module is ready

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """Classify documents into appropriate collections"""
    
    def __init__(self):
        # self.llm = LLMInference()  # TODO: Enable when LLM module is ready
        self._collection_patterns = self._build_collection_patterns()
        
    def _build_collection_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build regex patterns for quick classification"""
        patterns = {
            # Banking-specific patterns
            "regulatory_compliance": [
                re.compile(r"\b(?:basel|dodd.?frank|sox|sarbanes.?oxley|kyc|aml|bsa|ofac|fdic|occ|fed|federal.?reserve)\b", re.I),
                re.compile(r"\b(?:compliance|regulation|regulatory|requirement|mandate|directive|circular)\b", re.I),
                re.compile(r"\b(?:anti.?money.?laundering|know.?your.?customer|suspicious.?activity|sar|ctr)\b", re.I),
            ],
            "product_documentation": [
                re.compile(r"\b(?:product|service|feature|pricing|rate|fee|charge|interest)\b", re.I),
                re.compile(r"\b(?:account|loan|mortgage|credit.?card|debit.?card|deposit|savings|checking)\b", re.I),
                re.compile(r"\b(?:eligibility|benefits|terms|conditions|requirements)\b", re.I),
            ],
            "risk_management": [
                re.compile(r"\b(?:risk|threat|vulnerability|exposure|mitigation|control|assessment)\b", re.I),
                re.compile(r"\b(?:credit.?risk|market.?risk|operational.?risk|liquidity.?risk|compliance.?risk)\b", re.I),
                re.compile(r"\b(?:risk.?appetite|risk.?tolerance|risk.?framework|risk.?matrix)\b", re.I),
            ],
            "customer_support": [
                re.compile(r"\b(?:customer|client|support|service|help|assist|troubleshoot|resolve)\b", re.I),
                re.compile(r"\b(?:faq|frequently.?asked|common.?question|how.?to|guide|tutorial)\b", re.I),
                re.compile(r"\b(?:complaint|issue|problem|error|dispute|claim)\b", re.I),
            ],
            "audit_reports": [
                re.compile(r"\b(?:audit|review|inspection|examination|assessment|evaluation)\b", re.I),
                re.compile(r"\b(?:finding|observation|recommendation|remediation|corrective.?action)\b", re.I),
                re.compile(r"\b(?:internal.?audit|external.?audit|sox.?audit|regulatory.?exam)\b", re.I),
            ],
            "training_materials": [
                re.compile(r"\b(?:training|learning|education|course|module|certification)\b", re.I),
                re.compile(r"\b(?:onboarding|orientation|induction|employee.?handbook)\b", re.I),
                re.compile(r"\b(?:competency|skill|knowledge|assessment|exam|quiz)\b", re.I),
            ],
            # Original patterns
            "technical_docs": [
                re.compile(r"\b(?:API|SDK|code|function|class|method|implementation|architecture)\b", re.I),
                re.compile(r"\b(?:github|git|repository|commit|branch|merge)\b", re.I),
                re.compile(r"\b(?:docker|kubernetes|deployment|container|microservice)\b", re.I),
            ],
            "policies_procedures": [
                re.compile(r"\b(?:policy|procedure|guideline|standard|protocol|sop)\b", re.I),
                re.compile(r"\b(?:must|shall|required|mandatory|prohibited|forbidden)\b", re.I),
                re.compile(r"\b(?:approval|authorized|permitted|governance)\b", re.I),
            ],
            "meeting_notes": [
                re.compile(r"\b(?:meeting|minutes|attendees|agenda|action.?items|discussion)\b", re.I),
                re.compile(r"\b(?:decided|agreed|proposed|next.?steps|follow.?up)\b", re.I),
                re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*(?:meeting|call|session)", re.I),
            ],
            "contracts_legal": [
                re.compile(r"\b(?:agreement|contract|terms|conditions|legal|liability)\b", re.I),
                re.compile(r"\b(?:party|parties|signatory|witness|executed|enforceable)\b", re.I),
                re.compile(r"\b(?:warranty|indemnity|confidential|proprietary|disclaimer)\b", re.I),
            ],
            "partnership": [
                re.compile(r"\b(?:partnership|partner|strategic.?partner|business.?partner|alliance)\b", re.I),
                re.compile(r"\b(?:collaboration|joint.?venture|cooperative|co.?development|co.?innovation)\b", re.I),
                re.compile(r"\b(?:beyondsoft|alibaba|tencent|baidu|huawei|strategic.?alliance)\b", re.I),
                re.compile(r"\b(?:partnership.?agreement|memorandum.?of.?understanding|mou|joint.?agreement)\b", re.I),
                re.compile(r"\b(?:co.?creation|shared.?development|mutual.?cooperation|strategic.?cooperation)\b", re.I),
            ],
        }
        return patterns
    
    def classify_by_patterns(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Quick classification using regex patterns"""
        scores = {}
        
        # Check content against patterns
        for collection_type, patterns in self._collection_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(pattern.findall(content[:5000]))  # Check first 5000 chars
                score += min(matches * 0.1, 0.3)  # Cap contribution per pattern
            scores[collection_type] = min(score, 1.0)
        
        # Check metadata hints
        if metadata:
            doc_type = metadata.get("doc_type", "").lower()
            filename = metadata.get("source", "").lower()
            
            # Banking-specific metadata checks
            if any(term in doc_type + filename for term in ["compliance", "regulatory", "basel", "kyc", "aml"]):
                scores["regulatory_compliance"] = max(scores.get("regulatory_compliance", 0), 0.8)
            elif any(term in doc_type + filename for term in ["product", "pricing", "rate", "fee"]):
                scores["product_documentation"] = max(scores.get("product_documentation", 0), 0.8)
            elif any(term in doc_type + filename for term in ["risk", "control", "assessment"]):
                scores["risk_management"] = max(scores.get("risk_management", 0), 0.8)
            elif any(term in doc_type + filename for term in ["support", "faq", "help", "customer"]):
                scores["customer_support"] = max(scores.get("customer_support", 0), 0.8)
            elif any(term in doc_type + filename for term in ["audit", "review", "examination"]):
                scores["audit_reports"] = max(scores.get("audit_reports", 0), 0.8)
            elif any(term in doc_type + filename for term in ["training", "onboarding", "certification"]):
                scores["training_materials"] = max(scores.get("training_materials", 0), 0.8)
            elif any(term in doc_type + filename for term in ["partnership", "alliance", "collaboration", "joint", "strategic"]):
                scores["partnership"] = max(scores.get("partnership", 0), 0.8)
            # Original metadata checks
            elif "technical" in doc_type or "api" in doc_type:
                scores["technical_docs"] = max(scores.get("technical_docs", 0), 0.8)
            elif "policy" in doc_type or "procedure" in doc_type:
                scores["policies_procedures"] = max(scores.get("policies_procedures", 0), 0.8)
            elif "meeting" in doc_type or "minutes" in doc_type:
                scores["meeting_notes"] = max(scores.get("meeting_notes", 0), 0.8)
            elif "contract" in doc_type or "agreement" in doc_type:
                scores["contracts_legal"] = max(scores.get("contracts_legal", 0), 0.8)
        
        # Get best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.3:  # Minimum confidence threshold
                return best_type
        
        return "general", 0.5
    
    async def classify_with_llm(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, float]:
        """Use LLM for more intelligent classification"""
        try:
            # Get available collections
            collections = get_all_collections()
            collection_types = {}
            for col in collections:
                col_type = col["collection_type"]
                if col_type not in collection_types:
                    collection_types[col_type] = col["description"]
            
            # Build prompt
            prompt = f"""Classify the following document into one of these categories:

{chr(10).join([f"- {k}: {v}" for k, v in collection_types.items()])}

Document excerpt (first 2000 chars):
{content[:2000]}

Metadata:
{metadata}

Return only the category name and confidence score (0-1) in format: category_name|confidence
Example: technical_docs|0.85
"""
            
            response = await self.llm.generate(prompt, max_tokens=50)
            
            # Parse response
            if "|" in response:
                category, confidence = response.strip().split("|", 1)
                category = category.strip()
                confidence = float(confidence.strip())
                
                if category in collection_types:
                    return category, confidence
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
        
        # Fallback to pattern matching
        return self.classify_by_patterns(content, metadata)
    
    def classify_document(self, content: str, metadata: Dict[str, Any], use_llm: bool = False) -> str:
        """
        Classify a document into an appropriate collection.
        
        Args:
            content: Document content
            metadata: Document metadata
            use_llm: Whether to use LLM for classification (slower but more accurate)
            
        Returns:
            Collection type string
        """
        if use_llm:
            # This would need to be made async or run in a thread pool
            logger.warning("LLM classification requested but running synchronously")
        
        collection_type, confidence = self.classify_by_patterns(content, metadata)
        
        logger.info(f"Classified document as {collection_type} with confidence {confidence}")
        return collection_type
    
    def extract_domain_metadata(self, content: str, collection_type: str) -> Dict[str, Any]:
        """Extract collection-specific metadata from document content"""
        metadata = {}
        
        # Banking-specific metadata extraction
        if collection_type == "regulatory_compliance":
            # Extract regulation references
            regulations = re.findall(r"\b(?:basel\s+[iIvV]+|dodd.?frank|sox|kyc|aml|bsa|ofac|gdpr|ccpa)\b", content, re.I)
            if regulations:
                metadata["regulations"] = list(set(reg.upper() for reg in regulations[:5]))
            
            # Extract compliance year/version
            year_matches = re.findall(r"(?:compliance|regulatory)\s+(?:year|version|update)[:\s]+(\d{4})", content, re.I)
            if year_matches:
                metadata["compliance_year"] = year_matches[0]
                
        elif collection_type == "product_documentation":
            # Extract product names
            product_matches = re.findall(r"(?:product|service)[:\s]+([A-Za-z\s]+?)(?:\n|\.)", content[:2000], re.I)
            if product_matches:
                metadata["product_name"] = product_matches[0].strip()
            
            # Extract pricing/rates
            rate_matches = re.findall(r"(?:rate|apr|apy)[:\s]+(\d+\.?\d*%?)", content, re.I)
            if rate_matches:
                metadata["rates"] = rate_matches[:3]
                
        elif collection_type == "risk_management":
            # Extract risk types
            risk_types = re.findall(r"(?:credit|market|operational|liquidity|compliance|reputational)\s+risk", content, re.I)
            if risk_types:
                metadata["risk_types"] = list(set(risk.lower() for risk in risk_types[:5]))
            
            # Extract risk level
            risk_levels = re.findall(r"risk\s+(?:level|rating)[:\s]+(?:high|medium|low|critical)", content, re.I)
            if risk_levels:
                metadata["risk_level"] = risk_levels[0].split(":")[-1].strip()
                
        elif collection_type == "customer_support":
            # Extract issue category
            categories = ["account", "transaction", "card", "loan", "technical", "security"]
            for cat in categories:
                if re.search(rf"\b{cat}\s+(?:issue|problem|question)\b", content, re.I):
                    metadata["issue_category"] = cat
                    break
            
            # Extract resolution status
            if re.search(r"\b(?:resolved|closed|completed)\b", content, re.I):
                metadata["status"] = "resolved"
            elif re.search(r"\b(?:pending|open|in.?progress)\b", content, re.I):
                metadata["status"] = "pending"
                
        elif collection_type == "audit_reports":
            # Extract audit type
            audit_types = re.findall(r"(?:internal|external|sox|regulatory|compliance)\s+audit", content, re.I)
            if audit_types:
                metadata["audit_type"] = audit_types[0].lower()
            
            # Extract audit period
            period_matches = re.findall(r"audit\s+period[:\s]+([^\n]+)", content, re.I)
            if period_matches:
                metadata["audit_period"] = period_matches[0].strip()
                
        elif collection_type == "training_materials":
            # Extract training type
            training_types = ["onboarding", "compliance", "technical", "soft skills", "leadership", "security"]
            for tt in training_types:
                if re.search(rf"\b{tt}\s+(?:training|course|module)\b", content, re.I):
                    metadata["training_type"] = tt
                    break
            
            # Extract target audience
            audience_match = re.search(r"(?:target|intended)\s+audience[:\s]+([^\n]+)", content, re.I)
            if audience_match:
                metadata["target_audience"] = audience_match.group(1).strip()
                
        elif collection_type == "partnership":
            # Extract partner companies
            partner_companies = re.findall(r"\b(?:beyondsoft|alibaba|tencent|baidu|huawei|microsoft|google|amazon|oracle)\b", content, re.I)
            if partner_companies:
                metadata["partner_companies"] = list(set(company.lower() for company in partner_companies[:5]))
            
            # Extract partnership type
            partnership_types = ["strategic", "technology", "marketing", "distribution", "joint venture", "collaboration"]
            for ptype in partnership_types:
                if re.search(rf"\b{ptype}\s+(?:partnership|alliance|cooperation)\b", content, re.I):
                    metadata["partnership_type"] = ptype
                    break
            
            # Extract partnership objectives
            objectives_match = re.search(r"(?:objective|goal|purpose)[:\s]+([^\n\.]+)", content, re.I)
            if objectives_match:
                metadata["objectives"] = objectives_match.group(1).strip()
                
            # Extract partnership duration
            duration_match = re.search(r"(?:duration|term|period)[:\s]+(\d+\s+(?:year|month|day)s?)", content, re.I)
            if duration_match:
                metadata["duration"] = duration_match.group(1).strip()
                
        # Original metadata extraction
        elif collection_type == "technical_docs":
            # Extract programming languages
            languages = re.findall(r"\b(?:python|javascript|java|cpp|c\+\+|rust|go|typescript)\b", content, re.I)
            if languages:
                metadata["programming_languages"] = list(set(lang.lower() for lang in languages[:5]))
            
            # Extract frameworks
            frameworks = re.findall(r"\b(?:react|angular|vue|django|flask|fastapi|spring|express)\b", content, re.I)
            if frameworks:
                metadata["frameworks"] = list(set(fw.lower() for fw in frameworks[:5]))
                
        elif collection_type == "policies_procedures":
            # Extract department
            dept_matches = re.findall(r"(?:department|dept)[:\s]+([A-Za-z\s]+)", content, re.I)
            if dept_matches:
                metadata["department"] = dept_matches[0].strip()
            
            # Extract effective date
            date_matches = re.findall(r"effective\s+(?:date|from)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", content, re.I)
            if date_matches:
                metadata["effective_date"] = date_matches[0]
                
        elif collection_type == "meeting_notes":
            # Extract meeting date
            date_patterns = [
                r"(?:date|meeting\s+date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+meeting",
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, content[:1000], re.I)
                if matches:
                    metadata["meeting_date"] = matches[0]
                    break
            
            # Extract attendees
            attendee_match = re.search(r"attendees[:\s]+([^\n]+)", content, re.I)
            if attendee_match:
                attendees = [a.strip() for a in attendee_match.group(1).split(",")]
                metadata["attendees"] = attendees[:10]  # Limit to 10
                
        elif collection_type == "contracts_legal":
            # Extract contract type
            contract_types = ["service", "employment", "nda", "purchase", "lease", "license"]
            for ct in contract_types:
                if re.search(rf"\b{ct}\s+(?:agreement|contract)\b", content, re.I):
                    metadata["contract_type"] = ct
                    break
            
            # Extract parties
            party_match = re.findall(r"(?:between|party)[:\s]+([A-Za-z\s&,\.]+?)(?:and|,|\n)", content[:2000], re.I)
            if party_match:
                metadata["parties"] = [p.strip() for p in party_match[:3]]
        
        return metadata
    
    def get_target_collection(self, collection_type: str) -> Optional[str]:
        """Get the best available collection for a document type"""
        collections = get_all_collections()
        
        # First, try to find exact match by collection_type
        for col in collections:
            if col["collection_type"] == collection_type:
                return col["collection_name"]
        
        # Special handling for partnership - look for partnership collection name
        if collection_type == "partnership":
            for col in collections:
                if col["collection_name"] == "partnership":
                    return col["collection_name"]
        
        # If no specific collection exists, use general collections with smart selection
        general_collections = [c for c in collections if c["collection_type"] == "general"]
        if general_collections:
            # For partnership classification, prefer partnership collection if it exists
            if collection_type == "partnership":
                partnership_collections = [c for c in general_collections if "partnership" in c["collection_name"].lower() or "partnership" in c["description"].lower()]
                if partnership_collections:
                    return partnership_collections[0]["collection_name"]
            
            # Default to first general collection (likely default_knowledge)
            return general_collections[0]["collection_name"]
        
        # No suitable collection found
        logger.warning(f"No suitable collection found for type {collection_type}")
        return None

# Global instance
_classifier = None

def get_document_classifier() -> DocumentClassifier:
    """Get or create the global document classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier