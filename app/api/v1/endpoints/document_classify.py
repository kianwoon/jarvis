"""
Document Classification Endpoint

This endpoint allows documents to be classified without uploading them to the vector database.
Useful for previewing classification results and allowing users to confirm or override.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
import tempfile
import os
from pypdf import PdfReader
import logging
from app.core.document_classifier import get_document_classifier

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/classify")
async def classify_document(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Classify a document without storing it.
    
    Returns:
        Dictionary containing:
        - collection: Suggested collection name
        - collection_type: Collection type
        - confidence: Confidence score (0-1)
        - metadata: Extracted metadata
        - reason: Explanation for classification
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        # For now, only support PDF
        # Other file types can be added later
        raise HTTPException(
            status_code=400,
            detail="Currently only PDF files are supported for classification"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Extract text from PDF
        text_content = ""
        with open(tmp_file_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from first few pages for classification
            for page_num in range(min(5, num_pages)):  # Max 5 pages
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
        
        # Get classifier
        classifier = get_document_classifier()
        
        # Prepare metadata
        metadata = {
            "source": file.filename,
            "doc_type": "pdf",
            "num_pages": num_pages,
        }
        
        # Classify document
        collection_type, confidence = classifier.classify_by_patterns(text_content, metadata)
        
        # Get target collection
        target_collection = classifier.get_target_collection(collection_type)
        
        # Extract domain-specific metadata
        domain_metadata = classifier.extract_domain_metadata(text_content, collection_type)
        metadata.update(domain_metadata)
        
        # Determine classification reason
        reason = _get_classification_reason(collection_type, text_content, metadata, confidence)
        
        response = {
            "filename": file.filename,
            "collection": target_collection or "default_knowledge",
            "collection_type": collection_type,
            "confidence": confidence,
            "metadata": metadata,
            "reason": reason,
            "requires_confirmation": confidence < 0.5  # Suggest manual confirmation if low confidence
        }
        
        logger.info(f"Classified {file.filename} as {collection_type} with confidence {confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to classify document: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def _get_classification_reason(collection_type: str, content: str, metadata: Dict[str, Any], confidence: float) -> str:
    """Generate a human-readable reason for the classification"""
    
    reasons = []
    
    # Check filename hints
    filename = metadata.get("source", "").lower()
    
    if collection_type == "regulatory_compliance":
        if any(term in content.lower() for term in ["basel", "dodd-frank", "kyc", "aml"]):
            reasons.append("Contains regulatory terms (Basel, KYC, AML)")
        if "compliance" in filename:
            reasons.append("Filename suggests compliance document")
            
    elif collection_type == "product_documentation":
        if any(term in content.lower() for term in ["product", "pricing", "rate", "fee"]):
            reasons.append("Contains product and pricing information")
        if "product" in filename or "pricing" in filename:
            reasons.append("Filename indicates product documentation")
            
    elif collection_type == "risk_management":
        if any(term in content.lower() for term in ["risk", "control", "mitigation"]):
            reasons.append("Contains risk management terminology")
        if "risk" in filename:
            reasons.append("Filename suggests risk document")
            
    elif collection_type == "customer_support":
        if any(term in content.lower() for term in ["faq", "support", "help", "troubleshoot"]):
            reasons.append("Contains support and FAQ content")
        if "support" in filename or "faq" in filename:
            reasons.append("Filename indicates support documentation")
            
    elif collection_type == "audit_reports":
        if any(term in content.lower() for term in ["audit", "finding", "observation"]):
            reasons.append("Contains audit terminology")
        if "audit" in filename:
            reasons.append("Filename suggests audit report")
            
    elif collection_type == "training_materials":
        if any(term in content.lower() for term in ["training", "learning", "course"]):
            reasons.append("Contains training content")
        if "training" in filename:
            reasons.append("Filename indicates training material")
            
    elif collection_type == "technical_docs":
        if any(term in content.lower() for term in ["api", "code", "function", "implementation"]):
            reasons.append("Contains technical documentation")
        if "technical" in filename or "api" in filename:
            reasons.append("Filename suggests technical document")
            
    elif collection_type == "policies_procedures":
        if any(term in content.lower() for term in ["policy", "procedure", "guideline"]):
            reasons.append("Contains policy and procedure content")
        if "policy" in filename or "procedure" in filename:
            reasons.append("Filename indicates policy document")
            
    elif collection_type == "contracts_legal":
        if any(term in content.lower() for term in ["agreement", "contract", "legal"]):
            reasons.append("Contains legal terminology")
        if "contract" in filename or "agreement" in filename:
            reasons.append("Filename suggests legal document")
            
    elif collection_type == "meeting_notes":
        if any(term in content.lower() for term in ["meeting", "minutes", "agenda"]):
            reasons.append("Contains meeting-related content")
        if "meeting" in filename or "minutes" in filename:
            reasons.append("Filename indicates meeting notes")
    
    # Add confidence note
    if confidence < 0.3:
        reasons.append(f"Low confidence score ({confidence:.2f}) - manual selection recommended")
    elif confidence < 0.5:
        reasons.append(f"Medium confidence score ({confidence:.2f}) - please verify")
    else:
        reasons.append(f"High confidence score ({confidence:.2f})")
    
    # Default reason if none found
    if not reasons:
        reasons.append("No strong indicators found - defaulting to general collection")
    
    return " | ".join(reasons)