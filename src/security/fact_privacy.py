"""
Privacy and security controls for fact data.

This module implements comprehensive privacy protection, data classification,
and security controls for personal fact storage and retrieval.
"""

import re
import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for facts."""
    PUBLIC = "public"           # Non-sensitive, shareable
    INTERNAL = "internal"       # Internal use, limited sharing
    CONFIDENTIAL = "confidential"  # Sensitive, restricted access
    RESTRICTED = "restricted"   # Highly sensitive, minimal access


class PIIType(Enum):
    """Types of personally identifiable information."""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    FINANCIAL = "financial"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    MEDICAL = "medical"
    BEHAVIORAL = "behavioral"


class RedactionLevel(Enum):
    """Levels of data redaction."""
    NONE = "none"           # No redaction
    PARTIAL = "partial"     # Partial masking (e.g., email -> e***@***.com)
    FULL = "full"          # Complete replacement (e.g., email -> [EMAIL])
    HASH = "hash"          # Cryptographic hash
    REMOVE = "remove"      # Complete removal


@dataclass
class PrivacyPolicy:
    """Privacy policy configuration."""
    data_classification: DataClassification
    retention_days: int
    redaction_level: RedactionLevel
    allowed_operations: Set[str]
    geographic_restrictions: List[str]
    sharing_permissions: Dict[str, bool]
    audit_required: bool


@dataclass
class DataClassificationResult:
    """Result of data classification analysis."""
    classification: DataClassification
    pii_types: List[PIIType]
    confidence: float
    reasoning: str
    recommended_policy: PrivacyPolicy


@dataclass
class AccessAuditEntry:
    """Audit entry for fact access."""
    timestamp: float
    user_id: str
    operation: str
    fact_attribute: str
    classification: DataClassification
    access_granted: bool
    justification: str
    client_info: Dict[str, Any]


class PIIDetector:
    """
    Advanced PII detection with pattern matching and context analysis.
    """
    
    def __init__(self):
        # Compiled patterns for efficient matching
        self.patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            PIIType.SSN: re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            PIIType.ADDRESS: re.compile(r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl)\b', re.IGNORECASE),
            PIIType.FINANCIAL: re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),  # Credit card pattern
            PIIType.LOCATION: re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # ZIP code pattern
        }
        
        # Context indicators for enhanced detection
        self.context_indicators = {
            PIIType.NAME: ['name', 'called', 'i am', 'i\'m'],
            PIIType.EMAIL: ['email', 'contact', 'reach'],
            PIIType.PHONE: ['phone', 'number', 'call'],
            PIIType.ADDRESS: ['address', 'live', 'reside'],
            PIIType.MEDICAL: ['diagnosis', 'condition', 'medication', 'doctor'],
            PIIType.FINANCIAL: ['salary', 'income', 'account', 'balance'],
            PIIType.BEHAVIORAL: ['prefer', 'like', 'habit', 'routine']
        }
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIType]:
        """Detect PII types in text with context analysis."""
        detected_pii = []
        text_lower = text.lower()
        context_lower = context.lower()
        
        # Pattern-based detection
        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                detected_pii.append(pii_type)
        
        # Context-based detection
        for pii_type, indicators in self.context_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                # Special handling for names (look for capitalized words)
                if pii_type == PIIType.NAME:
                    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                        detected_pii.append(pii_type)
                # Check for behavioral patterns
                elif pii_type == PIIType.BEHAVIORAL:
                    if any(word in text_lower for word in ['like', 'prefer', 'enjoy', 'hate']):
                        detected_pii.append(pii_type)
        
        # Location detection (enhanced)
        location_indicators = ['live', 'from', 'located', 'city', 'state', 'country']
        if any(indicator in context_lower for indicator in location_indicators):
            # Look for location-like patterns
            if re.search(r'\b[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*\b', text):
                detected_pii.append(PIIType.LOCATION)
        
        return list(set(detected_pii))  # Remove duplicates


class DataClassifier:
    """
    Automatic data classification based on content analysis.
    """
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        
        # Classification rules
        self.classification_rules = {
            DataClassification.RESTRICTED: {
                'pii_types': [PIIType.SSN, PIIType.FINANCIAL, PIIType.MEDICAL, PIIType.BIOMETRIC],
                'keywords': ['password', 'secret', 'confidential', 'ssn', 'medical']
            },
            DataClassification.CONFIDENTIAL: {
                'pii_types': [PIIType.EMAIL, PIIType.PHONE, PIIType.ADDRESS],
                'keywords': ['personal', 'private', 'contact', 'address']
            },
            DataClassification.INTERNAL: {
                'pii_types': [PIIType.NAME, PIIType.LOCATION, PIIType.BEHAVIORAL],
                'keywords': ['preference', 'name', 'location']
            },
            DataClassification.PUBLIC: {
                'pii_types': [],
                'keywords': ['public', 'general', 'common']
            }
        }
    
    def classify_fact(self, attribute: str, value: Any, context: str = "") -> DataClassificationResult:
        """Classify a fact based on its content and context."""
        text_content = f"{attribute} {value} {context}".lower()
        detected_pii = self.pii_detector.detect_pii(str(value), f"{attribute} {context}")
        
        # Determine classification based on PII and keywords
        classification = DataClassification.PUBLIC
        confidence = 0.5
        reasoning_parts = []
        
        for level, rules in self.classification_rules.items():
            score = 0
            
            # Check PII types
            pii_matches = set(detected_pii).intersection(set(rules['pii_types']))
            if pii_matches:
                score += len(pii_matches) * 0.4
                reasoning_parts.append(f"Contains {', '.join(p.value for p in pii_matches)} PII")
            
            # Check keywords
            keyword_matches = [kw for kw in rules['keywords'] if kw in text_content]
            if keyword_matches:
                score += len(keyword_matches) * 0.2
                reasoning_parts.append(f"Keyword matches: {', '.join(keyword_matches)}")
            
            # Update classification if score is significant
            if score > 0.3:
                classification = level
                confidence = min(0.95, 0.5 + score)
                break
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No sensitive indicators detected"
        
        # Generate recommended policy
        recommended_policy = self._generate_policy(classification, detected_pii)
        
        return DataClassificationResult(
            classification=classification,
            pii_types=detected_pii,
            confidence=confidence,
            reasoning=reasoning,
            recommended_policy=recommended_policy
        )
    
    def _generate_policy(self, classification: DataClassification, pii_types: List[PIIType]) -> PrivacyPolicy:
        """Generate recommended privacy policy based on classification."""
        policy_templates = {
            DataClassification.RESTRICTED: {
                'retention_days': 30,
                'redaction_level': RedactionLevel.HASH,
                'allowed_operations': {'read', 'delete'},
                'audit_required': True
            },
            DataClassification.CONFIDENTIAL: {
                'retention_days': 90,
                'redaction_level': RedactionLevel.PARTIAL,
                'allowed_operations': {'read', 'write', 'delete'},
                'audit_required': True
            },
            DataClassification.INTERNAL: {
                'retention_days': 365,
                'redaction_level': RedactionLevel.NONE,
                'allowed_operations': {'read', 'write', 'delete', 'share_internal'},
                'audit_required': False
            },
            DataClassification.PUBLIC: {
                'retention_days': 1095,  # 3 years
                'redaction_level': RedactionLevel.NONE,
                'allowed_operations': {'read', 'write', 'delete', 'share_internal', 'share_external'},
                'audit_required': False
            }
        }
        
        template = policy_templates[classification]
        
        return PrivacyPolicy(
            data_classification=classification,
            retention_days=template['retention_days'],
            redaction_level=template['redaction_level'],
            allowed_operations=set(template['allowed_operations']),
            geographic_restrictions=[],  # Default: no restrictions
            sharing_permissions={
                'internal': 'share_internal' in template['allowed_operations'],
                'external': 'share_external' in template['allowed_operations']
            },
            audit_required=template['audit_required']
        )


class DataRedactor:
    """
    Privacy-preserving data redaction with multiple strategies.
    """
    
    def __init__(self, salt: str = "default_salt"):
        self.salt = salt
        self.pii_detector = PIIDetector()
    
    def redact_data(self, data: str, redaction_level: RedactionLevel, 
                   pii_types: Optional[List[PIIType]] = None) -> str:
        """Redact data according to specified level and PII types."""
        if redaction_level == RedactionLevel.NONE:
            return data
        
        if redaction_level == RedactionLevel.REMOVE:
            return ""
        
        # Detect PII if not provided
        if pii_types is None:
            pii_types = self.pii_detector.detect_pii(data)
        
        redacted = data
        
        for pii_type in pii_types:
            if pii_type == PIIType.EMAIL:
                redacted = self._redact_email(redacted, redaction_level)
            elif pii_type == PIIType.PHONE:
                redacted = self._redact_phone(redacted, redaction_level)
            elif pii_type == PIIType.NAME:
                redacted = self._redact_name(redacted, redaction_level)
            elif pii_type == PIIType.ADDRESS:
                redacted = self._redact_address(redacted, redaction_level)
            elif pii_type == PIIType.FINANCIAL:
                redacted = self._redact_financial(redacted, redaction_level)
        
        return redacted
    
    def _redact_email(self, text: str, level: RedactionLevel) -> str:
        """Redact email addresses."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        if level == RedactionLevel.PARTIAL:
            def partial_redact(match):
                email = match.group(0)
                if '@' in email:
                    local, domain = email.split('@', 1)
                    local_redacted = local[0] + '*' * (len(local) - 1) if len(local) > 1 else '*'
                    domain_parts = domain.split('.')
                    domain_redacted = '*' * len(domain_parts[0]) + '.' + '.'.join(domain_parts[1:])
                    return f"{local_redacted}@{domain_redacted}"
                return email
            return re.sub(pattern, partial_redact, text)
        
        elif level == RedactionLevel.FULL:
            return re.sub(pattern, '[EMAIL]', text)
        
        elif level == RedactionLevel.HASH:
            def hash_email(match):
                return self._hash_value(match.group(0))
            return re.sub(pattern, hash_email, text)
        
        return text
    
    def _redact_phone(self, text: str, level: RedactionLevel) -> str:
        """Redact phone numbers."""
        pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        
        if level == RedactionLevel.PARTIAL:
            return re.sub(pattern, r'\1(\2) ***-\4', text)
        elif level == RedactionLevel.FULL:
            return re.sub(pattern, '[PHONE]', text)
        elif level == RedactionLevel.HASH:
            def hash_phone(match):
                return self._hash_value(match.group(0))
            return re.sub(pattern, hash_phone, text)
        
        return text
    
    def _redact_name(self, text: str, level: RedactionLevel) -> str:
        """Redact person names."""
        # Simple name pattern (can be enhanced)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        
        if level == RedactionLevel.PARTIAL:
            def partial_name(match):
                name = match.group(0)
                parts = name.split()
                if len(parts) >= 2:
                    return f"{parts[0][0]}*** {parts[-1][0]}***"
                return name[0] + '*' * (len(name) - 1)
            return re.sub(pattern, partial_name, text)
        
        elif level == RedactionLevel.FULL:
            return re.sub(pattern, '[NAME]', text)
        
        elif level == RedactionLevel.HASH:
            def hash_name(match):
                return self._hash_value(match.group(0))
            return re.sub(pattern, hash_name, text)
        
        return text
    
    def _redact_address(self, text: str, level: RedactionLevel) -> str:
        """Redact addresses."""
        pattern = r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl)\b'
        
        if level == RedactionLevel.PARTIAL:
            def partial_address(match):
                addr = match.group(0)
                return '*** ' + addr.split()[-1]  # Keep street type
            return re.sub(pattern, partial_address, text, flags=re.IGNORECASE)
        
        elif level == RedactionLevel.FULL:
            return re.sub(pattern, '[ADDRESS]', text, flags=re.IGNORECASE)
        
        elif level == RedactionLevel.HASH:
            def hash_address(match):
                return self._hash_value(match.group(0))
            return re.sub(pattern, hash_address, text, flags=re.IGNORECASE)
        
        return text
    
    def _redact_financial(self, text: str, level: RedactionLevel) -> str:
        """Redact financial information."""
        pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'  # Credit card pattern
        
        if level == RedactionLevel.PARTIAL:
            def partial_financial(match):
                number = re.sub(r'[-\s]', '', match.group(0))
                return f"****-****-****-{number[-4:]}"
            return re.sub(pattern, partial_financial, text)
        
        elif level == RedactionLevel.FULL:
            return re.sub(pattern, '[FINANCIAL]', text)
        
        elif level == RedactionLevel.HASH:
            def hash_financial(match):
                return self._hash_value(match.group(0))
            return re.sub(pattern, hash_financial, text)
        
        return text
    
    def _hash_value(self, value: str) -> str:
        """Create a consistent hash of a value."""
        return hashlib.sha256((value + self.salt).encode()).hexdigest()[:8]


class PrivacyEnforcer:
    """
    Enforce privacy policies and access controls for fact data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.classifier = DataClassifier()
        self.redactor = DataRedactor(salt=self.config.get('redaction_salt', 'default'))
        
        # Audit trail
        self.audit_trail: List[AccessAuditEntry] = []
        self.max_audit_entries = self.config.get('max_audit_entries', 10000)
        
        # Default policies
        self.default_policies = {
            'retention_days': self.config.get('default_retention_days', 365),
            'redaction_level': RedactionLevel(self.config.get('default_redaction_level', 'none')),
            'audit_all_access': self.config.get('audit_all_access', False)
        }
    
    def apply_privacy_policy(self, attribute: str, value: Any, 
                           policy: Optional[PrivacyPolicy] = None) -> Tuple[Any, PrivacyPolicy]:
        """Apply privacy policy to fact data."""
        # Classify data if no policy provided
        if policy is None:
            classification_result = self.classifier.classify_fact(attribute, value)
            policy = classification_result.recommended_policy
        
        # Apply redaction
        if isinstance(value, str) and policy.redaction_level != RedactionLevel.NONE:
            classification_result = self.classifier.classify_fact(attribute, value)
            redacted_value = self.redactor.redact_data(
                value, policy.redaction_level, classification_result.pii_types
            )
            return redacted_value, policy
        
        return value, policy
    
    def check_access_permission(self, operation: str, attribute: str, value: Any,
                              user_context: Dict[str, Any], 
                              policy: Optional[PrivacyPolicy] = None) -> Tuple[bool, str]:
        """Check if access is permitted under privacy policy."""
        # Classify if no policy provided
        if policy is None:
            classification_result = self.classifier.classify_fact(attribute, value)
            policy = classification_result.recommended_policy
        
        # Check operation permission
        if operation not in policy.allowed_operations:
            return False, f"Operation '{operation}' not permitted for {policy.data_classification.value} data"
        
        # Check geographic restrictions
        user_location = user_context.get('location', 'unknown')
        if policy.geographic_restrictions and user_location in policy.geographic_restrictions:
            return False, f"Access restricted from location: {user_location}"
        
        # Check retention period
        fact_age_days = user_context.get('fact_age_days', 0)
        if fact_age_days > policy.retention_days:
            return False, f"Data exceeds retention period ({policy.retention_days} days)"
        
        return True, "Access permitted"
    
    def log_access(self, user_id: str, operation: str, attribute: str, 
                  classification: DataClassification, access_granted: bool,
                  justification: str, client_info: Optional[Dict[str, Any]] = None) -> None:
        """Log access attempt for audit trail."""
        entry = AccessAuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            operation=operation,
            fact_attribute=attribute,
            classification=classification,
            access_granted=access_granted,
            justification=justification,
            client_info=client_info or {}
        )
        
        self.audit_trail.append(entry)
        
        # Maintain size limit
        if len(self.audit_trail) > self.max_audit_entries:
            self.audit_trail = self.audit_trail[-self.max_audit_entries//2:]
        
        logger.info(f"Access {'granted' if access_granted else 'denied'}: {user_id} -> {attribute} ({operation})")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_entries = [e for e in self.audit_trail if e.timestamp >= cutoff_time]
        
        if not recent_entries:
            return {"message": "No audit entries in specified period"}
        
        # Aggregate statistics
        total_accesses = len(recent_entries)
        granted_accesses = sum(1 for e in recent_entries if e.access_granted)
        denied_accesses = total_accesses - granted_accesses
        
        # Group by classification
        by_classification = {}
        for entry in recent_entries:
            key = entry.classification.value
            if key not in by_classification:
                by_classification[key] = {'granted': 0, 'denied': 0}
            if entry.access_granted:
                by_classification[key]['granted'] += 1
            else:
                by_classification[key]['denied'] += 1
        
        # Group by operation
        by_operation = {}
        for entry in recent_entries:
            key = entry.operation
            if key not in by_operation:
                by_operation[key] = {'granted': 0, 'denied': 0}
            if entry.access_granted:
                by_operation[key]['granted'] += 1
            else:
                by_operation[key]['denied'] += 1
        
        return {
            "time_range_hours": hours,
            "total_accesses": total_accesses,
            "granted_accesses": granted_accesses,
            "denied_accesses": denied_accesses,
            "success_rate": granted_accesses / total_accesses if total_accesses > 0 else 0,
            "by_classification": by_classification,
            "by_operation": by_operation,
            "recent_denials": [
                {
                    "timestamp": entry.timestamp,
                    "user_id": entry.user_id,
                    "operation": entry.operation,
                    "attribute": entry.fact_attribute,
                    "reason": entry.justification
                }
                for entry in recent_entries[-10:] if not entry.access_granted
            ]
        }
    
    def export_privacy_report(self) -> Dict[str, Any]:
        """Export comprehensive privacy compliance report."""
        return {
            "privacy_configuration": {
                "default_retention_days": self.default_policies['retention_days'],
                "default_redaction_level": self.default_policies['redaction_level'].value,
                "audit_all_access": self.default_policies['audit_all_access']
            },
            "audit_summary_24h": self.get_audit_summary(24),
            "audit_summary_7d": self.get_audit_summary(168),  # 7 days
            "total_audit_entries": len(self.audit_trail),
            "supported_classifications": [c.value for c in DataClassification],
            "supported_redaction_levels": [r.value for r in RedactionLevel],
            "pii_types_detected": [p.value for p in PIIType]
        }


# Privacy-aware fact wrapper
class PrivacyAwareFact:
    """Wrapper for facts with integrated privacy controls."""
    
    def __init__(self, attribute: str, value: Any, privacy_enforcer: PrivacyEnforcer,
                 user_context: Optional[Dict[str, Any]] = None):
        self.attribute = attribute
        self._original_value = value
        self.privacy_enforcer = privacy_enforcer
        self.user_context = user_context or {}
        
        # Classify and apply privacy policy
        self._classification_result = privacy_enforcer.classifier.classify_fact(attribute, value)
        self._policy = self._classification_result.recommended_policy
        
        # Apply redaction
        self._processed_value, self._policy = privacy_enforcer.apply_privacy_policy(
            attribute, value, self._policy
        )
    
    @property
    def value(self) -> Any:
        """Get privacy-processed value."""
        return self._processed_value
    
    @property
    def classification(self) -> DataClassification:
        """Get data classification."""
        return self._classification_result.classification
    
    @property
    def policy(self) -> PrivacyPolicy:
        """Get privacy policy."""
        return self._policy
    
    def access_with_permission(self, operation: str, user_id: str,
                             client_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, str]:
        """Access fact value with permission checking."""
        # Check permissions
        permitted, reason = self.privacy_enforcer.check_access_permission(
            operation, self.attribute, self._original_value, self.user_context, self._policy
        )
        
        # Log access attempt
        self.privacy_enforcer.log_access(
            user_id, operation, self.attribute, self.classification,
            permitted, reason, client_info
        )
        
        if permitted:
            return True, self.value, reason
        else:
            return False, None, reason
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get fact metadata including privacy information."""
        return {
            "attribute": self.attribute,
            "classification": self.classification.value,
            "pii_types": [p.value for p in self._classification_result.pii_types],
            "confidence": self._classification_result.confidence,
            "reasoning": self._classification_result.reasoning,
            "redaction_level": self._policy.redaction_level.value,
            "retention_days": self._policy.retention_days,
            "audit_required": self._policy.audit_required
        }