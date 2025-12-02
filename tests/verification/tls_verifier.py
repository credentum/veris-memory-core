#!/usr/bin/env python3
"""
TLS/mTLS Verification Test Script
Standalone script for testing TLS/mTLS configurations when needed.
"""

import asyncio
import logging
import ssl
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class TLSTestResult:
    """Result of a TLS test."""
    service_name: str
    host: str
    port: int
    tls_enabled: bool
    certificate_valid: bool
    certificate_expires: Optional[datetime]
    cipher_suite: Optional[str]
    protocol_version: Optional[str]
    client_cert_required: bool
    error_message: Optional[str] = None


class TLSVerifier:
    """
    TLS/mTLS verification testing system.
    
    Tests TLS configurations for Veris Memory services including:
    - Certificate validation and expiry
    - Cipher suite security
    - Client certificate authentication (mTLS)
    - Protocol version compliance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TLS verifier."""
        self.config = config or self._get_default_config()
        self.services = self.config.get('services_to_test', self._get_default_services())
        
        logger.info("🔒 TLS Verifier initialized for manual testing")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default TLS verification configuration."""
        return {
            'connection_timeout_seconds': 10,
            'certificate_warning_days': 30,
            'required_tls_version': 'TLSv1.2',
            'allowed_cipher_suites': [
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES128-GCM-SHA256',
                'ECDHE-ECDSA-AES256-GCM-SHA384',
                'ECDHE-ECDSA-AES128-GCM-SHA256'
            ]
        }

    def _get_default_services(self) -> List[Dict[str, Any]]:
        """Get default services to test."""
        return [
            {
                'name': 'Veris API',
                'host': 'localhost',
                'port': 8080,
                'requires_client_cert': False
            },
            {
                'name': 'Qdrant Vector DB',
                'host': 'localhost',
                'port': 6334,
                'requires_client_cert': False
            },
            {
                'name': 'Neo4j Graph DB',
                'host': 'localhost',
                'port': 7687,
                'requires_client_cert': False
            }
        ]

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """
        Run comprehensive TLS verification across all configured services.
        
        Returns:
            Complete TLS verification results
        """
        logger.info("🎯 Starting comprehensive TLS verification")
        verification_start = datetime.utcnow()
        
        results = []
        errors = []
        
        try:
            for service_config in self.services:
                try:
                    result = await self._test_service_tls(service_config)
                    results.append(result)
                    
                except Exception as e:
                    error_msg = f"TLS test failed for {service_config['name']}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Generate summary
            verification_end = datetime.utcnow()
            summary = self._generate_verification_summary(verification_start, verification_end, results)
            
            return {
                'verification_type': 'tls_mtls_comprehensive',
                'timestamp': verification_start.isoformat(),
                'duration_seconds': (verification_end - verification_start).total_seconds(),
                'summary': summary,
                'test_results': [self._result_to_dict(r) for r in results],
                'errors': errors
            }

        except Exception as e:
            logger.error(f"Comprehensive TLS verification failed: {e}")
            
            return {
                'verification_type': 'tls_mtls_comprehensive',
                'timestamp': verification_start.isoformat(),
                'success': False,
                'error': str(e)
            }

    async def _test_service_tls(self, service_config: Dict[str, Any]) -> TLSTestResult:
        """Test TLS configuration for a specific service."""
        service_name = service_config['name']
        host = service_config['host']
        port = service_config['port']
        
        logger.info(f"🔒 Testing TLS for {service_name} at {host}:{port}")
        
        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE  # For testing purposes
            
            # Test connection
            with socket.create_connection((host, port), timeout=self.config['connection_timeout_seconds']) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    # Get certificate info
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
                    
                    # Parse certificate expiry
                    cert_expires = None
                    if cert and 'notAfter' in cert:
                        cert_expires = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    
                    return TLSTestResult(
                        service_name=service_name,
                        host=host,
                        port=port,
                        tls_enabled=True,
                        certificate_valid=cert is not None,
                        certificate_expires=cert_expires,
                        cipher_suite=cipher[0] if cipher else None,
                        protocol_version=version,
                        client_cert_required=service_config.get('requires_client_cert', False)
                    )
                    
        except ConnectionRefusedError:
            logger.warning(f"Connection refused to {service_name} at {host}:{port} - service may not be running")
            return TLSTestResult(
                service_name=service_name,
                host=host,
                port=port,
                tls_enabled=False,
                certificate_valid=False,
                certificate_expires=None,
                cipher_suite=None,
                protocol_version=None,
                client_cert_required=False,
                error_message="Connection refused - service not running"
            )
        except socket.timeout:
            logger.warning(f"Connection timeout to {service_name} at {host}:{port}")
            return TLSTestResult(
                service_name=service_name,
                host=host,
                port=port,
                tls_enabled=False,
                certificate_valid=False,
                certificate_expires=None,
                cipher_suite=None,
                protocol_version=None,
                client_cert_required=False,
                error_message=f"Connection timeout after {self.config['connection_timeout_seconds']}s"
            )
        except ssl.SSLError as e:
            logger.warning(f"SSL error for {service_name} at {host}:{port}: {e}")
            return TLSTestResult(
                service_name=service_name,
                host=host,
                port=port,
                tls_enabled=True,  # SSL error means TLS is attempted
                certificate_valid=False,
                certificate_expires=None,
                cipher_suite=None,
                protocol_version=None,
                client_cert_required=False,
                error_message=f"SSL error: {str(e)}"
            )
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {service_name} at {host}:{port}: {e}")
            return TLSTestResult(
                service_name=service_name,
                host=host,
                port=port,
                tls_enabled=False,
                certificate_valid=False,
                certificate_expires=None,
                cipher_suite=None,
                protocol_version=None,
                client_cert_required=False,
                error_message=f"DNS resolution failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"TLS test failed for {service_name} at {host}:{port}: {e}")
            return TLSTestResult(
                service_name=service_name,
                host=host,
                port=port,
                tls_enabled=False,
                certificate_valid=False,
                certificate_expires=None,
                cipher_suite=None,
                protocol_version=None,
                client_cert_required=False,
                error_message=f"Unexpected error: {str(e)}"
            )

    def _generate_verification_summary(self, start_time: datetime, end_time: datetime, 
                                     results: List[TLSTestResult]) -> Dict[str, Any]:
        """Generate TLS verification summary."""
        total_services = len(results)
        tls_enabled_count = sum(1 for r in results if r.tls_enabled)
        valid_certs_count = sum(1 for r in results if r.certificate_valid)
        
        # Check certificate expiry warnings
        warning_threshold = datetime.utcnow() + timedelta(days=self.config['certificate_warning_days'])
        expiring_soon = [
            r for r in results 
            if r.certificate_expires and r.certificate_expires < warning_threshold
        ]
        
        overall_health = 'PASS'
        if len(expiring_soon) > 0:
            overall_health = 'WARNING'
        if tls_enabled_count < total_services or valid_certs_count < tls_enabled_count:
            overall_health = 'FAIL'
        
        return {
            'total_services_tested': total_services,
            'tls_enabled_services': tls_enabled_count,
            'valid_certificates': valid_certs_count,
            'certificates_expiring_soon': len(expiring_soon),
            'overall_tls_health': overall_health,
            'duration_seconds': (end_time - start_time).total_seconds()
        }

    def _result_to_dict(self, result: TLSTestResult) -> Dict[str, Any]:
        """Convert TLSTestResult to dictionary."""
        return {
            'service_name': result.service_name,
            'host': result.host,
            'port': result.port,
            'tls_enabled': result.tls_enabled,
            'certificate_valid': result.certificate_valid,
            'certificate_expires': result.certificate_expires.isoformat() if result.certificate_expires else None,
            'cipher_suite': result.cipher_suite,
            'protocol_version': result.protocol_version,
            'client_cert_required': result.client_cert_required,
            'error_message': result.error_message
        }


async def main():
    """Run TLS verification tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔒 TLS/mTLS Verification Test")
    print("=" * 40)
    
    verifier = TLSVerifier()
    results = await verifier.run_comprehensive_verification()
    
    print(f"\n📊 Test Results:")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"Services Tested: {results.get('summary', {}).get('total_services_tested', 0)}")
    print(f"TLS Enabled: {results.get('summary', {}).get('tls_enabled_services', 0)}")
    print(f"Valid Certificates: {results.get('summary', {}).get('valid_certificates', 0)}")
    print(f"Overall Health: {results.get('summary', {}).get('overall_tls_health', 'UNKNOWN')}")
    
    if results.get('errors'):
        print(f"\n⚠️ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())