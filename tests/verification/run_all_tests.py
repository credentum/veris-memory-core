#!/usr/bin/env python3
"""
Run All Verification Tests
Comprehensive verification test runner for manual execution.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tls_verifier import TLSVerifier
from restore_drill import RestoreDrill


async def run_all_verification_tests():
    """Run all verification tests and generate comprehensive report."""
    print("🎯 COMPREHENSIVE VERIFICATION TEST SUITE")
    print("=" * 50)
    
    test_start = datetime.utcnow()
    all_results = {
        'test_suite': 'comprehensive_verification',
        'timestamp': test_start.isoformat(),
        'tests': {}
    }
    
    try:
        # 1. TLS/mTLS Verification
        print("\n1. 🔒 TLS/mTLS Verification Test")
        print("-" * 35)
        
        tls_verifier = TLSVerifier()
        tls_results = await tls_verifier.run_comprehensive_verification()
        all_results['tests']['tls_verification'] = tls_results
        
        tls_summary = tls_results.get('summary', {})
        print(f"   Services Tested: {tls_summary.get('total_services_tested', 0)}")
        print(f"   TLS Enabled: {tls_summary.get('tls_enabled_services', 0)}")
        print(f"   Valid Certificates: {tls_summary.get('valid_certificates', 0)}")
        print(f"   Overall Health: {tls_summary.get('overall_tls_health', 'UNKNOWN')}")
        
        # 2. Backup Restore Drill
        print("\n2. 💾 Backup Restore Drill Test")
        print("-" * 32)
        
        restore_drill = RestoreDrill()
        restore_results = await restore_drill.execute_comprehensive_drill()
        all_results['tests']['restore_drill'] = restore_results
        
        restore_summary = restore_results.get('summary', {})
        print(f"   Databases Tested: {restore_summary.get('databases_tested', 0)}")
        print(f"   Successful Restores: {restore_summary.get('successful_restores', 0)}")
        print(f"   Target Compliance: {'✅ YES' if restore_summary.get('target_compliance') else '❌ NO'}")
        print(f"   Data Integrity: {restore_summary.get('data_integrity_pass_rate', 0):.1f}%")
        
        # Overall Assessment
        test_end = datetime.utcnow()
        total_duration = (test_end - test_start).total_seconds()
        
        all_results['completion_time'] = test_end.isoformat()
        all_results['total_duration_seconds'] = total_duration
        
        # Determine overall status
        tls_healthy = tls_summary.get('overall_tls_health') in ['PASS', 'WARNING']
        restore_compliant = restore_summary.get('target_compliance', False)
        restore_integrity = restore_summary.get('data_integrity_pass_rate', 0) >= 95.0
        
        overall_status = 'PASS' if tls_healthy and restore_compliant and restore_integrity else 'FAIL'
        all_results['overall_status'] = overall_status
        
        print(f"\n📊 OVERALL TEST RESULTS")
        print("=" * 30)
        print(f"Total Duration: {total_duration:.1f}s")
        print(f"TLS Health: {tls_summary.get('overall_tls_health', 'UNKNOWN')}")
        print(f"Restore Compliance: {'✅ PASS' if restore_compliant else '❌ FAIL'}")
        print(f"Data Integrity: {'✅ PASS' if restore_integrity else '❌ FAIL'}")
        print(f"Overall Status: {'🎉 PASS' if overall_status == 'PASS' else '🚨 FAIL'}")
        
        # Recommendations
        recommendations = []
        
        if not tls_healthy:
            recommendations.append("Review TLS configuration and certificate status")
        
        if not restore_compliant:
            recommendations.append("Optimize backup restore procedures to meet <300s target")
        
        if not restore_integrity:
            recommendations.append("Investigate data integrity issues in restore process")
        
        if len(tls_results.get('errors', [])) > 0:
            recommendations.append("Address TLS verification errors")
        
        if len(restore_results.get('errors', [])) > 0:
            recommendations.append("Address restore drill errors")
        
        if not recommendations:
            recommendations.append("All verification tests meet requirements")
        
        all_results['recommendations'] = recommendations
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        all_results['success'] = False
        all_results['error'] = str(e)
        all_results['overall_status'] = 'ERROR'
        
        return all_results


async def run_quick_test():
    """Run a quick verification test with minimal output."""
    print("⚡ Quick Verification Test")
    print("-" * 25)
    
    try:
        # Quick TLS test
        tls_verifier = TLSVerifier()
        tls_results = await tls_verifier.run_comprehensive_verification()
        tls_health = tls_results.get('summary', {}).get('overall_tls_health', 'UNKNOWN')
        
        # Quick restore test  
        restore_drill = RestoreDrill()
        restore_results = await restore_drill.execute_comprehensive_drill()
        restore_compliance = restore_results.get('summary', {}).get('target_compliance', False)
        
        print(f"TLS Health: {tls_health}")
        print(f"Restore Compliance: {'✅ PASS' if restore_compliance else '❌ FAIL'}")
        
        overall_ok = tls_health in ['PASS', 'WARNING'] and restore_compliance
        print(f"Quick Status: {'🎉 OK' if overall_ok else '🚨 ISSUES'}")
        
        return overall_ok
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick test mode
        try:
            result = asyncio.run(run_quick_test())
            return 0 if result else 1
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
            return 130
        except Exception as e:
            print(f"\n💥 Test crashed: {e}")
            return 1
    else:
        # Full test suite
        try:
            results = asyncio.run(run_all_verification_tests())
            return 0 if results.get('overall_status') == 'PASS' else 1
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
            return 130
        except Exception as e:
            print(f"\n💥 Test crashed: {e}")
            return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    sys.exit(main())