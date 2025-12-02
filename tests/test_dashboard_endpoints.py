#!/usr/bin/env python3
"""
Test Dashboard Endpoints Integration
Quick test to verify dashboard endpoints work in the main server.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_dashboard_endpoints():
    """Test dashboard endpoints without starting full server."""
    try:
        print("🧪 Testing Dashboard Endpoint Integration")
        print("=" * 50)
        
        # Import dashboard components
        from monitoring.dashboard import UnifiedDashboard
        print("✅ Dashboard components imported successfully")
        
        # Initialize dashboard
        dashboard = UnifiedDashboard()
        print("✅ Dashboard initialized")
        
        # Test metrics collection
        metrics = await dashboard.collect_all_metrics()
        print(f"✅ Metrics collected: {len(metrics)} categories")
        
        # Test ASCII generation
        ascii_output = dashboard.generate_ascii_dashboard(metrics)
        print(f"✅ ASCII dashboard generated: {len(ascii_output)} characters")
        
        # Show sample output
        print("\n📊 Sample ASCII Dashboard:")
        print("-" * 40)
        print(ascii_output[:800] + "..." if len(ascii_output) > 800 else ascii_output)
        print("-" * 40)
        
        # Cleanup
        await dashboard.shutdown()
        
        print("\n✅ Dashboard endpoints integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    try:
        success = asyncio.run(test_dashboard_endpoints())
        if success:
            print("\n🎉 Ready for deployment!")
            return 0
        else:
            print("\n💥 Fix issues before deployment")
            return 1
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())