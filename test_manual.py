#!/usr/bin/env python3
"""
Manual test to verify SPIRED-Stab MCP tools work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_validate_files():
    """Test file validation tool."""
    print("Testing validate_input_files...")

    from server import validate_input_files

    # Test with small CSV
    result = validate_input_files("examples/test_small.csv")
    print(f"Result: {result}")

    return result.get("status") == "success"

def test_list_examples():
    """Test list example data tool."""
    print("Testing list_example_data...")

    from server import list_example_data

    result = list_example_data()
    print(f"Result: {result}")

    return result.get("status") == "success"

def test_job_listing():
    """Test job listing."""
    print("Testing list_jobs...")

    from server import list_jobs

    result = list_jobs()
    print(f"Result: {result}")

    return result.get("total") is not None

def main():
    """Run manual tests."""
    print("SPIRED-Stab MCP Manual Tests")
    print("=" * 40)

    tests = [
        ("File Validation", test_validate_files),
        ("List Examples", test_list_examples),
        ("Job Listing", test_job_listing),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print(f"\n{'='*40}")
    print(f"Manual Tests: {passed}/{len(tests)} passed")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)