#!/usr/bin/env python3
"""
Test runner script for the EmbeddingGemma project.

This script provides a convenient way to run different types of tests
with various configurations and options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description="Running command"):
    """Run a shell command and return the result."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def run_fast_tests():
    """Run fast unit tests only."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "unit", "-v", "--tb=short"]
    return run_command(cmd, "Running fast unit tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/test_integration.py", "-v", "--tb=short"]
    return run_command(cmd, "Running integration tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/test_performance.py", "-k", "not test_extreme_parameter_values", "-v", "--tb=short"]
    return run_command(cmd, "Running performance tests")


def run_all_tests():
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    return run_command(cmd, "Running all tests")


def run_specific_test(test_path):
    """Run a specific test file."""
    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
    return run_command(cmd, f"Running {test_path}")


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src/embeddinggemma", "--cov-report=html", "--cov-report=term", "-v", "--tb=short"]
    success = run_command(cmd, "Running tests with coverage")

    if success:
        print("\n📊 Coverage report generated in htmlcov/index.html")
        return True
    return False


def run_linting():
    """Run code linting and formatting checks."""
    success = True

    # Black formatting check
    if not run_command(["python", "-m", "black", "--check", "src/", "tests/"], "Checking code formatting"):
        success = False

    # Flake8 linting
    if not run_command(["python", "-m", "flake8", "src/", "tests/", "--max-line-length=100"], "Running linting"):
        success = False

    # MyPy type checking
    if not run_command(["python", "-m", "mypy", "src/", "--ignore-missing-imports"], "Running type checking"):
        success = False

    return success


def run_security_scan():
    """Run security vulnerability scan."""
    try:
        import safety
        cmd = ["python", "-m", "safety", "check"]
        return run_command(cmd, "Running security scan")
    except ImportError:
        print("⚠️  Safety not installed. Install with: pip install safety")
        return False


def run_ci_simulation():
    """Simulate the full CI pipeline locally."""
    print("\n🚀 Simulating full CI pipeline...")

    tests_passed = 0
    total_tests = 6

    # 1. Unit tests
    if run_fast_tests():
        tests_passed += 1

    # 2. Integration tests
    if run_integration_tests():
        tests_passed += 1

    # 3. Performance tests
    if run_performance_tests():
        tests_passed += 1

    # 4. Linting
    if run_linting():
        tests_passed += 1

    # 5. Security scan
    if run_security_scan():
        tests_passed += 1

    # 6. Coverage
    if run_tests_with_coverage():
        tests_passed += 1

    print(f"\n{'='*60}")
    print(f"CI SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 All CI checks passed!")
        return True
    else:
        print("❌ Some CI checks failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EmbeddingGemma Test Runner")

    parser.add_argument(
        "action",
        choices=["fast", "integration", "performance", "all", "coverage", "lint", "security", "ci", "specific"],
        help="Type of tests to run"
    )

    parser.add_argument(
        "--test-path",
        help="Specific test file to run (required for 'specific' action)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set environment for consistent testing
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")

    if args.action == "fast":
        success = run_fast_tests()
    elif args.action == "integration":
        success = run_integration_tests()
    elif args.action == "performance":
        success = run_performance_tests()
    elif args.action == "all":
        success = run_all_tests()
    elif args.action == "coverage":
        success = run_tests_with_coverage()
    elif args.action == "lint":
        success = run_linting()
    elif args.action == "security":
        success = run_security_scan()
    elif args.action == "ci":
        success = run_ci_simulation()
    elif args.action == "specific":
        if not args.test_path:
            print("❌ --test-path is required for 'specific' action")
            return 1
        success = run_specific_test(args.test_path)
    else:
        print(f"❌ Unknown action: {args.action}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())