#!/usr/bin/env python3
"""
Ilanya Test Runner

Runs all tests for both Desire Engine and Trait Engine with organized logging.
Provides a summary of test results and generates comprehensive logs.

Author: KleaSCM
Email: KleaSCM@gmail.com
License: MIT
Version: 0.1.0
"""

import sys
import os
import time
import subprocess
import unittest
from datetime import datetime

# Add the parent directory to the path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logger, log_test_start, log_test_end


def run_desire_engine_tests():
    """Run all desire engine tests."""
    logger = setup_logger(
        engine_type="desire",
        test_type="test",
        test_name="test_suite",
        test_target="all_modules",
        log_level="INFO"
    )
    
    start_time = time.time()
    log_test_start(logger, "desire_engine_test_suite", 
                  "Running all desire engine tests")
    
    try:
        # Add the IlanyaDesireEngine directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IlanyaDesireEngine'))
        
        # Import and run tests
        from Tests.test_emergent_desires import TestEmergentDesires
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEmergentDesires)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Log results
        logger.info(f"Desire Engine Tests - Ran: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
        
        if result.failures:
            for test, traceback in result.failures:
                logger.error(f"Test failure: {test} - {traceback}")
        
        if result.errors:
            for test, traceback in result.errors:
                logger.error(f"Test error: {test} - {traceback}")
        
        success = len(result.failures) == 0 and len(result.errors) == 0
        duration = time.time() - start_time
        log_test_end(logger, "desire_engine_test_suite", success, duration)
        
        return success, result.testsRun, len(result.failures), len(result.errors)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Desire engine test suite failed with error: {str(e)}")
        log_test_end(logger, "desire_engine_test_suite", False, duration)
        return False, 0, 0, 1


def run_trait_engine_tests():
    """Run all trait engine tests."""
    logger = setup_logger(
        engine_type="trait",
        test_type="test",
        test_name="test_suite",
        test_target="all_modules",
        log_level="INFO"
    )
    
    start_time = time.time()
    log_test_start(logger, "trait_engine_test_suite", 
                  "Running all trait engine tests")
    
    try:
        # Add the IlanyaTraitEngine directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IlanyaTraitEngine'))
        
        # Import pytest and run tests
        import pytest
        
        # Run pytest on the trait engine test file
        test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_trait_engine.py')
        result = pytest.main([test_file, '-v', '--tb=short'])
        
        # Log results (pytest returns 0 for success, 1 for failure)
        success = result == 0
        logger.info(f"Trait Engine Tests - Result: {'PASSED' if success else 'FAILED'}")
        
        duration = time.time() - start_time
        log_test_end(logger, "trait_engine_test_suite", success, duration)
        
        # Return simplified results for pytest
        return success, 1, 0, 0 if success else 1
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Trait engine test suite failed with error: {str(e)}")
        log_test_end(logger, "trait_engine_test_suite", False, duration)
        return False, 0, 0, 1


def run_demos():
    """Run all demos."""
    logger = setup_logger(
        engine_type="demo",
        test_type="demo",
        test_name="demo_suite",
        test_target="all_demos",
        log_level="INFO"
    )
    
    start_time = time.time()
    log_test_start(logger, "demo_suite", 
                  "Running all demos")
    
    try:
        demo_results = []
        
        # Run desire engine demo
        logger.info("Running desire engine demo...")
        try:
            from Demo.modular_demo import main as desire_demo_main
            desire_demo_main()
            demo_results.append(("Desire Engine Demo", True))
            logger.info("Desire engine demo completed successfully")
        except Exception as e:
            logger.error(f"Desire engine demo failed: {str(e)}")
            demo_results.append(("Desire Engine Demo", False))
        
        # Run trait engine demo
        logger.info("Running trait engine demo...")
        try:
            from Demo.demo import main as trait_demo_main
            trait_demo_main()
            demo_results.append(("Trait Engine Demo", True))
            logger.info("Trait engine demo completed successfully")
        except Exception as e:
            logger.error(f"Trait engine demo failed: {str(e)}")
            demo_results.append(("Trait Engine Demo", False))
        
        # Log results
        successful_demos = sum(1 for _, success in demo_results if success)
        total_demos = len(demo_results)
        
        logger.info(f"Demos - Completed: {successful_demos}/{total_demos}")
        
        for demo_name, success in demo_results:
            status = "PASSED" if success else "FAILED"
            logger.info(f"  {demo_name}: {status}")
        
        success = successful_demos == total_demos
        duration = time.time() - start_time
        log_test_end(logger, "demo_suite", success, duration)
        
        return success, total_demos, successful_demos
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Demo suite failed with error: {str(e)}")
        log_test_end(logger, "demo_suite", False, duration)
        return False, 0, 0


def main():
    """Main test runner function."""
    print("🚀 Ilanya Test Runner")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    overall_start_time = time.time()
    
    # Run tests
    print("🧪 Running Tests...")
    print("-" * 30)
    
    desire_success, desire_tests, desire_failures, desire_errors = run_desire_engine_tests()
    trait_success, trait_tests, trait_failures, trait_errors = run_trait_engine_tests()
    
    total_tests = desire_tests + trait_tests
    total_failures = desire_failures + trait_failures
    total_errors = desire_errors + trait_errors
    
    print(f"\n📊 Test Results:")
    print(f"  Desire Engine: {desire_tests} tests, {desire_failures} failures, {desire_errors} errors")
    print(f"  Trait Engine: {trait_tests} tests, {trait_failures} failures, {trait_errors} errors")
    print(f"  Total: {total_tests} tests, {total_failures} failures, {total_errors} errors")
    
    # Run demos
    print(f"\n🎬 Running Demos...")
    print("-" * 30)
    
    demo_success, total_demos, successful_demos = run_demos()
    
    print(f"\n📊 Demo Results:")
    print(f"  Completed: {successful_demos}/{total_demos} demos")
    
    # Overall summary
    overall_duration = time.time() - overall_start_time
    overall_success = desire_success and trait_success and demo_success
    
    print(f"\n🎯 Overall Summary:")
    print(f"  Tests: {'✅ PASSED' if total_failures == 0 and total_errors == 0 else '❌ FAILED'}")
    print(f"  Demos: {'✅ PASSED' if demo_success else '❌ FAILED'}")
    print(f"  Overall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    print(f"  Duration: {overall_duration:.2f} seconds")
    
    print(f"\n📁 Logs saved to:")
    print(f"  Logs/desire/tests/ - Desire engine test logs")
    print(f"  Logs/trait/tests/ - Trait engine test logs")
    print(f"  Logs/desire/demos/ - Desire engine demo logs")
    print(f"  Logs/trait/demos/ - Trait engine demo logs")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main() 