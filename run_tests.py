# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
from datetime import datetime


def run_tests(test_directory, log_directory="test_logs"):
    """
    Runs all pytest files in the given directory, logging each test's output separately.
    Creates a summary with pass/fail counts and specific error messages for failures.
    """
    # Ensure the log directory exists
    os.makedirs(log_directory, exist_ok=True)

    test_files = [f for f in os.listdir(test_directory) if f.startswith("test_") or f.endswith("_test.py")]
    test_files = sorted(test_files)
    summary = {"passed": 0, "failed": 0, "failures": {}}

    for test_file in test_files:
        test_path = os.path.join(test_directory, test_file)
        log_file = os.path.join(log_directory, f"{test_file}_log.txt")

        print(f"Running test: {test_file}")

        try:
            # Run each test file as a separate subprocess
            result = subprocess.run(["pytest", test_path], check=True, capture_output=True, text=True)

            # Log output to a file
            with open(log_file, "w") as f:
                if result.stderr:
                    f.write("=== STDERR ===\n")
                    f.write(result.stderr)
                if result.stdout:
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout)

            # Print pass message with clear formatting
            print(f"\tPassed")
            summary["passed"] += 1

        except subprocess.CalledProcessError as e:
            # Log output to a file
            with open(log_file, "w") as f:
                if e.stderr:
                    f.write("=== STDERR ===\n")
                    f.write(e.stderr)
                if e.stdout:
                    f.write("=== STDOUT ===\n")
                    f.write(e.stdout)

            error_message = e.stderr

            # Print fail message with clear formatting
            print(f"\tFailed")
            summary["failed"] += 1
            summary["failures"][test_file] = error_message

        except Exception as ex:
            print(f"An unexpected error occurred while running {test_file}: {ex}")

    # Print and log summary
    print("\n=== Test Summary ===")
    print(f"Total tests run: {len(test_files)}")
    print(f"Tests passed: {summary['passed']}")
    print(f"Tests failed: {summary['failed']}")

    if summary["failed"] > 0:
        print("\nFailed Tests:")
        for test, message in summary["failures"].items():
            print(f"- {test}: {message}")

    # Write summary to a file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(log_directory, f"summary_{timestamp}.txt")

    with open(summary_file, "w") as f:
        f.write(f"Total tests run: {len(test_files)}\n")
        f.write(f"Tests passed: {summary['passed']}\n")
        f.write(f"Tests failed: {summary['failed']}\n")

        if summary["failed"] > 0:
            f.write("\nFailed Tests:\n")
            for test, message in summary["failures"].items():
                f.write(f"- {test}: {message}\n")


if __name__ == "__main__":
    # Set your test directory here
    test_directory = "./generated_modules"  # Adjust this path to your test directory
    run_tests(test_directory)
