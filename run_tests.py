# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
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

        start_time = time.time()

        try:
            # Run each test file as a separate subprocess with a timeout of 30 seconds
            result = subprocess.run(["pytest", test_path], check=True, capture_output=True, text=True, timeout=30)

            # Log output to a file
            with open(log_file, "w") as f:
                if result.stderr:
                    f.write("=== STDERR ===\n")
                    f.write(result.stderr)
                if result.stdout:
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout)

            elapsed_time = time.time() - start_time
            # Print pass message with clear formatting
            print(f"\tPassed ({elapsed_time:.2f} seconds)")
            summary["passed"] += 1

        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start_time
            error_message = "Test timed out after 30 seconds"

            # Do WH warm reset (potentially hang occurred)
            print("\tWarm reset...")
            os.system("/home/software/syseng/wh/tt-smi -lr all")

            # Log timeout error to a file
            with open(log_file, "w") as f:
                f.write("=== TIMEOUT ===\n")
                f.write(error_message)

            # Print timeout message with clear formatting
            print(f"\tFailed ({elapsed_time:.2f} seconds) - {error_message}")
            summary["failed"] += 1
            summary["failures"][test_file] = error_message

        except subprocess.CalledProcessError as e:
            # Log output to a file
            with open(log_file, "w") as f:
                if e.stderr:
                    f.write("=== STDERR ===\n")
                    f.write(e.stderr)
                if e.stdout:
                    f.write("=== STDOUT ===\n")
                    f.write(e.stdout)

            elapsed_time = time.time() - start_time
            error_message = e.stderr

            # Print fail message with clear formatting
            print(f"\tFailed ({elapsed_time:.2f} seconds)")
            summary["failed"] += 1
            summary["failures"][test_file] = error_message

        except Exception as ex:
            elapsed_time = time.time() - start_time
            print(f"An unexpected error occurred while running {test_file}: {ex} ({elapsed_time:.2f} seconds)")

    # Print and log summary
    print("\n=== Test Summary ===")
    print(f"Total tests run: {len(test_files)}")
    print(f"Tests passed: {summary['passed']}")
    print(f"Tests failed: {summary['failed']}")

    # Write summary to a file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(log_directory, f"summary_{timestamp}.txt")

    with open(summary_file, "w") as f:
        f.write(f"Total tests run: {len(test_files)}\n")
        f.write(f"Tests passed: {summary['passed']}\n")
        f.write(f"Tests failed: {summary['failed']}\n")


if __name__ == "__main__":
    # Set your test directory here
    test_directory = "./generated_modules"  # Adjust this path to your test directory
    run_tests(test_directory)
