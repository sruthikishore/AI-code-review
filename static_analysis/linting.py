import subprocess
def run_linter(code_file):
    result = subprocess.run(["pylint", code_file], capture_output=True, text=True)
    return result.stdout
