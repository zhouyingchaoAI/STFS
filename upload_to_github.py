#!/usr/bin/env python3
"""
Script to upload code to GitHub repository
"""
import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def main():
    """Main function to upload to GitHub"""
    print("Starting GitHub upload process...")
    
    # Check if we're in a git repository
    returncode, stdout, stderr = run_command("git status")
    if returncode != 0:
        print("Error: Not in a git repository")
        return False
    
    # Check current status
    print("Checking git status...")
    returncode, stdout, stderr = run_command("git status --porcelain")
    if stdout.strip():
        print("There are uncommitted changes. Please commit them first.")
        return False
    
    # Check if we have commits to push
    print("Checking for commits to push...")
    returncode, stdout, stderr = run_command("git log --oneline origin/main..HEAD")
    if not stdout.strip():
        print("No commits to push")
        return True
    
    print(f"Found commits to push:\n{stdout}")
    
    # Try to push
    print("Attempting to push to GitHub...")
    returncode, stdout, stderr = run_command("git push origin main")
    
    if returncode == 0:
        print("Successfully pushed to GitHub!")
        return True
    else:
        print(f"Push failed: {stderr}")
        print("Authentication may be required.")
        print("Please set up GitHub authentication using one of these methods:")
        print("1. Personal Access Token: https://github.com/settings/tokens")
        print("2. SSH Key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh")
        print("3. GitHub CLI: https://cli.github.com/")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
