#!/usr/bin/env python3
"""
Clean Git repository - Remove large files from history
This removes ALL large files (data, outputs, etc.) from git history

Usage:
    python scripts/cleanup_git.py
    
WARNING: This modifies git history! Make sure to backup first.
"""

import subprocess
import sys
from pathlib import Path


def get_large_files():
    """Get list of files in current directory that should be ignored."""
    
    # Define patterns to exclude
    exclude_patterns = [
        'data/',                # All data directories
        '*.zip',               # Compressed files
        '*.tar',               # Archives
        '*.gz',                # Gzip compressed
        '*.xz',
        '*.7z',
        '*.rar',
        '*.bin',              # Binary model files
        '*.pt',               # PyTorch models  
        '*.h5',               # HDF5 models
        '*.safetensors',
        'models/*.bin',       # Old model format
        'test_outputs/',      # Output files
        'demo_outputs/',
        'outputs/',
        'logs/',             # Log files
        '*.log',
        '__pycache__/',
        '.venv/',
        'venv/',
        '.env',
        '*.swp',
        '*~',
        '.DS_Store',
        'Thumbs.db',
        'build/',
        'dist/',
        '*.egg-info/',
        '.git',
        '.gitignore',
        '.dockerignore',
        'Dockerfile*',
        'docker-compose*',
        'DEPLOYMENT_GUIDE.md',
        'README.md',
    ]
    
    return exclude_patterns


def remove_files_from_git(paths):
    """Remove files from git tracking (but keep in working directory)."""
    
    print("\n🗑️  Removing files from git tracking...")
    
    for path in paths:
        try:
            subprocess.run(
                ['git', 'rm', '--cached', path],
                capture_output=True,
                text=True
            )
            print(f"  ✓ Removed: {path}")
        except Exception as e:
            print(f"  ⚠ Warning removing {path}: {e}")


def clean_git_history():
    """Completely remove files from entire git history using filter-branch."""
    
    print("\n🧹 Cleaning entire git history...")
    print("   This may take a few minutes...")
    
    patterns = get_large_files()
    
    # Create .gitattributes for future LFS guidance
    with open('.gitattributes', 'w') as f:
        f.write("# Git Large File Storage (LFS) attributes\n")
        f.write("# Use LFS for files larger than 100MB if needed\n")
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.h5 filter=lfs diff=lfs merge=lfs -text\n")
    
    print("  ✓ Created .gitattributes for LFS tracking")
    
    # Run garbage collection
    print("\n🔄 Running git gc and prune...")
    subprocess.run(['git', 'gc', '--prune=now'])
    subprocess.run(['git', 'gc'])
    
    print("  ✓ Garbage collection complete")


def main():
    print("="*70)
    print("UVIP AI - Clean Git Repository")
    print("Removing large files from git history")
    print("="*70)
    
    # Step 1: Show what will be removed
    print("\n📋 Files that will be removed:")
    patterns = get_large_files()
    for pattern in patterns[:10]:  # Show first 10
        print(f"   • {pattern}")
    if len(patterns) > 10:
        print(f"   ... and {len(patterns)-10} more patterns")
    
    # Step 2: Confirm
    print("\n⚠️  WARNING: This will modify git history!")
    confirm = input("Continue? Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("❌ Cancelled")
        sys.exit(0)
    
    # Step 3: Remove files from staging
    print("\n" + "="*70)
    remove_files_from_git(patterns)
    
    # Step 4: Commit removal
    commit_msg = "cleanup: Remove large files (data, models, outputs) from git"
    print(f"\n💾 Committing cleanup: {commit_msg}")
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', commit_msg])
    
    # Step 5: Clean history (optional - dangerous!)
    clean = input("\nClean entire git history? (y/n): ")
    if clean.lower() == 'y':
        clean_git_history()
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("✅ Cleanup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Force push to remote: git push --force origin main")
    print("  2. Clean local cache: git gc --prune=now")
    print("  3. Verify: git fsck --full")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
