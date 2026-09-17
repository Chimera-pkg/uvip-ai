#!/usr/bin/env python3
"""
Helper script to upload model files from local to Vast.ai server via SSH/SFTP.

Usage:
    python scripts/upload_models.py --local-dir models/perception --remote-path /app/models/perception
"""

import argparse
import os
from pathlib import Path
import paramiko


def get_ssh_connection(host, port, username, key_file):
    """Create SSH connection to Vast.ai server."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=host,
        port=port,
        username=username,
        key_filename=key_file,
        timeout=10
    )
    return ssh


def upload_files(local_dir, remote_path, host, port, username, key_file):
    """Upload files to server."""
    
    print(f"\nConnecting to {host}:{port}...")
    ssh = get_ssh_connection(host, port, username, key_file)
    
    try:
        sftp = ssh.open_sftp()
        
        # Create remote directory if it doesn't exist
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            print(f"Creating remote directory: {remote_path}")
            sftp.mkdir(remote_path)
        
        local_path = Path(local_dir)
        total_files = 0
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Get relative path
                rel_path = file_path.relative_to(local_path)
                remote_full_path = f"{remote_path}/{rel_path}"
                
                # Create remote subdirectories
                remote_dir = str(Path(remote_full_path).parent)
                try:
                    sftp.stat(remote_dir)
                except FileNotFoundError:
                    print(f"  Creating directory: {remote_dir}")
                    sftp.mkdir(remote_dir)
                
                # Upload file
                print(f"  Uploading: {rel_path}")
                sftp.put(str(file_path), remote_full_path)
                total_files += 1
                
                # Show progress
                size = file_path.stat().st_size / (1024 * 1024)
                print(f"    ✓ Uploaded ({size:.2f} MB)")
        
        print(f"\n✅ Successfully uploaded {total_files} files!")
        
    finally:
        ssh.close()


def main():
    parser = argparse.ArgumentParser(description="Upload model files to Vast.ai server")
    parser.add_argument("--local-dir", type=str, required=True,
                       help="Local directory containing files to upload")
    parser.add_argument("--remote-path", type=str, default="/app/models/perception",
                       help="Remote path on server (default: /app/models/perception)")
    parser.add_argument("--host", type=str, default="182.224.239.168",
                       help="Vast.ai server IP address")
    parser.add_argument("--port", type=int, default=22,
                       help="SSH port (default: 22)")
    parser.add_argument("--username", type=str, default="root",
                       help="SSH username (default: root)")
    parser.add_argument("--key-file", type=str, required=True,
                       help="Path to SSH private key file")
    
    args = parser.parse_args()
    
    # Validate local directory
    if not os.path.exists(args.local_dir):
        print(f"❌ Local directory not found: {args.local_dir}")
        exit(1)
    
    # Upload files
    upload_files(
        local_dir=args.local_dir,
        remote_path=args.remote_path,
        host=args.host,
        port=args.port,
        username=args.username,
        key_file=args.key_file
    )


if __name__ == "__main__":
    main()
