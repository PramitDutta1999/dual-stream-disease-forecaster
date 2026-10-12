"""Hugging Face Model Hub Upload Script

Programmatically creates a public model repository on Hugging Face and pushes
the trained DualStreamEHRTransformer checkpoint artifacts to the Hub.
"""

from __future__ import annotations

import sys

# Prevent scripts/tokenize.py from shadowing Python's standard library tokenize
_scripts_dir = sys.path[0] if sys.path else ""
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = os.getenv("HF_REPO_ID", "PramitDutta1999/ehr-dual-stream-transformer")


def get_default_project_root() -> Path:
    """Returns project root directory assuming this script is in scripts/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        return script_dir.parent
    return Path.cwd()


def get_hf_token(project_root: Path) -> str | None:
    """Resolves HF_TOKEN from environment variable, falling back to secrets.txt or api_key.txt if present."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for fname in ["secrets.txt", "api_key.txt"]:
        key_file = project_root / fname
        if key_file.exists():
            try:
                for line in key_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload trained DualStreamEHRTransformer model to Hugging Face Model Hub."
    )
    project_root = get_default_project_root()
    token_default = get_hf_token(project_root)

    parser.add_argument(
        "--token",
        type=str,
        default=token_default,
        help="Hugging Face API User Access Token (defaults to HF_TOKEN env var or api_key.txt)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face Repository ID (default: PramitDutta1999/ehr-dual-stream-transformer)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=project_root / "checkpoints" / "best_model",
        help="Local directory containing model checkpoint files",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned DualStreamEHRTransformer checkpoint and evaluation summary",
        help="Commit message for the Hugging Face upload",
    )

    args = parser.parse_args()

    # 1. Validation & Initialization
    if not args.checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found at: {args.checkpoint_dir}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("        HUGGING FACE MODEL HUB DEPLOYMENT PIPELINE        ")
    print("=" * 75)
    print(f"Target Repository ID : {args.repo_id}")
    print(f"Local Checkpoint Dir : {args.checkpoint_dir}")
    print(f"Files to Upload      :")
    for f in sorted(args.checkpoint_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    # Initialize HfApi client with authentication token
    print(f"\n[1/3] Authenticating with Hugging Face Hub using token...")
    api = HfApi(token=args.token)

    auth_username = None
    try:
        user_info = api.whoami()
        auth_username = user_info.get("name")
        print(f"      Authenticated user: @{auth_username} ({user_info.get('fullname', auth_username)})")
    except Exception as e:
        print(f"      Authentication notice: Proceeding with token validation ({e})")

    # 2. Repository Creation
    target_repo_id = args.repo_id
    print(f"\n[2/3] Programmatically creating public repository: {target_repo_id}...")
    try:
        repo_url = api.create_repo(
            repo_id=target_repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"      Repository ready: {repo_url}")
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg and "You don't have the rights to create a model under the namespace" in err_msg and auth_username:
            print(f"      Notice: Token belongs to user '@{auth_username}'.")
            model_name = target_repo_id.split("/")[-1]
            target_repo_id = f"{auth_username}/{model_name}"
            print(f"      Switching repository target to authenticated namespace: {target_repo_id}...")
            repo_url = api.create_repo(
                repo_id=target_repo_id,
                repo_type="model",
                exist_ok=True,
                private=False,
            )
            print(f"      Repository successfully created: {repo_url}")
        else:
            print(f"      Notice: {e}")

    # 3. Model Upload
    print(f"\n[3/3] Uploading checkpoint folder to {target_repo_id}...")
    try:
        upload_info = api.upload_folder(
            folder_path=str(args.checkpoint_dir),
            repo_id=target_repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
        print(f"      Folder upload successful!")
    except Exception as e:
        print(f"Error uploading folder: {e}")
        sys.exit(1)

    # 4. Final Output & Technical Report URL
    final_url = f"https://huggingface.co/{target_repo_id}"
    sep = "=" * 75
    print("\n" + sep)
    print(" DEPLOYMENT SUCCESSFUL: MODEL PUSHED TO HUGGING FACE HUB!")
    print(sep)
    print(f"\nPublic Repository URL:")
    print(f"  --> {final_url}\n")
    print("You can copy and paste the above URL directly into your LaTeX technical report:")
    print(f"  \\url{{{final_url}}}")
    print("\n" + sep + "\n")


if __name__ == "__main__":
    main()
