#!/usr/bin/env python
"""
Check which datasets in an organization are missing the codebase version tag.

Usage:
    python scripts/check_dataset_tags.py --org ai2-cortex
    python scripts/check_dataset_tags.py --org ai2-cortex --fix  # Also fix missing tags
"""

import argparse

from huggingface_hub import HfApi, list_datasets
from huggingface_hub.errors import RepositoryNotFoundError


def get_repo_version_tags(hub_api: HfApi, repo_id: str) -> list[str]:
    """Get all version tags (starting with 'v') for a dataset."""
    try:
        refs = hub_api.list_repo_refs(repo_id, repo_type="dataset")
        return [tag.name for tag in refs.tags if tag.name.startswith("v")]
    except RepositoryNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Check datasets for missing version tags")
    parser.add_argument("--org", required=True, help="Organization name to check")
    parser.add_argument("--fix", action="store_true", help="Create missing version tags")
    parser.add_argument("--version", default="v3.0", help="Version tag to check/create (default: v3.0)")
    args = parser.parse_args()

    hub_api = HfApi()
    
    print(f"Fetching datasets for organization: {args.org}")
    datasets = list(list_datasets(author=args.org))
    print(f"Found {len(datasets)} datasets\n")

    missing_tags = []
    has_tags = []

    for ds in datasets:
        repo_id = ds.id
        tags = get_repo_version_tags(hub_api, repo_id)
        
        if not tags:
            missing_tags.append(repo_id)
            print(f"❌ {repo_id} - NO VERSION TAGS")
        else:
            has_tags.append(repo_id)
            print(f"✓  {repo_id} - tags: {tags}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total datasets: {len(datasets)}")
    print(f"  With version tags: {len(has_tags)}")
    print(f"  Missing version tags: {len(missing_tags)}")

    if missing_tags:
        print(f"\nDatasets missing version tags:")
        for repo_id in missing_tags:
            print(f"  - {repo_id}")

        if args.fix:
            print(f"\nFixing missing tags (creating {args.version})...")
            for repo_id in missing_tags:
                try:
                    hub_api.create_tag(repo_id, tag=args.version, repo_type="dataset")
                    print(f"  ✓ Created {args.version} tag for {repo_id}")
                except Exception as e:
                    print(f"  ❌ Failed to create tag for {repo_id}: {e}")
        else:
            print(f"\nTo fix these, run with --fix flag:")
            print(f"  python scripts/check_dataset_tags.py --org {args.org} --fix")


if __name__ == "__main__":
    main()

