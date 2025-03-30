import sys
import re
from pathlib import Path

# CONFIG — path to your version file
VERSION_FILE = Path("pretab/__version__.py")
VERSION_PATTERN = r'__version__\s*=\s*["\'](\d+)\.(\d+)\.(\d+)["\']'

def parse_version():
    text = VERSION_FILE.read_text()
    match = re.search(VERSION_PATTERN, text)
    if not match:
        raise ValueError("Could not find __version__ in version file.")
    return list(map(int, match.groups()))

def write_version(version):
    major, minor, patch = version
    new_version_line = f'__version__ = "{major}.{minor}.{patch}"\n'
    new_content = re.sub(VERSION_PATTERN, new_version_line, VERSION_FILE.read_text())
    VERSION_FILE.write_text(new_content)
    print(f"Bumped version to {major}.{minor}.{patch}")

def determine_bump_type(commits):
    commits = commits.lower()
    if "breaking" in commits:
        return "major"
    elif "feat" in commits:
        return "minor"
    elif "fix" in commits:
        return "patch"
    else:
        return None

def bump(version, bump_type):
    major, minor, patch = version
    if bump_type == "major":
        return [major + 1, 0, 0]
    elif bump_type == "minor":
        return [major, minor + 1, 0]
    elif bump_type == "patch":
        return [major, minor, patch + 1]
    else:
        return version  # no change

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bump_version.py \"commit messages\"")
        sys.exit(1)

    commit_messages = sys.argv[1]
    current_version = parse_version()
    bump_type = determine_bump_type(commit_messages)

    if bump_type is None:
        print("No bump type detected from commit messages.")
        sys.exit(0)

    new_version = bump(current_version, bump_type)
    if new_version != current_version:
        write_version(new_version)
