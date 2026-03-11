"""
PR Review Agent - Uses Claude (Anthropic API) to perform a deep review of pull requests.

What this script does:
  1. Fetches PR metadata and the full diff from GitHub
  2. Collects file-level context (full file content for changed files)
  3. Sends everything to Claude with a structured review prompt
  4. Posts the review back as a PR comment

Key review areas Claude checks:
  - Logic correctness and impact on existing processes
  - Edge cases that could cause failures or halted processes
  - Logging: missing logs on external interactions, or overly verbose logs
  - PR description coverage: code changes not mentioned in the PR body
  - General code quality, security, and maintainability
"""

import os
import sys
import json
import textwrap
import anthropic
from github import Github, GithubException


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum characters of diff to send to Claude.
# Very large diffs are truncated to avoid exceeding token limits.
MAX_DIFF_CHARS = 80_000

# Maximum characters for a single file's full content sent as context.
MAX_FILE_CONTENT_CHARS = 15_000

# Maximum number of changed files for which we fetch full content.
# For PRs touching hundreds of files we only grab the most relevant ones.
MAX_CONTEXT_FILES = 20

# Claude model to use - claude-sonnet gives a great balance of speed and depth
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Label added to the start of every review comment so old reviews can be
# identified and deleted before posting a fresh one on re-runs.
REVIEW_MARKER = "<!-- claude-pr-review-agent -->"


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def get_pr_diff(repo, pr) -> str:
    """
    Fetch the raw unified diff for the pull request.
    Returns the diff as a string, truncated if it's extremely large.
    """
    import requests

    headers = {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3.diff",
    }
    url = f"https://api.github.com/repos/{repo.full_name}/pulls/{pr.number}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    diff = response.text
    if len(diff) > MAX_DIFF_CHARS:
        # Warn the reviewer that the diff was cut off
        truncation_note = (
            f"\n\n[⚠️ DIFF TRUNCATED: original diff was {len(diff):,} chars, "
            f"showing first {MAX_DIFF_CHARS:,} chars only]"
        )
        diff = diff[:MAX_DIFF_CHARS] + truncation_note

    return diff


def get_changed_file_contents(repo, pr) -> dict[str, str]:
    """
    For each file changed in the PR, fetch its CURRENT full content from the
    HEAD branch. This gives Claude enough context to understand the surrounding
    code, not just the changed lines.

    Returns a dict: { filepath: file_content_string }
    """
    file_contents: dict[str, str] = {}
    files = list(pr.get_files())

    # Prioritize files that were modified or added (not just renamed/deleted)
    relevant_files = [
        f for f in files
        if f.status in ("modified", "added") and not f.filename.endswith(
            # Skip binary / generated / lock files - they add noise with no value
            (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
             ".woff", ".woff2", ".ttf", ".eot",
             "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock")
        )
    ][:MAX_CONTEXT_FILES]

    head_sha = pr.head.sha

    for file in relevant_files:
        try:
            content_file = repo.get_contents(file.filename, ref=head_sha)
            # GitHub returns base64-encoded content; decode it
            raw = content_file.decoded_content.decode("utf-8", errors="replace")
            # Truncate very long files to keep the payload manageable
            if len(raw) > MAX_FILE_CONTENT_CHARS:
                raw = (
                    raw[:MAX_FILE_CONTENT_CHARS]
                    + f"\n\n[... FILE TRUNCATED at {MAX_FILE_CONTENT_CHARS:,} chars ...]"
                )
            file_contents[file.filename] = raw
        except (GithubException, UnicodeDecodeError):
            # Binary file or API error - skip silently
            pass

    return file_contents


def delete_previous_review_comments(pr) -> None:
    """
    Remove any existing review comments left by this bot on previous runs.
    This keeps the PR timeline clean - only the latest review is shown.
    """
    for comment in pr.get_issue_comments():
        if REVIEW_MARKER in comment.body:
            try:
                comment.delete()
            except GithubException:
                # Not critical if deletion fails - we'll just post alongside it
                pass


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_review_prompt(
    pr_title: str,
    pr_body: str,
    pr_author: str,
    base_branch: str,
    head_branch: str,
    diff: str,
    file_contents: dict[str, str],
) -> str:
    """
    Build the full prompt that is sent to Claude.
    The prompt is structured so Claude returns a well-formatted Markdown review.
    """

    # Format full file contents as a readable block
    file_context_section = ""
    if file_contents:
        parts = []
        for filepath, content in file_contents.items():
            parts.append(
                f"### File: `{filepath}`\n```\n{content}\n```"
            )
        file_context_section = (
            "## Full File Contents (for context)\n\n"
            + "\n\n".join(parts)
        )

    prompt = textwrap.dedent(f"""
    You are a senior software engineer performing a thorough code review of a pull request.
    Your job is to protect the stability, reliability, and maintainability of the system.

    ## Pull Request Information

    - **Title:** {pr_title}
    - **Author:** {pr_author}
    - **Base branch:** {base_branch}
    - **Head branch:** {head_branch}

    ## PR Description (written by the author)

    {pr_body or "_No description provided._"}

    ## The Diff

    ```diff
    {diff}
    ```

    {file_context_section}

    ---

    ## Your Review Task

    Perform a meticulous review covering ALL of the following areas.
    Be specific: reference file names and line numbers (from the diff) whenever possible.

    ### 1. 🔍 Impact Analysis
    - How do these changes affect existing behavior and processes?
    - Are there downstream consumers, callers, or dependent services that could break?
    - Are database schema changes, API contract changes, or config changes handled safely?

    ### 2. ⚠️ Edge Cases & Failure Modes
    - Identify inputs, states, or conditions that could cause the process to halt, crash, or
      produce incorrect results.
    - Check for: null/undefined/empty values, off-by-one errors, race conditions, timeout
      scenarios, missing error handling, unhandled promise rejections, resource leaks.
    - Flag any place where an exception is silently swallowed.

    ### 3. 📋 Logging & Observability
    - Every interaction with the "outside world" (HTTP calls, database queries, queue
      messages, file I/O, third-party SDKs) MUST be logged at the appropriate level:
        - Start of call (DEBUG or INFO)
        - Success result (DEBUG or INFO, with relevant identifiers but NOT full payloads)
        - Error / exception (ERROR with full context)
    - Flag any external interaction that has no logging.
    - Also flag logging that is TOO verbose: logging full request/response bodies,
      logging inside tight loops, DEBUG logs that would flood production, or any log that
      could grow unboundedly and fill up disk / overwhelm a log aggregator.
    - Check that sensitive data (tokens, passwords, PII) is NEVER logged.

    ### 4. 📝 PR Description Coverage
    - Compare the code changes against the PR description.
    - List any significant changes in the diff that are NOT mentioned or explained in the
      PR description. Suggest what the author should add to the description.

    ### 5. 🛡️ Security & Data Safety
    - Input validation and sanitization
    - Authentication / authorization checks
    - Secrets or credentials accidentally committed
    - SQL injection, command injection, path traversal risks

    ### 6. ✅ Code Quality & Maintainability
    - Readability and clarity of the code
    - Missing or inadequate comments on complex logic
    - Dead code, duplicated logic, or overly complex implementations
    - Test coverage: are the changes covered by tests? Are edge cases tested?

    ---

    ## Output Format

    Structure your review EXACTLY as follows (use these emoji headers):

    ### 🔴 Critical Issues
    Problems that MUST be fixed before merging. Number each issue.

    ### 🟡 Warnings
    Things that should be addressed but are not blocking. Number each item.

    ### 🔵 Suggestions
    Nice-to-haves, style improvements, or minor observations. Number each item.

    ### 📋 PR Description — Missing Coverage
    List changes in the code that are not explained in the PR description.
    If everything is covered, write: "✅ All significant changes are described in the PR."

    ### 📊 Summary
    A 3-5 sentence overall assessment of the PR: quality, risk level, and recommendation
    (Approve / Request Changes / Needs Discussion).

    Be direct and precise. Do not pad with generic advice. Every point must reference
    specific code from this PR.
    """).strip()

    return prompt


# ── Main entry point ───────────────────────────────────────────────────────────

def main():
    # ── Read configuration from environment variables (injected by GitHub Actions)
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    pr_number_str = os.environ.get("PR_NUMBER")
    repo_full_name = os.environ.get("REPO_FULL_NAME")
    pr_title = os.environ.get("PR_TITLE", "")
    pr_body = os.environ.get("PR_BODY", "")
    base_branch = os.environ.get("PR_BASE_BRANCH", "main")
    head_branch = os.environ.get("PR_HEAD_BRANCH", "")
    pr_author = os.environ.get("PR_AUTHOR", "unknown")

    # Validate that all required secrets / env vars are present
    missing = []
    if not anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not github_token:
        missing.append("GITHUB_TOKEN")
    if not pr_number_str:
        missing.append("PR_NUMBER")
    if not repo_full_name:
        missing.append("REPO_FULL_NAME")
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    pr_number = int(pr_number_str)

    print(f"🔍 Starting PR review for PR #{pr_number} in {repo_full_name}")

    # ── Connect to GitHub ──────────────────────────────────────────────────────
    gh = Github(github_token)
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    # ── Gather PR data ─────────────────────────────────────────────────────────
    print("📥 Fetching PR diff...")
    diff = get_pr_diff(repo, pr)
    print(f"   Diff size: {len(diff):,} characters")

    print("📂 Fetching changed file contents for context...")
    file_contents = get_changed_file_contents(repo, pr)
    print(f"   Loaded {len(file_contents)} files for context")

    # ── Build the prompt ───────────────────────────────────────────────────────
    print("🧠 Building review prompt...")
    prompt = build_review_prompt(
        pr_title=pr_title,
        pr_body=pr_body,
        pr_author=pr_author,
        base_branch=base_branch,
        head_branch=head_branch,
        diff=diff,
        file_contents=file_contents,
    )
    print(f"   Prompt size: {len(prompt):,} characters")

    # ── Call Claude API ────────────────────────────────────────────────────────
    print("🤖 Sending to Claude for review...")
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,   # Reviews can be long - give Claude enough room
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    review_text = message.content[0].text
    print(f"   Review received: {len(review_text):,} characters")

    # ── Format the final comment ───────────────────────────────────────────────
    # The REVIEW_MARKER is an invisible HTML comment used to identify and
    # delete this bot's comments on future re-runs (keeps PR timeline clean).
    comment_body = f"""{REVIEW_MARKER}
## 🤖 Claude PR Review Agent

> **PR:** {pr_title}
> **Branch:** `{head_branch}` → `{base_branch}`
> **Reviewed by:** Claude (`{CLAUDE_MODEL}`)

---

{review_text}

---
<sub>🔄 This review is automatically regenerated on each push. Previous reviews are removed.</sub>
"""

    # ── Post to GitHub ─────────────────────────────────────────────────────────
    print("🗑️  Removing previous review comments...")
    delete_previous_review_comments(pr)

    print("💬 Posting review comment to PR...")
    pr.create_issue_comment(comment_body)

    print(f"✅ Review posted successfully to PR #{pr_number}")


if __name__ == "__main__":
    main()
