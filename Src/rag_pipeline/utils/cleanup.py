
import re
    
def clean_context_to_bullets_pinecone(raw_context: str) -> str:
    """
    Clean raw context and format it as bullet points, without over-removing content.
    """
    lines = raw_context.splitlines()
    cleaned_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are ONLY numbers (slide/page numbers)
        if re.fullmatch(r'\d+', line):
            continue

        # Normalize internal whitespace
        line = re.sub(r'\s+', ' ', line)

        # Skip exact duplicates
        if line in seen:
            continue
        seen.add(line)

        cleaned_lines.append(line)

    # Turn each cleaned line into a bullet
    bullet_text = "\n".join(f"- {line}" for line in cleaned_lines)
    return bullet_text
