#!/usr/bin/env python3
"""
Standalone test script to demonstrate the diff filter functionality.
This doesn't require Django to be installed.
"""

import difflib


def text_diff(new_text, original_text):
    """
    Generate an inline HTML diff between original and new text.
    Shows additions in green and deletions in red with strikethrough.
    """
    if not original_text or not new_text:
        return f'<span>{new_text}</span>'

    # Split texts into words for better readability
    original_words = original_text.split()
    new_words = new_text.split()

    # Generate diff using SequenceMatcher
    matcher = difflib.SequenceMatcher(None, original_words, new_words)

    result = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            # No change - display normally
            result.append(' '.join(new_words[j1:j2]))
        elif opcode == 'delete':
            # Text was deleted - show in red with strikethrough
            deleted = ' '.join(original_words[i1:i2])
            result.append(f'<span class="diff-delete">{deleted}</span>')
        elif opcode == 'insert':
            # Text was added - show in green
            inserted = ' '.join(new_words[j1:j2])
            result.append(f'<span class="diff-insert">{inserted}</span>')
        elif opcode == 'replace':
            # Text was replaced - show deletion and insertion
            deleted = ' '.join(original_words[i1:i2])
            inserted = ' '.join(new_words[j1:j2])
            result.append(f'<span class="diff-delete">{deleted}</span> <span class="diff-insert">{inserted}</span>')

    return ' '.join(result)


def test_diff_filter():
    """Test the diff filter with sample texts."""

    print("=" * 80)
    print("DIFF FILTER TEST")
    print("=" * 80)

    # Sample data from the study
    source = "I thick that book are better than TV is it is better i can put you in a whole norther wold and it is educational"

    rewrite_a = "I think that books are better than TV, as it is better i can put you in a whole norther wold and it is educational"

    rewrite_b = "I think that books are better than TV because you can get a much better immersive experience when reading. and it is also educational."

    print("\n📄 SOURCE (Original):")
    print("-" * 80)
    print(source)

    print("\n\n✏️  REWRITE A (with diff - HTML):")
    print("-" * 80)
    diff_a = text_diff(rewrite_a, source)
    print(diff_a)

    print("\n\n✏️  REWRITE B (with diff - HTML):")
    print("-" * 80)
    diff_b = text_diff(rewrite_b, source)
    print(diff_b)

    print("\n\n" + "=" * 80)
    print("Generating HTML preview file...")
    print("=" * 80)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Diff Preview</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .source {{
            background-color: #ebebeb;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            text-align: center;
        }}
        .rewrites {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }}
        .rewrite {{
            flex: 1;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
        }}
        .rewrite-a {{
            background-color: aliceblue;
        }}
        .rewrite-b {{
            background-color: bisque;
        }}
        h4 {{
            background-color: rgba(0,0,0,0.1);
            padding: 10px;
            margin: -20px -20px 20px -20px;
            border-radius: 5px 5px 0 0;
        }}
        .rewrite-a h4 {{
            background-color: #97ceff;
        }}
        .rewrite-b h4 {{
            background-color: #ffc887;
        }}
        .diff-insert {{
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }}
        .diff-delete {{
            background-color: #f8d7da;
            color: #721c24;
            text-decoration: line-through;
            padding: 2px 4px;
            border-radius: 3px;
            opacity: 0.8;
        }}
        .diff-container {{
            line-height: 1.8;
            word-wrap: break-word;
        }}
        h2, h3 {{
            color: #333;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #519459;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .info {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🎨 Diff Display Preview - Annotation Interface</h2>

        <div class="info">
            <strong>ℹ️ About:</strong> This preview shows how the annotation interface will display diffs between
            the original argument and the rewrites. Annotators will see changes highlighted to make
            comparisons easier.
        </div>

        <div class="legend">
            <strong>Legend:</strong>
            <span class="legend-item"><span class="diff-insert">Green = Text Added</span></span>
            <span class="legend-item"><span class="diff-delete">Red Strikethrough = Text Deleted</span></span>
        </div>

        <h3>Context: Original Argument</h3>
        <p style="font-size: 0.9em; color: #666; font-style: italic;">
            The original argument that was considered inappropriate by a discussion participant.
        </p>
        <div class="source">
            <p>{source}</p>
        </div>

        <h3>Comparison: Which rewrite do you prefer?</h3>
        <div class="rewrites">
            <div class="rewrite rewrite-a">
                <h4>Rewrite A</h4>
                <p class="diff-container">{diff_a}</p>
            </div>
            <div class="rewrite rewrite-b">
                <h4>Rewrite B</h4>
                <p class="diff-container">{diff_b}</p>
            </div>
        </div>

        <div class="info" style="margin-top: 30px;">
            <strong>📊 Sample from:</strong> Tv is better than books (post_id: 8)
        </div>
    </div>
</body>
</html>
"""

    # Save HTML preview
    preview_file = "diff_preview.html"
    with open(preview_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML preview saved to: {preview_file}")
    print("   Open this file in a browser to see the styled diff display!")
    print(f"\n   Full path: {os.path.abspath(preview_file)}")


if __name__ == "__main__":
    import os
    try:
        test_diff_filter()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
