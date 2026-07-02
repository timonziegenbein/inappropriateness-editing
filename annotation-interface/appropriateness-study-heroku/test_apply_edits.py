#!/usr/bin/env python3
"""
Test script to demonstrate the apply_edits filter with actual model edits.
"""

import json
import csv
from html import escape


def apply_edits(source_text, edits_json):
    """
    Apply actual model edits to source text and highlight changes.
    """
    if not source_text:
        return ''

    # Parse edits if it's a JSON string
    if isinstance(edits_json, str):
        try:
            edits = json.loads(edits_json) if edits_json else []
        except json.JSONDecodeError:
            edits = []
    else:
        edits = edits_json if edits_json else []

    if not edits:
        return escape(source_text)

    # Collect all edit positions
    edits_with_positions = []
    for edit in edits:
        inappropriate_part = edit.get('inappropriate_part', '')
        rewritten_part = edit.get('rewritten_part', '')

        if not inappropriate_part:
            continue

        # Find the first occurrence
        pos = source_text.find(inappropriate_part)
        if pos != -1:
            edits_with_positions.append({
                'start': pos,
                'end': pos + len(inappropriate_part),
                'original': inappropriate_part,
                'replacement': rewritten_part
            })

    if not edits_with_positions:
        return escape(source_text)

    # Sort by position
    edits_with_positions.sort(key=lambda x: x['start'])

    # Build result
    result_parts = []
    last_pos = 0

    for edit_info in edits_with_positions:
        # Add text before this edit
        if last_pos < edit_info['start']:
            result_parts.append(escape(source_text[last_pos:edit_info['start']]))

        # Add the edit (highlighted)
        original_escaped = escape(edit_info['original'])
        replacement_escaped = escape(edit_info['replacement'])

        if edit_info['replacement']:
            result_parts.append(f'<span class="diff-delete">{original_escaped}</span> <span class="diff-insert">{replacement_escaped}</span>')
        else:
            result_parts.append(f'<span class="diff-delete">{original_escaped}</span>')

        last_pos = edit_info['end']

    # Add remaining text
    if last_pos < len(source_text):
        result_parts.append(escape(source_text[last_pos:]))

    return ''.join(result_parts)


def test_with_csv_data():
    """Test the filter with actual CSV data."""
    print("=" * 80)
    print("APPLY_EDITS FILTER TEST - Using Actual Model Edits")
    print("=" * 80)

    # Read first few rows from CSV
    csv_path = 'data/study_pairs.csv'

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:2]  # Get first 2 rows

        for i, row in enumerate(rows, 1):
            print(f"\n{'=' * 80}")
            print(f"EXAMPLE {i}")
            print('=' * 80)

            source = row['source']
            edits_a = row['edits_a']
            edits_b = row['edits_b']

            print(f"\n📄 SOURCE (Original):")
            print("-" * 80)
            print(source[:200] + "..." if len(source) > 200 else source)

            print(f"\n✏️  REWRITE A Edits:")
            print("-" * 80)
            edits_a_parsed = json.loads(edits_a)
            for j, edit in enumerate(edits_a_parsed, 1):
                print(f"{j}. '{edit['inappropriate_part']}' → '{edit['rewritten_part']}'")

            print(f"\n✨ REWRITE A (with highlighting):")
            print("-" * 80)
            result_a = apply_edits(source, edits_a)
            print(result_a[:500] + "..." if len(result_a) > 500 else result_a)

            print(f"\n✏️  REWRITE B Edits:")
            print("-" * 80)
            edits_b_parsed = json.loads(edits_b)
            if edits_b_parsed:
                for j, edit in enumerate(edits_b_parsed, 1):
                    print(f"{j}. '{edit['inappropriate_part']}' → '{edit['rewritten_part']}'")
            else:
                print("(No edits)")

            print(f"\n✨ REWRITE B (with highlighting):")
            print("-" * 80)
            result_b = apply_edits(source, edits_b)
            print(result_b[:500] + "..." if len(result_b) > 500 else result_b)

        # Generate HTML preview
        print("\n\n" + "=" * 80)
        print("Generating HTML preview...")
        print("=" * 80)

        first_row = rows[0]
        source = first_row['source']
        result_a = apply_edits(source, first_row['edits_a'])
        result_b = apply_edits(source, first_row['edits_b'])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Apply Edits Preview - Using Actual Model Edits</title>
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
        .edit-list {{
            background-color: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            text-align: left;
            font-size: 0.9em;
        }}
        .edit-list h5 {{
            margin-top: 0;
            color: #856404;
        }}
        .edit-item {{
            margin: 5px 0;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🎨 Apply Edits Preview - Using Actual Model Edits</h2>

        <div class="info">
            <strong>ℹ️ About:</strong> This preview shows actual edits made by the models, extracted from
            the prediction files. Each edit shows the exact text that was identified as inappropriate and
            what it was replaced with.
        </div>

        <div class="legend">
            <strong>Legend:</strong>
            <span class="legend-item"><span class="diff-insert">Green = Replacement Text</span></span>
            <span class="legend-item"><span class="diff-delete">Red Strikethrough = Inappropriate Text</span></span>
        </div>

        <h3>Context: Original Argument</h3>
        <div class="source">
            <p>{escape(source[:500] + "..." if len(source) > 500 else source)}</p>
        </div>

        <h3>Comparison: Actual Model Edits</h3>
        <div class="rewrites">
            <div class="rewrite rewrite-a">
                <h4>Rewrite A</h4>
                <div class="edit-list">
                    <h5>Edits Applied:</h5>
                    {''.join([f'<div class="edit-item">{i}. <span class="diff-delete">{escape(e["inappropriate_part"][:50])}</span> → <span class="diff-insert">{escape(e["rewritten_part"][:50])}</span></div>' for i, e in enumerate(json.loads(first_row['edits_a'])[:5], 1)])}
                </div>
                <p class="diff-container">{result_a}</p>
            </div>
            <div class="rewrite rewrite-b">
                <h4>Rewrite B</h4>
                <div class="edit-list">
                    <h5>Edits Applied:</h5>
                    {f'<div class="edit-item">(No edits - text unchanged)</div>' if not json.loads(first_row['edits_b']) else ''.join([f'<div class="edit-item">{i}. <span class="diff-delete">{escape(e["inappropriate_part"][:50])}</span> → <span class="diff-insert">{escape(e["rewritten_part"][:50])}</span></div>' for i, e in enumerate(json.loads(first_row['edits_b'])[:5], 1)])}
                </div>
                <p class="diff-container">{result_b}</p>
            </div>
        </div>

        <div class="info" style="margin-top: 30px;">
            <strong>✨ Key Difference:</strong> This version uses the <em>actual edits</em> from the model predictions
            (inappropriate_part → rewritten_part), not word-level diffs. This shows exactly what the model identified
            as inappropriate and how it chose to rewrite it.
        </div>
    </div>
</body>
</html>
"""

        preview_file = "apply_edits_preview.html"
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n✅ HTML preview saved to: {preview_file}")
        print("   Open this file in a browser to see the actual model edits with highlighting!")

    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("   Make sure you're running this script from the annotation-interface/appropriateness-study-heroku directory")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    test_with_csv_data()
