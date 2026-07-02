"""
Custom template filters for displaying text diffs using actual model edits.
"""
import difflib
import re
import json
from django import template
from django.utils.safestring import mark_safe
from django.utils.html import escape

register = template.Library()


@register.filter(name='apply_edits')
def apply_edits(source_text, edits_json):
    """
    Apply actual model edits to source text and highlight changes.
    If no edits available, returns escaped source text.

    Args:
        source_text: Original text
        edits_json: JSON string or list of edits with 'inappropriate_part' and 'rewritten_part'

    Returns:
        HTML string with highlighted edits
    """
    if not source_text:
        return mark_safe('')

    # Parse edits if it's a JSON string
    if isinstance(edits_json, str):
        try:
            edits = json.loads(edits_json) if edits_json else []
        except json.JSONDecodeError:
            edits = []
    else:
        edits = edits_json if edits_json else []

    if not edits:
        # No edits available - return original text (will use smart_diff fallback)
        return None

    # Collect all edit positions (find first occurrence of each edit)
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
        return mark_safe(escape(source_text))

    # Sort by position
    edits_with_positions.sort(key=lambda x: x['start'])

    # Build result by iterating through text and applying edits
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
            # Text was replaced
            result_parts.append(f'<span class="diff-delete">{original_escaped}</span> <span class="diff-insert">{replacement_escaped}</span>')
        else:
            # Text was deleted (replacement is empty)
            result_parts.append(f'<span class="diff-delete">{original_escaped}</span>')

        last_pos = edit_info['end']

    # Add remaining text
    if last_pos < len(source_text):
        result_parts.append(escape(source_text[last_pos:]))

    return mark_safe(''.join(result_parts))


@register.filter(name='text_diff')
def text_diff(new_text, original_text):
    """
    Generate an inline HTML diff between original and new text.
    Shows additions in green and deletions in red with strikethrough.

    This is a fallback for when edit information is not available.
    """
    if not original_text or not new_text:
        return mark_safe(f'<span>{escape(new_text)}</span>')

    # Split texts into words for better readability
    original_words = original_text.split()
    new_words = new_text.split()

    # Generate diff using SequenceMatcher
    matcher = difflib.SequenceMatcher(None, original_words, new_words)

    result = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            # No change - display normally
            result.append(escape(' '.join(new_words[j1:j2])))
        elif opcode == 'delete':
            # Text was deleted - show in red with strikethrough
            deleted = escape(' '.join(original_words[i1:i2]))
            result.append(f'<span class="diff-delete">{deleted}</span>')
        elif opcode == 'insert':
            # Text was added - show in green
            inserted = escape(' '.join(new_words[j1:j2]))
            result.append(f'<span class="diff-insert">{inserted}</span>')
        elif opcode == 'replace':
            # Text was replaced - show deletion and insertion
            deleted = escape(' '.join(original_words[i1:i2]))
            inserted = escape(' '.join(new_words[j1:j2]))
            result.append(f'<span class="diff-delete">{deleted}</span> <span class="diff-insert">{inserted}</span>')

    return mark_safe(' '.join(result))


@register.filter(name='smart_diff')
def smart_diff(rewrite_text, source_and_edits):
    """
    Smart diff that uses actual edits if available, otherwise computes word-level diff.

    Usage: {{post.rewrite_a|smart_diff:post}}

    Args:
        rewrite_text: The rewritten text
        source_and_edits: The post object containing source, edits_a, edits_b

    Returns:
        HTML string with highlighted changes
    """
    # Extract the appropriate edits field based on which rewrite we're showing
    # This is a bit hacky but works for the template context
    source_text = source_and_edits.get('source', '') if isinstance(source_and_edits, dict) else getattr(source_and_edits, 'source', '')

    # Try to find which rewrite this is (a or b)
    rewrite_a = source_and_edits.get('rewrite_a', '') if isinstance(source_and_edits, dict) else getattr(source_and_edits, 'rewrite_a', '')
    rewrite_b = source_and_edits.get('rewrite_b', '') if isinstance(source_and_edits, dict) else getattr(source_and_edits, 'rewrite_b', '')

    if rewrite_text == rewrite_a:
        edits_json = source_and_edits.get('edits_a', '[]') if isinstance(source_and_edits, dict) else getattr(source_and_edits, 'edits_a', '[]')
    else:
        edits_json = source_and_edits.get('edits_b', '[]') if isinstance(source_and_edits, dict) else getattr(source_and_edits, 'edits_b', '[]')

    # Try to apply actual edits first
    result = apply_edits(source_text, edits_json)

    # If no edits available (returns None), fall back to word-level diff
    if result is None:
        return text_diff(rewrite_text, source_text)

    return result


@register.filter(name='char_diff')
def char_diff(new_text, original_text):
    """
    Generate a character-level HTML diff between original and new text.
    More granular than word-level diff.
    """
    if not original_text or not new_text:
        return mark_safe(f'<span>{new_text}</span>')

    # Generate character-level diff
    matcher = difflib.SequenceMatcher(None, original_text, new_text)

    result = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            result.append(new_text[j1:j2])
        elif opcode == 'delete':
            deleted = original_text[i1:i2]
            result.append(f'<span class="diff-delete">{deleted}</span>')
        elif opcode == 'insert':
            inserted = new_text[j1:j2]
            result.append(f'<span class="diff-insert">{inserted}</span>')
        elif opcode == 'replace':
            deleted = original_text[i1:i2]
            inserted = new_text[j1:j2]
            result.append(f'<span class="diff-delete">{deleted}</span><span class="diff-insert">{inserted}</span>')

    return mark_safe(''.join(result))
