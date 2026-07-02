"""
Custom template filters for displaying text edits with diff highlighting.
"""
from django import template
from django.utils.safestring import mark_safe
from django.utils.html import escape
import diff_match_patch as dmp_module

register = template.Library()


@register.simple_tag
def highlight_with_char_diff(source_text, inappropriate_part, rewritten_part):
    """
    Highlights the inappropriate part in the source text, with character-level
    diff against the rewritten part and a tooltip.
    """
    if source_text is None:
        return mark_safe('')

    source_text = str(source_text)
    inappropriate_part = str(inappropriate_part) if inappropriate_part is not None else ''
    rewritten_part = str(rewritten_part) if rewritten_part is not None else ''

    if inappropriate_part in ('nan', 'None', '') or not source_text:
        return mark_safe(escape(source_text))

    if rewritten_part in ('nan', 'None'):
        rewritten_part = ''

    if inappropriate_part not in source_text:
        return mark_safe(escape(source_text))

    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(inappropriate_part, rewritten_part)
    dmp.diff_cleanupSemantic(diffs)

    highlighted_inappropriate = ''
    for op, data in diffs:
        text = escape(data)
        if op == dmp.DIFF_DELETE:
            highlighted_inappropriate += f'<b class="diff-change">{text}</b>'
        elif op == dmp.DIFF_EQUAL:
            highlighted_inappropriate += text

    tooltip_text = "The text with the red dotted underline will be edited. The bold red parts will be removed."
    before, _, after = source_text.partition(inappropriate_part)
    result = (
        f'{escape(before)}'
        f'<span class="highlight-inappropriate" uk-tooltip="{tooltip_text}">{highlighted_inappropriate}</span>'
        f'{escape(after)}'
    )
    return mark_safe(result)


@register.simple_tag
def display_revised_diff(source_text, inappropriate_part, rewritten_part):
    """
    Displays the revised text, highlighting the edited portion with a green
    dotted underline and a tooltip, and bolding the added characters within it.
    """
    source_text = str(source_text) if source_text is not None else ''
    inappropriate_part = str(inappropriate_part) if inappropriate_part is not None else ''
    rewritten_part = str(rewritten_part) if rewritten_part is not None else ''

    if inappropriate_part in ('nan', 'None', '') or not source_text:
        return mark_safe(escape(source_text))

    if rewritten_part in ('nan', 'None'):
        rewritten_part = ''

    # Diff the parts to find additions
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(inappropriate_part, rewritten_part)
    dmp.diff_cleanupSemantic(diffs)

    # Build the HTML for the rewritten part
    highlighted_rewritten = ''
    for op, data in diffs:
        text = escape(data)
        if op == dmp.DIFF_INSERT:
            highlighted_rewritten += f'<b class="diff-add">{text}</b>'
        elif op == dmp.DIFF_EQUAL:
            highlighted_rewritten += text

    if inappropriate_part not in source_text:
        # Fallback: if the part to be replaced isn't in the source, just return the rewritten text.
        revised_text = source_text.replace(inappropriate_part, rewritten_part)
        return mark_safe(escape(revised_text))

    tooltip_text = "This is the edited version of the text. The bold green parts are newly added text."
    # Partition the source text and insert the new highlighted part
    before, _, after = source_text.partition(inappropriate_part)
    result = (
        f'{escape(before)}'
        f'<span class="highlight-appropriate" uk-tooltip="{tooltip_text}">{highlighted_rewritten}</span>'
        f'{escape(after)}'
    )
    return mark_safe(result)
