# Diff Display Feature

## Overview

The annotation interface has been updated to display **diffs** between the source argument and the rewrites (Rewrite A and Rewrite B), instead of showing plain text. This makes it easier for annotators to quickly identify what changes each model made.

## Changes Made

### 1. Custom Template Filter (`study/templatetags/diff_filters.py`)

Created a custom Django template filter that uses Python's built-in `difflib` library to generate HTML diffs:

- **`text_diff`**: Word-level diff (recommended for readability)
- **`char_diff`**: Character-level diff (more granular)

The filter compares the rewritten text with the source and highlights:
- **Insertions** (green background): Text added in the rewrite
- **Deletions** (red background with strikethrough): Text removed from the source
- **Replacements**: Both deletion and insertion shown together

### 2. Updated Template (`study/templates/study/annotation.html`)

- Added `{% load diff_filters %}` to load the custom filter
- Changed `{{post.rewrite_a}}` to `{{post.rewrite_a|text_diff:post.source}}`
- Changed `{{post.rewrite_b}}` to `{{post.rewrite_b|text_diff:post.source}}`
- Added CSS styling for diff display

### 3. CSS Styling

Added the following CSS classes:

```css
.diff-insert {
  background-color: #d4edda;  /* Light green */
  color: #155724;             /* Dark green text */
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: 500;
}

.diff-delete {
  background-color: #f8d7da;  /* Light red */
  color: #721c24;             /* Dark red text */
  text-decoration: line-through;
  padding: 2px 4px;
  border-radius: 3px;
  opacity: 0.8;
}

.diff-container {
  line-height: 1.8;           /* Better spacing for diff display */
  word-wrap: break-word;
}
```

## Visual Examples

### Before (Plain Text)
```
Rewrite A: I think that books are better than TV because they are educational.
```

### After (Diff Display)
```
Rewrite A: I [think] that [books] are better than [TV] [because] [they are educational].
```
Where:
- `[green text]` = additions
- `~~red strikethrough~~` = deletions

## Benefits

1. **Faster annotation**: Annotators can quickly see what changed without reading entire texts
2. **Better comparison**: Easier to identify which model made which types of edits
3. **Improved accuracy**: Less cognitive load leads to more accurate annotations
4. **No dependencies**: Uses Python's built-in `difflib` library

## Technical Details

- **Filter type**: Word-level diff (splits on spaces)
- **Comparison**: Uses `difflib.SequenceMatcher` for optimal diff generation
- **Output**: HTML with `mark_safe()` to render in template
- **Performance**: Efficient for typical argument lengths (< 1000 words)

## Alternative: Character-Level Diff

If you prefer character-level diffs (more granular), change the template to use:

```django
{{post.rewrite_a|char_diff:post.source}}
{{post.rewrite_b|char_diff:post.source}}
```

This will show character-by-character changes instead of word-level changes.
