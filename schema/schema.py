from helper import EVENT_TYPES, ORDER_MAIN, SUBJECT_FIELDS, TREATMENT_FIELDS
import re
import sys
import os

# Add root directory to sys.path to import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def escape_except_space(text):
    """Escapes all regex meta-characters except space. Handles special cases for '&' and '#' which are often unnecessary to escape and cause warnings in some engines."""
    escaped = re.escape(text).replace(r'\ ', ' ')
    # Some regex engines (like vLLM's underlying ones) warn about \ & and \ #
    return escaped.replace(r'\&', '&').replace(r'\#', '#')


def phrases_with_signs(valid_phrases):
    """Returns the list of valid phrases. Placeholder for additional processing."""
    return valid_phrases


def get_schema_main(valid_phrases, order=ORDER_MAIN, allow_empty=True):
    """
    Returns a regex pattern for the 'main' format: 
    a list of tuples based on the specified order.

    Default structure: [('Event_type', 'subject', 'treatment', 'effect'), ...]
    Where Event_type is one of EVENT_TYPES from helper.py.
    Other fields are either 'null' or one of the valid phrases from the text.
    """
    event_types = "|".join(EVENT_TYPES)
    # Allows both single and double quotes for event types: 'Type' or "Type"
    event_type_pattern = rf"(['\"]({event_types})['\"])"

    # For extraction fields, we allow 'null' or any valid phrase
    # escaping them for use in regex.
    escaped_phrases = [escape_except_space(
        p) for p in phrases_with_signs(valid_phrases)]
    phrases_pattern = "|".join(p for p in escaped_phrases if p)

    # Pattern for a field that can be 'null' or a quoted valid phrase
    # Supports 'phrase', "phrase", 'null', or "null"
    field_pattern = rf"((['\"]null['\"])|(['\"]({phrases_pattern})['\"]))"

    # Build the tuple pattern based on the order
    parts = []
    for field in order:
        if field == "event_type":
            parts.append(event_type_pattern)
        else:
            parts.append(field_pattern)

    tuple_pattern = rf"\(" + ", ".join(parts) + rf"\)"

    # list of tuples: [(...), (...)]
    if allow_empty:
        pattern = rf"\[({tuple_pattern}(, {tuple_pattern})*)?\]"
    else:
        pattern = rf"\[{tuple_pattern}(, {tuple_pattern})*\]"

    # Return pattern with optional newline at the end
    return pattern + r"\n?"


def get_schema_sub(valid_phrases, order_main=ORDER_MAIN,
                   order_subject=SUBJECT_FIELDS,
                   order_treatment=TREATMENT_FIELDS,
                   allow_empty=True):
    """
    Returns a regex pattern for the 'sub' format: 
    a list of dictionaries based on the specified orders.

    Structure: [{"event_type": "...", "subject": ("...", ...), "treatment": ("...", ...), "effect": "..."}, ...]
    """
    event_types = "|".join(EVENT_TYPES)
    # Allows both single and double quotes for event types: 'Type' or "Type"
    event_type_pattern = rf"(['\"]({event_types})['\"])"

    # For extraction fields, we allow 'null' or any valid phrase
    escaped_phrases = [escape_except_space(
        p) for p in phrases_with_signs(valid_phrases)]
    phrases_pattern = "|".join(p for p in escaped_phrases if p)

    # Pattern for a field that can be 'null' or a quoted valid phrase
    field_pattern = rf"((['\"]null['\"])|(['\"]({phrases_pattern})['\"]))"

    # helper for tuple patterns
    def get_tuple_pattern(num_fields):
        parts = [field_pattern] * num_fields
        return rf"\(" + ", ".join(parts) + rf"\)"

    # Build dictionary pattern
    dict_parts = []
    for key in order_main:
        # JSON keys usually use double quotes
        key_pattern = rf"\"{key}\""

        if key == "event_type":
            val_pattern = event_type_pattern
        elif key == "subject":
            val_pattern = get_tuple_pattern(len(order_subject))
        elif key == "treatment":
            val_pattern = get_tuple_pattern(len(order_treatment))
        else:
            val_pattern = field_pattern

        dict_parts.append(rf"{key_pattern}:\s*{val_pattern}")

    dict_pattern = rf"\{{" + ", ".join(dict_parts) + rf"\}}"

    # list of dictionaries: [{}, {}]
    if allow_empty:
        pattern = rf"\[({dict_pattern}(, {dict_pattern})*)?\]"
    else:
        pattern = rf"\[{dict_pattern}(, {dict_pattern})*\]"

    return pattern + r"\n?"
