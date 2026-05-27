import re


def find_valid_phrases_list(text: str, max_chars_in_phrase: int | None = None) -> list[str]:
    """
    Extract all valid phrases from a given text based on punctuation and formatting rules.

    Args:
        text (str): The input text.
        max_chars_in_phrase (int, optional): Maximum number of characters allowed per phrase. 
                                              If None, no limit is applied.
    Returns:
        list[str]: List of cleaned, unique phrases.
    """
    text = f" {text.strip()} "
    phrases = []

    # Collect split positions
    split_positions = {0, len(text)}

    # Define patterns for splits (merged for clarity)
    split_patterns = [
        # before punctuation (without hyphen and period)
        r'(?<=\w)(?=[,!?;:\(\)])',
        r'(?<=[,!?;:\(\)])(?=\w)',      # after punctuation (without period)
        r'(?<=\))(?=\.)',               # between ) and .
        r'(?<=\))(?=\,)',               # between ) and ,
        r'\s+',                          # spaces
        r'[\$"\'""\/…]',                 # before/after special chars
        r'(?<=[a-z])(?=[A-Z])',          # camel-case boundary
        r'[^\x00-\x7F]',                 # non-ASCII chars
        r'(?<=\w)(?=-)',                 # before hyphen
        r'(?<=-)(?=\w)',                 # after hyphen
        r'(?<=\w)(?=\*)',                # before asterisk
        r'(?<=\*)',                      # after asterisk
        r'(?<=\w)(?=\.)',                # before period
        r'(?<=\.)(?=\w)',                # after period
    ]

    # Collect split indices
    for pattern in split_patterns:
        for match in re.finditer(pattern, text):
            split_positions.update({match.start(), match.end()})

    split_positions = sorted(split_positions)

    # Set character limit
    if max_chars_in_phrase is None:
        max_chars_in_phrase = len(text)

    # Generate phrases
    for i, start in enumerate(split_positions):
        for end in split_positions[i + 1:]:
            phrase = text[start:end].strip()
            if not phrase:
                continue
            if len(phrase) <= max_chars_in_phrase:
                phrases.append(phrase)

    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique_phrases.append(phrase)
    
    unique_phrases = [phrase for phrase in unique_phrases if not (phrase.startswith(('.', ',', ';', ':', '!', '?', '(')))]

    return unique_phrases