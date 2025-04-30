"""
Linguistic features module for sentence fragment detection and analysis.
Contains common word lists and regex patterns used across multiple scripts.
"""

# 1. Auxiliary Verbs (verbs indicating tense, state, or activity)
AUXILIARY_VERBS = [
    "is", "are", "was", "were", "be", "been", "am", "being", 
    "has", "have", "had", "does", "do", "did",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "ought", "need", "dare", "used", "going", "get", "gets", "got", "gotten",
    "keep", "keeps", "kept", "seem", "seems", "seemed", "become", "becomes", "became",
    "remain", "remains", "remained", "stay", "stays", "stayed", "appear", "appears", "appeared"
]

# 2. Common Expressions / Exclamations
COMMON_EXPRESSIONS = [
    "wow", "oops", "amazing", "fantastic", "unbelievable", "incredible", 
    "interesting", "sadly", "fortunately", "surprisingly", "finally",
    "oh", "ah", "ugh", "yikes", "gosh", "goodness", "jeez", "alas", "phew",
    "hmm", "huh", "meh", "yay", "hooray", "bingo", "bravo", "congrats", "darn",
    "excellent", "great", "wonderful", "terrific", "brilliant", "splendid", "marvelous",
    "unfortunately", "regrettably", "honestly", "frankly", "truthfully", "admittedly",
    "basically", "essentially", "literally", "virtually", "practically", "technically",
    "ironically", "curiously", "strangely", "oddly", "weirdly", "bizarrely"
]

# 3. Conjunctions (links between ideas)
CONJUNCTIONS = [
    "and", "but", "or", "so", "because", "although", "however", 
    "yet", "still", "therefore", "meanwhile", "whereas",
    "since", "unless", "until", "while", "though", "if", "whether",
    "as", "when", "where", "whenever", "wherever", "once", "before",
    "after", "than", "that", "which", "who", "whom", "whose", "what",
    "whatever", "whichever", "whoever", "whomever", "nor", "for", "plus",
    "furthermore", "moreover", "additionally", "consequently", "hence",
    "thus", "accordingly", "otherwise", "nevertheless", "nonetheless",
    "instead", "alternatively", "conversely", "similarly", "likewise"
]

# 4. Temporal Words (expressions indicating time changes)
TEMPORAL_WORDS = [
    "then", "after", "before", "later", "suddenly", "soon", 
    "eventually", "earlier", "now", "today", "tonight", "tomorrow",
    "yesterday", "morning", "afternoon", "evening", "night", "midnight",
    "dawn", "dusk", "weekly", "monthly", "yearly", "daily", "hourly",
    "instantly", "immediately", "promptly", "currently", "presently",
    "previously", "formerly", "lately", "recently", "nowadays", "momentarily",
    "temporarily", "briefly", "shortly", "occasionally", "frequently", "regularly",
    "periodically", "constantly", "continuously", "perpetually", "eternally",
    "forever", "always", "never", "ever", "seldom", "rarely", "sometimes",
    "often", "usually", "generally", "typically", "historically", "traditionally",
    "initially", "ultimately", "finally", "lastly", "meanwhile", "simultaneously",
    "concurrently", "subsequently", "consequently", "accordingly", "henceforth"
]

# 5. Adverbs of Opinion / Degree - Common explicit ones
ADVERBS_OPINION = [
    "clearly", "obviously", "probably", "certainly", "absolutely", 
    "seriously", "undoubtedly", "definitely", "arguably", "presumably",
    "apparently", "evidently", "seemingly", "supposedly", "allegedly",
    "conceivably", "possibly", "perhaps", "maybe", "likely", "unlikely",
    "surely", "truly", "really", "actually", "honestly", "frankly",
    "admittedly", "unfortunately", "fortunately", "surprisingly", "amazingly",
    "astonishingly", "shockingly", "disappointingly", "regrettably", "sadly",
    "happily", "gladly", "hopefully", "mercifully", "thankfully", "luckily",
    "incredibly", "remarkably", "notably", "significantly", "substantially",
    "considerably", "essentially", "fundamentally", "basically", "primarily",
    "mainly", "largely", "mostly", "generally", "typically", "usually",
    "normally", "commonly", "frequently", "occasionally", "rarely", "seldom"
]

# 6. Sentence Starters / Fillers
SENTENCE_STARTERS = [
    "well", "so", "anyway", "besides", "actually", "by the way", 
    "in fact", "as a matter of fact", "to be honest", "honestly",
    "frankly", "to tell the truth", "truthfully", "admittedly",
    "obviously", "clearly", "evidently", "apparently", "seemingly",
    "interestingly", "surprisingly", "remarkably", "notably", "significantly",
    "importantly", "essentially", "basically", "fundamentally", "generally",
    "typically", "usually", "normally", "commonly", "frequently", "occasionally",
    "first", "firstly", "second", "secondly", "third", "thirdly", "finally", "lastly",
    "meanwhile", "subsequently", "consequently", "therefore", "thus", "hence",
    "accordingly", "as a result", "for this reason", "due to this", "because of this",
    "nevertheless", "nonetheless", "however", "on the other hand", "conversely",
    "in contrast", "alternatively", "instead", "rather", "in addition", "furthermore",
    "moreover", "similarly", "likewise", "in the same way", "for example", "for instance",
    "specifically", "in particular", "namely", "to illustrate", "such as", "including",
    "in conclusion", "to conclude", "to summarize", "in summary", "overall", "ultimately",
    "in the end", "eventually", "after all", "all in all", "on the whole", "by and large"
]

# Regex patterns for linguistic features
ADVERB_PATTERNS = [
    r'\b\w+ly\b',         # Words ending with 'ly' (carefully, quickly, etc.)
    r'\b\w+ily\b',        # Words ending with 'ily' (happily, easily, etc.)
    r'\b\w+ally\b',       # Words ending with 'ally' (basically, naturally, etc.)
    r'\b\w+ically\b',     # Words ending with 'ically' (specifically, dramatically, etc.)
]

# Past tense verb patterns
PAST_VERB_PATTERNS = [
    r'\b\w{3,}ed\b',      # Regular past tense (played, walked, etc.)
    r'\b\w{3,}en\b',      # Past participles (taken, broken, etc.)
    r'\b\w+ied\b',        # Words ending with 'ied' (cried, tried, etc.)
    r'\b\w+ought\b',      # Irregular past tense (thought, bought, etc.)
    r'\b\w+aught\b',      # Irregular past tense (caught, taught, etc.)
    r'\bwent\b',          # Irregular past tense of 'go'
    r'\bsaw\b',           # Irregular past tense of 'see'
    r'\bcame\b',          # Irregular past tense of 'come'
    r'\btook\b',          # Irregular past tense of 'take'
    r'\bgave\b',          # Irregular past tense of 'give'
    r'\bmade\b',          # Irregular past tense of 'make'
    r'\bsaid\b',          # Irregular past tense of 'say'
    r'\bfelt\b',          # Irregular past tense of 'feel'
    r'\bheld\b',          # Irregular past tense of 'hold'
    r'\bfound\b',         # Irregular past tense of 'find'
    r'\bknew\b',          # Irregular past tense of 'know'
    r'\bgot\b',           # Irregular past tense of 'get'
    r'\bput\b',           # Irregular past tense of 'put'
    r'\bset\b',           # Irregular past tense of 'set'
    r'\bran\b',           # Irregular past tense of 'run'
    r'\bwrote\b',         # Irregular past tense of 'write'
]

# Gerund patterns
GERUND_PATTERNS = [
    r'\b\w{3,}ing\b',     # Standard gerunds (playing, walking, etc.)
    r'\b\w+ying\b',       # Words ending with 'ying' (trying, crying, etc.)
    r'\b\w+ling\b',       # Words ending with 'ling' (handling, cycling, etc.)
    r'\b\w+ting\b',       # Words ending with 'ting' (sitting, getting, etc.)
    r'\b\w+ping\b',       # Words ending with 'ping' (shopping, clapping, etc.)
    r'\b\w+ning\b',       # Words ending with 'ning' (running, planning, etc.)
    r'\b\w+ming\b',       # Words ending with 'ming' (swimming, coming, etc.)
    r'\b\w+ding\b',       # Words ending with 'ding' (reading, building, etc.)
    r'\b\w+cing\b',       # Words ending with 'cing' (dancing, racing, etc.)
    r'\b\w+king\b',       # Words ending with 'king' (talking, making, etc.)
    r'\b\w+ging\b',       # Words ending with 'ging' (hanging, bringing, etc.)
]

# Common adverbs that don't end with 'ly' but should be excluded from pattern matching
NON_SPLITTING_ADVERBS = [
    "very", "quite", "rather", "too", "so", "just", "only",
    "almost", "nearly", "really", "pretty", "even"
]

# Feature descriptions for documentation and display
FEATURE_DESCRIPTIONS = {
    "has_auxiliary": "Contains auxiliary verbs (is, are, was, were, etc.)",
    "has_fullstop": "Contains a period (.)",
    "has_question_mark": "Contains a question mark (?)",
    "has_exclamation_mark": "Contains an exclamation mark (!)",
    "has_comma": "Contains a comma (,)",
    "has_semicolon": "Contains a semicolon (;)",
    "has_colon": "Contains a colon (:)",
    "has_quotation": "Contains quotation marks (' or \")",
    "has_expression": "Contains common expressions or exclamations",
    "has_conjunction": "Contains conjunctions (and, but, or, etc.)",
    "has_temporal": "Contains temporal words (then, after, before, etc.)",
    "has_opinion_adverb": "Contains adverbs of opinion (clearly, obviously, etc.)",
    "has_adverb": "Contains adverbs (words ending with -ly, -ily, etc.)",
    "has_starter": "Contains sentence starters or fillers (well, so, etc.)",
    "has_past_verb": "Contains past tense verbs (words ending with -ed or -en)",
    "has_gerund": "Contains gerunds/present participles (words ending with -ing)",
    "word_count": "Number of words in the sentence",
    "char_count": "Number of characters in the sentence",
    "starts_capitalized": "Sentence starts with a capital letter",
    "all_caps_word": "Contains a word in all capital letters"
} 