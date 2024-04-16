
def wrap_text(text, n_words = 5, indent = ''):
    """
    Wrap text with a newline character every n_words with an 
    optional indent on each line

    inputs:
        -text: string to be wrapped into multiple lines
        -n_words: the max number of words in a line
        -indent: the string indent on subsecuent lines (Examples: '    ', '\t')
    """
    sent = text.split(' ')
    for i in range(n_words, len(sent), n_words):
        sent.insert(i, '\n'+indent)
    return ' '.join(sent)