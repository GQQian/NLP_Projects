import os
def process(file):
    """
    folder: name of folder
    TODO: wrong way to compile all files together, test files should be parsed separately,
    """
    compiled_content = []
    raw_content = open(file, 'r').read()
    compiled_content = raw_content.split('\n')

    while '' in compiled_content:
        compiled_content.remove('')

    for i, unit in enumerate(compiled_content):
        compiled_content[i] = tuple(unit.split('\t'))
    return compiled_content

def generate_path(folder):
    return os.getcwd() + "/" + folder + "/"