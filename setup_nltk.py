"""
Script to download required NLTK data
"""
import nltk

print("Downloading required NLTK data...")

resources = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords')
]

# Output is plain ASCII on purpose. This is the one setup step the README asks
# a new engineer to run by hand, and it used to print check marks: with stdout
# redirected to a file on a Windows console whose code page is not UTF-8 --
# cp1254 on a Turkish install -- printing one raises UnicodeEncodeError and the
# documented step fails with a traceback, having done its work. A setup script
# that cannot report its own success is worse than a plain one.
for path, name in resources:
    try:
        nltk.data.find(path)
        print(f"ok       {name} already installed")
    except LookupError:
        print(f"fetching {name}...")
        nltk.download(name, quiet=False)
        print(f"ok       {name} downloaded")

print("\nAll NLTK resources ready.")

