from unidecode import unidecode
from string import punctuation, digits, whitespace, ascii_lowercase


count = lambda l1,l2: sum([1 for x in l1 if x in l2])

def countAlphabet(file):
    myFile = open(file, 'r', encoding='utf-8')
    file = myFile.read()
    decoded = unidecode(file.lower())

    for letter in ascii_lowercase:
        freq = round(decoded.count(letter) / len(decoded), 5)
        print(letter + ": " + str(freq))

    others = punctuation + digits + whitespace
    others_freq = round(count(decoded, others) / len(decoded), 5)
    print('- :' + str(others_freq))

if __name__ == "__main__":
    print('\nNoli Me Tangere (English ver.)')
    countAlphabet('noli_eng.txt')
    print('\nNoli Me Tangere (Tagalog ver.)')
    countAlphabet('noli_tag.txt')
