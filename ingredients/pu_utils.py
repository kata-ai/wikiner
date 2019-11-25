
from ingredients.crf_utils import wordshape

def word2pufeatures(sent, idx, window_size):
    word = sent[idx][0]
    
    features = {
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'contains_digit': any(char.isdigit() for char in word),
        'word.shape': wordshape(word),
    }
    
    for j in range(window_size):
        if idx-j-1 >= 0:
            word = sent[idx-j-1][0]
            features.update({
                str(j-1) + ':word.istitle()': word.istitle(),
                str(j-1) + ':word.isupper()': word.isupper(),
                str(j-1) + ':word.isdigit()': word.isdigit(),
                str(j-1) + ':contains_digit': any(char.isdigit() for char in word),

            })
            if idx-j-2 >= 0:
                wordprev = sent[idx-j-2][0]
                features.update({
                    f'{str(j-2)}-{str(j-1)}:word.lower()': f'{word.lower()}-{wordprev.lower()}',
                })

        if idx+j+1 < len(sent):
            word = sent[idx+j+1][0]
            features.update({
                str(j+1) + ':word.istitle()': word.istitle(),
                str(j+1) + ':word.isupper()': word.isupper(),
                str(j+1) + ':word.isdigit()': word.isdigit(),
                str(j+1) + ':contains_digit': any(char.isdigit() for char in word),
            })
            if idx+j+2 < len(sent):
                wordnext = sent[idx+j+2][0]
                features.update({
                    f'{str(j-2)}-{str(j-1)}:word.lower()': f'{word.lower()}-{wordnext.lower()}',
                })
    
    if idx == 0:
        features['BOS'] = True

    if idx == len(sent)-1:
        features['EOS'] = True

    return features

def sent2pufeatures(sent, window_size):
    return [word2pufeatures(sent, i, window_size) for i in range(len(sent))]
