import re


def tag(sentence, tokens, predicted):
    results, prev, last_idx = [], None, 0

    for i, _ in enumerate(tokens):
        pred = "O"
        pred_label = "O"
        try:
            pred = predicted[i]
            pred_label = predicted[i].split("-")
            # idx = np.argmax(pred)
            # score = pred[idx]
        except Exception as e:
            pass

        if len(pred_label) == 2:
            prefix, label = pred_label
        else:
            prefix = 'O'
            label = pred

        start_idx = last_idx + sentence[last_idx:].index(tokens[i])
        end_idx = start_idx + len(tokens[i])

        if prefix in ['I', 'E', 'O']:
            if label == prev:
                results[-1]['end'] = end_idx
            else:
                # mislabelled or 'O'
                results.append({
                    'start': start_idx,
                    'end': end_idx,
                    'tagname': label,
                })
        elif prefix in ['B', 'S']:
            results.append({
                'start': start_idx,
                'end': end_idx,
                'tagname': label,
            })

        last_idx = end_idx
        prev = label

    for i, pred in enumerate(results):
        results[i]['span'] = sentence[pred['start']:pred['end']]
    return results


def word_tokenize(sentence, sep=r'(\W+)?'):
    return [x.strip() for x in re.split(sep, sentence) if x.strip()]