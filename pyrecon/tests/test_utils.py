import re


def decode_eval_str(s):
    # change ${col} => col, and return list of columns
    toret = str(s)
    columns = []
    for replace in re.finditer('(\${.*?})',s):
        value = replace.group(1)
        col = value[2:-1]
        toret = toret.replace(value,col)
        if col not in columns: columns.append(col)
    return toret, columns


def test_decode_eval_str():
    s = '(${RA}>0.) & (${RA}<30.) & (${DEC}>0.) & (${DEC}<30.)'
    s,cols = decode_eval_str(s)
    print(s,cols)


if __name__ == '__main__':

    test_decode_eval_str()
