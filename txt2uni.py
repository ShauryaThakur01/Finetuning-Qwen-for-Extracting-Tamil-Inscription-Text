#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
# Author : Arulalan.T <arulalant@gmail.com>                                  #
# Date : 04.08.2014                                                          #
#                                                                            #
# Refactored for Python 3 Compatibility                                      #
#                                                                            #
##############################################################################

from collections import OrderedDict  # Moved to Python's built-in library
from encode2utf8 import anjal2utf8, bamini2utf8, boomi2utf8, \
    dinakaran2utf8, dinamani2utf8, dinathanthy2utf8, \
    kavipriya2utf8, murasoli2utf8, mylai2utf8, nakkeeran2utf8, \
    roman2utf8, tab2utf8, tam2utf8, tscii2utf8, pallavar2utf8, \
    indoweb2utf8, koeln2utf8, libi2utf8, oldvikatan2utf8, webulagam2utf8, \
    diacritic2utf8, shreelipi2utf8, softview2utf8, tace2utf8, vanavil2utf8

__all__ = ['anjal2unicode', 'bamini2unicode', 'boomi2unicode', 
    'dinakaran2unicode', 'dinathanthy2unicode', 'kavipriya2unicode',
    'murasoli2unicode', 'mylai2unicode', 'nakkeeran2unicode',
    'roman2unicode', 'tab2unicode', 'tam2unicode', 'tscii2unicode',
    'indoweb2unicode', 'koeln2unicode', 'libi2unicode', 'oldvikatan2unicode',
    'webulagam2unicode', 'auto2unicode', 'dinamani2unicode', 
    'pallavar2unicode', 'diacritic2unicode', 'shreelipi2unicode',
    'softview2unicode', 'tace2unicode', 'vanavil2unicode']
    
_all_encodes_ = OrderedDict([
    ('anjal2utf8', anjal2utf8), ('bamini2utf8', bamini2utf8), 
    ('boomi2utf8', boomi2utf8), ('dinakaran2utf8', dinakaran2utf8), 
    ('dinamani2utf8', dinamani2utf8), ('dinathanthy2utf8', dinathanthy2utf8), 
    ('kavipriya2utf8', kavipriya2utf8), ('murasoli2utf8', murasoli2utf8),
    ('mylai2utf8', mylai2utf8), ('nakkeeran2utf8', nakkeeran2utf8),
    ('roman2utf8', roman2utf8), ('tab2utf8', tab2utf8),
    ('tam2utf8', tam2utf8), ('tscii2utf8', tscii2utf8), 
    ('pallavar2utf8', pallavar2utf8), ('indoweb2utf8', indoweb2utf8),
    ('koeln2utf8', koeln2utf8), ('libi2utf8', libi2utf8),
    ('oldvikatan2utf8', oldvikatan2utf8), ('webulagam2utf8', webulagam2utf8),    
    ('diacritic2utf8', diacritic2utf8), ('shreelipi2utf8', shreelipi2utf8),
    ('softview2utf8', softview2utf8),  ('tace2utf8', tace2utf8),
    ('vanavil2utf8', vanavil2utf8),
])

# By enabling this flag, it will write individual encodes unique & common
# characters in text file.
__WRITE_CHARS_TXT = False


def encode2unicode(text, charmap):
    '''
    charmap : dictionary which has both encode as key, unicode as value
    '''
    if isinstance(text, (list, tuple)):
        unitxt = ''
        for line in text:
            for key, val in charmap.items():  # Py3: iteritems() -> items()
                if key in line:
                    line = line.replace(key, val)
            unitxt += line
        return unitxt
    elif isinstance(text, str):
        for key, val in charmap.items():  # Py3: iteritems() -> items()
            if key in text:
                text = text.replace(key, val)
        return text

def anjal2unicode(text): return encode2unicode(text, anjal2utf8)
def bamini2unicode(text): return encode2unicode(text, bamini2utf8)
def boomi2unicode(text): return encode2unicode(text, boomi2utf8)
def dinakaran2unicode(text): return encode2unicode(text, dinakaran2utf8)
def dinamani2unicode(text): return encode2unicode(text, dinamani2utf8)
def dinathanthy2unicode(text): return encode2unicode(text, dinathanthy2utf8)
def kavipriya2unicode(text): return encode2unicode(text, kavipriya2utf8)
def murasoli2unicode(text): return encode2unicode(text, murasoli2utf8)
def mylai2unicode(text): return encode2unicode(text, mylai2utf8)
def nakkeeran2unicode(text): return encode2unicode(text, nakkeeran2utf8)
def roman2unicode(text): return encode2unicode(text, roman2utf8)
def tab2unicode(text): return encode2unicode(text, tab2utf8)
def tam2unicode(text): return encode2unicode(text, tam2utf8)
def tscii2unicode(text): return encode2unicode(text, tscii2utf8)
def pallavar2unicode(text): return encode2unicode(text, pallavar2utf8)
def indoweb2unicode(text): return encode2unicode(text, indoweb2utf8)
def koeln2unicode(text): return encode2unicode(text, koeln2utf8)
def libi2unicode(text): return encode2unicode(text, libi2utf8)
def oldvikatan2unicode(text): return encode2unicode(text, oldvikatan2utf8)
def webulagam2unicode(text): return encode2unicode(text, webulagam2utf8)
def diacritic2unicode(text): return encode2unicode(text, diacritic2utf8)
def shreelipi2unicode(text): return encode2unicode(text, shreelipi2utf8)
def softview2unicode(text): return encode2unicode(text, softview2utf8)
def tace2unicode(text): return encode2unicode(text, tace2utf8)
def vanavil2unicode(text): return encode2unicode(text, vanavil2utf8)

def _get_unique_ch(text, all_common_encodes):
    """
        text : encode sample strings
        returns unique word / characters from input text encode strings.
    """
    unique_chars = ''
    if isinstance(text, str):
        text = text.split("\n")
    elif isinstance(text, (list, tuple)):
        pass

    special_chars = ['.', ',', ';', ':','', ' ', '\r', '\t', '=', '\n']
    for line in text:
        for word in line.split(' '):
            # Py3 natively handles Unicode; removed unicode(word, 'utf-8') cast
            for ch in all_common_encodes:
                if ch in word: word = word.replace(ch, '')

            if not word: continue

            for ch in word:
                if ch.isdigit() or ch in special_chars:
                    word = word.replace(ch, '')
                    continue
                return word
    return ''

def _get_unique_common_encodes():
    """
    Returns unique_encodes and common_encodes as tuple.
    """
    _all_unique_encodes_ = []
    _all_unicode_encodes_ = {}
    _all_common_encodes_ = set([])
    _all_common_encodes_single_char_ = set([])

    for name, encode in _all_encodes_.items():
        # Py3 natively handles Unicode strings; removed unicode() cast
        encode_utf8 = set([ch for ch in encode.keys()])
        _all_unicode_encodes_[name] = encode_utf8

    _all_unique_encodes_full_ = _all_unicode_encodes_.copy()

    for supname, super_encode in _all_unicode_encodes_.items():
        for subname, sub_encode in _all_unicode_encodes_.items():
            if supname == subname: continue
            super_encode = super_encode - sub_encode
        
        common = _all_unique_encodes_full_[supname] - super_encode
        _all_common_encodes_ = _all_common_encodes_.union(common)
        _all_unique_encodes_.append((supname, super_encode))

    for ch in _all_common_encodes_:
        if len(ch) == 1: _all_common_encodes_single_char_.add(ch)

    _all_common_encodes_ -= _all_common_encodes_single_char_


    if __WRITE_CHARS_TXT:
        # Py3: Added encoding='utf-8' and safe context managers ('with' blocks)
        with open('all.encodes.common.chars.txt', 'w', encoding='utf-8') as f:
            for ch in _all_common_encodes_:
                uni = ''
                for encode_keys in _all_encodes_.values():
                    if ch in encode_keys:
                        uni = encode_keys[ch]
                        break
                f.write(ch + '  =>  ' + uni + '\n')

        for encode_name, encode_keys in _all_unique_encodes_:
            with open(encode_name + '.unique.chars.txt', 'w', encoding='utf-8') as f:
                for ch in encode_keys:
                    uni = _all_encodes_[encode_name][ch]
                    f.write(ch + '  =>  ' + uni + '\n')

    return (_all_unique_encodes_, _all_common_encodes_)


def auto2unicode(text):
    """
    Tries to identify encode in available encodings.
    """
    _all_unique_encodes_, _all_common_encodes_ = _get_unique_common_encodes()
    unique_chars = _get_unique_ch(text, _all_common_encodes_)
    clen = len(_all_common_encodes_)
    
    # Py3: Modern f-string formatting
    msg = "Sorry, couldn't find encode :-(\n"
    msg += f"Need more words to find unique encode out side of {clen} common compound characters"
    
    if not unique_chars:
        print(msg)
        return ''

    for encode_name, encode_keys in _all_unique_encodes_:
        if not len(encode_keys): continue
        for ch in encode_keys:
            if ch in unique_chars:
                print("Whola! found encode : ", encode_name)
                encode = _all_encodes_[encode_name]
                return encode2unicode(text, encode)
    else:
        print(msg)
        return ''