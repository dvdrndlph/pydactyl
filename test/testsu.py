#!/usr/bin/env python3

GRAMMAR = """
        @@grammar::abcDF

        sequence = upper:staff ['@' lower:staff] ;

        staff = '&'.{line};
        line = {score_fingering}* ;

        score_fingering = orn:ornamental ["/" alt_orn:ornamental] segmenter:[segmenter]
        | pf:pedaled_fingering ['/' alt_pf:pedaled_fingering] segmenter:[segmenter]
        | p:pedaling ['/' alt_p:pedaling] segmenter:[segmenter]
        ;

        ornamental = ornaments:('(' {pedaled_fingering}+ ')') ;

        pedaled_fingering = soft:[soft] fingering:fingering damper:[damper] ;
        pedaling = soft:{soft}+ wildcard damper:{damper}+ ;

        fingering = strike:finger ['-' release:finger]
        | wildcard;
        finger = hand:[hand] digit:digit ;

        segmenter = "," | ";" | "." ;
        damper = '_' | '^' ;
        soft = /[pf]/ ;
        hand = '<' | '>' ;
        digit = /[1-5]/ ;
        wildcard = /x/ ;
    """

from pprint import pprint
from tatsu import parse

GRAMMAR = """
@@grammar :: test
@@nameguard :: False
@@namechars :: '12345'

start = sequence $ ;
sequence = {digit}+ ;
digit = 'x' | '1' | '2' | '3' | '4' | '5' ;"""

test = "23"
ast = parse(GRAMMAR, test)
pprint(ast)  # Prints ['2', '3']

test = "xx"
ast = parse(GRAMMAR, test, nameguard=False)
pprint(ast)
# fingering = [hand] digit ;
# hand = '<' | '>' ;
from pprint import pprint
from tatsu import parse

# test = "2xx&1x2@1&2"
# ast = parse(GRAMMAR, test)
# pprint(ast)  # Prints ['2', '3']

# exit(0)

GRAMMAR2 = """
@@grammar :: test
@@nameguard :: False

start = sequence $ ;
sequence = {digit}*;
digit = 'x' ;"""

test = "xx"
ast = parse(GRAMMAR2, test, start="start")  # tatsu.exceptions.FailedToken: (1:1) expecting 'x' :

# digit = /x/ ;"""

# test = "23"
# ast = parse(GRAMMAR2, test)
# pprint(ast)  # Prints ['2', '3']
