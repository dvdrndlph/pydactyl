__author__ = 'David Randolph'
# Copyright (c) 2014-2018 David A. Randolph.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
from tatsu import compile
import re
import copy
from pydactyl.dactyler import Constant
from pprint import pprint


class DAnnotation:
    GRAMMAR = """
        @@grammar :: abcDF
        @@nameguard :: False
        
        start = sequence $ ;
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
        | wildcard ;
        finger = hand:[hand] digit:digit ;

        segmenter = /[,;\.]/ ; 
        damper = '_' | '^' ;
        soft = /[pf]/ ;
        hand = '<' | '>' ;
        digit = /[1-5]/ ;
        wildcard = /x/ ;
    """
    _parser = compile(GRAMMAR)

    FINGERING_STR = '% abcD fingering '
    AUTHORITY_STR = '% Authority: '
    TRANSCRIBER_STR = '% Transcriber: '
    TRANSCRIPTION_DATE_STR = '% Transcription date: '

    @staticmethod
    def abcdf_to_ast(abcdf):
        ast = DAnnotation._parser.parse(abcdf)
        return ast

    def parse(self):
        return self._ast

    def score_fingering_at_index(self, index, staff="upper"):
        if staff != "upper" and staff != "lower":
            raise Exception("Score fingerings ordered within upper and lower staff only.")
        if staff == "upper":
            lines = self._ast.upper
        else:
            lines = self._ast.lower

        offset = 0
        for line in lines:
            if index < len(line) + offset:
                adjusted_index = index - offset
                return line[adjusted_index]
            offset += len(line)
        return None

    def score_fingerings(self, staff="both"):
        if staff != "upper" and staff != "lower":
            raise Exception("Score fingerings ordered within upper and lower staff only.")
        if staff == "upper":
            lines = self._ast.upper
        else:
            lines = self._ast.lower

        # Each staff is parsed into an array of lines. Each
        # line is an array of "score fingerings," or note
        # fingerings with all the trimmings.
        sfs = []
        for line in lines:
            for sf in line:
                sfs.append(sf)
        # pprint(sfs)
        return sfs

    def strike_digit_at_index(self, index, staff="upper"):
        sf = self.score_fingering_at_index(staff=staff, index=index)
        if isinstance(sf.pf.fingering, str):
            return None
        return sf.pf.fingering.strike.digit

    def phrase_mark_at_index(self, index, staff="upper"):
        sf = self.score_fingering_at_index(staff=staff, index=index)
        if not sf:
            raise Exception("No note at index {}.".format(index))
        segger = sf.segmenter
        return segger

    def parse_upper(self):
        upper_abcdf = self.upper_abcdf()
        return DAnnotation.abcdf_to_ast(upper_abcdf)

    def parse_lower(self):
        lower_abcdf = self.upper_abcdf()
        return DAnnotation.abcdf_to_ast(lower_abcdf)

    def score_fingering_count(self, staff="both"):
        # print(staff + ":::" + self.abcdf() + "\n")
        ast = self.parse()
        # print(ast)
        count = 0
        # Each staff is parsed into an array of lines. Each
        # line is an array of "score fingerings," or note
        # fingerings with all the trimmings.
        if 'upper' in ast and (staff == "upper" or staff == "both"):
            lines = ast['upper']
            for line in lines:
                count += len(line)
        if 'lower' in ast and (staff == "lower" or staff == "both"):
            lines = ast['lower']
            for line in lines:
                count += len(line)
        return count

    @staticmethod
    def ast_to_segregated_strike_digits(ast, staff="upper", hand=None):
        """
        :return: String of digits (1-5), assuming all fingerings are
                 are for the specified hand (">" or right for the
                 upper staff by default).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        if staff not in ("upper", "lower"):
            raise Exception("Invalid input: staff must be 'upper' or 'lower'.")

        if not hand:
            hand = ">"
            if staff == "lower":
                hand = "<"

        digits = []
        if staff == "upper":
            lines = ast.upper
        else:
            lines = ast.lower

        for line in lines:
            for score_fingering in line:
                if isinstance(score_fingering.pf.fingering, str):
                    digits.append('x')
                    continue
                strike = score_fingering.pf.fingering.strike
                current_hand = strike.hand
                digit = strike.digit
                if current_hand and current_hand != hand:
                    return None
                digits.append(digit)
        digit_str = "".join(digits)
        return digit_str

    @staticmethod
    def default_hand(staff, hand=None):
        if hand is not None:
            default_hand = hand
        elif staff == "upper":
            default_hand = ">"
        else:
            default_hand = "<"
        return default_hand

    @staticmethod
    def hand_and_digit(action, staff="upper", hand=None):
        digit = None
        if action:
            current_hand = action.hand
            digit = int(action.digit)
        else:
            current_hand = None
        if not current_hand:
            current_hand = DAnnotation.default_hand(staff=staff, hand=hand)
        return current_hand, digit

    @staticmethod
    def handed_digit(action, staff="upper", hand=None):
        current_hand, digit = DAnnotation.hand_and_digit(action, staff=staff, hand=hand)
        if digit:
            handed_digit = current_hand + str(digit)
        else:
            handed_digit = None
        return handed_digit

    @staticmethod
    def hands_and_digits_for_ornaments(orns, staff="upper", hand=None):
        """
        Return an array of tuples specifying the fingerings for the input ornamentals.

        :param orns: The list of ornamental AST elements to process.
        :param staff: The staff ("upper" or "lower") from which sf was taken. Used to determine
        default hand to use if not overridden by the hand parameter.
        :param hand: The "prevailing" hand in use for fingerings prior to the score fingering
        in the complete staff annotation sequence.
        :return: Hash of list of (hand, digit) tuples, with keys "strike" and "release."
        While the hand will always be set, the digit may be set to None if an "x" is
        specified for the score fingering.
        """
        hands_and_digits = {'strike': [], 'release': []}
        for orn in orns:
            act = orn.fingering.strike
            strike_hand, strike_digit = DAnnotation.hand_and_digit(action=act, staff=staff, hand=hand)
            hands_and_digits['strike'].append((strike_hand, strike_digit))
            act = orn.fingering.release
            rel_hand, rel_digit = DAnnotation.hand_and_digit(action=act, staff=staff, hand=strike_hand)
            if not rel_digit:
                rel_hand = strike_hand
                rel_digit = strike_digit
                hand = strike_hand
            else:
                hand = rel_hand
            hands_and_digits['release'].append((rel_hand, rel_digit))
        return hands_and_digits

    @staticmethod
    def pedals_for_ornaments(orns, damper="^", soft="f"):
        dampers = []
        softs = []
        for orn in orns:
            damper = orn.damper or damper
            dampers.append(damper)
            soft = orn.soft or soft
            softs.append(soft)
        return dampers, softs

    @staticmethod
    def pedals_for_score_fingering(sf, damper="^", soft="f"):
        dampers = []
        softs = []
        if sf.orn:
            dampers, softs = DAnnotation.pedals_for_ornaments(
                orns=sf.orn.ornaments[1], damper=damper, soft=soft)
        elif sf.pedaling:
            damper = sf.pedaling.damper or damper
            soft = sf.pedaling.soft or soft
            dampers.append(damper)
            softs.append(soft)
        elif sf.pf:
            damper = sf.pf.damper or damper
            soft = sf.pf.soft or soft
            dampers.append(damper)
            softs.append(soft)
        return dampers, softs

    @staticmethod
    def hands_and_digits_for_score_fingering(sf, staff="upper", hand=None):
        """
        Return an array of tuples specifying the fingerings for the input score_fingering

        :param sf: The score_fingering AST to process.
        :param staff: The staff ("upper" or "lower") from which sf was taken. Used to determine
        default hand to use if not overridden by the hand parameter.
        :param hand: The "prevailing" hand in use for fingerings prior to the score fingering
        in the complete staff annotation sequence.
        :return: Hash of lists of (hand, digit) tuples, with keys "strike" and "release."
        While the hand will always be set, the digit may be set to None if an "x" is
        specified for the score fingering.
        """
        if sf is None:
            return None

        hands_and_digits = {'strike': [], 'release': []}
        if sf.orn:
            hands_and_digits = DAnnotation.hands_and_digits_for_ornaments(
                orns=sf.orn.ornaments[1], staff=staff, hand=hand)
        elif isinstance(sf.pf.fingering, str):
            hand = DAnnotation.default_hand(staff=staff)
            hd = (hand, None)
            hands_and_digits['strike'].append(hd)
            hands_and_digits['release'].append(hd)
        else:
            act = sf.pf.fingering.strike
            stk_hand, stk_digit = DAnnotation.hand_and_digit(action=act, staff=staff, hand=hand)
            hands_and_digits['strike'].append((stk_hand, stk_digit))
            act = sf.pf.fingering.release
            rel_hand, rel_digit = DAnnotation.hand_and_digit(action=act, staff=staff, hand=stk_hand)
            if not rel_digit:
                rel_hand = stk_hand
                rel_digit = stk_digit
            hands_and_digits['release'].append((rel_hand, rel_digit))
        return hands_and_digits

    @staticmethod
    def ast_to_strike_release_data(ast, staff="upper"):
        if staff not in ("upper", "lower"):
            raise Exception("Invalid input: staff must be 'upper' or 'lower'.")

        if staff == "upper":
            lines = ast.upper
            hand = ">"
        else:
            lines = ast.lower
            hand = "<"
        sr_data = []
        for line in lines:
            for score_fingering in line:
                hands_and_digits = DAnnotation.hands_and_digits_for_score_fingering(
                    sf=score_fingering, staff=staff)
                sr_data.append(hands_and_digits)
        return sr_data

    @staticmethod
    def hand_digit(digit, staff):
        """
        Determine the handed digit for the input digit string.
        If the string already has an assigned digit, just return it.
        Otherwise assign the default hand for the specified staff.
        :return: The abcDF hand digit (">2" or "<3").
        """
        if digit is None:
            return None

        handed_re = re.compile('^[<>]\d$')
        if handed_re.match(str(digit)):
            return digit

        staff_prefix = ">"
        if staff == "lower":
            staff_prefix = "<"
        handed_digit = staff_prefix + str(digit)
        return handed_digit

    @staticmethod
    def digit_hand(handed_digit):
        """
        Determine the hand specified in the handed digit.
        :return: The abcDF hand specifier (">" or "<").
        """
        handed_re = re.compile('^([<>]{1})\d$')
        mat = handed_re.match(str(handed_digit))
        hand = mat.group(1)
        if hand != "<" and hand != ">":
            raise Exception("Ill-formed handed digit: {0}".format(handed_digit))
        return hand

    @staticmethod
    def digit_only(handed_digit):
        """
        Return the digit specified in the handed digit as an integer.
        :param handed_digit:
        :return:
        """
        handed_re = re.compile('^[<>]{1}(\d)$')
        mat = handed_re.match(str(handed_digit))
        digit = mat.group(1)
        if not digit:
            raise Exception("Ill-formed handed digit: {0}".format(handed_digit))
        return int(digit)

    @staticmethod
    def strike_distance_cost(gold_handed_digit, test_handed_digit, method="hamming"):
        test_digit = DAnnotation.digit_only(test_handed_digit)
        test_hand = DAnnotation.digit_hand(test_handed_digit)
        gold_digit = DAnnotation.digit_only(gold_handed_digit)
        gold_hand = DAnnotation.digit_hand(gold_handed_digit)
        if method == "hamming":
            if gold_digit == 'x' or test_digit == 'x':
                return 0
            if test_digit != gold_digit or test_hand != gold_hand:
                return 1
            else:
                return 0

        one = str(gold_hand) + str(gold_digit)
        other = str(test_hand) + str(test_digit)
        if method == "natural":
            cost = Constant.NATURAL_EDIT_DISTANCES[(one, other)]
            return cost
        elif method == "pivot":
            cost = Constant.PIVOT_EDIT_DISTANCES[(one, other)]
            return cost
        else:
            raise Exception("Unsupported method: {0}".format(method))

    def segregated_strike_digits(self, staff="upper", hand=None):
        """
        :return: String of digits (1-5), assuming all fingerings are
                 are for the specified hand (by default, ">" or right for the
                 upper staff and "<" or left for the lower staff).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        ast = self.parse()
        return DAnnotation.ast_to_segregated_strike_digits(ast=ast, staff=staff, hand=hand)

    @staticmethod
    def abcdf_to_segregated_strike_digits(abcdf, staff="upper", hand=None):
        """
        :return: String of digits (1-5), assuming all fingerings are
                 are for the specified hand (">" or right for the
                 upper staff by default).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        ast = DAnnotation._parser.parse(abcdf)
        return DAnnotation.ast_to_segregated_strike_digits(ast=ast, staff=staff, hand=hand)

    @staticmethod
    def ast_to_handed_strike_digits(ast, staff="upper"):
        if staff not in ("upper", "lower"):
            raise Exception("Invalid input: staff must be 'upper' or 'lower'.")

        if staff == "upper":
            lines = ast.upper
            hand = ">"
        else:
            lines = ast.lower
            hand = "<"
        handed_digits = []
        for line in lines:
            for score_fingering in line:
                if score_fingering.orn is not None:
                    # We take the strike digit of the first note in the ornament
                    # and ignore the rest.
                    ped_f = score_fingering.orn.ornaments[1][0]
                elif isinstance(score_fingering.pf.fingering, str):
                    handed_digits.append('x')
                    continue
                else:
                    ped_f = score_fingering.pf
                strike = ped_f.fingering.strike
                if strike.hand and strike.hand != hand:
                    hand = strike.hand
                digit = strike.digit
                handed_digit = hand + str(digit)
                if digit == "x":
                    handed_digit = digit
                handed_digits.append(handed_digit)
        return handed_digits

    @staticmethod
    def abcdf_to_handed_strike_digits(abcdf, staff="upper"):
        """
        :return: Array of strings, each of which is either "x" or is
                 composed of a hand ("<" or ">") identifier and a digit (1-5).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        ast = DAnnotation._parser.parse(abcdf)
        return DAnnotation.ast_to_handed_strike_digits(ast=ast, staff=staff)

    def handed_strike_digits(self, staff="upper"):
        """
        :return: Array of strings, each of which is either "x" or is
                 composed of a hand ("<" or ">") identifier and a digit (1-5).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        ast = self.parse()
        return DAnnotation.ast_to_handed_strike_digits(ast=ast, staff=staff)

    def strike_release_data(self, staff="upper"):
        """
        :return: Array of strings, each of which is either "x" or is
                 composed of a hand ("<" or ">") identifier and a digit (1-5).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        ast = self.parse()
        return DAnnotation.ast_to_strike_release_data(ast=ast, staff=staff)

    def __init__(self, abcdf=None, authority=None, authority_year=None, transcriber=None,
                 transcription_date=None, abcdf_id=None, comments=''):
        self._authority = authority
        self._authority_year = authority_year
        self._transcriber = transcriber
        self._transcription_date = transcription_date
        self._ast = None
        self._abcdf = None
        if abcdf:
            self.abcdf(abcdf)
        self._abcdf_id = abcdf_id
        self._comments = comments

    def __str__(self):
        abcdf_id = 1
        if self._abcdf_id:
            abcdf_id = self._abcdf_id
        annot_str = DAnnotation.FINGERING_STR + str(abcdf_id) + ': ' + self.abcdf() + "\n"
        if self._authority:
            annot_str += DAnnotation.AUTHORITY_STR + self._authority + "\n"
            if self._authority_year:
                annot_str += '(' + self._authority_year + ')' + "\n"
        if self._transcriber:
            annot_str += DAnnotation.TRANSCRIBER_STR + self._transcriber + "\n"
        if self._transcription_date:
            annot_str += DAnnotation.TRANSCRIPTION_DATE_STR + self._transcription_date + "\n"
        if self._comments:
            annot_str += '% ' + self.comments() + "\n"
        return annot_str

    def authority(self, authority=None):
        if authority:
            self._authority = authority
        return self._authority

    def authority_year(self, authority_year=None):
        if authority_year:
            self._authority_year = authority_year
        return self._authority_year

    def transcriber(self, transcriber=None):
        if transcriber:
            self._transcriber = transcriber
        return self._transcriber

    def transcription_date(self, transcription_date=None):
        if transcription_date:
            self._transcription_date = transcription_date
        return self._transcription_date

    @staticmethod
    def _flatten(abcdf):
        flattened = abcdf.replace("&", "")
        return flattened

    def abcdf(self, abcdf=None, staff=None, flat=False):
        if abcdf:
            self._abcdf = abcdf
            self._ast = DAnnotation.abcdf_to_ast(abcdf)
        if staff == "upper":
            return self.upper_abcdf(flat=flat)
        elif staff == "lower":
            return self.lower_abcdf(flat=flat)
        if flat:
            return DAnnotation._flatten(self._abcdf)
        return self._abcdf

    def upper_abcdf(self, flat=False):
        (upper, lower) = self.abcdf().split('@')
        if flat:
            return DAnnotation._flatten(upper)
        return upper

    def lower_abcdf(self, flat=False):
        (upper, lower) = self.abcdf().split('@')
        if flat:
            return DAnnotation._flatten(lower)
        return lower

    def abcdf_id(self, abcdf_id=None):
        if abcdf_id:
            self._abcdf_id = abcdf_id
        return self._abcdf_id

    def comments(self, comments=None):
        if comments:
            self._comments = comments
        return self._comments.rstrip()

    def add_comment_line(self, comment):
        if self._comments:
            self._comments += "\n"
        self._comments += comment.rstrip()
