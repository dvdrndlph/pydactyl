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
from tatsu import parse


class DAnnotation:
    GRAMMAR = """
        @@grammar::Calc

        sequence = upper:staff ['@' lower:staff] ;

        staff = '&'.{line};
        line = {score_fingering}* ;

        score_fingering = orn:ornamental ["/" alt_orn:ornamental]
        | pf:pedaled_fingering ['/' alt_pf:pedaled_fingering]
        | p:pedaling ['/' alt_p:pedaling]
        ;

        ornamental = ornaments:('(' {pedaled_fingering}+ ')') ;

        pedaled_fingering = soft:[soft] fingering:fingering damper:[damper] ;
        pedaling = soft:{soft}+ 'x' damper:{damper}+ ;

        fingering = strike:finger ['-' release:finger] ;
        finger = hand:[hand] digit:digit ;

        damper = '_' | '^' ;
        soft = 'p' | 'f' ;
        hand = '<' | '>' ;
        digit = '1' | '2' | '3' | '4' | '5' ;
    """

    @staticmethod
    def ast_for_abcdf(abcdf):
        ast = parse(DAnnotation.GRAMMAR, abcdf)
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

    def strike_digit_at_index(self, index, staff="upper"):
        sf = self.score_fingering_at_index(staff=staff, index=index)
        strike = sf.pf.fingering.strike
        return strike.digit

    def parse_upper(self):
        upper_abcdf = self.upper_abcdf()
        return DAnnotation.ast_for_abcdf(upper_abcdf)

    def parse_lower(self):
        lower_abcdf = self.upper_abcdf()
        return DAnnotation.ast_for_abcdf(lower_abcdf)

    def score_fingering_count(self, staff="both"):
        ast = self.parse()
        count = 0
        # Each staff is parsed into an array of lines. Each
        # line is an array of "score fingerings," or note
        # fingerings with all the trimmings.
        if staff == "upper" or staff == "both":
            lines = ast['upper']
            for line in lines:
                count += len(line)
        if staff == "lower" or staff == "both":
            # print(self.abcdf())
            lines = ast['lower']
            for line in lines:
                count += len(line)
        return count

    def segregated_strike_digits(self, staff="upper", hand=None):
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
        ast = self.parse()
        if staff == "upper":
            lines = ast.upper
        else:
            lines = ast.lower

        for line in lines:
            for score_fingering in line:
                strike = score_fingering.pf.fingering.strike
                current_hand = strike.hand
                digit = strike.digit
                if current_hand and current_hand != hand:
                    return None
                digits.append(digit)
        digit_str = "".join(digits)
        return digit_str

    def handed_strike_digits(self, staff="upper"):
        """
        :return: Array of string, each of which is either "x" or is
                 composed of a hand ("<" or ">") identifief and a digit (1-5).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        if staff not in ("upper", "lower"):
            raise Exception("Invalid input: staff must be 'upper' or 'lower'.")

        ast = self.parse()
        if staff == "upper":
            lines = ast.upper
            hand = ">"
        else:
            lines = ast.lower
            hand = "<"

        handed_digits = []
        for line in lines:
            for score_fingering in line:
                strike = score_fingering.pf.fingering.strike
                if strike.hand and strike.hand != hand:
                    hand = strike.hand
                digit = strike.digit
                handed_digit = hand + str(digit)
                handed_digits.append(handed_digit)
        return handed_digits

    def __init__(self, abcdf=None):
        self._authority = None
        self._authority_year = None
        self._transcriber = None
        self._transcription_date = None
        self._ast = None
        self._abcdf = None
        if abcdf:
            self.abcdf(abcdf)
        self._abcdf_id = None
        self._comments = ''

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
            self._ast = DAnnotation.ast_for_abcdf(abcdf)
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
        self._comments += comment + "\n"
