# ================== IMPORTS ==================
import re


# ================== QUERY PARSER ==================
class QueryParser:
    def __init__(self, query):
        self.tokens = self.tokenize(query)
        self.pos = 0

    def tokenize(self, query):
        # Splits by quotes and operators while keeping phrases
        token_pattern = r'"[^"]+"|\(|\)|AND|OR|NOT|PHRASE'
        tokens = re.findall(token_pattern, query)
        return [t.strip() for t in tokens if t.strip()]

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self):
        return self.parse_or()

    # Lowest precedence
    def parse_or(self):
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            left = {"OR": [left, right]}
        return left

    # Medium precedence
    def parse_and(self):
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_not()
            left = {"AND": [left, right]}
        return left

    # Next precedence
    def parse_not(self):
        if self.peek() == "NOT":
            self.consume()
            operand = self.parse_phrase()
            return {"NOT": operand}
        return self.parse_phrase()

    # Highest precedence (PHRASE, parentheses, terms)
    def parse_phrase(self):
        token = self.peek()
        if token == "(":
            self.consume()
            expr = self.parse_or()
            assert self.consume() == ")", "Missing closing parenthesis"
            return expr

        if token == "PHRASE":
            self.consume()
            right = self.parse_phrase()
            return {"PHRASE": right}

        term = self.consume()
        if term.startswith('"') and term.endswith('"'):
            return {"TERM": term.strip('"')}

        raise ValueError(f"Unexpected token: {token}")
