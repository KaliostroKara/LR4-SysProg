import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set

def parse_grammar_text(grammar_text):
    """
    Parses a grammar from its textual representation into a dictionary.
    Format:
    A -> aBc | d
    B -> e | ε
    """
    grammar = {}
    lines = grammar_text.strip().split('\n')

    for line in lines:
        left, right = line.split('->')
        left = left.strip()
        productions = right.strip().split('|')
        grammar[left] = [prod.strip() for prod in productions]

    return grammar


def ll1_parse(input_string, grammar, parsing_table):
    input_string += '$'
    stack = ['$', 'A']  # Assuming 'A' is the start symbol
    pointer = 0
    while len(stack) > 0:
        top = stack[-1]
        current_input = input_string[pointer]

        if top == current_input:
            stack.pop()
            pointer += 1
        elif top in parsing_table and current_input in parsing_table[top]:
            production = parsing_table[top][current_input]
            stack.pop()
            for symbol in reversed(production[0]):
                if symbol != 'ε':
                    stack.append(symbol)
        else:
            print(f"Failed: Expected one of {list(parsing_table[top].keys()) if top in parsing_table else []}, found {current_input}")
            return False
    return pointer == len(input_string)



def find_epsilon_producing_non_terminal(grammar):
    """
    Finds the non-terminals which produce epsilon in the given grammar.
    """
    epsilon_producers = set()

    # Initial scan for direct producers
    for non_terminal, productions in grammar.items():
        for production in productions:
            if production == 'ε' or production == '':
                epsilon_producers.add(non_terminal)

    # Iterative scan for indirect producers
    size_before = 0
    while size_before != len(epsilon_producers):
        size_before = len(epsilon_producers)

        for non_terminal, productions in grammar.items():
            for production in productions:
                if all(symbol in epsilon_producers for symbol in production):
                    epsilon_producers.add(non_terminal)
                    break

    return epsilon_producers


# Test the functions
# sample_grammar_text = """
# A -> aBc | d
# B -> e | ε
# """
#
# grammar = parse_grammar_text(sample_grammar_text)
# epsilon_producers = find_epsilon_producing_non_terminal(grammar)
#
# grammar, epsilon_producers


def compute_first(grammar, epsilon_producers):
    """
    Computes the FIRST set for each non-terminal in the grammar.
    """
    first = {non_terminal: set() for non_terminal in grammar.keys()}

    # Initialization for terminals
    for non_terminal, productions in grammar.items():
        for production in productions:
            if production[0] not in grammar.keys():  # if it's a terminal
                first[non_terminal].add(production[0])
            elif production == 'ε':
                first[non_terminal].add('ε')

    # Iteratively compute FIRST sets
    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for symbol in production:
                    # If symbol is terminal
                    if symbol not in grammar.keys():
                        break
                    # If symbol is non-terminal
                    first[non_terminal].update(first[symbol])
                    if symbol not in epsilon_producers:
                        break
                    if symbol == production[-1] and symbol in epsilon_producers:
                        first[non_terminal].add('ε')

                new_items_count = len(first[non_terminal])
                if new_items_count > len(first[non_terminal]):
                    changed = True

    return first


def compute_follow(grammar, first, epsilon_producers):
    """
    Computes the FOLLOW set for each non-terminal in the grammar.
    """
    follow = {non_terminal: set() for non_terminal in grammar.keys()}
    follow[next(iter(grammar.keys()))].add('$')  # Add $ to the start symbol

    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar.keys():  # If symbol is a non-terminal
                        # All but the last symbol
                        if i < len(production) - 1:
                            next_symbol = production[i + 1]
                            if next_symbol in grammar.keys():
                                follow[symbol].update(first[next_symbol] - {'ε'})
                                if next_symbol in epsilon_producers:
                                    follow[symbol].update(follow[non_terminal])
                            else:
                                follow[symbol].add(next_symbol)
                        # If the symbol is the last one in the production
                        else:
                            follow[symbol].update(follow[non_terminal])

                new_items_count = len(follow[non_terminal])
                if new_items_count > len(follow[non_terminal]):
                    changed = True

    return follow


# # Test the functions
# first_sets = compute_first(grammar, epsilon_producers)
# follow_sets = compute_follow(grammar, first_sets, epsilon_producers)
#
# first_sets, follow_sets


def build_parsing_tabl(grammar, first, follow, epsilon_producers):
    """
    Builds the LL(1) parsing table.
    """
    table = {}
    for non_terminal in grammar.keys():
        table[non_terminal] = {}

        for production in grammar[non_terminal]:
            first_symbols = set()

            for symbol in production:
                if symbol in grammar.keys():
                    first_symbols.update(first[symbol])
                    if symbol not in epsilon_producers:
                        break
                else:
                    first_symbols.add(symbol)
                    break

            for symbol in first_symbols:
                if symbol != 'ε':
                    if symbol not in table[non_terminal]:
                        table[non_terminal][symbol] = []
                    table[non_terminal][symbol].append(production)

            if 'ε' in first_symbols or production == 'ε':
                for symbol in follow[non_terminal]:
                    if symbol not in table[non_terminal]:
                        table[non_terminal][symbol] = []
                    table[non_terminal][symbol].append(production)

    return table


# Test the function
# parsing_table = build_parsing_tabl(grammar, first_sets, follow_sets, epsilon_producers)
# parsing_table


def ast_visual(grammar, parsing_table, input_string):
    """
    Builds and visualizes the Abstract Syntax Tree (AST) for the given input string using the provided grammar
    and parsing table.
    """
    stack = ['$']
    start_symbol = next(iter(grammar.keys()))
    stack.append(start_symbol)

    graph = nx.DiGraph()
    graph.add_node(start_symbol)

    prev_node = start_symbol
    idx = 0
    while stack:
        top = stack[-1]

        if idx >= len(input_string):
            break

        if top in parsing_table and input_string[idx] in parsing_table[top]:
            production = parsing_table[top][input_string[idx]][0]
            stack.pop()

            for symbol in reversed(production):
                if symbol != 'ε':
                    stack.append(symbol)
                    graph.add_node(symbol)
                    graph.add_edge(prev_node, symbol)
                    prev_node = symbol

            if production == 'ε':
                epsilon_node = 'ε' + str(idx)
                graph.add_node(epsilon_node, label='ε')
                graph.add_edge(prev_node, epsilon_node)
                prev_node = epsilon_node

        else:
            if top == input_string[idx]:
                stack.pop()
                idx += 1
            else:
                raise ValueError(f"Parsing error: Unexpected token '{input_string[idx]}' at position {idx}.")

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=1500, node_color='pink')
    plt.title("Abstract Syntax Tree (AST)")
    plt.show()


# Test the function with the sample input string "aec"
#ast_visual(grammar, parsing_table, "aec")


# 1. parse_grammar_text_k
def parse_grammar_text_k(text: str) -> Dict[str, List[List[str]]]:
    grammar = {}
    for line in text.strip().split('\n'):
        left, right = line.split('->')
        productions = [prod.split() for prod in right.split('|')]
        grammar[left.strip()] = productions
    return grammar


# 2. llk_parse
def llk_parse(input_text: str, grammar: Dict[str, List[List[str]]], k: int) -> bool:
    table = build_parsing_tabl_k(grammar, k)
    input_tokens = input_text.split() + ["$"]  # Adding end of input symbol
    stack = [list(grammar.keys())[0], "$"]

    cursor = 0
    while stack:
        top = stack[-1]

        # If top of stack is a terminal
        if top not in grammar:
            if top == input_tokens[cursor]:
                stack.pop()
                cursor += 1
            else:
                return False
        else:
            lookahead = tuple(input_tokens[cursor:cursor + k])
            key = (top, ' '.join(lookahead))

            if key in table:
                production = table[key]
                stack.pop()
                if production != ["ε"]:
                    stack.extend(production[::-1])  # Pushing symbols of production onto stack in reverse order
            else:
                return False

    return cursor == len(input_tokens)


# 3. find_epsilon_producing_non_terminal_k
def find_epsilon_producing_non_terminal_k(grammar: Dict[str, List[List[str]]]) -> Set[str]:
    epsilon_producers = set()
    changes_made = True

    while changes_made:
        changes_made = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                if all(symbol in epsilon_producers or symbol == "ε" for symbol in production):
                    if non_terminal not in epsilon_producers:
                        epsilon_producers.add(non_terminal)
                        changes_made = True

    return epsilon_producers

# 4. compute_first_k
def compute_first_k(grammar: Dict[str, List[List[str]]], k: int) -> Dict[str, Set[str]]:
    first = {non_terminal: set() for non_terminal in grammar}
    epsilon_producers = find_epsilon_producing_non_terminal_k(grammar)

    for non_terminal, productions in grammar.items():
        for production in productions:
            if production[0] not in grammar:
                first[non_terminal].add(production[0])
            else:
                tokens = []
                for symbol in production:
                    if symbol in grammar:
                        tokens.extend(list(first[symbol]))
                    else:
                        tokens.append(symbol)
                    if symbol not in epsilon_producers:
                        break
                first[non_terminal].update(tokens[:k])

    return first


# 5. compute_follow_k
def compute_follow_k(grammar: Dict[str, List[List[str]]], k: int) -> Dict[str, Set[str]]:
    first = compute_first_k(grammar, k)
    epsilon_producers = find_epsilon_producing_non_terminal_k(grammar)

    follow = {non_terminal: set() for non_terminal in grammar}
    start_symbol = list(grammar.keys())[0]
    follow[start_symbol].add('$')

    changes_made = True
    while changes_made:
        changes_made = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar:
                        next_symbols = production[i+1:i+1+k]
                        tokens = []
                        for next_symbol in next_symbols:
                            if next_symbol in grammar:
                                tokens.extend(list(first[next_symbol]))
                            else:
                                tokens.append(next_symbol)
                            if next_symbol not in epsilon_producers:
                                break
                        if len(tokens) < k:
                            tokens.extend(list(follow[non_terminal])[:k-len(tokens)])
                        if not follow[symbol].issuperset(tokens):
                            follow[symbol].update(tokens)
                            changes_made = True

    return follow


# 6. build_parsing_tabl_k
def build_parsing_tabl_k(grammar: Dict[str, List[List[str]]], k: int) -> Dict[Tuple[str, str], List[str]]:
    table = {}
    first = compute_first_k(grammar, k)
    follow = compute_follow_k(grammar, k)
    epsilon_producers = find_epsilon_producing_non_terminal_k(grammar)

    for non_terminal, productions in grammar.items():
        for production in productions:
            first_set = set()

            j = 0
            while j < len(production) and production[j] in epsilon_producers:
                first_set.update(first[production[j]])
                j += 1

            if j < len(production) and production[j] in first:
                first_set.update(first[production[j]])

            for item in first_set:
                key = (non_terminal, ' '.join(item.split()[:k]))
                table[key] = production

            if "ε" in first[non_terminal]:
                for item in follow[non_terminal]:
                    key = (non_terminal, ' '.join(item.split()[:k]))
                    if key not in table:
                        table[key] = production

    return table


def manual_llk_parse(tokens: List[str], index: int = 0) -> bool:
    if index >= len(tokens):
        return False

    if tokens[index] == "id":
        index += 1
        index = parse_E_prime(tokens, index)
        if index == len(tokens):
            return True
    return False

def parse_E_prime(tokens: List[str], index: int) -> int:
    if index < len(tokens) and tokens[index] == "++":
        index += 1
        if index < len(tokens) and tokens[index] == "id":
            index += 1
            return parse_E_prime(tokens, index)
    return index


class RecursiveDescentParserWithAST:
    def __init__(self, input_string):
        self.input = input_string
        self.index = 0
        self.ast = nx.DiGraph()
        self.node_count = 0

    def lookahead(self):
        return self.input[self.index] if self.index < len(self.input) else None

    def consume(self, char):
        if self.lookahead() == char:
            self.index += 1
            return True
        return False

    def E(self):
        node = self.new_node('E')
        if self.T():
            child1 = self.last_node
            if self.consume('+'):
                child2 = self.new_node('+')
                if self.E():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            elif self.consume('-'):
                child2 = self.new_node('-')
                if self.E():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            else:
                self.ast.add_edge(node, child1)
                self.last_node = node
                return True
        return False

    def T(self):
        node = self.new_node('T')
        if self.F():
            child1 = self.last_node
            if self.consume('*'):
                child2 = self.new_node('*')
                if self.T():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            elif self.consume('/'):
                child2 = self.new_node('/')
                if self.T():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            else:
                self.ast.add_edge(node, child1)
                self.last_node = node
                return True
        return False

    def F(self):
        node = self.new_node('F')
        if self.consume('('):
            child1 = self.new_node('(')
            if self.E():
                child2 = self.last_node
                if self.consume(')'):
                    child3 = self.new_node(')')
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
        elif self.lookahead().isalnum():
            child1 = self.new_node(self.lookahead())
            self.index += 1
            self.ast.add_edge(node, child1)
            self.last_node = node
            return True
        return False

    def new_node(self, label):
        self.node_count += 1
        self.ast.add_node(self.node_count, label=label)
        return self.node_count

    def parse(self):
        result = self.E() and self.index == len(self.input)
        return result, self.ast if result else None


# Test the recursive descent parser
# parser = RecursiveDescentParserWithAST("a+b*c")
# success, ast = parser.parse()

# Visualize the AST
# if success:
#     pos = nx.spring_layout(ast)
#     labels = {node: ast.nodes[node]['label'] for node in ast.nodes()}
#     nx.draw(ast, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=15)
# else:
#     print("Parsing failed.")


def user_input_prompt():
    """Prompt the user for input and return the selected option."""
    options = ["1) LL(1)", "2) LL(k)", "3) RecursiveParser", "4) Exit"]
    print("\n".join(options))
    choice = input("Select an option: ")
    return choice


def main():
    while True:
        choice = user_input_prompt()

        if choice == "1":
            # LL(1) analyzer
            input_string = input("Enter the string for LL(1) analysis: ")

            # This is a placeholder grammar. You would probably have your own grammar definition somewhere.
            # grammar_text = """
            # E -> T E'
            # E' -> + T E' | ε
            # T -> F T'
            # T' -> * F T' | ε
            # F -> ( E ) | id
            # """
            grammar_text = """
            A -> aBc | d
            B -> e | ε
            """
            grammar = parse_grammar_text(grammar_text)
            epsilon_producers = find_epsilon_producing_non_terminal(grammar)
            first = compute_first(grammar, epsilon_producers)
            follow = compute_follow(grammar, first, epsilon_producers)
            parsing_table = build_parsing_tabl(grammar, first, follow, epsilon_producers)
            print("Parsing Table:")
            for non_terminal, entries in parsing_table.items():
                print(f"{non_terminal}: {entries}")
            is_parsed = ll1_parse(input_string, grammar, parsing_table)
            ast_visual(grammar, parsing_table, input_string)

            # Here you would visualize the AST or print the parsing results as required
            print(f"String is parsed: {is_parsed}")

        elif choice == "2":
            # LL(k) analyzer
            try:
                input_string = input("Enter the string for LL(k) analysis: ")

                # Updated grammar for the LL(k) analyzer
                grammar_text_k = """
                        E -> T E'
                        E' -> ++ T E' | ε
                        T -> id
                        """
                k = 2
                grammar_k = parse_grammar_text_k(grammar_text_k)
                is_parsed = manual_llk_parse(input_string.split())
                print(f"String is parsed: {is_parsed}")

            except Exception as e:
                print(f"Error occurred: {e}")



        elif choice == "3":
            # RecursiveParser logic
            input_string = input("Enter the string for Recursive Descent Parsing: ")
            # Create an instance of the RecursiveDescentParserWithAST class
            parser_instance = RecursiveDescentParserWithAST(input_string)
            # Call the parse method or other necessary methods
            success, ast = parser_instance.parse()

            if success:
                print("Parsing succeeded.")
                pos = nx.spring_layout(ast)
                labels = {node: ast.nodes[node]['label'] for node in ast.nodes()}
                nx.draw(ast, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=15)
                plt.show()

            else:
                print("Parsing failed.")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please try again.")


main()
