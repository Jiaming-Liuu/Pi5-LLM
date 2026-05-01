"""GBNF grammar that constrains the LLM to a single tool-call JSON object.

Three skills: read_status (no args), read_PDF (path), write_file (path + content).
"""

from llama_cpp import LlamaGrammar

TOOL_CALL_GBNF = r"""
root              ::= "{" ws "\"skill\"" ws ":" ws skill ws "," ws "\"args\"" ws ":" ws args ws "}"
skill             ::= "\"read_status\"" | "\"read_PDF\"" | "\"read_file\"" | "\"list_dir\"" | "\"write_file\"" | "\"done\""
args              ::= empty-args | path-args | path-content-args
empty-args        ::= "{" ws "}"
path-args         ::= "{" ws "\"path\"" ws ":" ws string ws "}"
path-content-args ::= "{" ws "\"path\"" ws ":" ws string ws "," ws "\"content\"" ws ":" ws string ws "}"
string            ::= "\"" char* "\""
char              ::= [^"\\] | "\\" ["\\/bfnrt]
ws                ::= [ \t\n]*
"""


def tool_call_grammar() -> LlamaGrammar:
    return LlamaGrammar.from_string(TOOL_CALL_GBNF, verbose=False)
