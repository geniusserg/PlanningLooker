####
# Tools for text preprocessing]
####

import spacy
nlp = spacy.load("en_core_web_sm")


# Extract roots, which childrens contain DATE or TIME tokens
def extract_date_roots(sents: list[str]):
    docs_collection = []
    for sent in sents:
        roots_collection = set()
        doc = nlp(sent)
        token_head = None
        for doctok in doc:
            tok = None
            if (doctok.ent_type_ in ["DATE", "TIME"]):
                tok = doctok
            else:
                continue
            tok_seq = []
            token_head = tok
            while True:
                filtered_tok_seq = [] # filter extra dependencies of root word
                head_tok_seq = [*[i for i in token_head.lefts if i not in tok_seq], token_head, *[i for i in token_head.rights if i not in tok_seq]]
                for j, seqtok in enumerate(head_tok_seq):
                    if (seqtok.dep_ in ["dobj", "nsubj", "nummod", "ROOT", "neg"]) or (seqtok is token_head):
                        filtered_tok_seq.append(head_tok_seq[j])
                tok_seq.extend(filtered_tok_seq)
                if (token_head.dep_ == "ROOT"):
                    break
                token_head = token_head.head
            if (token_head not in roots_collection):
                roots_collection = set(tok_seq)
            else:
                roots_collection = set.union(roots_collection, set(tok_seq))
        if (roots_collection == set()):
            roots_collection = None
        else:
            roots_collection = " ".join([t.text for t in roots_collection])
        docs_collection.append([" ".join([t.text for t in doc]), token_head, roots_collection])
    return docs_collection

