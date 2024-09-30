que_type_models = ["iekt","qdkt","qikt","lpkt", "rkt", "akt_que", "akt_que", "iekt_que", "qikt_que", "simplekt_que", "sparsekt_que", "kqn_que", "atkt_que", "deep_irt_que", "dkvmn_que", "skvmn_que", "dkt_plus_que", "atdkt_que", "sakt_que", "saint_que"]

qikt_ab_models = ["qikt_ab_a+b+c","qikt_ab_a+b+c+irt","qikt_ab_a+b+irt","qikt_ab_a+c+irt","qikt_ab_a+irt","qikt_ab_b+irt"]

que_type_models += qikt_ab_models

## We add a dictionary to find the mapping between original model and their implemented que_type versions. 
dict_baseline_to_model = {
    'akt': 'akt_que',
    'atdkt': 'atdkt_que',
    'atkt': 'atkt_que',
    'deep_irt': 'deep_irt_que',
    'dkt': 'qdkt',
    'dkt+': 'dkt_plus_que',
    'dkvmn': 'dkvmn_que',
    'iekt': 'iekt_que',
    'kqn': 'kqn_que',
    'qdkt': 'qdkt',
    'qikt': 'qikt_que',
    'saint': 'saint_que',
    'sakt': 'sakt_que',
    'simplekt': 'simplekt_que',
    'sparsekt': 'sparsekt_que'
}

# Reversing the above dictionary
dict_model_to_baseline = {v: k for k, v in dict_baseline_to_model.items()}
