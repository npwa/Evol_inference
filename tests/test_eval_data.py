from evol_inference.eval_data import scored_tokens


def test_scored_tokens_matches_llama_perplexity_second_half():
    assert scored_tokens(2048, 1) == 1023  # positions 1024..2046 predict tokens 1025..2047
    assert scored_tokens(2048, 2) == 2046
    assert scored_tokens(512, 1) == 255
