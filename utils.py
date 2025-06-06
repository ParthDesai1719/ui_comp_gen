def clean_response(decoded):
    for tok in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|eos|>"]:
        decoded = decoded.replace(tok, "")
    decoded = decoded.strip()
    lines = decoded.splitlines()
    cleaned = [line for line in lines if not line.strip().lower().startswith("generate a")]
    return "\n".join(cleaned)
