import hf_jacobian as hj

model, tok = hj.load("gpt2")
hj.print_model(model)

ids  = hj.tokenize(tok, "Hello world")
jacs = hj.position_jacobians(model, ids, layer_idx=0, sublayer="attn")

print(f"\nresult: {jacs.shape}  — jacs[b, pos, i, j] = d(output[b,pos,i]) / d(input[b,pos,j])")
