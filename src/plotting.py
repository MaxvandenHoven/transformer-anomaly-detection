import matplotlib.pyplot as plt

def plot_attention_weights(attn_weights, idx):
    sample_attn_weights = attn_weights[idx]
    num_layers = sample_attn_weights.shape[0]
    nhead = sample_attn_weights.shape[1]

    fig, axes = plt.subplots(num_layers, nhead, figsize=(nhead, num_layers+2), dpi=200)

    for layer_idx in range(num_layers):
        for head_idx in range(nhead):
            attn_weights = sample_attn_weights[layer_idx, head_idx]
            axes[layer_idx, head_idx].imshow(attn_weights)
            axes[layer_idx, head_idx].axis("off")
            axes[layer_idx, head_idx].set_title(
                f"Layer {layer_idx + 1} \nHead {head_idx + 1}",
                fontdict={
                    "fontsize": 10
                }
            )

        
    fig.show()