# Patch embedding from CNN feature map + Transformer + MLP head
class PatchEmbeddingFromFeatureMap(nn.Module):
    def __init__(self, in_channels, embed_dim, spatial_size=(7, 7), use_cls_token=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.use_cls_token = use_cls_token
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls_token else None
        num_patches = spatial_size[0] * spatial_size[1]
        nb_tokens = num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, nb_tokens, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return x + self.pos_embed
